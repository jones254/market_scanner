"""
Multi-timeframe signal engine for the Gold Scalper.

Architecture
------------
The signal is driven by the **macro composite** (the 7-market weighted
score from `scoring.composite_score`), not by an ad-hoc DXY/VIX/SP500
formula.  This keeps the signal engine consistent with the rest of
the system: whatever the composite says, the signal uses.

Two timeframes are evaluated: 15m and 1h.  Each timeframe runs the
full multi-market engine, producing a composite score in [-100, +100]
and a 5-bucket classification (Strong Bullish → Strong Bearish).

BUY trigger
    1. 15m composite > 1h composite   (momentum building, not fading)
    2. 15m forecast = Bullish on short AND medium AND long
    3. 15m composite > 0   (sanity guard: don't buy when macro is bearish)
    4. Probability > 50%

SELL trigger
    1. 15m composite < 1h composite   (downward momentum)
    2. 15m forecast = Bearish on all three horizons
    3. 15m composite < 0
    4. Probability > 50%

Entry / SL / TP
    When a BUY or SELL fires, we also compute:
    - Entry: pullback to slow EMA on the 15m chart
    - Stop: 1.5× ATR below/above entry
    - Target: 3.0× ATR above/below entry

Chart snapshots use matplotlib (no kaleido dependency) so they
work on any environment.
"""

from __future__ import annotations
import io
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

try:
    from .indicators import ema, atr
    from .scoring import composite_score, _classify
except ImportError:
    from indicators import ema, atr
    from scoring import composite_score, _classify


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class SignalConfig:
    # Buy / sell composite thresholds (in composite-score units, -100..+100)
    min_composite_buy: float = 0.0       # 15m composite must be > this for BUY
    min_composite_sell: float = 0.0      # 15m composite must be < this for SELL

    # Momentum filter: 15m composite must be >= 1h composite (BUY) by this much
    momentum_gap_required: bool = True
    momentum_min_gap: float = 0.0         # points of composite-score difference

    # Forecast alignment: how many of the 3 forecasts must agree
    # 3 = strict (all three Bullish/Bearish), 2 = medium, 1 = lenient
    forecast_alignment_required: int = 3

    # Minimum probability to fire
    min_probability: float = 0.50

    # Cooldown between successive signals (in 1h bars)
    cooldown_bars: int = 6

    # Entry / SL / TP parameters
    k_sl: float = 1.5
    k_tp: float = 3.0
    pullback_atr_frac: float = 0.5


# -----------------------------------------------------------------------------
# Result
# -----------------------------------------------------------------------------
@dataclass
class SignalState:
    """Persists between Streamlit reruns (in session_state)."""
    last_signal: str = "NOACTION"
    last_probability: float = 0.0
    last_check_ts: object = None
    last_15m_composite: float = 0.0
    last_1h_composite: float = 0.0
    last_price: float = 0.0
    history: list = None

    def __post_init__(self):
        if self.history is None:
            self.history = []


# -----------------------------------------------------------------------------
@dataclass
class SignalResult:
    signal: str                 # BUY / SELL / WAITBUY / WAITSELL / NOACTION
    probability: float
    composite_15m: float
    composite_1h: float
    forecast_15m: Dict[str, str]    # horizon -> label
    forecast_1h:  Dict[str, str]
    last_price: float
    momentum_ok: bool
    forecast_agreement: int
    reasons: List[str]
    # Trade setup (entry/SL/TP) — only present on BUY/SELL
    entry: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    risk_reward: Optional[float] = None


# -----------------------------------------------------------------------------
# Forecast classification helper
# -----------------------------------------------------------------------------
def _composite_to_forecast_label(score: float) -> str:
    """Map a composite score to a 3-bucket forecast label."""
    if score >  40: return "Bullish"
    if score < -40: return "Bearish"
    return "Neutral"


def _horizon_forecast_from_composite(close: pd.Series, ema_fast: int, ema_slow: int, rsi_len: int) -> str:
    """
    Build a per-horizon forecast label by running a small trend+momentum
    blend on gold's own close.  We re-use the same shape as the main
    `scoring.forecast_score` but as a single-label classifier.
    """
    if len(close) < max(ema_slow, rsi_len) + 5:
        return "Neutral"
    e_fast = ema(close, ema_fast).iloc[-1]
    e_slow = ema(close, ema_slow).iloc[-1]
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(com=rsi_len - 1, adjust=False, min_periods=rsi_len).mean().iloc[-1]
    avg_loss = loss.ewm(com=rsi_len - 1, adjust=False, min_periods=rsi_len).mean().iloc[-1]
    if avg_loss == 0 or np.isnan(avg_loss):
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - 100.0 / (1.0 + rs)
    t_score = 100.0 if e_fast > e_slow else (-100.0 if e_fast < e_slow else 0.0)
    pos = max(0.0, (rsi - 60.0) * 2.5)
    neg = -max(0.0, (40.0 - rsi) * 2.5)
    soft = (rsi - 50.0) * 0.5
    m_score = pos if rsi > 60 else (neg if rsi < 40 else soft)
    raw = t_score * 0.70 + m_score * 0.30
    if   raw >  40: return "Bullish"
    if   raw < -40: return "Bearish"
    return "Neutral"


# -----------------------------------------------------------------------------
# Probability estimate
# -----------------------------------------------------------------------------
def probability(macro_15m: float, macro_1h: float, agreement: int) -> float:
    """
    Confidence in [0.50, 0.95].

      - magnitude: how strong the 15m composite is (away from 0)
      - momentum:  how much bigger 15m is vs 1h
      - agreement: how many of 3 forecasts agree
    """
    mag = min(abs(macro_15m), 80.0) / 80.0          # 0..1
    gap = abs(macro_15m - macro_1h)
    mom = min(gap, 30.0) / 30.0                     # 0..1
    agr = agreement / 3.0                            # 0..1

    score = 0.50 * mag + 0.30 * mom + 0.20 * agr
    return 0.50 + score * 0.45                       # 0.50..0.95


# -----------------------------------------------------------------------------
# Entry / SL / TP computation
# -----------------------------------------------------------------------------
def _compute_entry_sltp(
    data_15m: Dict[str, pd.DataFrame],
    side: int,
    cfg: SignalConfig,
) -> Tuple[Optional[float], Optional[float], Optional[float], float]:
    """
    Returns (entry, stop, target, atr_value) using the 15m gold chart.
    Entry = pullback to the slow EMA, capped at the ATR-pulled-back price.
    Stop = entry ± k_sl * ATR, target = entry ± k_tp * ATR.
    """
    if "gold" not in data_15m or len(data_15m["gold"]) < 30:
        return None, None, None, 0.0
    gold = data_15m["gold"]
    atr_series = atr(gold["High"], gold["Low"], gold["Close"], 14)
    a = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0
    if a <= 0:
        return None, None, None, 0.0
    price = float(gold["Close"].iloc[-1])
    ema_slow = float(ema(gold["Close"], 50).iloc[-1]) if len(gold) >= 50 else price

    pb_price  = price - side * cfg.pullback_atr_frac * a
    # Cap at the slow EMA so the limit is structural
    if side == 1 and pb_price < ema_slow:
        pb_price = ema_slow
    if side == -1 and pb_price > ema_slow:
        pb_price = ema_slow

    entry = pb_price
    stop  = entry - side * cfg.k_sl * a
    target= entry + side * cfg.k_tp * a
    return entry, stop, target, a


# -----------------------------------------------------------------------------
# Per-horizon entry / SL / TP (short = 15m, medium = 1h, long = 1d)
# -----------------------------------------------------------------------------
def horizon_entry_sltp(
    gold_df: Optional[pd.DataFrame],
    side: int,
    cfg: SignalConfig,
    horizon: str = "short",
) -> Dict[str, Any]:
    """
    Compute entry / stop / target on the gold chart for a given horizon.
    Returns a dict with `entry`, `stop`, `target`, `atr`, `ema_slow`,
    `last_price`, `label` so the UI can render per-horizon cards.

    horizon: "short" (15m), "medium" (1h), "long" (1d)
    """
    out = {
        "entry": None, "stop": None, "target": None, "atr": 0.0,
        "ema_slow": None, "ema_fast": None, "last_price": None,
        "score": 0.0, "label": "—", "conf": 0.0,
    }
    if gold_df is None or len(gold_df) < 30:
        return out
    close = gold_df["Close"]
    high  = gold_df["High"]
    low   = gold_df["Low"]
    # Use longer ATR/EMA for longer horizons, so the SL/TP scales sensibly
    horizon_params = {
        "short":  {"ema_fast": 10, "ema_slow": 20, "atr": 14},
        "medium": {"ema_fast": 20, "ema_slow": 50, "atr": 14},
        "long":   {"ema_fast": 50, "ema_slow": 200, "atr": 14},
    }
    p = horizon_params.get(horizon, horizon_params["short"])
    a_series = atr(high, low, close, p["atr"])
    a = float(a_series.iloc[-1]) if not pd.isna(a_series.iloc[-1]) else 0.0
    ema_fast_s = ema(close, p["ema_fast"])
    ema_slow_s = ema(close, p["ema_slow"])
    ema_fast = float(ema_fast_s.iloc[-1]) if not pd.isna(ema_fast_s.iloc[-1]) else None
    ema_slow = float(ema_slow_s.iloc[-1]) if not pd.isna(ema_slow_s.iloc[-1]) else None
    price    = float(close.iloc[-1])
    out["atr"] = a
    out["ema_fast"] = ema_fast
    out["ema_slow"] = ema_slow
    out["last_price"] = price
    if a <= 0 or ema_slow is None:
        return out
    pb_price = price - side * cfg.pullback_atr_frac * a
    if side == 1 and pb_price < ema_slow:
        pb_price = ema_slow
    if side == -1 and pb_price > ema_slow:
        pb_price = ema_slow
    out["entry"]  = pb_price
    out["stop"]   = pb_price - side * cfg.k_sl * a
    out["target"] = pb_price + side * cfg.k_tp * a
    return out


# -----------------------------------------------------------------------------
# Main evaluation
# -----------------------------------------------------------------------------
def evaluate(
    data_15m: Dict[str, pd.DataFrame],
    data_1h:  Dict[str, pd.DataFrame],
    config,
    cfg: SignalConfig = SignalConfig(),
) -> SignalResult:
    """
    Run the multi-TF macro-composite signal engine.
    `data_15m` and `data_1h` are the full 7-market dicts on their
    respective timeframes (so the macro composite can be computed).
    """
    # ---- 1. Compute the macro composite on each timeframe --------------
    try:
        res_15m = composite_score(data_15m, config)
        macro_15m = float(res_15m.composite.iloc[-1])
    except Exception:
        return _noaction("15m composite failed")
    try:
        res_1h  = composite_score(data_1h, config)
        macro_1h = float(res_1h.composite.iloc[-1])
    except Exception:
        return _noaction("1h composite failed")

    if "gold" not in data_1h or len(data_1h["gold"]) == 0:
        return _noaction("no gold data")
    last_price = float(data_1h["gold"]["Close"].iloc[-1])

    # ---- 2. Per-horizon forecast on the 15m chart ------------------------
    forecast_15m = {
        "short":  _horizon_forecast_from_composite(data_15m["gold"]["Close"], 10, 20, 7),
        "medium": _horizon_forecast_from_composite(data_15m["gold"]["Close"], 20, 50, 14),
        "long":   _horizon_forecast_from_composite(data_15m["gold"]["Close"], 50, 200, 21),
    }
    forecast_1h = {
        "short":  _horizon_forecast_from_composite(data_1h["gold"]["Close"], 10, 20, 7),
        "medium": _horizon_forecast_from_composite(data_1h["gold"]["Close"], 20, 50, 14),
        "long":   _horizon_forecast_from_composite(data_1h["gold"]["Close"], 50, 200, 21),
    }

    # ---- 3. BUY / SELL / WAIT logic -------------------------------------
    reasons: List[str] = []

    momentum_buy_ok  = (not cfg.momentum_gap_required) or (macro_15m > macro_1h + cfg.momentum_min_gap)
    momentum_sell_ok = (not cfg.momentum_gap_required) or (macro_15m < macro_1h - cfg.momentum_min_gap)

    bullish_15m_count = sum(1 for v in forecast_15m.values() if v == "Bullish")
    bearish_15m_count = sum(1 for v in forecast_15m.values() if v == "Bearish")

    buy_aligned  = bullish_15m_count >= cfg.forecast_alignment_required
    sell_aligned = bearish_15m_count >= cfg.forecast_alignment_required

    buy_ok  = (macro_15m >  cfg.min_composite_buy)  and momentum_buy_ok  and buy_aligned
    sell_ok = (macro_15m < -cfg.min_composite_sell) and momentum_sell_ok and sell_aligned

    agreement = bullish_15m_count if buy_ok else (bearish_15m_count if sell_ok else 0)
    prob = probability(macro_15m, macro_1h, agreement)

    if buy_ok and prob >= cfg.min_probability:
        entry, stop, target, _ = _compute_entry_sltp(data_15m, 1, cfg)
        rr = (target - entry) / max(entry - stop, 1e-9) if entry and stop and target else None
        return SignalResult(
            signal="BUY",
            probability=prob,
            composite_15m=macro_15m,
            composite_1h=macro_1h,
            forecast_15m=forecast_15m,
            forecast_1h=forecast_1h,
            last_price=last_price,
            momentum_ok=momentum_buy_ok,
            forecast_agreement=bullish_15m_count,
            reasons=["all BUY conditions met"],
            entry=entry, stop=stop, target=target, risk_reward=rr,
        )

    if sell_ok and prob >= cfg.min_probability:
        entry, stop, target, _ = _compute_entry_sltp(data_15m, -1, cfg)
        rr = (entry - target) / max(stop - entry, 1e-9) if entry and stop and target else None
        return SignalResult(
            signal="SELL",
            probability=prob,
            composite_15m=macro_15m,
            composite_1h=macro_1h,
            forecast_15m=forecast_15m,
            forecast_1h=forecast_1h,
            last_price=last_price,
            momentum_ok=momentum_sell_ok,
            forecast_agreement=bearish_15m_count,
            reasons=["all SELL conditions met"],
            entry=entry, stop=stop, target=target, risk_reward=rr,
        )

    # WAITBUY: 15m macro > 0 and 1h macro > 0 but momentum not aligned
    if macro_15m > 0 and macro_1h > 0 and not buy_ok:
        reasons_wait = []
        if not momentum_buy_ok:
            reasons_wait.append("momentum not building (15m ≤ 1h)")
        if not buy_aligned:
            reasons_wait.append(f"only {bullish_15m_count}/3 forecasts bullish")
        if not reasons_wait:
            reasons_wait.append("probability below threshold")
        return SignalResult(
            signal="WAITBUY",
            probability=probability(macro_15m, macro_1h, bullish_15m_count),
            composite_15m=macro_15m,
            composite_1h=macro_1h,
            forecast_15m=forecast_15m,
            forecast_1h=forecast_1h,
            last_price=last_price,
            momentum_ok=momentum_buy_ok,
            forecast_agreement=bullish_15m_count,
            reasons=reasons_wait,
        )
    if macro_15m < 0 and macro_1h < 0 and not sell_ok:
        reasons_wait = []
        if not momentum_sell_ok:
            reasons_wait.append("momentum not building (15m ≥ 1h)")
        if not sell_aligned:
            reasons_wait.append(f"only {bearish_15m_count}/3 forecasts bearish")
        if not reasons_wait:
            reasons_wait.append("probability below threshold")
        return SignalResult(
            signal="WAITSELL",
            probability=probability(macro_15m, macro_1h, bearish_15m_count),
            composite_15m=macro_15m,
            composite_1h=macro_1h,
            forecast_15m=forecast_15m,
            forecast_1h=forecast_1h,
            last_price=last_price,
            momentum_ok=momentum_sell_ok,
            forecast_agreement=bearish_15m_count,
            reasons=reasons_wait,
        )

    return SignalResult(
        signal="NOACTION",
        probability=0.0,
        composite_15m=macro_15m,
        composite_1h=macro_1h,
        forecast_15m=forecast_15m,
        forecast_1h=forecast_1h,
        last_price=last_price,
        momentum_ok=False,
        forecast_agreement=0,
        reasons=[f"composite_15m={macro_15m:+.1f}, composite_1h={macro_1h:+.1f}, "
                 f"bullish={bullish_15m_count}/3, bearish={bearish_15m_count}/3"],
    )


def _noaction(reason: str) -> SignalResult:
    return SignalResult(
        signal="NOACTION",
        probability=0.0,
        composite_15m=0.0, composite_1h=0.0,
        forecast_15m={}, forecast_1h={},
        last_price=0.0,
        momentum_ok=False, forecast_agreement=0,
        reasons=[reason],
    )


# -----------------------------------------------------------------------------
# Telegram message formatter
# -----------------------------------------------------------------------------
def format_signal_message(res: SignalResult) -> Tuple[str, str]:
    """Return (caption, parse_mode)."""
    if res.signal == "BUY":
        header = f"🟢 *BUY SIGNAL* — Gold (prob {res.probability*100:.0f}%)"
    elif res.signal == "SELL":
        header = f"🔴 *SELL SIGNAL* — Gold (prob {res.probability*100:.0f}%)"
    elif res.signal == "WAITBUY":
        header = f"⏳ *Waiting for BUY* — bias {res.probability*100:.0f}%"
    elif res.signal == "WAITSELL":
        header = f"⏳ *Waiting for SELL* — bias {res.probability*100:.0f}%"
    else:
        header = "💤 *No action*"

    body = (
        f"\n\n"
        f"• Last: `{res.last_price:,.2f}`\n"
        f"• 15m composite: `{res.composite_15m:+.1f}`\n"
        f"• 1h  composite: `{res.composite_1h:+.1f}`\n"
        f"• Momentum: {'building' if res.momentum_ok else 'fading'}\n"
        f"• 15m forecasts: "
        f"short `{res.forecast_15m.get('short','?')}`, "
        f"medium `{res.forecast_15m.get('medium','?')}`, "
        f"long `{res.forecast_15m.get('long','?')}`\n"
    )
    if res.entry is not None:
        body += (
            f"\n📍 *Setup:*\n"
            f"  Entry: `{res.entry:,.2f}`\n"
            f"  Stop:  `{res.stop:,.2f}`\n"
            f"  Target:`{res.target:,.2f}`\n"
            f"  R:R = {res.risk_reward:.2f}\n"
        )
    if res.reasons and res.signal in ("NOACTION", "WAITBUY", "WAITSELL"):
        body += f"\n_Reason:_ {', '.join(res.reasons)}"
    return header + body, "Markdown"


# -----------------------------------------------------------------------------
# Chart snapshot — matplotlib-based (no kaleido dependency)
# -----------------------------------------------------------------------------
def render_chart_snapshot(
    data_1h: Dict[str, pd.DataFrame],
    data_15m: Dict[str, pd.DataFrame],
    res: SignalResult,
    last_n_bars_1h: int = 60,
    last_n_bars_15m: int = 96,
) -> Optional[bytes]:
    """
    Render a 2-panel chart (1h + 15m gold) to PNG bytes.
    Uses matplotlib — no system-level dependencies required.
    Returns None if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if "gold" not in data_1h or "gold" not in data_15m:
        return None

    gold_1h  = data_1h["gold"]["Close"].tail(last_n_bars_1h)
    gold_15m = data_15m["gold"]["Close"].tail(last_n_bars_15m)

    color = {"BUY": "#2E8B57", "SELL": "#B22222", "WAITBUY": "#90EE90",
             "WAITSELL": "#FFA07A", "NOACTION": "#888"}.get(res.signal, "#888")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), dpi=110,
                                    gridspec_kw={"hspace": 0.35})
    # 1h panel
    ax1.plot(gold_1h.index, gold_1h.values, color="#DAA520", linewidth=1.8, label="1h")
    ax1.set_title("Gold — 1h", fontsize=11, loc="left")
    ax1.grid(True, alpha=0.3)
    if res.entry is not None:
        ax1.axhline(res.entry,   color="#1f77b4", linestyle="--", linewidth=1.2, label=f"Entry {res.entry:.0f}")
        ax1.axhline(res.stop,    color="#B22222", linestyle="--", linewidth=1.2, label=f"Stop {res.stop:.0f}")
        ax1.axhline(res.target,  color="#006400", linestyle="--", linewidth=1.2, label=f"Target {res.target:.0f}")
    ax1.legend(loc="upper left", fontsize=8)

    # 15m panel
    ax2.plot(gold_15m.index, gold_15m.values, color="#888", linewidth=1.4, label="15m")
    ax2.set_title("Gold — 15m", fontsize=11, loc="left")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8)

    fig.suptitle(f"{res.signal} · p={res.probability*100:.0f}% · "
                 f"15m {res.composite_15m:+.1f} / 1h {res.composite_1h:+.1f}",
                 color=color, fontsize=12, fontweight="bold")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
