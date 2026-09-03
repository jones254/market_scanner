"""
Backtest engine for the macro-composite signal engine.

Walks historical 15m / 1h bars and asks: at each bar, would the
signal engine have fired BUY or SELL?  Then measures the forward
return of gold over a configurable holding window.

Walk-forward rules (no look-ahead)
----------------------------------
* At each step T, we use bars up to T for the 15m / 1h lookbacks.
* The 1h bar at T is the "entry" and we measure forward return on
  the 1h series from T to T+holding.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from .config import Config
    from .signals import SignalConfig, evaluate as signal_evaluate
except ImportError:
    from config import Config
    from signals import SignalConfig, evaluate as signal_evaluate


@dataclass
class SignalBacktestConfig:
    start: str = "2024-01-01"
    end:   str = "2025-12-31"   # past date; date_input will clamp against today
    holding_bars: int = 12
    cooldown_bars: int = 6
    step_bars: int = 1
    max_triggers: int = 5000


@dataclass
class SignalBacktestResult:
    triggers: pd.DataFrame
    equity_curve: pd.Series
    benchmark: pd.Series
    metrics: Dict[str, float]
    bucket_stats: pd.DataFrame
    coverage_warning: Optional[str] = None


def run_signal_backtest(
    data_15m: Dict[str, pd.DataFrame],
    data_1h:  Dict[str, pd.DataFrame],
    config:   Config,
    sig_cfg:  SignalConfig = SignalConfig(),
    bt_cfg:   SignalBacktestConfig = SignalBacktestConfig(),
    initial_capital: float = 10_000.0,
) -> SignalBacktestResult:
    if "gold" not in data_1h or len(data_1h["gold"]) < 100:
        raise RuntimeError("Not enough 1h data to backtest.")

    gold_1h = data_1h["gold"]["Close"].copy()
    start_ts = pd.Timestamp(bt_cfg.start)
    end_ts   = pd.Timestamp(bt_cfg.end)
    if start_ts > gold_1h.index[-1] or end_ts < gold_1h.index[0]:
        raise RuntimeError(
            f"Backtest window {bt_cfg.start} → {bt_cfg.end} outside available data."
        )
    start_idx = max(gold_1h.index.get_indexer([start_ts], method="nearest")[0], 0)
    end_idx   = min(gold_1h.index.get_indexer([end_ts], method="nearest")[0],
                    len(gold_1h) - 1)
    if end_idx - start_idx < 50:
        raise RuntimeError("Backtest window too narrow.")

    coverage_warning = None
    if data_15m and "gold" in data_15m and len(data_15m["gold"]) > 0:
        first_15m = data_15m["gold"].index[0]
        if start_ts < first_15m:
            coverage_warning = (
                f"15m data only available from {first_15m.date()}; "
                f"earlier bars use only 1h. yfinance 15m has a 30-day limit."
            )

    triggers: List[Dict] = []
    last_trigger_idx = -10_000
    n_evaluated = 0

    d15_close = {mkt: df["Close"] for mkt, df in data_15m.items()
                 if "Close" in df.columns}

    for i in range(start_idx, end_idx - bt_cfg.holding_bars, bt_cfg.step_bars):
        n_evaluated += 1
        if n_evaluated > bt_cfg.max_triggers:
            break
        if (i - last_trigger_idx) < bt_cfg.cooldown_bars:
            continue

        ts = gold_1h.index[i]

        d1_slice: Dict[str, pd.DataFrame] = {}
        for mkt, df in data_1h.items():
            if "Close" not in df.columns:
                continue
            sliced = df.loc[df.index <= ts]
            if len(sliced) >= 30:
                d1_slice[mkt] = sliced

        d15_slice: Dict[str, pd.DataFrame] = {}
        for mkt, s in d15_close.items():
            sliced = s.loc[s.index <= ts]
            if len(sliced) >= 30:
                d15_slice[mkt] = pd.DataFrame({"Close": sliced})

        if not d1_slice or not d15_slice:
            continue

        try:
            res = signal_evaluate(d15_slice, d1_slice, config, sig_cfg)
        except Exception:
            continue

        if res.signal not in ("BUY", "SELL"):
            continue

        entry_price = float(gold_1h.iloc[i])
        exit_price  = float(gold_1h.iloc[i + bt_cfg.holding_bars])
        if entry_price <= 0 or not np.isfinite(exit_price):
            continue
        side = 1 if res.signal == "BUY" else -1
        fwd_ret = (exit_price - entry_price) / entry_price
        signed_ret = side * fwd_ret

        triggers.append({
            "entry_time":    gold_1h.index[i],
            "exit_time":     gold_1h.index[i + bt_cfg.holding_bars],
            "side":          "long" if side == 1 else "short",
            "signal":        res.signal,
            "entry":         entry_price,
            "exit":          exit_price,
            "fwd_return":    fwd_ret,
            "signed_return": signed_ret,
            "composite_15m": res.composite_15m,
            "composite_1h":  res.composite_1h,
            "probability":   res.probability,
        })
        last_trigger_idx = i

    if not triggers:
        return SignalBacktestResult(
            triggers=pd.DataFrame(),
            equity_curve=pd.Series(dtype=float),
            benchmark=pd.Series(dtype=float),
            metrics={},
            bucket_stats=pd.DataFrame(),
            coverage_warning=coverage_warning,
        )

    trig_df = pd.DataFrame(triggers).set_index("entry_time").sort_index()

    bench_ret   = gold_1h.pct_change().fillna(0.0)
    strat_daily = bench_ret.copy()
    for t in triggers:
        i_entry = gold_1h.index.get_loc(t["entry_time"])
        i_exit  = gold_1h.index.get_loc(t["exit_time"])
        strat_daily.iloc[i_exit] = (1 + strat_daily.iloc[i_exit]) * (1 + t["signed_return"]) - 1

    eq_curve = (1 + strat_daily).cumprod() * initial_capital
    bench    = (1 + bench_ret).cumprod() * initial_capital

    metrics = _compute_metrics(trig_df, eq_curve, initial_capital)
    bucket_stats = _bucket_stats(trig_df)

    return SignalBacktestResult(
        triggers=trig_df,
        equity_curve=eq_curve,
        benchmark=bench,
        metrics=metrics,
        bucket_stats=bucket_stats,
        coverage_warning=coverage_warning,
    )


def _compute_metrics(trig_df, eq, initial_capital) -> Dict[str, float]:
    if trig_df.empty:
        return {
            "n_triggers": 0, "hit_rate_pct": 0.0, "expectancy": 0.0,
            "avg_R_pct": 0.0, "profit_factor": 0.0, "max_dd_pct": 0.0,
            "total_return_pct": 0.0, "annualized_pct": 0.0,
        }
    r = trig_df["signed_return"]
    wins = r[r > 0]; losses = r[r < 0]
    gp = wins.sum() if len(wins) else 0.0
    gl = -losses.sum() if len(losses) else 0.0
    pf = float(gp / gl) if gl > 0 else float("inf")
    peak = eq.cummax(); dd = (eq - peak) / peak
    max_dd = float(dd.min() * 100) if len(dd) else 0.0
    final = float(eq.iloc[-1]) if len(eq) else initial_capital
    total = (final / initial_capital - 1) * 100
    days = (eq.index[-1] - eq.index[0]).days if len(eq) > 1 else 0
    ann = ((final / initial_capital) ** (365 / max(days, 1)) - 1) * 100 if days > 0 else 0.0
    return {
        "n_triggers":        int(len(trig_df)),
        "hit_rate_pct":      round(float((r > 0).mean() * 100), 2),
        "expectancy":        round(float(r.mean() * 100), 3),
        "avg_R_pct":         round(float(r.mean() * 100), 3),
        "profit_factor":     round(pf, 2) if np.isfinite(pf) else 99.99,
        "max_dd_pct":        round(max_dd, 2),
        "total_return_pct":  round(total, 2),
        "annualized_pct":    round(ann, 2),
    }


def _bucket_stats(trig_df: pd.DataFrame) -> pd.DataFrame:
    if trig_df.empty:
        return pd.DataFrame()
    g = trig_df.groupby("signal")["signed_return"]
    out = pd.DataFrame({
        "n":           g.count(),
        "hit_rate_%":  (g.apply(lambda x: (x > 0).mean()) * 100).round(1),
        "avg_ret_%":   (g.mean() * 100).round(3),
        "total_ret_%": (g.sum() * 100).round(2),
        "max_win_%":   (g.max() * 100).round(2),
        "max_loss_%":  (g.min() * 100).round(2),
    })
    return out.reindex([b for b in ("BUY", "SELL") if b in out.index])
