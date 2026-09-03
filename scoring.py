"""
Composite scoring engine for the Gold Scalper.

Pipeline:
    1. Compute Trend / Momentum / Strength for each market
    2. Blend 40/35/25 -> asset score
    3. Invert the sign of negatively-correlated assets (DXY)
    4. Weighted sum, normalised -> composite in [-100, +100]
    5. Classify into 5 buckets (Strong Bull -> Strong Bear)
    6. Compute three forecast horizons on Gold
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    from .indicators import ema, rsi, roc
    from .config import Config, MARKET_LABELS, NEGATIVE_CORRELATIONS
except ImportError:
    from indicators import ema, rsi, roc
    from config import Config, MARKET_LABELS, NEGATIVE_CORRELATIONS


def neg_corr_for(symbol: str) -> set:
    """Return the negative-correlation set for an instrument symbol.
    Accepts either the display symbol (e.g. "EUR/USD") or the yfinance
    key (e.g. "eurusd").  Falls back to the legacy flat
    NEGATIVE_CORRELATIONS for backwards-compat.
    """
    if isinstance(NEGATIVE_CORRELATIONS, set):
        return NEGATIVE_CORRELATIONS
    if isinstance(NEGATIVE_CORRELATIONS, dict):
        # Try the display symbol first
        if symbol in NEGATIVE_CORRELATIONS:
            return NEGATIVE_CORRELATIONS[symbol]
        # Translate display -> yfinance key
        try:
            from config import INSTRUMENT_BY_SYMBOL
        except ImportError:
            from .config import INSTRUMENT_BY_SYMBOL
        mset = INSTRUMENT_BY_SYMBOL.get(symbol)
        if mset is not None:
            yf_key = mset.target
            if yf_key in NEGATIVE_CORRELATIONS:
                return NEGATIVE_CORRELATIONS[yf_key]
        return set()
    return set()


# -----------------------------------------------------------------------------
# Per-market asset score
# -----------------------------------------------------------------------------
def asset_score(close: pd.Series, periods: Dict[str, int]) -> pd.Series:
    """40% Trend + 35% Momentum + 25% Strength, clamped to [-100, +100]."""
    e1 = close.ewm(span=periods["ema_fast"], adjust=False, min_periods=periods["ema_fast"]).mean()
    e2 = close.ewm(span=periods["ema_slow"], adjust=False, min_periods=periods["ema_slow"]).mean()
    t_score = np.where(e1 > e2, 100.0, np.where(e1 < e2, -100.0, 0.0))

    r = rsi(close, periods["rsi_len"])
    pos = ((r - 60.0) * (100.0 / 40.0)).clip(0.0, 100.0)
    neg = -((40.0 - r) * (100.0 / 40.0)).clip(0.0, 100.0)
    soft = (r - 50.0) * 5.0 * 0.10
    m_score = np.where(r > 60, pos, np.where(r < 40, neg, soft))

    rc = roc(close, periods["roc_len"])
    s_score = (rc * 20.0).clip(-100.0, 100.0)

    raw = t_score * 0.40 + m_score * 0.35 + s_score * 0.25
    return pd.Series(raw, index=close.index).clip(-100.0, 100.0).fillna(0.0)


# -----------------------------------------------------------------------------
# Composite + classification
# -----------------------------------------------------------------------------
@dataclass
class ScoreResult:
    composite: pd.Series
    per_market: Dict[str, pd.Series]
    contributions: pd.DataFrame
    label: pd.Series
    confidence: pd.Series


def composite_score(
    data: Dict[str, pd.DataFrame],
    config: Config,
    neg_corr: set = None,
) -> ScoreResult:
    """Compute the composite score for every bar in `data[config.target]`.
    `neg_corr` is the set of market keys whose raw score should be
    sign-inverted before weighting (e.g. DXY for gold).  If None, we use
    the legacy flat NEGATIVE_CORRELATIONS or look it up from the active
    instrument via `neg_corr_for(config.active_instrument)`.
    """
    if neg_corr is None:
        # Backwards-compat: if NEGATIVE_CORRELATIONS is a flat set, use it
        if isinstance(NEGATIVE_CORRELATIONS, set):
            neg_corr = NEGATIVE_CORRELATIONS
        else:
            neg_corr = neg_corr_for(getattr(config, "active_instrument", ""))
    weights = config.normalized_weights()
    per_market: Dict[str, pd.Series] = {}
    contributions: Dict[str, pd.Series] = {}

    for mkt, df in data.items():
        if mkt not in weights:
            continue
        close = df["Close"]
        score = asset_score(close, config.periods)
        if mkt in neg_corr:
            score = -score
        per_market[mkt] = score
        contributions[mkt] = score * weights[mkt]

    contrib_df = pd.DataFrame(contributions)
    composite = contrib_df.sum(axis=1).clip(-100.0, 100.0).rename("composite")

    label, confidence = _classify(composite)
    return ScoreResult(
        composite=composite,
        per_market=per_market,
        contributions=contrib_df,
        label=label,
        confidence=confidence,
    )


def _classify(score: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """5-bucket classification with confidence per the spec."""
    label = pd.Series("Neutral", index=score.index, dtype=object)
    conf  = pd.Series(62.5, index=score.index, dtype=float)

    label = label.mask(score >  70, "Strong Bullish")
    label = label.mask((score >  40) & (score <=  70), "Bullish")
    label = label.mask((score >= -40) & (score <=  40), "Neutral")
    label = label.mask((score >= -70) & (score <  -40), "Bearish")
    label = label.mask(score <  -70, "Strong Bearish")

    conf = conf.mask(score >  70, 95.0)
    conf = conf.mask((score >  40) & (score <=  70), 82.5)
    conf = conf.mask((score >= -40) & (score <=  40), 62.5)
    conf = conf.mask((score >= -70) & (score <  -40), 82.5)
    conf = conf.mask(score <  -70, 95.0)
    return label, conf


# -----------------------------------------------------------------------------
# Forecast horizons
# -----------------------------------------------------------------------------
def forecast_score(close: pd.Series, ema_fast: int, ema_slow: int, rsi_len: int) -> pd.Series:
    """70% trend (EMA fast/slow) + 30% momentum (RSI), clamped to [-100, +100]."""
    e1 = ema(close, ema_fast)
    e2 = ema(close, ema_slow)
    t = np.where(e1 > e2, 100.0, np.where(e1 < e2, -100.0, 0.0))
    t_series = pd.Series(t, index=close.index).fillna(0.0)

    r = rsi(close, rsi_len)
    pos = ((r - 60.0) * (100.0 / 40.0)).clip(0.0, 100.0)
    neg = -((40.0 - r) * (100.0 / 40.0)).clip(0.0, 100.0)
    soft = (r - 50.0) * 5.0 * 0.10
    m_series = pd.Series(np.where(r > 60, pos, np.where(r < 40, neg, soft)),
                         index=close.index).fillna(0.0)

    return (t_series * 0.70 + m_series * 0.30).clip(-100.0, 100.0)


def forecasts(data: Dict[str, pd.DataFrame], config: Config) -> Dict[str, pd.Series]:
    """Compute the three forecast horizon scores for the active instrument.
    Falls back to `data['gold']` if `config.target` is missing (legacy).
    """
    target = getattr(config, "target", None) or getattr(config, "active_instrument", "Gold")
    target_key = "gold"
    if target in data:
        target_key = target
    elif target in {"EUR/USD": "eurusd", "GBP/USD": "gbpusd", "USD/JPY": "usdjpy",
                    "AUD/USD": "audusd", "USD/CAD": "usdcad", "USD/CHF": "usdchf",
                    "NZD/USD": "nzdusd", "S&P 500": "sp500", "Bitcoin": "btc",
                    "Ethereum": "eth", "Solana": "sol", "Gold": "gold"}:
        target_key = {"EUR/USD": "eurusd", "GBP/USD": "gbpusd", "USD/JPY": "usdjpy",
                      "AUD/USD": "audusd", "USD/CAD": "usdcad", "USD/CHF": "usdchf",
                      "NZD/USD": "nzdusd", "S&P 500": "sp500", "Bitcoin": "btc",
                      "Ethereum": "eth", "Solana": "sol", "Gold": "gold"}[target]
    if target_key not in data:
        # legacy fallback
        target_key = "gold" if "gold" in data else next(iter(data.keys()))
    close = data[target_key]["Close"]
    out: Dict[str, pd.Series] = {}
    for name, cfg in config.forecasts.items():
        out[name] = forecast_score(
            close,
            ema_fast=cfg["ema_fast"],
            ema_slow=cfg["ema_slow"],
            rsi_len=cfg["rsi"],
        )
    return out


def forecast_classify(score: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Per-horizon label + confidence in [50, 95]%."""
    label = pd.Series("Neutral", index=score.index, dtype=object)
    conf  = pd.Series(60.0, index=score.index, dtype=float)

    bull = score >  40
    bear = score < -40

    label = label.mask(bull, "Bullish")
    label = label.mask(bear, "Bearish")

    conf = conf.mask(bull, 75.0 + ((score - 40.0) / 60.0).clip(0.0, 1.0) * 20.0)
    conf = conf.mask(bear, 75.0 + ((-score - 40.0) / 60.0).clip(0.0, 1.0) * 20.0)
    neutral = ~bull & ~bear
    conf = conf.mask(neutral, 50.0 + (1.0 - score.abs() / 40.0).clip(0.0, 1.0) * 25.0)
    return label, conf


# -----------------------------------------------------------------------------
# Market regime (Gold-Friendly / Gold-Hostile / Transition)
# -----------------------------------------------------------------------------
def market_regime(per_market: Dict[str, pd.Series]) -> pd.Series:
    """
    Gold-Friendly : DXY < -20  AND VIX < -10  AND SP500 >  10
    Gold-Hostile  : DXY >  20  AND VIX >  10  AND SP500 < -10
    Else          : Transition

    DXY is sign-inverted upstream (NEGATIVE_CORRELATIONS), so a negative
    DXY-score here = dollar weakening = gold-friendly.
    """
    sp  = per_market.get("sp500", pd.Series(0.0))
    vix = per_market.get("vix",   pd.Series(0.0))
    dxy = per_market.get("dxy",   pd.Series(0.0))

    idx = sp.index
    vix = vix.reindex(idx).fillna(0.0)
    dxy = dxy.reindex(idx).fillna(0.0)

    on  = (dxy < -20) & (vix < -10) & (sp >  10)
    off = (dxy >  20) & (vix >  10) & (sp < -10)

    regime = pd.Series("Transition", index=idx, dtype=object)
    regime = regime.mask(on,  "Gold-Friendly")
    regime = regime.mask(off, "Gold-Hostile")
    return regime


# -----------------------------------------------------------------------------
# Institutional flow meter  (0-100)   — Gold-tuned
# -----------------------------------------------------------------------------
def flow_meter(
    composite: pd.Series,
    vix_close: pd.Series,
    dxy_close: pd.Series,
    silver_score: pd.Series = None,
) -> pd.Series:
    """
    0-100: higher = more gold-friendly.

      50% composite
      20% inverted VIX      (vol up = gold-friendly)
      15% inverted DXY      (dollar up = gold-hostile)
      15% Silver score      (sister-metal confirmation)
    """
    composite_n = (composite + 100.0) / 2.0

    vix_v = vix_close.clip(lower=0.0).fillna(15.0)
    vix_n = ((1.0 - vix_v / 40.0).clip(0.0, 1.0)) * 100.0

    dxy_v = dxy_close.clip(lower=0.0).fillna(100.0)
    dxy_n = ((1.0 - (dxy_v - 90.0) / 20.0).clip(0.0, 1.0)) * 100.0

    if silver_score is None:
        silver_score = pd.Series(0.0, index=composite.index)
    silver_n = (silver_score + 100.0) / 2.0

    raw = (composite_n * 0.50 + vix_n * 0.20 + dxy_n * 0.15 + silver_n * 0.15)
    return raw.clip(0.0, 100.0).fillna(50.0)
