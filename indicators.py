"""
Pine-Script-equivalent technical indicators on top of pandas.

Every function:
  - takes a `pd.Series` of closes (or any price series)
  - returns a `pd.Series` aligned to the same index
  - leaves a leading run of NaNs for the warmup period (same as TradingView)

Implemented:
  ema()  : Exponential Moving Average (Wilder-equivalent)
  rsi()  : Relative Strength Index (Wilder smoothing, matches TradingView)
  roc()  : Rate of Change (%)
  atr()  : Average True Range (helper, useful for stops in backtest)
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# EMA
# -----------------------------------------------------------------------------
def ema(series: pd.Series, length: int) -> pd.Series:
    """
    Exponential moving average.

    Uses `ewm(span=length, adjust=False)`.  This is the same recursive form
    Pine's `ta.ema()` uses (`alpha = 2 / (length + 1)`).  The first
    `length - 1` values are NaN to mirror Pine's behaviour.
    """
    s = pd.to_numeric(series, errors="coerce")
    return s.ewm(span=length, adjust=False, min_periods=length).mean()


# -----------------------------------------------------------------------------
# RSI  (Wilder smoothing, matches TradingView)
# -----------------------------------------------------------------------------
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """
    Relative Strength Index, Wilder smoothing.

    Equivalent to Pine `ta.rsi(src, length)`.  Returns 50 on the first bar
    (TradingView's behaviour) and 100 once the loss becomes zero.
    """
    s = pd.to_numeric(series, errors="coerce")
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder smoothing is `ewm(com=length-1, adjust=False)`
    avg_gain = gain.ewm(com=length - 1, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(com=length - 1, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.fillna(50.0)  # neutral when both averages are zero
    return out


# -----------------------------------------------------------------------------
# ROC
# -----------------------------------------------------------------------------
def roc(series: pd.Series, length: int = 20) -> pd.Series:
    """
    Rate of Change in percent.

    `ta.roc(src, len)` in Pine returns `(src - src[len]) / src[len] * 100`.
    """
    s = pd.to_numeric(series, errors="coerce")
    prev = s.shift(length)
    return (s - prev) / prev * 100.0


# -----------------------------------------------------------------------------
# ATR  (helper, used by the backtest)
# -----------------------------------------------------------------------------
def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder ATR.  Useful for position sizing / stop distance in backtests."""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=length - 1, adjust=False, min_periods=length).mean()


# -----------------------------------------------------------------------------
# Score components (pure functions, no state)
# -----------------------------------------------------------------------------
def trend_score(close: pd.Series, ema_fast: int, ema_slow: int) -> pd.Series:
    """+100 if EMA_fast > EMA_slow, -100 if below, 0 if equal.  Matches Pine."""
    e1 = ema(close, ema_fast)
    e2 = ema(close, ema_slow)
    out = np.where(e1 > e2, 100.0, np.where(e1 < e2, -100.0, 0.0))
    return pd.Series(out, index=close.index).fillna(0.0)


def momentum_score(close: pd.Series, rsi_len: int) -> pd.Series:
    """
    Map RSI to [-100, +100].

    Pine logic:
        r > 60  -> clip((r-60) * 100/40, 0, 100)
        r < 40  -> -clip((40-r) * 100/40, 0, 100)
        else    -> (r-50) * 5 * 0.10   (soft dead-zone)
    """
    r = rsi(close, rsi_len)
    pos = ((r - 60.0) * (100.0 / 40.0)).clip(0.0, 100.0)
    neg = -((40.0 - r) * (100.0 / 40.0)).clip(0.0, 100.0)
    soft = (r - 50.0) * 5.0 * 0.10
    out = np.where(r > 60, pos, np.where(r < 40, neg, soft))
    return pd.Series(out, index=close.index).fillna(0.0)


def strength_score(close: pd.Series, roc_len: int) -> pd.Series:
    """
    Map ROC(roc_len) to [-100, +100] with a 5% saturation point.
    """
    rc = roc(close, roc_len)
    return (rc * 20.0).clip(-100.0, 100.0).fillna(0.0)
