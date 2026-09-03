"""
Historical backtest engine for the Gold Scalper engine.

Strategy
--------
For each bar T in the backtest window:
    1. Compute the composite score using market closes up to bar T
       (no look-ahead, same as the Pine Script's `close[1]`).
    2. Hold a position according to the bucketed signal for `holding_period`
       bars, starting at the next available bar.
    3. Measure the realised forward return of GOLD over the holding window.

We track:
    - Hit rate (% of signals whose forward return agreed with direction)
    - Mean / median / std return per bucket
    - Equity curve of a long-flat strategy: long Gold when the model says
      "Bull" or "Strong Bull", cash otherwise.
    - Buy & hold baseline for comparison
    - Sharpe ratio, max drawdown, CAGR
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from .scoring import composite_score, _classify
    from .config import Config
except ImportError:
    from scoring import composite_score, _classify
    from config import Config


# -----------------------------------------------------------------------------
# Configuration & result containers
# -----------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    start: str = "2018-01-01"
    end:   str = "2030-12-31"
    holding_period: int = 5           # forward window in bars
    initial_capital: float = 10_000.0
    long_only_when: List[str] = field(
        default_factory=lambda: ["Bullish", "Strong Bullish"]
    )


@dataclass
class BacktestResult:
    signals: pd.DataFrame         # index = date, columns = score, signal, fwd_return
    equity: pd.Series             # long-flat strategy equity curve
    benchmark: pd.Series          # buy & hold Gold
    summary: pd.DataFrame         # per-bucket statistics
    metrics: Dict[str, float]     # headline metrics
    n_signals: int


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def run_backtest(
    data: Dict[str, pd.DataFrame],
    config: Config,
    bt: BacktestConfig,
    target: str = "gold",
) -> BacktestResult:
    """
    Run the historical backtest and return everything the UI needs.

    `data` is the full multi-market dict from the DataSource.  The
    `target` argument (default "gold" for backwards-compat) picks which
    market is treated as the tradable instrument; the backtest measures
    forward return on `data[target]["Close"]`.
    """
    target = target or "gold"
    # ----- 1. Compute composite for every available bar ---------------------
    res = composite_score(data, config)
    df = pd.DataFrame({
        "composite": res.composite,
        "signal":    res.label,
    })
    df = df.dropna(subset=["composite"])

    # ----- 2. Trim to the backtest window -----------------------------------
    df = df.loc[bt.start:bt.end].copy()
    if df.empty:
        raise RuntimeError(
            f"No data in backtest window {bt.start} → {bt.end}.  "
            "Pick a wider date range or reduce the lookback."
        )

    # ----- 3. Forward return of the target instrument -----------------------
    if target not in data or len(data[target]) == 0:
        raise RuntimeError(f"Target market '{target}' not in data.")
    target_close = data[target]["Close"].reindex(df.index)
    fwd = target_close.shift(-bt.holding_period) / target_close - 1.0
    df["fwd_return"] = fwd

    # ----- 4. Per-bucket statistics -----------------------------------------
    grouped = df.dropna(subset=["fwd_return"]).groupby("signal")["fwd_return"]
    summary = pd.DataFrame({
        "count":        grouped.count(),
        "mean_return":  grouped.mean() * 100,
        "median_return":grouped.median() * 100,
        "stdev":        grouped.std() * 100,
        "win_rate":     grouped.apply(lambda x: (x > 0).mean() * 100),
    }).round(2)
    order = ["Strong Bullish", "Bullish", "Neutral", "Bearish", "Strong Bearish"]
    summary = summary.reindex([o for o in order if o in summary.index])

    # ----- 5. Long-flat equity curve ----------------------------------------
    long_mask = df["signal"].isin(bt.long_only_when)
    daily_ret = target_close.pct_change().fillna(0.0)
    strat_ret = daily_ret.where(long_mask, 0.0)

    equity = (1.0 + strat_ret).cumprod() * bt.initial_capital
    bench   = (1.0 + daily_ret).cumprod() * bt.initial_capital

    # ----- 6. Headline metrics ----------------------------------------------
    metrics = _compute_metrics(equity, bench, strat_ret, daily_ret)

    return BacktestResult(
        signals=df,
        equity=equity,
        benchmark=bench,
        summary=summary,
        metrics=metrics,
        n_signals=int(len(df.dropna(subset=["fwd_return"]))),
    )


# -----------------------------------------------------------------------------
# Performance metrics
# -----------------------------------------------------------------------------
def _compute_metrics(
    equity: pd.Series,
    bench:  pd.Series,
    strat_ret: pd.Series,
    bench_ret:  pd.Series,
) -> Dict[str, float]:
    def _sharpe(r: pd.Series) -> float:
        if r.std() == 0 or np.isnan(r.std()):
            return 0.0
        return float(r.mean() / r.std() * np.sqrt(252))

    def _max_dd(eq: pd.Series) -> float:
        peak = eq.cummax()
        dd = (eq - peak) / peak
        return float(dd.min() * 100.0)  # percent

    def _cagr(eq: pd.Series) -> float:
        if len(eq) < 2 or eq.iloc[0] == 0:
            return 0.0
        days = (eq.index[-1] - eq.index[0]).days
        if days <= 0:
            return 0.0
        return float((eq.iloc[-1] / eq.iloc[0]) ** (365.0 / days) - 1.0) * 100

    return {
        "Strategy CAGR %":      round(_cagr(equity), 2),
        "Benchmark CAGR %":     round(_cagr(bench), 2),
        "Strategy Sharpe":      round(_sharpe(strat_ret), 2),
        "Benchmark Sharpe":     round(_sharpe(bench_ret), 2),
        "Strategy Max DD %":    round(_max_dd(equity), 2),
        "Benchmark Max DD %":   round(_max_dd(bench), 2),
        "Strategy Total Ret %": round((equity.iloc[-1] / equity.iloc[0] - 1) * 100, 2),
        "Benchmark Total Ret %":round((bench.iloc[-1]   / bench.iloc[0]   - 1) * 100, 2),
    }


# -----------------------------------------------------------------------------
# Convenience: pretty summary of latest signal vs forward windows
# -----------------------------------------------------------------------------
def quick_signal_breakdown(result: BacktestResult) -> pd.DataFrame:
    return result.signals.tail(30).iloc[::-1]
