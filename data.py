"""
Data layer for the multi-asset composite engine.

Each provider exposes a single method:
    fetch_all(lookback_days, interval, target=None, universe=None) -> {mkt: DataFrame}

If `target` and `universe` are given, only those tickers are fetched.  If
not, the legacy 7-market gold set is fetched (backwards-compat).

A separate `fetch_multi_instrument` helper fetches every instrument in
the registry in parallel (each in its own thread) and returns a nested
dict: {symbol: {mkt: DataFrame}}.  This is what powers the Dashboard tab.
"""

from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

try:
    from .config import Config, INSTRUMENT_BY_SYMBOL, TICKERS, KNOWN_TICKERS
except ImportError:
    from config import Config, INSTRUMENT_BY_SYMBOL, TICKERS, KNOWN_TICKERS


# -----------------------------------------------------------------------------
# Legacy ticker maps (kept for backwards-compat with the gold-only code path)
# -----------------------------------------------------------------------------
YFINANCE_TICKERS: Dict[str, str] = {
    "dxy":    "DX-Y.NYB",
    "ief":    "IEF",
    "silver": "SI=F",
    "sp500":  "SPY",
    "eurusd": "EURUSD=X",
    "vix":    "^VIX",
    "gold":   "GC=F",
}
TWELVE_DATA_TICKERS: Dict[str, str] = {
    "dxy":    "DXY", "ief": "IEF", "silver": "XAG/USD",
    "sp500":  "SPY", "eurusd": "EUR/USD", "vix": "VIX", "gold": "XAU/USD",
}


# -----------------------------------------------------------------------------
# OHLC normalization
# -----------------------------------------------------------------------------
def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().title() for c in df.columns]
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep]
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.dropna(how="all")


# -----------------------------------------------------------------------------
# Alignment helper
# -----------------------------------------------------------------------------
def _align(data: Dict[str, pd.DataFrame], target_key: str = "gold") -> Dict[str, pd.DataFrame]:
    """
    Reindex every market onto the union of dates, forward-filling gaps.
    `target_key` is the master instrument that drives the index (defaults
    to "gold" for backwards-compat with the original gold-only code).
    """
    if not data:
        return data
    # Pick the first non-empty key if `target_key` is missing
    if target_key not in data or len(data[target_key]) == 0:
        for k, v in data.items():
            if len(v) > 0:
                target_key = k
                break
        else:
            return data
    master_idx = data[target_key].index
    out = {target_key: data[target_key]}
    for mkt, df in data.items():
        if mkt == target_key:
            continue
        try:
            aligned = df.reindex(master_idx).ffill()
        except Exception:
            continue
        out[mkt] = aligned
    return out


# -----------------------------------------------------------------------------
# Interval configuration
# -----------------------------------------------------------------------------
INTERVAL_CONFIG: Dict[str, Dict] = {
    "1d":  {"lookback_days": 365 * 3, "bar_label": "day",   "bars_per_day": 1},
    "1h":  {"lookback_days": 60,      "bar_label": "hour",  "bars_per_day": 24},
    "30m": {"lookback_days": 30,      "bar_label": "30min", "bars_per_day": 32},
    "15m": {"lookback_days": 30,      "bar_label": "15min", "bars_per_day": 64},
    "5m":  {"lookback_days": 30,      "bar_label": "5min",  "bars_per_day": 192},
    "1m":  {"lookback_days": 5,       "bar_label": "1min",  "bars_per_day": 960},
}
SUPPORTED_INTERVALS = list(INTERVAL_CONFIG.keys())


def interval_lookback_days(interval: str) -> int:
    cfg = INTERVAL_CONFIG.get(interval)
    if cfg is None:
        raise ValueError(f"Unsupported interval '{interval}'. Supported: {SUPPORTED_INTERVALS}")
    return cfg["lookback_days"]


def interval_bar_label(interval: str) -> str:
    return INTERVAL_CONFIG.get(interval, {}).get("bar_label", interval)


# -----------------------------------------------------------------------------
# Abstract base
# -----------------------------------------------------------------------------
class DataSource:
    name: str = "abstract"

    def fetch_all(
        self,
        lookback_days: int = 365 * 3,
        interval: str = "1d",
        target: Optional[str] = None,
        universe: Optional[Tuple[str, ...]] = None,
    ) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError


# -----------------------------------------------------------------------------
# yfinance implementation
# -----------------------------------------------------------------------------
class YFinanceSource(DataSource):
    name = "yfinance"

    def _download(self, ticker: str, start, end, interval: str) -> pd.DataFrame:
        import yfinance as yf
        return yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
        )

    def fetch_all(
        self,
        lookback_days: int = 365 * 3,
        interval: str = "1d",
        target: Optional[str] = None,
        universe: Optional[Tuple[str, ...]] = None,
    ) -> Dict[str, pd.DataFrame]:
        end = datetime.utcnow().date()
        start = end - timedelta(days=lookback_days)

        # Pick the ticker map
        if target is not None and universe is not None:
            markets = [target] + [m for m in universe if m != target]
            ticker_map = {m: TICKERS.get(m, "") for m in markets if TICKERS.get(m)}
        else:
            # Legacy gold-only path
            ticker_map = dict(YFINANCE_TICKERS)

        out: Dict[str, pd.DataFrame] = {}
        for mkt, ticker in ticker_map.items():
            if not ticker or ticker.endswith("_ORE") or ticker.endswith("_placeholder"):
                continue
            try:
                df = self._download(ticker, start, end, interval)
                df = _normalize_ohlc(df)
                if not df.empty:
                    out[mkt] = df
            except Exception as e:
                print(f"[yfinance] {mkt} ({ticker}) failed: {e}")
        return _align(out, target_key=target or "gold")


# -----------------------------------------------------------------------------
# Twelve Data implementation
# -----------------------------------------------------------------------------
class TwelveDataSource(DataSource):
    BASE = "https://api.twelvedata.com/time_series"
    name = "twelvedata"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Twelve Data API key is required (set in the sidebar).")
        self.api_key = api_key

    def _fetch_one(self, symbol: str, lookback_days: int, interval: str) -> pd.DataFrame:
        td_interval = {
            "1d": "1day", "1h": "1h", "30m": "30min", "15m": "15min",
            "5m": "5min", "1m": "1min",
        }.get(interval, interval)
        params = {
            "symbol": symbol, "interval": td_interval,
            "outputsize": max(80, lookback_days),
            "apikey": self.api_key, "format": "JSON", "order": "ASC",
        }
        r = requests.get(self.BASE, params=params, timeout=15)
        r.raise_for_status()
        js = r.json()
        if "values" not in js:
            raise RuntimeError(f"Twelve Data error for {symbol}: {js.get('message', js)}")
        df = pd.DataFrame(js["values"]).rename(columns={
            "datetime": "Date", "open": "Open", "high": "High",
            "low": "Low", "close": "Close", "volume": "Volume",
        })
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def fetch_all(
        self,
        lookback_days: int = 365 * 3,
        interval: str = "1d",
        target: Optional[str] = None,
        universe: Optional[Tuple[str, ...]] = None,
    ) -> Dict[str, pd.DataFrame]:
        if target is not None and universe is not None:
            markets = [target] + [m for m in universe if m != target]
            ticker_map = {m: TICKERS.get(m, "") for m in markets if TICKERS.get(m)}
        else:
            ticker_map = dict(TWELVE_DATA_TICKERS)

        out: Dict[str, pd.DataFrame] = {}
        for mkt, symbol in ticker_map.items():
            if not symbol:
                continue
            for attempt in range(3):
                try:
                    df = self._fetch_one(symbol, lookback_days, interval)
                    df = _normalize_ohlc(df)
                    if not df.empty:
                        out[mkt] = df
                    break
                except Exception as e:
                    print(f"[twelvedata] {symbol} attempt {attempt+1} failed: {e}")
                    time.sleep(2 ** attempt)
        return _align(out, target_key=target or "gold")


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------
class DataSourceFactory:
    @staticmethod
    def create(config: Config) -> DataSource:
        if config.data_source == "yfinance":
            return YFinanceSource()
        if config.data_source == "twelvedata":
            return TwelveDataSource(config.twelvedata_api_key)
        raise ValueError(f"Unknown data source: {config.data_source}")


# -----------------------------------------------------------------------------
# Multi-timeframe fetch helper (used by the signal engine)
# -----------------------------------------------------------------------------
def fetch_multi_timeframe(
    config: Config,
    intervals: tuple = ("15m", "1h", "1d"),
    show_progress: bool = False,
    target: Optional[str] = None,
    universe: Optional[Tuple[str, ...]] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Fetch the active instrument's universe on multiple timeframes.
    Returns a nested dict: {interval: {market_key: DataFrame}}
    """
    source = DataSourceFactory.create(config)
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for interval in intervals:
        if interval not in INTERVAL_CONFIG:
            continue
        lookback = INTERVAL_CONFIG[interval]["lookback_days"]
        try:
            out[interval] = source.fetch_all(
                lookback_days=lookback, interval=interval,
                target=target, universe=universe,
            )
        except Exception as e:
            print(f"[fetch_multi] {interval} failed: {e}")
            out[interval] = {}
    return out


# -----------------------------------------------------------------------------
# Multi-instrument fetch (for the Dashboard tab) — parallel yfinance
# -----------------------------------------------------------------------------
def fetch_instrument_one(
    symbol: str,
    source: DataSource,
    lookback_days: int,
    interval: str,
) -> Tuple[str, Dict[str, pd.DataFrame]]:
    """Fetch a single instrument.  Thread-safe."""
    mset = INSTRUMENT_BY_SYMBOL.get(symbol)
    if mset is None:
        return symbol, {}
    target = mset.target
    universe = mset.available_universe()
    try:
        data = source.fetch_all(
            lookback_days=lookback_days, interval=interval,
            target=target, universe=universe,
        )
    except Exception as e:
        print(f"[fetch_instrument_one] {symbol} failed: {e}")
        data = {}
    return symbol, data


def fetch_all_instruments(
    config: Config,
    symbols: Optional[List[str]] = None,
    max_workers: int = 6,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Fetch every instrument in `symbols` (default = all) in parallel.
    Returns {symbol: {mkt: DataFrame}}.

    Uses a ThreadPoolExecutor to overlap yfinance network calls.
    """
    if symbols is None:
        symbols = [m.symbol for m in INSTRUMENT_BY_SYMBOL.values()]

    source = DataSourceFactory.create(config)
    lookback = interval_lookback_days(config.interval)

    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(fetch_instrument_one, sym, source, lookback, config.interval): sym
            for sym in symbols
        }
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                sym, data = fut.result(timeout=60)
                out[sym] = data
            except Exception as e:
                print(f"[fetch_all_instruments] {sym} failed: {e}")
                out[sym] = {}
    return out
