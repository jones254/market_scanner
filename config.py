"""
Configuration for the **Multi-Asset Composite Engine**.

This module defines the registry of supported instruments.  Each instrument
has its own *universe* (the set of driving markets) and its own *weight
table* (how much each driver matters).  The same scoring algorithm
(``composite_score`` from ``scoring.py``) is applied to every instrument —
only the universe and weights differ.

Source: the design PDF "Composite Market Sets — Multi-Asset Scoring Engine",
authored by the user.  Each instrument's market set mirrors the rationale
documented in that PDF.

The gold set is unchanged from the previous version (DXY 25, IEF 20, Silver
15, SP500 15, EUR/USD 10, VIX 10, Gold 5) so existing backtests, weights
sliders, and tests all stay valid.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple


# =============================================================================
# 1.  Tickers — yfinance symbol for every market used anywhere
# =============================================================================
# Master ticker map.  Keys are short names (used in weight dicts) and
# values are the yfinance symbols to fetch.
TICKERS: Dict[str, str] = {
    # --- Forex majors (DXY components + carry) ---
    "dxy":          "DX-Y.NYB",
    "eurusd":       "EURUSD=X",
    "usdjpy":       "JPY=X",
    "gbpusd":       "GBPUSD=X",
    "audusd":       "AUDUSD=X",
    "usdcad":       "CAD=X",
    "usdchf":       "CHF=X",
    "nzdusd":       "NZDUSD=X",
    # --- Crosses ---
    "eurgbp":       "EURGBP=X",
    "eurjpy":       "EURJPY=X",
    "gbpjpy":       "GBPJPY=X",
    "euraud":       "EURAUD=X",
    "cadjpy":       "CADJPY=X",
    "eurchf":       "EURCHF=X",
    # --- US / global equity indices ---
    "sp500":        "^GSPC",
    "nasdaq":       "^IXIC",
    "dow":          "^DJI",
    "dax":          "^GDAXI",
    "nikkei":       "^N225",
    "ftse":         "^FTSE",
    # --- Commodities ---
    "gold":         "GC=F",
    "silver":       "SI=F",
    "crude_oil":    "CL=F",
    "brent":        "BZ=F",
    "natgas":       "NG=F",
    "ttf_gas":      "TTF=F",          # Dutch TTF — may be unavailable on yfinance
    "copper":       "HG=F",
    "wheat":        "ZW=F",
    "corn":         "ZC=F",
    "soybean":      "ZS=F",
    "coffee":       "KC=F",
    "sugar":        "SB=F",
    "palm_oil":     "FCPO=F",         # may be unavailable
    "iron_ore":     "IRON_ORE",       # not on yfinance; placeholder
    # --- Treasuries & yields (price) ---
    "ief":          "IEF",            # 7-10Y US Treasuries ETF
    "tlt":          "TLT",            # 20+Y US Treasuries ETF
    "us2y":         "^IRX",           # 13W T-bill proxy for 2Y area
    "us10y":        "^TNX",           # 10Y nominal yield
    "us30y":        "^TYX",           # 30Y nominal yield
    "de10y":        "DEXUSEU",        # not on yfinance; placeholder
    "jp10y":        "DEXJPUS",        # not on yfinance; placeholder
    "uk10y":        "DEXGBUS",        # placeholder
    "ca10y":        "DEXCAUS",        # placeholder
    "au10y":        "DEXAUS",         # placeholder
    "ch10y":        "DEXSZUS",        # placeholder
    "brl":          "BRL=X",          # Brazilian real
    # --- Volatility ---
    "vix":          "^VIX",
    "vstoxx":       "^V2TX",          # may be unavailable
    # --- Crypto ---
    "btc":          "BTC-USD",
    "eth":          "ETH-USD",
    "sol":          "SOL-USD",
    "xrp":          "XRP-USD",
    "ada":          "ADA-USD",
    "dot":          "DOT-USD",
    "link":         "LINK-USD",
    "avax":         "AVAX-USD",
}

# A subset of tickers that we *know* are available on yfinance right now.
# Anything else will fall back to "Data unavailable" on that instrument.
# (We still list them in the registry so the user can see the design and
#  switch to Twelve Data for the missing ones.)
KNOWN_TICKERS: Dict[str, str] = {
    k: v for k, v in TICKERS.items() if k in {
        "dxy", "eurusd", "usdjpy", "gbpusd", "audusd", "usdcad", "usdchf",
        "nzdusd", "eurgbp", "eurjpy", "gbpjpy", "euraud", "cadjpy", "eurchf",
        "sp500", "nasdaq", "dow", "dax", "nikkei",
        "gold", "silver", "crude_oil", "brent", "natgas", "copper",
        "wheat", "corn", "soybean", "coffee", "sugar",
        "ief", "tlt", "us2y", "us10y", "us30y", "brl",
        "vix", "btc", "eth", "sol", "xrp", "ada", "dot", "link", "avax",
    }
}


# =============================================================================
# 2.  Market labels (for display)
# =============================================================================
MARKET_LABELS: Dict[str, str] = {
    # Forex
    "dxy":    "DXY",
    "eurusd": "EUR/USD",
    "usdjpy": "USD/JPY",
    "gbpusd": "GBP/USD",
    "audusd": "AUD/USD",
    "usdcad": "USD/CAD",
    "usdchf": "USD/CHF",
    "nzdusd": "NZD/USD",
    "eurgbp": "EUR/GBP",
    "eurjpy": "EUR/JPY",
    "gbpjpy": "GBP/JPY",
    "euraud": "EUR/AUD",
    "cadjpy": "CAD/JPY",
    "eurchf": "EUR/CHF",
    # Indices
    "sp500":  "S&P 500",
    "nasdaq": "Nasdaq 100",
    "dow":    "Dow 30",
    "dax":    "DAX 40",
    "nikkei": "Nikkei 225",
    "ftse":   "FTSE 100",
    # Commodities
    "gold":      "Gold",
    "silver":    "Silver",
    "crude_oil": "Crude Oil (WTI)",
    "brent":     "Brent Crude",
    "natgas":    "Natural Gas (Henry Hub)",
    "ttf_gas":   "Natural Gas (TTF)",
    "copper":    "Copper",
    "wheat":     "Wheat",
    "corn":      "Corn",
    "soybean":   "Soybeans",
    "coffee":    "Coffee",
    "sugar":     "Sugar",
    "palm_oil":  "Palm Oil",
    "iron_ore":  "Iron Ore",
    # Treasuries / yields
    "ief":   "US 7-10Y Treasuries (IEF)",
    "tlt":   "US 20Y+ Treasuries (TLT)",
    "us2y":  "US 2Y Yield",
    "us10y": "US 10Y Yield",
    "us30y": "US 30Y Yield",
    "de10y": "DE 10Y Yield",
    "jp10y": "JP 10Y Yield",
    "uk10y": "UK 10Y Yield",
    "ca10y": "CA 10Y Yield",
    "au10y": "AU 10Y Yield",
    "ch10y": "CH 10Y Yield",
    "brl":   "Brazilian Real",
    # Volatility
    "vix":    "VIX",
    "vstoxx": "VSTOXX",
    # Crypto
    "btc":  "Bitcoin",
    "eth":  "Ethereum",
    "sol":  "Solana",
    "xrp":  "XRP",
    "ada":  "Cardano",
    "dot":  "Polkadot",
    "link": "Chainlink",
    "avax": "Avalanche",
}


# =============================================================================
# 3.  Negative correlations per instrument
# =============================================================================
# For each instrument, which of its driving markets have an *inverted*
# relationship?  E.g. for gold, DXY strength = gold weakness → invert.
# The scoring algorithm reads this set and flips the sign of those markets'
# raw scores before weighting.
NEGATIVE_CORRELATIONS: Dict[str, set] = {
    # --- Gold: classic inverse-USD ---
    "gold":      {"dxy", "us10y", "usdjpy"},
    "silver":    {"dxy", "us10y"},
    # --- FX majors: USD pairs have DXY in numerator ---
    "eurusd":    {"dxy"},
    "gbpusd":    {"dxy"},
    "audusd":    {"dxy"},
    "usdcad":    {"dxy"},
    "usdchf":    {"dxy"},
    "nzdusd":    {"dxy"},
    # --- JPY crosses: USD/JPY strength = JPY weakness = crosses falling ---
    "usdjpy":    {"dxy", "sp500", "nikkei"},
    # --- Crosses ---
    "eurgbp":    set(),  # both legs are priced in USD so no inversion
    "eurjpy":    {"usdjpy"},
    "gbpjpy":    {"usdjpy"},
    "euraud":    {"audusd"},
    "cadjpy":    {"usdjpy"},
    "eurchf":    set(),
    # --- Equities ---
    "sp500":     {"us10y", "vix", "dxy"},
    "nasdaq":    {"us10y", "vix", "dxy"},
    "dow":       {"us10y", "vix", "dxy"},
    "dax":       {"dxy", "ttf_gas"},
    "nikkei":    {"usdjpy", "us10y"},
    "ftse":      {"dxy", "us10y"},
    # --- Commodities: USD-priced ---
    "crude_oil": {"dxy", "us10y"},
    "brent":     {"dxy", "us10y"},
    "natgas":    {"dxy", "us10y"},
    "copper":    {"dxy", "us10y"},
    "wheat":     {"dxy", "us10y"},
    "corn":      {"dxy", "us10y"},
    "soybean":   {"dxy", "us10y"},
    "coffee":    {"dxy", "us10y"},
    "sugar":     {"dxy", "us10y"},
    # --- Treasuries: yield up = bond price down ---
    "ief":       {"us10y", "us2y", "vix"},
    "tlt":       {"us30y", "us10y", "vix"},
    # --- Vol ---
    "vix":       {"sp500"},  # VIX inversely tracks SPX
    # --- Crypto: anti-fiat ---
    "btc":       {"dxy", "us10y"},
    "eth":       {"dxy", "us10y", "btc"},
    "sol":       {"dxy", "us10y", "btc", "eth"},
    "xrp":       {"dxy", "us10y", "btc"},
    "ada":       {"dxy", "us10y", "btc", "eth"},
    "dot":       {"dxy", "us10y", "btc", "eth"},
    "link":      {"dxy", "us10y", "btc", "eth"},
    "avax":      {"dxy", "us10y", "btc", "eth"},
}


# =============================================================================
# 4.  INSTRUMENT REGISTRY — every tradable we support
# =============================================================================
@dataclass(frozen=True)
class MarketSet:
    """
    A market set = an instrument + the universe of markets that drive it.
    `target` is the yfinance key of the instrument itself (e.g. "gold",
    "eurusd").  `universe` is the list of market keys that should be
    fetched and scored.  `weights` maps each universe key to a percentage
    (they are auto-normalized to sum=1 at scoring time).  `asset_class`
    is just for grouping on the dashboard.
    """
    symbol: str            # display name: "EUR/USD", "Gold", "BTC/USD"
    target: str            # yfinance key of the instrument
    universe: Tuple[str, ...]
    weights: Dict[str, float]
    asset_class: str       # "FX Major", "FX Cross", "Index", "Commodity", "Crypto"
    rationale: str = ""    # one-line explanation
    target_ticker_override: str = ""  # optional override for the target ticker

    def available_universe(self) -> Tuple[str, ...]:
        """Return only the universe keys whose tickers are confirmed on yfinance."""
        return tuple(k for k in self.universe if k in KNOWN_TICKERS)

    def is_available(self) -> bool:
        """True if both the target and at least 60% of the universe are fetchable."""
        target_ok = (self.target in KNOWN_TICKERS) or bool(self.target_ticker_override)
        if not target_ok:
            return False
        avail = self.available_universe()
        return len(avail) >= max(3, int(0.6 * len(self.universe)))

    def missing_markets(self) -> List[str]:
        """Markets in the universe that we can't currently fetch."""
        return [k for k in self.universe if k not in KNOWN_TICKERS]


# -----------------------------------------------------------------------------
# Build the registry.  Numbers come from the user's PDF.
# -----------------------------------------------------------------------------
INSTRUMENTS: List[MarketSet] = [
    # =====================  FOREX MAJORS  =====================
    MarketSet(
        symbol="EUR/USD", target="eurusd", asset_class="FX Major",
        universe=("dxy", "us10y", "sp500", "vix", "eurusd", "us2y", "de10y"),
        weights={"dxy": 30, "us10y": 25, "sp500": 15, "vix": 10, "eurusd": 10, "us2y": 5, "de10y": 5},
        rationale="DXY (57.6% of basket) + US-DE 10Y spread + SP500 risk + VIX regime",
    ),
    MarketSet(
        symbol="GBP/USD", target="gbpusd", asset_class="FX Major",
        universe=("dxy", "us10y", "uk10y", "sp500", "vix", "gbpusd", "ftse"),
        weights={"dxy": 25, "us10y": 20, "uk10y": 20, "sp500": 10, "vix": 10, "gbpusd": 10, "ftse": 5},
        rationale="DXY + US-UK 10Y spread + risk-off (SP500, VIX) + FTSE",
    ),
    MarketSet(
        symbol="USD/JPY", target="usdjpy", asset_class="FX Major",
        universe=("us10y", "jp10y", "dxy", "sp500", "vix", "usdjpy", "nikkei"),
        weights={"us10y": 30, "jp10y": 25, "dxy": 15, "sp500": 10, "vix": 5, "usdjpy": 10, "nikkei": 5},
        rationale="US-JP 10Y spread (BoJ YCC vs Fed) + DXY + risk-on (SP500, Nikkei)",
    ),
    MarketSet(
        symbol="AUD/USD", target="audusd", asset_class="FX Major",
        universe=("iron_ore", "copper", "dxy", "sp500", "vix", "audusd", "au10y"),
        weights={"iron_ore": 20, "copper": 15, "dxy": 20, "sp500": 15, "vix": 5, "audusd": 5, "au10y": 20},
        rationale="AUD commodity anchor (iron ore 20%, copper 15%) + DXY + China risk proxy",
    ),
    MarketSet(
        symbol="USD/CAD", target="usdcad", asset_class="FX Major",
        universe=("crude_oil", "dxy", "us10y", "ca10y", "sp500", "vix", "usdcad"),
        weights={"crude_oil": 25, "dxy": 20, "us10y": 15, "ca10y": 15, "sp500": 5, "vix": 5, "usdcad": 5, "au10y": 10},
        rationale="WTI is #1 CAD driver (Canada is oil exporter) + US-CA 10Y spread",
    ),
    MarketSet(
        symbol="USD/CHF", target="usdchf", asset_class="FX Major",
        universe=("dxy", "us10y", "ch10y", "sp500", "vix", "usdchf", "eurusd"),
        weights={"dxy": 30, "us10y": 20, "ch10y": 15, "sp500": 10, "vix": 5, "usdchf": 5, "eurusd": 15},
        rationale="DXY + US-CH 10Y + EUR/USD cross-check",
    ),
    MarketSet(
        symbol="NZD/USD", target="nzdusd", asset_class="FX Major",
        universe=("dxy", "audusd", "dairy", "sp500", "vix", "nzdusd", "au10y"),
        weights={"dxy": 25, "audusd": 20, "dairy": 15, "sp500": 10, "vix": 5, "nzdusd": 5, "au10y": 20},
        rationale="DXY + AUD correlation + dairy prices + AU/NZ 10Y",
    ),
    # =====================  FOREX CROSSES  =====================
    MarketSet(
        symbol="EUR/GBP", target="eurgbp", asset_class="FX Cross",
        universe=("eurusd", "gbpusd", "us10y", "vix", "sp500", "eurgbp"),
        weights={"eurusd": 30, "gbpusd": 30, "us10y": 15, "vix": 5, "sp500": 5, "eurgbp": 15},
        rationale="Ratio of EUR/USD and GBP/USD — both legs matter",
    ),
    MarketSet(
        symbol="EUR/JPY", target="eurjpy", asset_class="FX Cross",
        universe=("eurusd", "usdjpy", "us10y", "jp10y", "vix", "sp500", "eurjpy"),
        weights={"eurusd": 30, "usdjpy": 25, "us10y": 15, "jp10y": 10, "vix": 5, "sp500": 5, "eurjpy": 10},
        rationale="Product of EUR/USD × USD/JPY; JPY carry dominates",
    ),
    MarketSet(
        symbol="GBP/JPY", target="gbpjpy", asset_class="FX Cross",
        universe=("gbpusd", "usdjpy", "us10y", "jp10y", "vix", "gbpjpy", "ftse"),
        weights={"gbpusd": 25, "usdjpy": 25, "us10y": 15, "jp10y": 15, "vix": 5, "ftse": 5, "gbpjpy": 10},
        rationale="GBP × JPY cross; UK-JP 10Y spread (carry)",
    ),
    MarketSet(
        symbol="EUR/AUD", target="euraud", asset_class="FX Cross",
        universe=("eurusd", "audusd", "iron_ore", "copper", "us10y", "vix", "euraud"),
        weights={"eurusd": 25, "audusd": 20, "iron_ore": 15, "copper": 10, "us10y": 10, "vix": 5, "euraud": 15},
        rationale="EUR/USD ÷ AUD/USD; iron ore / copper as AUD commodity anchor",
    ),
    MarketSet(
        symbol="CAD/JPY", target="cadjpy", asset_class="FX Cross",
        universe=("usdcad", "usdjpy", "crude_oil", "us10y", "jp10y", "cadjpy"),
        weights={"usdcad": 25, "usdjpy": 25, "crude_oil": 20, "us10y": 10, "jp10y": 10, "cadjpy": 10},
        rationale="USD/JPY ÷ USD/CAD; WTI as CAD anchor",
    ),
    MarketSet(
        symbol="EUR/CHF", target="eurchf", asset_class="FX Cross",
        universe=("eurusd", "usdchf", "us10y", "ch10y", "vix", "sp500", "eurchf"),
        weights={"eurusd": 30, "usdchf": 25, "us10y": 15, "ch10y": 15, "vix": 5, "sp500": 5, "eurchf": 5},
        rationale="EUR/USD ÷ USD/CHF; SNB intervention via CHF rate",
    ),
    # =====================  INDICES  =====================
    MarketSet(
        symbol="S&P 500", target="sp500", asset_class="Index",
        universe=("us10y", "dxy", "vix", "sp500", "us2y", "ief", "crude_oil"),
        weights={"us10y": 25, "dxy": 20, "vix": 15, "sp500": 10, "us2y": 10, "ief": 10, "crude_oil": 10},
        rationale="Discount rate (10Y) + earnings translation (DXY) + vol regime (VIX)",
    ),
    MarketSet(
        symbol="Nasdaq 100", target="nasdaq", asset_class="Index",
        universe=("us10y", "dxy", "vix", "nasdaq", "us2y", "btc", "sp500"),
        weights={"us10y": 30, "dxy": 15, "vix": 15, "nasdaq": 10, "us2y": 10, "btc": 10, "sp500": 10},
        rationale="Long-duration tech — most sensitive to real yields; BTC risk correlation",
    ),
    MarketSet(
        symbol="Dow 30", target="dow", asset_class="Index",
        universe=("us10y", "dxy", "vix", "dow", "us2y", "crude_oil", "sp500"),
        weights={"us10y": 25, "dxy": 20, "vix": 10, "dow": 10, "us2y": 10, "crude_oil": 10, "sp500": 15},
        rationale="Cyclical value — DXY + yields + oil as industrial demand proxy",
    ),
    MarketSet(
        symbol="DAX 40", target="dax", asset_class="Index",
        universe=("dxy", "ttf_gas", "eurusd", "sp500", "vix", "dax", "us10y"),
        weights={"dxy": 20, "ttf_gas": 15, "eurusd": 15, "sp500": 15, "vix": 10, "dax": 10, "us10y": 15},
        rationale="Export-heavy German index — DXY + EU energy (TTF) + EUR leg",
    ),
    MarketSet(
        symbol="Nikkei 225", target="nikkei", asset_class="Index",
        universe=("usdjpy", "us10y", "dxy", "sp500", "vix", "nikkei", "jp10y"),
        weights={"usdjpy": 25, "us10y": 20, "dxy": 15, "sp500": 10, "vix": 5, "nikkei": 10, "jp10y": 15},
        rationale="JPY translation + global risk (SP500) + BoJ policy (JP10Y)",
    ),
    # =====================  COMMODITIES  =====================
    MarketSet(
        symbol="Gold", target="gold", asset_class="Commodity",
        universe=("dxy", "ief", "silver", "sp500", "eurusd", "vix", "gold"),
        weights={"dxy": 25, "ief": 20, "silver": 15, "sp500": 15, "eurusd": 10, "vix": 10, "gold": 5},
        rationale="#1 driver DXY (neg corr), IEF yields (neg), silver confirmation, VIX fear",
    ),
    MarketSet(
        symbol="Silver", target="silver", asset_class="Commodity",
        universe=("dxy", "ief", "gold", "sp500", "eurusd", "vix", "silver", "copper"),
        weights={"dxy": 20, "ief": 15, "gold": 20, "sp500": 10, "eurusd": 5, "vix": 5, "silver": 5, "copper": 20},
        rationale="Gold correlation + industrial copper — silver sits between gold and copper",
    ),
    MarketSet(
        symbol="WTI Crude", target="crude_oil", asset_class="Commodity",
        universe=("dxy", "us10y", "sp500", "vix", "crude_oil", "brent", "natgas", "usdcad"),
        weights={"dxy": 20, "us10y": 15, "sp500": 15, "vix": 5, "crude_oil": 10, "brent": 15, "natgas": 10, "usdcad": 10},
        rationale="USD-priced, Brent as benchmark, natgas as energy complex, SP500 risk",
    ),
    MarketSet(
        symbol="Brent Crude", target="brent", asset_class="Commodity",
        universe=("dxy", "us10y", "sp500", "vix", "brent", "crude_oil", "eurusd"),
        weights={"dxy": 20, "us10y": 15, "sp500": 10, "vix": 5, "brent": 15, "crude_oil": 20, "eurusd": 15},
        rationale="Brent is Europe-priced — EUR leg matters + WTI spread",
    ),
    MarketSet(
        symbol="Nat Gas (HH)", target="natgas", asset_class="Commodity",
        universe=("dxy", "us10y", "natgas", "ttf_gas", "sp500", "vix", "us2y"),
        weights={"dxy": 15, "us10y": 10, "natgas": 25, "ttf_gas": 15, "sp500": 10, "vix": 5, "us2y": 20},
        rationale="Mean-reverting; TTF as global price anchor; weather as inventory driver",
    ),
    MarketSet(
        symbol="Copper", target="copper", asset_class="Commodity",
        universe=("dxy", "us10y", "sp500", "vix", "copper", "iron_ore", "audusd", "eurusd"),
        weights={"dxy": 15, "us10y": 10, "sp500": 10, "vix": 5, "copper": 15, "iron_ore": 20, "audusd": 15, "eurusd": 10},
        rationale="Dr. Copper — China/global industrial cycle + iron ore as substitute",
    ),
    MarketSet(
        symbol="Wheat", target="wheat", asset_class="Commodity",
        universe=("dxy", "us10y", "sp500", "vix", "wheat", "corn", "soybean", "eurusd"),
        weights={"dxy": 15, "us10y": 10, "sp500": 10, "vix": 5, "wheat": 20, "corn": 20, "soybean": 10, "eurusd": 10},
        rationale="Grain complex — corn as substitute, soybean as oilseed pair",
    ),
    MarketSet(
        symbol="Corn", target="corn", asset_class="Commodity",
        universe=("dxy", "us10y", "sp500", "vix", "corn", "wheat", "soybean", "natgas"),
        weights={"dxy": 15, "us10y": 10, "sp500": 10, "vix": 5, "corn": 20, "wheat": 20, "soybean": 10, "natgas": 10},
        rationale="Ethanol demand + grain substitute complex (wheat) + natgas for fertilizer",
    ),
    MarketSet(
        symbol="Soybeans", target="soybean", asset_class="Commodity",
        universe=("dxy", "us10y", "sp500", "vix", "soybean", "corn", "wheat", "palm_oil"),
        weights={"dxy": 15, "us10y": 10, "sp500": 10, "vix": 5, "soybean": 20, "corn": 15, "wheat": 10, "palm_oil": 15},
        rationale="China import cycle + palm oil substitute + corn/wheat complex",
    ),
    MarketSet(
        symbol="Coffee", target="coffee", asset_class="Commodity",
        universe=("dxy", "us10y", "sp500", "vix", "coffee", "sugar", "brl", "eurusd"),
        weights={"dxy": 15, "us10y": 10, "sp500": 10, "vix": 5, "coffee": 20, "sugar": 10, "brl": 20, "eurusd": 10},
        rationale="Brazil producer currency (BRL) + sugar as softs pair + weather",
    ),
    MarketSet(
        symbol="Sugar", target="sugar", asset_class="Commodity",
        universe=("dxy", "us10y", "sp500", "vix", "sugar", "coffee", "brl", "natgas"),
        weights={"dxy": 15, "us10y": 10, "sp500": 10, "vix": 5, "sugar": 20, "coffee": 10, "brl": 20, "natgas": 10},
        rationale="BRL producer + softs pair (coffee) + ethanol arbitrage via natgas",
    ),
    # =====================  TREASURIES  =====================
    MarketSet(
        symbol="US 7-10Y (IEF)", target="ief", asset_class="Treasury",
        universe=("us2y", "us10y", "us30y", "vix", "sp500", "ief", "dxy"),
        weights={"us2y": 25, "us10y": 25, "us30y": 15, "vix": 10, "sp500": 10, "ief": 5, "dxy": 10},
        rationale="Fed policy anchor (2Y) + 10Y + risk regime (VIX, SP500)",
    ),
    MarketSet(
        symbol="US 20Y+ (TLT)", target="tlt", asset_class="Treasury",
        universe=("us10y", "us30y", "vix", "sp500", "tlt", "dxy", "ief"),
        weights={"us10y": 25, "us30y": 30, "vix": 10, "sp500": 10, "tlt": 5, "dxy": 10, "ief": 10},
        rationale="Long duration — most sensitive to inflation + 30Y + VIX fear",
    ),
    # =====================  VOLATILITY  =====================
    MarketSet(
        symbol="VIX", target="vix", asset_class="Volatility",
        universe=("sp500", "vix", "us10y", "vstoxx", "dxy", "ief", "gold"),
        weights={"sp500": 35, "vix": 10, "us10y": 15, "vstoxx": 15, "dxy": 10, "ief": 10, "gold": 5},
        rationale="Inverse SP500 mechanical correlation + VSTOXX as global vol + flight-to-quality (gold)",
    ),
    # =====================  CRYPTO  =====================
    MarketSet(
        symbol="Bitcoin", target="btc", asset_class="Crypto",
        universe=("dxy", "us10y", "sp500", "vix", "btc", "eth", "nasdaq", "gold"),
        weights={"dxy": 20, "us10y": 20, "sp500": 15, "vix": 10, "btc": 5, "eth": 10, "nasdaq": 10, "gold": 10},
        rationale="Digital gold — DXY + real yields + risk asset correlation (SP500/NQ)",
    ),
    MarketSet(
        symbol="Ethereum", target="eth", asset_class="Crypto",
        universe=("dxy", "us10y", "sp500", "vix", "btc", "eth", "nasdaq"),
        weights={"dxy": 15, "us10y": 20, "sp500": 15, "vix": 10, "btc": 20, "eth": 5, "nasdaq": 15},
        rationale="BTC anchor + risk asset (SP500/NQ) + DXY",
    ),
    MarketSet(
        symbol="Solana", target="sol", asset_class="Crypto",
        universe=("dxy", "us10y", "sp500", "vix", "btc", "eth", "sol", "nasdaq"),
        weights={"dxy": 10, "us10y": 15, "sp500": 10, "vix": 5, "btc": 25, "eth": 15, "sol": 5, "nasdaq": 15},
        rationale="BTC beta ~1.5 + ETH competitor + tech sentiment (Nasdaq)",
    ),
    MarketSet(
        symbol="XRP", target="xrp", asset_class="Crypto",
        universe=("dxy", "us10y", "sp500", "vix", "btc", "eth", "xrp"),
        weights={"dxy": 10, "us10y": 15, "sp500": 10, "vix": 5, "btc": 30, "eth": 15, "xrp": 5, "nasdaq": 10},
        rationale="BTC anchor + ETH + regulatory-news-driven momentum",
    ),
    MarketSet(
        symbol="Cardano", target="ada", asset_class="Crypto",
        universe=("dxy", "us10y", "sp500", "vix", "btc", "eth", "ada"),
        weights={"dxy": 10, "us10y": 15, "sp500": 10, "vix": 5, "btc": 30, "eth": 15, "ada": 5, "nasdaq": 10},
        rationale="BTC beta ~1.4 + ETH competitor + tech sentiment",
    ),
    MarketSet(
        symbol="Polkadot", target="dot", asset_class="Crypto",
        universe=("dxy", "us10y", "sp500", "vix", "btc", "eth", "dot", "nasdaq"),
        weights={"dxy": 10, "us10y": 15, "sp500": 10, "vix": 5, "btc": 25, "eth": 15, "dot": 5, "nasdaq": 15},
        rationale="BTC anchor + ETH alt + tech risk sentiment",
    ),
    MarketSet(
        symbol="Chainlink", target="link", asset_class="Crypto",
        universe=("dxy", "us10y", "sp500", "vix", "btc", "eth", "link", "nasdaq"),
        weights={"dxy": 10, "us10y": 15, "sp500": 10, "vix": 5, "btc": 25, "eth": 20, "link": 5, "nasdaq": 10},
        rationale="ETH ecosystem token + BTC anchor + DeFi correlation",
    ),
    MarketSet(
        symbol="Avalanche", target="avax", asset_class="Crypto",
        universe=("dxy", "us10y", "sp500", "vix", "btc", "eth", "avax", "nasdaq"),
        weights={"dxy": 10, "us10y": 15, "sp500": 10, "vix": 5, "btc": 25, "eth": 15, "avax": 5, "nasdaq": 15},
        rationale="ETH L1 competitor + BTC anchor + tech sentiment",
    ),
]


# Index by symbol for quick lookup
INSTRUMENT_BY_SYMBOL: Dict[str, MarketSet] = {m.symbol: m for m in INSTRUMENTS}

# Group by asset class for the dashboard
INSTRUMENTS_BY_CLASS: Dict[str, List[MarketSet]] = {}
for m in INSTRUMENTS:
    INSTRUMENTS_BY_CLASS.setdefault(m.asset_class, []).append(m)

# Sort each class for stable display
for cls in INSTRUMENTS_BY_CLASS:
    INSTRUMENTS_BY_CLASS[cls].sort(key=lambda m: m.symbol)


# =============================================================================
# 5.  Backwards-compat with the original gold-only code
# =============================================================================
# `MARKET_LABELS`, `DEFAULT_WEIGHTS`, `NEGATIVE_CORRELATIONS` are still used
# by `scoring.py` / `data.py` directly.  We keep the gold versions as the
# defaults so the original code paths work unchanged.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "dxy":    25.0,
    "ief":    20.0,
    "silver": 15.0,
    "sp500":  15.0,
    "eurusd": 10.0,
    "vix":    10.0,
    "gold":    5.0,
}

DEFAULT_PERIODS = {
    "ema_fast": 20,
    "ema_slow": 50,
    "rsi_len":  14,
    "roc_len":  20,
}

DEFAULT_FORECASTS = {
    "short":  {"ema_fast": 10, "ema_slow": 20,  "rsi": 7},
    "medium": {"ema_fast": 20, "ema_slow": 50,  "rsi": 14},
    "long":   {"ema_fast": 50, "ema_slow": 200, "rsi": 21},
}


# =============================================================================
# 6.  Config dataclass
# =============================================================================
@dataclass
class Config:
    data_source: str = "yfinance"
    twelvedata_api_key: str = ""
    interval: str = "1d"
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    periods: Dict[str, int]  = field(default_factory=lambda: dict(DEFAULT_PERIODS))
    forecasts: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        k: dict(v) for k, v in DEFAULT_FORECASTS.items()
    })
    active_instrument: str = "Gold"   # display name; the registry key

    def weight_sum(self) -> float:
        return sum(max(0.0, v) for v in self.weights.values()) or 1.0

    def normalized_weights(self) -> Dict[str, float]:
        s = self.weight_sum()
        return {k: max(0.0, v) / s for k, v in self.weights.items()}

    def as_dict(self) -> dict:
        return asdict(self)

    def load_instrument(self, symbol: str) -> bool:
        """Switch the active instrument; load its weights/periods/forecasts."""
        mset = INSTRUMENT_BY_SYMBOL.get(symbol)
        if mset is None:
            return False
        self.active_instrument = symbol
        # Load the instrument's weight table
        self.weights = {k: float(v) for k, v in mset.weights.items()}
        return True

    @classmethod
    def from_sidebar(cls) -> "Config":
        """Build a Config from the Streamlit sidebar."""
        import streamlit as st
        cfg = cls()

        with st.sidebar:
            st.subheader("Data Source")
            src = st.selectbox(
                "Provider",
                ["yfinance", "twelvedata"],
                index=0,
                help=(
                    "yfinance is free, no key, covers ~70% of the registry.  "
                    "Twelve Data needs a free API key from twelvedata.com "
                    "and supports more cross / yield tickers."
                ),
            )
            cfg.data_source = src
            if src == "twelvedata":
                cfg.twelvedata_api_key = st.text_input(
                    "Twelve Data API key", value=cfg.twelvedata_api_key, type="password"
                )

            # Active instrument selector
            symbol_options = [m.symbol for m in INSTRUMENTS]
            current = st.session_state.get("active_instrument", cfg.active_instrument)
            if current not in symbol_options:
                current = cfg.active_instrument
            chosen = st.selectbox(
                "Active instrument",
                symbol_options,
                index=symbol_options.index(current),
                help="All tabs (Live / Backtest / Trades / Signals / Signal-BT) "
                     "re-render for this instrument.",
                key="sidebar_active_instrument",
            )
            cfg.active_instrument = chosen
            # Load the instrument's weight table
            cfg.load_instrument(chosen)

            # Data interval
            try:
                from data import INTERVAL_CONFIG
            except ImportError:
                from data import INTERVAL_CONFIG
            interval_opts = list(INTERVAL_CONFIG.keys())
            current_idx = interval_opts.index(cfg.interval) if cfg.interval in interval_opts else 0
            cfg.interval = st.selectbox(
                "Data interval (bar size)",
                interval_opts,
                index=current_idx,
                help=(
                    "1d = swing trading (years of history).  "
                    "1h = intraday (60-day lookback).  "
                    "15m/5m/1m = scalping (7-30 day lookback)."
                ),
                key="interval_select",
            )
            cfg._interval_lookback = INTERVAL_CONFIG[cfg.interval]["lookback_days"]

            # Weights for the ACTIVE instrument
            st.subheader(f"Weights for {chosen} (% — auto-normalized)")
            mset = INSTRUMENT_BY_SYMBOL.get(chosen)
            universe = mset.available_universe() if mset else ()
            new_w = {}
            for k in universe:
                label = MARKET_LABELS.get(k, k)
                new_w[k] = st.slider(
                    label, 0, 100, int(cfg.weights.get(k, 0)),
                    step=1, key=f"w_{chosen}_{k}",
                )
            cfg.weights = new_w

            # Indicator periods
            with st.expander("Indicator periods", expanded=False):
                cfg.periods["ema_fast"] = st.number_input(
                    "EMA fast", 2, 200, cfg.periods["ema_fast"])
                cfg.periods["ema_slow"] = st.number_input(
                    "EMA slow", 5, 400, cfg.periods["ema_slow"])
                cfg.periods["rsi_len"]  = st.number_input(
                    "RSI length", 2, 100, cfg.periods["rsi_len"])
                cfg.periods["roc_len"]  = st.number_input(
                    "ROC length", 1, 200, cfg.periods["roc_len"])

            # Forecast horizons
            with st.expander("Forecast horizons", expanded=False):
                st.caption("Short (15-30m)")
                cfg.forecasts["short"]["ema_fast"]  = st.number_input("S EMA fast", 2, 50, cfg.forecasts["short"]["ema_fast"])
                cfg.forecasts["short"]["ema_slow"]  = st.number_input("S EMA slow", 5, 100, cfg.forecasts["short"]["ema_slow"])
                cfg.forecasts["short"]["rsi"]       = st.number_input("S RSI",      2, 50,  cfg.forecasts["short"]["rsi"])
                st.caption("Medium (1-4h)")
                cfg.forecasts["medium"]["ema_fast"] = st.number_input("M EMA fast", 2, 100, cfg.forecasts["medium"]["ema_fast"])
                cfg.forecasts["medium"]["ema_slow"] = st.number_input("M EMA slow", 5, 200, cfg.forecasts["medium"]["ema_slow"])
                cfg.forecasts["medium"]["rsi"]      = st.number_input("M RSI",      2, 50,  cfg.forecasts["medium"]["rsi"])
                st.caption("Long (1-3d)")
                cfg.forecasts["long"]["ema_fast"]   = st.number_input("L EMA fast", 5, 200, cfg.forecasts["long"]["ema_fast"])
                cfg.forecasts["long"]["ema_slow"]   = st.number_input("L EMA slow", 10, 400, cfg.forecasts["long"]["ema_slow"])
                cfg.forecasts["long"]["rsi"]        = st.number_input("L RSI",      2, 100, cfg.forecasts["long"]["rsi"])

            if st.button("Reset to defaults"):
                st.rerun()

        return cfg
