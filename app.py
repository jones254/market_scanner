"""
Streamlit entry point for the Gold Scalper engine.

Multi-market macro model for **Gold (GC=F / XAU/USD)** using DXY,
US Treasuries (IEF), Silver, S&P 500, EUR/USD, and VIX as inputs.

Run locally:    streamlit run app.py
Deploy:         push to GitHub, connect on share.streamlit.io

Flat layout: all .py files at the repo root, no engine/ subfolder.
"""

from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple

# Make repo root importable
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------------------------
# Robust import — try flat layout first, fall back to engine.* package
# ---------------------------------------------------------------------------
def _import_local():
    """
    Try each module individually in flat layout, then engine.* if flat
    fails.  Returns (modules_dict, layout_name) or raises with a clear
    list of which files are missing.
    """
    flat = {
        "auto_refresh":       "auto_refresh",
        "telegram_alerts":    "telegram_alerts",
        "telegram_commands":  "telegram_commands",
        "signals":            "signals",
        "data":               "data",
        "signal_backtest":    "signal_backtest",
        "indicators":         "indicators",
    }
    engine = {
        "auto_refresh":       "engine.auto_refresh",
        "telegram_alerts":    "engine.telegram_alerts",
        "telegram_commands":  "engine.telegram_commands",
        "signals":            "engine.signals",
        "data":               "engine.data",
        "signal_backtest":    "engine.signal_backtest",
        "indicators":         "engine.indicators",
    }
    import importlib
    errors = {}
    for layout_name, mapping in [("flat", flat), ("engine", engine)]:
        out = {}
        layout_errors = {}
        for short, mod in mapping.items():
            try:
                out[short] = importlib.import_module(mod)
            except ImportError as e:
                layout_errors[short] = str(e)
        if not layout_errors:
            return out, layout_name
        errors[layout_name] = layout_errors
    raise ImportError(
        "Could not import engine modules in either flat or engine.* layout.\n\n"
        "Flat layout errors:\n  " +
        "\n  ".join(f"{k}: {v}" for k, v in errors.get("flat", {}).items()) +
        "\n\nEngine.* package errors:\n  " +
        "\n  ".join(f"{k}: {v}" for k, v in errors.get("engine", {}).items()) +
        "\n\nMake sure these files exist at the repo root: " +
        ", ".join(flat.keys())
    )


_LOCALS, _LAYOUT = _import_local()

# Engine modules
setup_auto_refresh = _LOCALS["auto_refresh"].setup_auto_refresh
tg_alerts          = _LOCALS["telegram_alerts"]
TgConfig           = _LOCALS["telegram_alerts"].TelegramConfig
TgState            = _LOCALS["telegram_alerts"]._State
AlertHistory       = _LOCALS["telegram_alerts"].AlertHistory
tg_is_configured   = _LOCALS["telegram_alerts"].is_configured
tg_send            = _LOCALS["telegram_alerts"].send_message
tg_send_photo      = _LOCALS["telegram_alerts"].send_photo
_tg_process_commands = _LOCALS["telegram_commands"].process_pending_commands
_tg_send_open_signal = _LOCALS["telegram_commands"].send_open_signal
TgCommandState     = _LOCALS["telegram_commands"].CommandState
SignalConfig       = _LOCALS["signals"].SignalConfig
SignalState        = _LOCALS["signals"].SignalState
SignalResult       = _LOCALS["signals"].SignalResult
evaluate_signal    = _LOCALS["signals"].evaluate
format_signal_message = _LOCALS["signals"].format_signal_message
render_chart_snapshot  = _LOCALS["signals"].render_chart_snapshot
horizon_entry_sltp    = _LOCALS["signals"].horizon_entry_sltp
fetch_multi_timeframe  = _LOCALS["data"].fetch_multi_timeframe
SignalBacktestConfig  = _LOCALS["signal_backtest"].SignalBacktestConfig
SignalBacktestResult  = _LOCALS["signal_backtest"].SignalBacktestResult
run_signal_backtest    = _LOCALS["signal_backtest"].run_signal_backtest

# Engine modules from the original DE40-era code (still needed for the
# Live tab composite, regime, flow, etc.)
try:
    from config import (
        Config, MARKET_LABELS,
        INSTRUMENTS, INSTRUMENTS_BY_CLASS, INSTRUMENT_BY_SYMBOL,  # noqa: F401
    )
    from data import DataSourceFactory       # noqa: F401
    from scoring import (                    # noqa: F401
        composite_score, forecasts, forecast_classify,
        market_regime, flow_meter, neg_corr_for,
    )
    from backtest import BacktestConfig, run_backtest  # noqa: F401
    from trades import run_trade_engine, TradeConfig as EngineTradeConfig  # noqa: F401
    from trade_ui import render_trade_dashboard        # noqa: F401
    from indicators import atr as _atr, ema as _ema
except ImportError:
    from engine.config import Config, MARKET_LABELS
    from engine.data import DataSourceFactory
    from engine.scoring import (
        composite_score, forecasts, forecast_classify,
        market_regime, flow_meter,
    )
    from engine.backtest import BacktestConfig, run_backtest
    from engine.trades import run_trade_engine, TradeConfig as EngineTradeConfig
    from engine.trade_ui import render_trade_dashboard
    from engine.indicators import atr as _atr, ema as _ema


# -----------------------------------------------------------------------------
# Page config & theming
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Gold Scalper Engine",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
        h1, h2, h3 { letter-spacing: -0.01em; }
        .stMetric > div { padding: 0.5rem 0.75rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px; border-radius: 8px 8px 0 0; font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
SIGNAL_COLOURS = {
    "Strong Bullish":  "#006400",
    "Bullish":         "#2E8B57",
    "Neutral":         "#DAA520",
    "Bearish":         "#B22222",
    "Strong Bearish":  "#8B0000",
}
REGIME_COLOURS = {
    "Gold-Friendly":  "#2E8B57",
    "Gold-Hostile":   "#B22222",
    "Transition":     "#DAA520",
}


@st.cache_data(ttl=300, show_spinner="Fetching market data…")
def _fetch(source_name: str, api_key: str, interval: str, lookback_days: int):
    cfg = Config(data_source=source_name, twelvedata_api_key=api_key, interval=interval)
    return DataSourceFactory.create(cfg).fetch_all(
        lookback_days=lookback_days, interval=interval
    )


@st.cache_data(ttl=300, show_spinner="Fetching active instrument…")
def _fetch_active(
    source_name: str, api_key: str, interval: str,
    target: str, universe: Tuple[str, ...],
):
    """Fetch data for a specific instrument (target + universe)."""
    cfg = Config(data_source=source_name, twelvedata_api_key=api_key, interval=interval)
    return DataSourceFactory.create(cfg).fetch_all(
        lookback_days=interval_lookback_days(interval),
        interval=interval, target=target, universe=tuple(universe),
    )


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
config = Config.from_sidebar()


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
try:
    from data import interval_lookback_days, interval_bar_label, INTERVAL_CONFIG
except ImportError:
    from engine.data import interval_lookback_days, interval_bar_label, INTERVAL_CONFIG

INTERVAL_LABELS = {
    "1d":  "Daily (1d) — swing trading",
    "1h":  "Hourly (1h) — intraday",
    "30m": "30-min — intraday",
    "15m": "15-min — intraday / scalping",
    "5m":  "5-min — scalping",
    "1m":  "1-min — scalping (limited history)",
}

with st.spinner(f"Loading {config.interval} data for {config.active_instrument}…"):
    try:
        # Get the active instrument's universe
        mset = INSTRUMENT_BY_SYMBOL.get(config.active_instrument)
        if mset is None:
            st.error(f"Unknown instrument: {config.active_instrument}")
            st.stop()
        target = mset.target
        universe = mset.available_universe()
        # Use the dashboard pre-fetch if available (faster)
        cached = st.session_state.get("dashboard_data", {}).get(config.active_instrument)
        if cached and len(cached.get(target, pd.DataFrame())) > 30:
            data = cached
        else:
            data = _fetch_active(
                config.data_source, config.twelvedata_api_key, config.interval,
                target=target, universe=universe,
            )
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()
        st.stop()

missing = [m for m in MARKET_LABELS if m not in data or len(data[m]) == 0]
if missing:
    st.warning(f"Missing data for: {', '.join(missing)}. Try a different data source.")
    # The active instrument is the critical one
    if mset.target not in data or len(data[mset.target]) == 0:
        st.error(f"{config.active_instrument} data is required. Cannot continue.")
        st.stop()


# ---------------------------------------------------------------------------
# Multi-timeframe data fetch (for the signal engine)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner="Loading 15m / 1h / 1d for signal engine…")
def _fetch_multi_tf(source_name: str, api_key: str, target: str, universe: Tuple[str, ...]):
    cfg = Config(data_source=source_name, twelvedata_api_key=api_key, interval="1d")
    return fetch_multi_timeframe(
        cfg, intervals=("15m", "1h", "1d"),
        target=target, universe=universe,
    )


multi_tf_data: Dict[str, Dict[str, pd.DataFrame]] = {"15m": {}, "1h": {}, "1d": {}}
try:
    multi_tf_data = _fetch_multi_tf(
        config.data_source, config.twelvedata_api_key,
        target=mset.target, universe=mset.available_universe(),
    )
except Exception as e:
    st.warning(f"Multi-timeframe data fetch failed: {e}. Signal engine disabled.")

active_target = mset.target
multi_tf_ready = all(
    active_target in multi_tf_data.get(tf, {}) and len(multi_tf_data[tf][active_target]) > 30
    for tf in ("15m", "1h")
)
# 1d is optional — only the per-horizon long card needs it
long_tf_ready = (
    active_target in multi_tf_data.get("1d", {}) and len(multi_tf_data["1d"].get(active_target, pd.DataFrame())) > 30
)


# -----------------------------------------------------------------------------
# Compute composite + forecasts + regime + flow
# -----------------------------------------------------------------------------
with st.spinner("Computing composite score…"):
    score_result = composite_score(data, config)
    fcasts = forecasts(data, config)
    f_labels, f_conf = {}, {}
    for name, s in fcasts.items():
        lbl, conf = forecast_classify(s)
        f_labels[name] = lbl
        f_conf[name]   = conf
    regime = market_regime(score_result.per_market)
    flow   = flow_meter(
        composite   = score_result.composite,
        vix_close   = data["vix"]["Close"]   if "vix"   in data else pd.Series(15.0,  index=score_result.composite.index),
        dxy_close   = data["dxy"]["Close"]   if "dxy"   in data else pd.Series(100.0, index=score_result.composite.index),
        silver_score= score_result.per_market.get("silver", pd.Series(0.0, index=score_result.composite.index)),
    )


# -----------------------------------------------------------------------------
# Humanize bar counts to wall-clock time
# -----------------------------------------------------------------------------
_blabel = INTERVAL_CONFIG[config.interval]["bar_label"]

def _humanize(bars: int) -> str:
    if config.interval == "1d":
        return f"~{bars} days"
    if config.interval == "1h":
        return f"~{bars/24:.1f} days" if bars >= 24 else f"~{bars} hours"
    hours = bars * {"30m": 0.5, "15m": 0.25, "5m": 5/60, "1m": 1/60}.get(config.interval, 1.0)
    if hours < 1:
        return f"~{int(hours*60)} min"
    if hours < 24:
        return f"~{hours:.1f} hours"
    return f"~{hours/24:.1f} days"


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title(f"🥇 {config.active_instrument} — Multi-Asset Composite Engine")
st.caption(
    f"Interval: **{INTERVAL_LABELS.get(config.interval, config.interval)}** · "
    f"Data: **{config.data_source}** · "
    f"Last bar: **{data[mset.target].index[-1].strftime('%Y-%m-%d %H:%M')}** · "
    f"Markets loaded: **{len(data)}/{len(MARKET_LABELS)}**"
)


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_dashboard, tab_live, tab_backtest, tab_trades, tab_signals, tab_sigbt, tab_alerts, tab_about = st.tabs(
    ["🌐  Dashboard", "📊  Live", "🧪  Backtest", "🎯  Trades", "⚡  Signals", "📈  Signal-BT", "🔔  Alerts", "ℹ️  About"]
)


# =============================================================================
# Gauge helper (used by Live tab)
# =============================================================================
def _gauge(value, title, min_val=0, max_val=100, suffix=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14}},
        number={"suffix": suffix, "font": {"size": 28}},
        gauge={
            "axis": {"range": [min_val, max_val]},
            "bar":  {"color": "#DAA520"},
            "steps": [
                {"range": [min_val, 40],         "color": "#f8d7da"},
                {"range": [40,         60],      "color": "#fff3cd"},
                {"range": [60,         max_val], "color": "#d4edda"},
            ],
        },
    ))
    fig.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=0))
    return fig


# =============================================================================
# DASHBOARD TAB — multi-instrument overview
# =============================================================================
@st.cache_data(ttl=600, show_spinner="Loading all instruments…")
def _fetch_all_instruments(source_name: str, api_key: str, interval: str):
    from config import INSTRUMENTS as _REG
    cfg = Config(data_source=source_name, twelvedata_api_key=api_key, interval=interval)
    symbols = [m.symbol for m in _REG]
    from data import fetch_all_instruments
    return fetch_all_instruments(cfg, symbols=symbols, max_workers=6)


with tab_dashboard:
    st.subheader("🌐 Multi-asset dashboard")
    st.caption(
        f"All instruments in the registry, scored on the **{config.interval}** "
        f"interval using **{config.data_source}**.  Click a card to drill into "
        f"that instrument."
    )

    # Optional: choose a subset of asset classes
    asset_classes = sorted(INSTRUMENTS_BY_CLASS.keys())
    selected_classes = st.multiselect(
        "Asset classes", asset_classes, default=asset_classes,
        key="dash_classes",
    )

    if st.button("🔄  Refresh all instruments", type="primary", key="dash_refresh"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Fetching data for all instruments in parallel…"):
        all_data = _fetch_all_instruments(
            config.data_source, config.twelvedata_api_key, config.interval,
        )

    if not all_data:
        st.error("No data returned. Check the data source / network.")
    else:
        st.session_state["dashboard_data"] = all_data

        # ---- Score every instrument ----
        st.session_state.setdefault("dashboard_scores", {})
        st.session_state.setdefault("dashboard_signals", {})
        if not st.session_state["dashboard_scores"] or st.button("Rescore", key="dash_rescore"):
            from scoring import composite_score, neg_corr_for, forecasts as scoring_forecasts
            from signals import evaluate_single_tf, SignalConfig
            scores = {}
            signals = {}
            progress = st.progress(0.0, text="Scoring…")
            n = len(all_data)
            for i, (sym, data) in enumerate(all_data.items()):
                mset = INSTRUMENT_BY_SYMBOL.get(sym)
                if mset is None or not data:
                    scores[sym] = None
                    signals[sym] = None
                    continue
                # Build a per-instrument Config
                icfg = Config(
                    data_source=config.data_source,
                    twelvedata_api_key=config.twelvedata_api_key,
                    interval=config.interval,
                    weights={k: float(v) for k, v in mset.weights.items()},
                    periods=dict(config.periods),
                    forecasts={k: dict(v) for k, v in config.forecasts.items()},
                    active_instrument=sym,
                )
                # Restrict data to the universe + target
                avail = mset.available_universe()
                target = mset.target
                restricted = {m: data[m] for m in ([target] + [u for u in avail if u != target]) if m in data}
                if target not in restricted or not restricted.get(target, pd.DataFrame()).size:
                    scores[sym] = None
                    signals[sym] = None
                    continue
                try:
                    sc = composite_score(restricted, icfg, neg_corr=neg_corr_for(sym))
                    scores[sym] = sc
                except Exception as e:
                    scores[sym] = None
                    signals[sym] = None
                    progress.progress((i + 1) / max(n, 1), text=f"Scored {i+1}/{n} instruments")
                    continue
                # ---- Per-instrument signal (single-TF, same logic as gold) ----
                try:
                    last_score = float(sc.composite.iloc[-1])
                    fcasts = scoring_forecasts(restricted, icfg)
                    f_dict = {h: str(forecast_classify(s)[0].iloc[-1]) for h, s in fcasts.items()}
                    last_price = float(restricted[target]["Close"].iloc[-1])
                    sig_cfg = SignalConfig()
                    signals[sym] = evaluate_single_tf(last_score, f_dict, last_price, sig_cfg)
                except Exception as e:
                    signals[sym] = None
                progress.progress((i + 1) / max(n, 1), text=f"Scored {i+1}/{n} instruments")
            progress.empty()
            st.session_state["dashboard_scores"] = scores
            st.session_state["dashboard_signals"] = signals

        scores = st.session_state["dashboard_scores"]
        signals = st.session_state["dashboard_signals"]

        # ---- Render cards by asset class ----
        SIGNAL_COLORS_DASH = {
            # Macro labels (used when no trade signal is computed)
            "Strong Bullish": "#006400", "Bullish": "#2E8B57", "Neutral": "#DAA520",
            "Bearish": "#B22222", "Strong Bearish": "#8B0000",
            # Trade signals (BUY / SELL / WAITBUY / WAITSELL / NOACTION)
            "BUY":      "#006400", "SELL":    "#8B0000",
            "WAITBUY":  "#2E8B57", "WAITSELL":"#B22222",
            "NOACTION": "#888",
        }

        def _card(symbol: str, sc, sig):
            mset = INSTRUMENT_BY_SYMBOL.get(symbol)
            if sc is None or mset is None:
                avail = mset.available_universe() if mset else ()
                missing = mset.missing_markets() if mset else []
                st.markdown(
                    f"""<div style="border:1px dashed #aaa;border-radius:8px;
                                padding:10px;height:140px;background:#f8f8f8">
                          <div style="font-weight:700;color:#666">{symbol}</div>
                          <div style="font-size:0.75rem;color:#999;margin-top:6px">
                            Data unavailable
                          </div>
                          {f'<div style="font-size:0.65rem;color:#bbb;margin-top:4px">Missing: {", ".join(missing[:3])}</div>' if missing else ''}
                        </div>""",
                    unsafe_allow_html=True,
                )
                return
            # ---- Trade signal (primary) ----
            if sig is not None:
                signal    = sig["signal"]
                prob      = sig["probability"]
                score     = sig["score"]
                bull_n    = sig["bull_count"]
                bear_n    = sig["bear_count"]
                arrow     = {"BUY": "▲", "SELL": "▼", "WAITBUY": "△", "WAITSELL": "▽", "NOACTION": "◆"}.get(signal, "◆")
                color     = SIGNAL_COLORS_DASH.get(signal, "#888")
                # Subtitle: count of bullish / bearish forecasts
                forecast_summary = f"{bull_n}↑ / {bear_n}↓"
            else:
                # Fallback to macro label if signal not computed
                last = sc.composite.index[-1]
                label = str(sc.label.loc[last])
                score = float(sc.composite.loc[last])
                conf  = float(sc.confidence.loc[last])
                prob  = 0.50 + abs(score) / 100.0 * 0.45
                prob  = min(0.95, max(0.50, prob))
                arrow = "▲" if score > 5 else ("▼" if score < -5 else "◆")
                color = SIGNAL_COLORS_DASH.get(label, "#DAA520")
                signal = label
                forecast_summary = f"conf {conf:.0f}%"
            st.markdown(
                f"""<div style="border:2px solid {color};border-radius:8px;
                            padding:10px;min-height:140px;background:{color}0d">
                      <div style="display:flex;justify-content:space-between;align-items:start">
                        <div style="font-weight:800;color:{color};font-size:0.95rem">{symbol}</div>
                        <div style="font-size:0.7rem;color:#888">{mset.asset_class}</div>
                      </div>
                      <div style="font-size:1.5rem;font-weight:800;color:{color};margin-top:4px">
                        {arrow} {signal}
                      </div>
                      <div style="font-size:0.75rem;color:#444;margin-top:4px">
                        Score {score:+.1f} · p={prob*100:.0f}% · {forecast_summary}
                      </div>
                      <div style="font-size:0.65rem;color:#888;margin-top:4px">
                        {mset.rationale[:60]}{'…' if len(mset.rationale) > 60 else ''}
                      </div>
                    </div>""",
                unsafe_allow_html=True,
            )

        for asset_class in selected_classes:
            st.markdown(f"#### {asset_class}")
            instruments_in_class = INSTRUMENTS_BY_CLASS.get(asset_class, [])
            # Render in 4-column grid
            cols_per_row = 4
            for row_start in range(0, len(instruments_in_class), cols_per_row):
                row = instruments_in_class[row_start:row_start + cols_per_row]
                cols = st.columns(cols_per_row)
                for col, mset in zip(cols, row):
                    with col:
                        _card(mset.symbol, scores.get(mset.symbol), signals.get(mset.symbol))
                        if scores.get(mset.symbol) is not None:
                            if st.button(f"Open {mset.symbol}", key=f"open_{mset.symbol}",
                                          use_container_width=True):
                                st.session_state.active_instrument = mset.symbol
                                st.rerun()
                        else:
                            st.caption("—")

        # ---- Bulk stats ----
        st.markdown("#### 📊 Bulk stats")
        scored = [(sym, sc, signals.get(sym)) for sym, sc in scores.items() if sc is not None]
        if scored:
            # Count by trade signal
            n_buy  = sum(1 for _, _, sg in scored if sg and sg["signal"] == "BUY")
            n_sell = sum(1 for _, _, sg in scored if sg and sg["signal"] == "SELL")
            n_wb   = sum(1 for _, _, sg in scored if sg and sg["signal"] == "WAITBUY")
            n_ws   = sum(1 for _, _, sg in scored if sg and sg["signal"] == "WAITSELL")
            n_no   = sum(1 for _, _, sg in scored if sg and sg["signal"] == "NOACTION")
            n_unk  = sum(1 for _, _, sg in scored if sg is None)
            bc1, bc2, bc3, bc4, bc5 = st.columns(5)
            bc1.metric("BUY",      n_buy,  delta=f"{n_buy*100/len(scored):.0f}% of universe")
            bc2.metric("WAITBUY",  n_wb,   delta=f"{n_wb*100/len(scored):.0f}% of universe")
            bc3.metric("NOACTION", n_no,   delta=f"{n_no*100/len(scored):.0f}% of universe", delta_color="inverse")
            bc4.metric("WAITSELL", n_ws,   delta=f"{n_ws*100/len(scored):.0f}% of universe", delta_color="inverse")
            bc5.metric("SELL",     n_sell, delta=f"{n_sell*100/len(scored):.0f}% of universe", delta_color="inverse")
            if n_unk:
                st.caption(f"({n_unk} instruments have no signal computed)")

            # Top 5 BUY candidates
            st.markdown("**Top 5 BUY candidates**")
            buy_rows = [
                {"Symbol": sym, "Score": float(sc.composite.iloc[-1]),
                 "Signal": sg["signal"] if sg else "—",
                 "Prob":   f"{(sg['probability']*100 if sg else 0):.0f}%",
                 "Conf":   f"{float(sc.confidence.iloc[-1]):.0f}%"}
                for sym, sc, sg in scored
                if sg and sg["signal"] == "BUY"
            ]
            if buy_rows:
                st.dataframe(pd.DataFrame(buy_rows).sort_values("Score", ascending=False).head(5),
                             use_container_width=True, height=200)
            else:
                st.caption("No BUY signals active right now.")

            st.markdown("**Top 5 SELL candidates**")
            sell_rows = [
                {"Symbol": sym, "Score": float(sc.composite.iloc[-1]),
                 "Signal": sg["signal"] if sg else "—",
                 "Prob":   f"{(sg['probability']*100 if sg else 0):.0f}%",
                 "Conf":   f"{float(sc.confidence.iloc[-1]):.0f}%"}
                for sym, sc, sg in scored
                if sg and sg["signal"] == "SELL"
            ]
            if sell_rows:
                st.dataframe(pd.DataFrame(sell_rows).sort_values("Score", ascending=True).head(5),
                             use_container_width=True, height=200)
            else:
                st.caption("No SELL signals active right now.")
        else:
            st.info("No instruments scored yet.")


# =============================================================================
# LIVE TAB
# =============================================================================
with tab_live:
    setup_auto_refresh(interval_seconds=900)  # 15 min

    # Session-state for Telegram
    if "tg_state" not in st.session_state:
        st.session_state.tg_state = TgState()
    if "tg_history" not in st.session_state:
        st.session_state.tg_history = AlertHistory()
    if "tg_last_fired_msg" not in st.session_state:
        st.session_state.tg_last_fired_msg = None
    if "tg_cmd_state" not in st.session_state:
        st.session_state.tg_cmd_state = TgCommandState()

    # Two-way Telegram: process commands + send open signal
    if tg_is_configured():
        def _get_signal_result():
            if not multi_tf_ready:
                return None
            try:
                return evaluate_signal(
                    data_15m=multi_tf_data["15m"],
                    data_1h =multi_tf_data["1h"],
                    config  =config,
                    cfg     =st.session_state.get("sig_cfg", SignalConfig()),
                )
            except Exception:
                return None

        ctx = {
            "config":            config,
            "data":              data,
            "score_result":      score_result,
            "regime":            regime,
            "flow":              flow,
            "multi_tf_data":     multi_tf_data,
            "get_signal_result": _get_signal_result,
            "render_chart":      render_chart_snapshot,
        }
        try:
            for cmd_text, ok in _tg_process_commands(st.session_state.tg_cmd_state, ctx):
                st.session_state.tg_history.add(
                    f"[CMD] {cmd_text} — {'ok' if ok else 'failed'}",
                    "sent" if ok else "failed",
                )
        except Exception as e:
            st.session_state.tg_history.add(f"Command processing failed: {e}", "failed", error=str(e))

        if multi_tf_ready and st.session_state.get("tg_open_signal_enabled", True):
            try:
                sent = _tg_send_open_signal(
                    state                =st.session_state.tg_cmd_state,
                    get_signal_result    =_get_signal_result,
                    format_signal_message=format_signal_message,
                    render_chart_fn      =render_chart_snapshot,
                    multi_tf_data        =multi_tf_data,
                )
                if sent:
                    st.session_state.tg_history.add("[OPEN] " + sent[:80], "sent")
            except Exception as e:
                st.session_state.tg_history.add(f"Open-signal send failed: {e}", "failed", error=str(e))

    # Auto-check alert conditions
    if st.session_state.get("tg_cfg_enabled", False) and tg_is_configured() and multi_tf_ready and "sig_cfg" in st.session_state:
        try:
            sig_res = evaluate_signal(
                data_15m=multi_tf_data["15m"],
                data_1h =multi_tf_data["1h"],
                config  =config,
                cfg     =st.session_state.sig_cfg,
            )
            last_sig = st.session_state.get("last_auto_signal", None)
            if sig_res.signal in ("BUY", "SELL") and sig_res.signal != last_sig:
                caption, _ = format_signal_message(sig_res)
                png = render_chart_snapshot(multi_tf_data["1h"], multi_tf_data["15m"], sig_res)
                if png is not None:
                    r = tg_send_photo(png, caption=caption[:1024])
                else:
                    r = tg_send(caption)
                st.session_state.tg_history.add(
                    f"[AUTO] {caption[:80]}", "sent" if r["ok"] else "failed",
                    error=r.get("error") if not r["ok"] else None,
                )
                st.session_state.last_auto_signal = sig_res.signal
        except Exception as e:
            st.session_state.tg_history.add(f"Multi-TF signal check failed: {e}", "failed", error=str(e))

    if st.session_state.tg_last_fired_msg:
        st.success(f"📨 Telegram alert sent: {st.session_state.tg_last_fired_msg[:120]}…")

    # ---- Composite / forecasts / regime / flow cards (legacy) --------
    latest = score_result.composite.index[-1]
    last_score   = float(score_result.composite.loc[latest])
    last_label   = str(score_result.label.loc[latest])
    last_conf    = float(score_result.confidence.loc[latest])
    last_color   = SIGNAL_COLOURS[last_label]
    last_flow    = float(flow.loc[latest])

    col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 1])
    with col_a:
        st.markdown(
            f"""<div style="background:{last_color}1a;border-left:5px solid {last_color};
                        padding:10px 14px;border-radius:6px;display:flex;align-items:center;gap:14px">
                  <div>
                    <div style="font-size:0.7rem;color:#666;letter-spacing:0.04em">MACRO (7-MARKET COMPOSITE)</div>
                    <div style="font-size:1.15rem;font-weight:700;color:{last_color};line-height:1.1">
                      {last_label}
                    </div>
                  </div>
                  <div style="font-size:0.85rem;color:#888">score {last_score:+.1f} · conf {last_conf:.0f}%</div>
                </div>""",
            unsafe_allow_html=True,
        )
        st.caption("ℹ️ Composite is the macro read.  Action signals come from the three forecast horizons below.")
    with col_b:
        st.metric("Institutional Flow", f"{last_flow:.0f}/100", delta=f"{last_flow - 50:+.0f} vs neutral")
    with col_c:
        st.metric("Composite score", f"{last_score:+.1f}")
    with col_d:
        st.metric("Last bar", latest.strftime('%Y-%m-%d %H:%M'))

    # Per-market breakdown
    st.subheader("Market breakdown")
    contribs = score_result.contributions
    weights  = config.normalized_weights()
    try:
        from config import NEGATIVE_CORRELATIONS
    except ImportError:
        from engine.config import NEGATIVE_CORRELATIONS
    rows = []
    for mkt, label in MARKET_LABELS.items():
        if mkt not in score_result.per_market:
            continue
        s = float(score_result.per_market[mkt].loc[latest])
        w = weights[mkt] * 100
        c = float(contribs[mkt].loc[latest])
        direction = "▼" if c < 0 else "▲"
        is_flipped = mkt in NEGATIVE_CORRELATIONS
        raw = -s if is_flipped else s
        flip_badge = " 🔄" if is_flipped else ""
        rows.append({
            "Market":       f"{label}{flip_badge}",
            "Raw score":    f"{raw:+.1f}",
            "Sign-flip":    "inverted" if is_flipped else "—",
            "Asset Score":  f"{s:+.1f}",
            "Weight":       f"{w:.1f}%",
            "Contribution": f"{c:+.1f}",
            "Direction":    direction,
        })
    if rows:
        df_mkts = pd.DataFrame(rows).set_index("Market")
        st.caption(
            f"🔄 = sign-inverted for gold (currently: "
            f"{', '.join(sorted(NEGATIVE_CORRELATIONS)) or 'none'})"
        )
        st.dataframe(df_mkts, use_container_width=True, height=290)

    # Underlying price action
    st.subheader("📈 Underlying price action")
    st.caption("Normalised close of each market (rebased to 100 at the start of the window) so you can compare trend direction at a glance.")
    try:
        # Rebase each market to 100 at the start of the window
        n_bars = min(120, len(score_result.composite))
        rebased = {}
        for mkt, df in data.items():
            if df is None or len(df) == 0 or "Close" not in df.columns:
                continue
            close = df["Close"].dropna()
            if len(close) < 2:
                continue
            close = close.tail(n_bars)
            base = float(close.iloc[0]) if float(close.iloc[0]) != 0 else 1.0
            rebased[MARKET_LABELS.get(mkt, mkt)] = close / base * 100.0
        if rebased:
            fig_price = go.Figure()
            for label, s in rebased.items():
                fig_price.add_trace(go.Scatter(
                    x=s.index, y=s.values, name=label, mode="lines", line=dict(width=1.6),
                ))
            fig_price.add_hline(y=100, line_dash="dot", line_color="#888", line_width=1)
            fig_price.update_layout(
                height=380, template="plotly_white",
                yaxis_title="Rebased = 100",
                margin=dict(l=10, r=10, t=20, b=10),
                hovermode="x unified",
                legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left"),
            )
            st.plotly_chart(fig_price, use_container_width=True)
    except Exception as e:
        st.warning(f"Price action chart failed: {e}")

    # Forecast horizons (per market)
    st.subheader("🔮 Forecast horizons")
    st.caption("Short / medium / long-horizon forecast for gold (latest bar).")
    try:
        # f_labels and f_conf are dicts {"short": Series, "medium": Series, "long": Series}
        rows = []
        for h_name in ("short", "medium", "long"):
            if h_name not in f_labels:
                continue
            lbl_series = f_labels[h_name]
            conf_series = f_conf[h_name]
            label_at_last = str(lbl_series.iloc[-1]) if len(lbl_series) else "—"
            conf_at_last  = float(conf_series.iloc[-1]) if len(conf_series) else 0.0
            rows.append({
                "Horizon":  h_name,
                "Label":    label_at_last,
                "Conf %":   int(round(conf_at_last)),
                "Score":    f"{float(fcasts[h_name].iloc[-1]):+.1f}" if h_name in fcasts else "—",
            })
        if rows:
            fdf = pd.DataFrame(rows).set_index("Horizon")
            def _color(val):
                if val in ("Strong Bullish", "Bullish"):
                    return "background-color: #2E8B5733"
                if val in ("Strong Bearish", "Bearish"):
                    return "background-color: #B2222233"
                if val in ("Neutral",):
                    return "background-color: #DAA52033"
                return ""
            st.dataframe(
                fdf.style.map(_color, subset=["Label"]),
                use_container_width=True, height=160,
            )
            # Forecast score over time (all 3 horizons on one chart)
            fig_f = go.Figure()
            for h_name, color in [("short", "#1f77b4"), ("medium", "#DAA520"), ("long", "#7f7f7f")]:
                if h_name in fcasts:
                    s = fcasts[h_name].tail(120)
                    fig_f.add_trace(go.Scatter(
                        x=s.index, y=s.values, name=f"{h_name}-horizon",
                        line=dict(color=color, width=1.8),
                    ))
            fig_f.add_hline(y=0, line_dash="dot", line_color="#888")
            fig_f.update_layout(
                height=260, template="plotly_white",
                yaxis_title="Forecast score (−100…+100)",
                margin=dict(l=10, r=10, t=20, b=10),
                hovermode="x unified",
                legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left"),
            )
            st.plotly_chart(fig_f, use_container_width=True)
    except Exception as e:
        st.warning(f"Forecast horizons table failed: {e}")

    # Composite history
    st.subheader("📊 Composite history")
    st.caption("Time series of the 7-market macro composite (last 200 bars).")
    try:
        n_bars = min(200, len(score_result.composite))
        comp_tail = score_result.composite.tail(n_bars)
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(
            x=comp_tail.index, y=comp_tail.values,
            name="Composite", line=dict(color="#DAA520", width=2),
            fill="tozeroy", fillcolor="rgba(218,165,32,0.15)",
        ))
        fig_c.add_hline(y=0, line_dash="dot", line_color="#888", line_width=1)
        fig_c.add_hrect(y0=-20, y1=20, line_width=0, fillcolor="rgba(128,128,128,0.08)")
        fig_c.update_layout(
            height=300, template="plotly_white",
            yaxis_title="Composite (−100…+100)",
            margin=dict(l=10, r=10, t=20, b=10),
            hovermode="x unified",
        )
        st.plotly_chart(fig_c, use_container_width=True)
    except Exception as e:
        st.warning(f"Composite history chart failed: {e}")

    # ---- 🎯 Forecast horizons — pickable cards + Live entry setup ----
    if multi_tf_ready and "sig_cfg" in st.session_state:
        sig_cfg_live = st.session_state.sig_cfg
        # Lazily evaluate the signal result for the Live tab
        try:
            res_live = evaluate_signal(
                data_15m=multi_tf_data["15m"],
                data_1h =multi_tf_data["1h"],
                config  =config,
                cfg     =sig_cfg_live,
            )
        except Exception as e:
            st.warning(f"Signal evaluation failed: {e}")
            res_live = None

        if res_live is not None:
            st.markdown("##### 🎯 Forecast horizons — pick one for the entry setup")

            if "active_horizon" not in st.session_state:
                st.session_state.active_horizon = "medium"

            base_gold = data.get("gold") if "gold" in data else None
            interval_label = config.interval  # e.g. "1d", "1h", "15m", etc.
            horizon_meta = {
                "short":  {"tf": interval_label, "data": base_gold,
                           "label": f"SHORT · EMA 10/20 · {interval_label}",  "use": "Use short for entry"},
                "medium": {"tf": interval_label, "data": base_gold,
                           "label": f"MEDIUM · EMA 20/50 · {interval_label}", "use": "Use medium for entry"},
                "long":   {"tf": interval_label, "data": base_gold,
                           "label": f"LONG · EMA 50/200 · {interval_label}",   "use": "Use long for entry"},
            }

            side = 1 if res_live.signal == "BUY" else (-1 if res_live.signal == "SELL" else 0)

            # 3 cards: short / medium / long
            hcols = st.columns(3)
            for i, h in enumerate(("short", "medium", "long")):
                with hcols[i]:
                    meta = horizon_meta[h]
                    gold_df = meta["data"]
                    if gold_df is None or len(gold_df) < 30:
                        st.markdown(
                            f"""<div style="border:1px dashed #aaa;border-radius:8px;
                                        padding:10px;height:160px;background:#f5f5f5">
                                  <div style="font-weight:700">{meta['label']}</div>
                                  <div style="color:#999;font-size:0.85rem;margin-top:6px">
                                    Data unavailable
                                  </div>
                                </div>""",
                            unsafe_allow_html=True,
                        )
                        continue
                    setup = horizon_entry_sltp(gold_df, side if side else 1, sig_cfg_live, h)
                    fc_label = "—"
                    fc_score = 0.0
                    fc_conf  = 0.0
                    try:
                        fc_label = str(f_labels[h].iloc[-1])
                        fc_score = float(fcasts[h].iloc[-1])
                        fc_conf  = float(f_conf[h].iloc[-1])
                    except Exception:
                        pass
                    bg     = "#d4edda" if fc_label == "Bullish" else ("#f8d7da" if fc_label == "Bearish" else "#fff3cd")
                    border = "#2E8B57" if fc_label == "Bullish" else ("#B22222" if fc_label == "Bearish" else "#DAA520")
                    is_active = (st.session_state.active_horizon == h)
                    active_badge = (
                        f'<div style="margin-top:6px;font-size:0.7rem;color:#fff;background:{border};'
                        f'padding:2px 6px;border-radius:3px;display:inline-block">✓ Active</div>'
                        if is_active else
                        f'<div style="margin-top:6px;font-size:0.7rem;color:#666">—</div>'
                    )
                    ema_part = " — "
                    if setup["ema_fast"] is not None and setup["ema_slow"] is not None:
                        ema_part = f"{setup['ema_fast']:.1f} → {setup['ema_slow']:.1f}"
                    fc_score_str = f"+{fc_score:.1f}" if fc_score >= 0 else f"{fc_score:.1f}"
                    st.markdown(
                        f"""<div style="border:2px solid {border};background:{bg};
                                    border-radius:8px;padding:10px;min-height:170px">
                              <div style="font-size:0.7rem;font-weight:700;color:#444;letter-spacing:0.05em">
                                {meta['label'].split(' · ')[0]}
                              </div>
                              <div style="font-size:0.7rem;color:#666;margin-top:2px">{ema_part}</div>
                              <div style="font-size:1.2rem;font-weight:800;color:{border};margin-top:6px">
                                {fc_label}
                              </div>
                              <div style="font-size:0.78rem;color:#444;margin-top:4px">
                                Score {fc_score_str} · Conf {int(round(fc_conf*100))}%
                              </div>
                              {active_badge}
                            </div>""",
                        unsafe_allow_html=True,
                    )
                    if not is_active:
                        if st.button(meta["use"], key=f"live_use_{h}", use_container_width=True):
                            st.session_state.active_horizon = h
                            st.rerun()
                    else:
                        st.button("✓ Selected", key=f"live_sel_{h}", use_container_width=True, disabled=True)

            # Reference table
            st.markdown("")
            try:
                ref_rows = []
                for h in ("short", "medium", "long"):
                    gold_df = horizon_meta[h]["data"]
                    if gold_df is None or len(gold_df) < 30:
                        ref_rows.append({"Horizon": h.upper(), "Forecast": "—", "EMA fast": "—", "EMA slow": "—", "ATR": "—"})
                        continue
                    setup = horizon_entry_sltp(gold_df, side if side else 1, sig_cfg_live, h)
                    fc_label = "—"
                    try:
                        fc_label = str(f_labels[h].iloc[-1])
                    except Exception:
                        pass
                    ref_rows.append({
                        "Horizon":   h.upper(),
                        "Forecast":  fc_label,
                        "EMA fast":  f"{setup['ema_fast']:.1f}" if setup['ema_fast'] is not None else "—",
                        "EMA slow":  f"{setup['ema_slow']:.1f}" if setup['ema_slow'] is not None else "—",
                        "ATR":       f"{setup['atr']:.2f}",
                    })
                rdf = pd.DataFrame(ref_rows).set_index("Horizon")
                def _color_live(val):
                    if val == "Bullish":  return "background-color: #d4edda"
                    if val == "Bearish":  return "background-color: #f8d7da"
                    if val == "Neutral":  return "background-color: #fff3cd"
                    return ""
                st.dataframe(rdf.style.map(_color_live, subset=["Forecast"]), use_container_width=True, height=110)
            except Exception as e:
                st.warning(f"Reference table failed: {e}")

            # Active horizon entry-setup chart
            active = st.session_state.active_horizon
            active_meta = horizon_meta[active]
            active_df = active_meta["data"]
            if active_df is None or len(active_df) < 5:
                st.info(f"Gold chart for {active_meta['tf']} is not available.")
            else:
                side_label = "Long" if side == 1 else ("Short" if side == -1 else "—")
                st.markdown(f"##### 🎯 Live entry setup — {active} horizon")
                try:
                    setup = horizon_entry_sltp(active_df, side if side else 1, sig_cfg_live, active)
                    am1, am2, am3, am4, am5 = st.columns(5)
                    am1.metric("Action", side_label, delta=("↑ Long" if side == 1 else ("↓ Short" if side == -1 else "—")))
                    if setup["entry"] is not None:
                        am2.metric("Entry",   f"{setup['entry']:,.2f}",   delta=f"{setup['entry'] - setup['last_price']:+.1f}")
                        am3.metric("Stop",    f"{setup['stop']:,.2f}",    delta=f"-{abs(setup['entry'] - setup['stop']):.1f} pts", delta_color="inverse")
                        am4.metric("Target",  f"{setup['target']:,.2f}",  delta=f"+{abs(setup['target'] - setup['entry']):.1f} pts")
                        risk   = abs(setup['entry'] - setup['stop'])
                        reward = abs(setup['target'] - setup['entry'])
                        am5.metric("R:R",     f"{reward/risk:.2f}" if risk > 0 else "—")
                    fc_label = "—"
                    fc_score = 0.0
                    fc_conf  = 0.0
                    try:
                        fc_label = str(f_labels[active].iloc[-1])
                        fc_score = float(fcasts[active].iloc[-1])
                        fc_conf  = float(f_conf[active].iloc[-1])
                    except Exception:
                        pass
                    bias = "fast > slow" if (setup["ema_fast"] and setup["ema_slow"] and setup["ema_fast"] > setup["ema_slow"]) else "fast < slow"
                    fast_v = f"{setup['ema_fast']:.1f}" if setup["ema_fast"] else "—"
                    slow_v = f"{setup['ema_slow']:.1f}" if setup["ema_slow"] else "—"
                    last_v = f"{setup['last_price']:.1f}" if setup["last_price"] else "—"
                    stop_v = f"{setup['stop']:.1f}" if setup["stop"] else "—"
                    tgt_v  = f"{setup['target']:.1f}" if setup["target"] else "—"
                    st.caption(
                        f"**{active.upper()} forecast:** {fc_label} "
                        f"(score {fc_score:+.1f}, conf {fc_conf*100:.1f}%) · "
                        f"**EMA bias:** {bias} (fast={fast_v} slow={slow_v}) · "
                        f"Last {last_v} · Stop {stop_v} · Target {tgt_v} · R:R "
                        f"{(abs(setup['target']-setup['entry'])/max(abs(setup['entry']-setup['stop']),1e-9)):.2f}"
                    )
                    tail = active_df.tail(120).copy()
                    fig_e = go.Figure()
                    fig_e.add_trace(go.Candlestick(
                        x=tail.index,
                        open=tail["Open"], high=tail["High"],
                        low=tail["Low"], close=tail["Close"],
                        name="Gold", increasing_line_color="#2E8B57",
                        decreasing_line_color="#B22222",
                    ))
                    from indicators import ema as _ema_fn_local
                    ef_len = {"short": 10, "medium": 20, "long": 50}.get(active, 20)
                    es_len = {"short": 20, "medium": 50, "long": 200}.get(active, 50)
                    ef_full = _ema_fn_local(tail["Close"], ef_len)
                    es_full = _ema_fn_local(tail["Close"], es_len)
                    fig_e.add_trace(go.Scatter(
                        x=tail.index, y=ef_full.values,
                        name="EMA fast", line=dict(color="#1f77b4", width=1.4, dash="dot"),
                    ))
                    fig_e.add_trace(go.Scatter(
                        x=tail.index, y=es_full.values,
                        name="EMA slow", line=dict(color="#ff7f0e", width=1.6),
                    ))
                    if setup["entry"] is not None:
                        line_color = {"BUY": "#2E8B57", "SELL": "#B22222"}.get(res_live.signal, "#DAA520")
                        fig_e.add_hline(
                            y=setup["entry"], line_dash="solid", line_color=line_color, line_width=2,
                            annotation_text=f"Entry {setup['entry']:,.2f}",
                            annotation_position="right", annotation_font_color=line_color,
                        )
                        fig_e.add_hrect(
                            y0=min(setup["entry"], setup["stop"]), y1=max(setup["entry"], setup["stop"]),
                            fillcolor="rgba(178,34,34,0.12)", line_width=0,
                        )
                        fig_e.add_hline(
                            y=setup["stop"], line_dash="dash", line_color="#B22222", line_width=1.5,
                            annotation_text=f"Stop {setup['stop']:,.2f}",
                            annotation_position="right", annotation_font_color="#B22222",
                        )
                        fig_e.add_hrect(
                            y0=min(setup["entry"], setup["target"]), y1=max(setup["entry"], setup["target"]),
                            fillcolor="rgba(46,139,87,0.12)", line_width=0,
                        )
                        fig_e.add_hline(
                            y=setup["target"], line_dash="dash", line_color="#2E8B57", line_width=1.5,
                            annotation_text=f"Target {setup['target']:,.2f}",
                            annotation_position="right", annotation_font_color="#2E8B57",
                        )
                    fig_e.update_layout(
                        height=460, template="plotly_white",
                        yaxis_title="Gold price",
                        xaxis_rangeslider_visible=False,
                        margin=dict(l=10, r=10, t=20, b=10),
                        hovermode="x unified",
                        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left"),
                    )
                    st.plotly_chart(fig_e, use_container_width=True)
                except Exception as e:
                    st.warning(f"Entry-setup chart failed: {e}")
    else:
        if not multi_tf_ready:
            pass  # already warned above
        else:
            st.info("Set up a signal config in the **Signals** tab first to enable the Live entry setup.")

    # Regime + flow
    st.subheader("Regime & flow")
    r1, r2 = st.columns([1, 1])
    with r1:
        reg_now = regime.loc[latest]
        rc = REGIME_COLOURS.get(reg_now, "#DAA520")
        st.markdown(
            f"""<div style="background:{rc}1a;border-left:5px solid {rc};
                        padding:10px 12px;border-radius:6px">
                  <div style="font-size:0.78rem;color:#666">Market Regime</div>
                  <div style="font-size:1.1rem;font-weight:700;color:{rc}">{reg_now}</div>
                </div>""",
            unsafe_allow_html=True,
        )
        st.caption("Gold-Friendly: DXY<−20 ∧ VIX<−10 ∧ SP500>10 · Gold-Hostile: DXY>20 ∧ VIX>10 ∧ SP500<−10 · otherwise Transition")
    with r2:
        st.plotly_chart(_gauge(last_flow, "Institutional Flow Meter"), use_container_width=True)


# =============================================================================
# BACKTEST TAB
# =============================================================================
with tab_backtest:
    st.subheader("🧪 Historical backtest")
    st.caption(
        f"Strategy: long Gold when the model says **Bullish / Strong Bullish**, "
        f"cash otherwise. Returns on **{config.interval} bars**. "
        f"No look-ahead — signal at bar T only uses bars up to T."
    )

    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        start = st.date_input("Start", value=datetime(2018, 1, 1), key="bt_start")
    with bc2:
        end   = st.date_input("End", value=datetime.now(), key="bt_end")
    with bc3:
        _max_hold = {"1d": 250, "1h": 500, "30m": 500, "15m": 500, "5m": 1000, "1m": 1000}.get(config.interval, 500)
        _default_hold = {"1d": 5, "1h": 12, "30m": 24, "15m": 32, "5m": 96, "1m": 240}.get(config.interval, 5)
        hold = st.number_input(f"Holding period ({_blabel}s)", 1, _max_hold, _default_hold, key="bt_hold")

    bt_cfg = BacktestConfig(start=start.isoformat(), end=end.isoformat(), holding_period=int(hold))
    with st.spinner("Running backtest…"):
        try:
            result = run_backtest(data, config, bt_cfg, target=mset.target)
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            st.stop()

    m = result.metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Strategy CAGR",  f"{m['Strategy CAGR %']:.2f}%",
               delta=f"{m['Strategy CAGR %'] - m['Benchmark CAGR %']:.2f}% vs BH")
    mc2.metric("Strategy Sharpe", f"{m['Strategy Sharpe']:.2f}",
               delta=f"{m['Strategy Sharpe'] - m['Benchmark Sharpe']:.2f}")
    mc3.metric("Strategy Max DD", f"{m['Strategy Max DD %']:.2f}%",
               delta=f"{m['Strategy Max DD %'] - m['Benchmark Max DD %']:.2f}% vs BH",
               delta_color="inverse")
    mc4.metric("Strategy Total Ret", f"{m['Strategy Total Ret %']:.2f}%",
               delta=f"{m['Strategy Total Ret %'] - m['Benchmark Total Ret %']:.2f}% vs BH")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.equity.index, y=result.equity.values,
                             name="Strategy (long-flat)", line=dict(color="#DAA520", width=2)))
    fig.add_trace(go.Scatter(x=result.benchmark.index, y=result.benchmark.values,
                             name="Buy & Hold Gold", line=dict(color="#888", width=1.5, dash="dot")))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=0),
                      yaxis_title="Equity ($)", template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Per-signal statistics (forward return, %)")
    st.dataframe(result.summary, use_container_width=True)

    st.markdown("##### Last 30 signals")
    st.dataframe(
        result.signals.tail(30).iloc[::-1]
            .assign(composite=lambda d: d["composite"].round(1),
                    fwd_return=lambda d: (d["fwd_return"] * 100).round(2))
            .rename(columns={"fwd_return": "fwd_ret_%"}),
        use_container_width=True, height=380,
    )


# =============================================================================
# TRADES TAB
# =============================================================================
with tab_trades:
    st.subheader("🎯 Trade Engine — R-based execution")
    st.caption("Pullback entries · ATR stops · structural stop blend · automatic breakeven at +1R · risk-based sizing · both directions.")

    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        risk_pct = st.number_input("Risk per trade (% equity)", 0.1, 5.0, 1.0, step=0.1, key="t_risk") / 100.0
    with tc2:
        k_sl = st.number_input("Stop (× ATR)", 0.5, 5.0, 1.5, step=0.1, key="t_sl")
    with tc3:
        k_tp = st.number_input("Target (× ATR)", 1.0, 10.0, 3.0, step=0.1, key="t_tp")
    with tc4:
        min_rr = st.number_input("Min R:R filter", 1.0, 5.0, 1.5, step=0.1, key="t_rr")
    tc5, tc6, tc7, tc8 = st.columns(4)
    with tc5:
        use_be = st.checkbox("Breakeven at +1R", value=True, key="t_be")
    with tc6:
        swing_lb = st.number_input("Swing lookback (bars)", 0, 50, 10, step=1, key="t_swing")
    with tc7:
        pb_frac = st.number_input("Pullback depth (× ATR)", 0.0, 2.0, 0.5, step=0.1, key="t_pb")
    with tc8:
        entry_win = st.number_input("Entry window (bars)", 1, 20, 3, step=1, key="t_ew")

    trade_cfg = EngineTradeConfig(
        risk_per_trade=float(risk_pct), k_sl=float(k_sl), k_tp=float(k_tp),
        min_rr=float(min_rr), use_breakeven_be=bool(use_be),
        swing_lookback=int(swing_lb), pullback_atr_frac=float(pb_frac),
        entry_window=int(entry_win),
    )

    with st.spinner("Running trade engine…"):
        try:
            trade_log = run_trade_engine(
                target=data[mset.target], composite=score_result.composite,
                labels=score_result.label, confidence=score_result.confidence,
                cfg=trade_cfg, initial_equity=10_000.0,
            )
            render_trade_dashboard(trade_log, data[mset.target], config)
        except Exception as e:
            st.error(f"Trade engine failed: {e}")


# =============================================================================
# SIGNALS TAB  (new macro-composite engine)
# =============================================================================
with tab_signals:
    st.subheader("⚡ Macro-composite signal engine")
    st.caption(
        "**BUY** when 15m macro > 1h macro AND all 3 of {short, medium, long} "
        "15m forecasts = Bullish. **SELL** mirror.  Entry / SL / TP included."
    )

    if not multi_tf_ready:
        st.error("Multi-timeframe data not ready.  Check data source / network.")
    else:
        if "sig_cfg" not in st.session_state:
            st.session_state.sig_cfg = SignalConfig()
        if "sig_state" not in st.session_state:
            st.session_state.sig_state = SignalState()
        sig_cfg = st.session_state.sig_cfg

        with st.expander("⚙️  Signal thresholds", expanded=False):
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                sig_cfg.min_composite_buy  = st.number_input("Min 15m composite (BUY)",  -100.0, 100.0, sig_cfg.min_composite_buy,  5.0)
                sig_cfg.min_composite_sell = st.number_input("Max 15m composite (SELL)", -100.0, 100.0, -sig_cfg.min_composite_sell, 5.0)
            with cc2:
                sig_cfg.forecast_alignment_required = st.selectbox(
                    "Forecast alignment required", [1, 2, 3],
                    index=sig_cfg.forecast_alignment_required - 1,
                    help="How many of the 3 15m forecasts must agree.  3 = strict, 1 = lenient.",
                )
                sig_cfg.min_probability = st.slider("Min probability", 0.0, 1.0, sig_cfg.min_probability, 0.05)
            with cc3:
                sig_cfg.momentum_gap_required = st.checkbox(
                    "Require 15m > 1h (momentum building)",
                    value=sig_cfg.momentum_gap_required,
                )

        try:
            res = evaluate_signal(
                data_15m=multi_tf_data["15m"],
                data_1h =multi_tf_data["1h"],
                config  =config,
                cfg     =sig_cfg,
            )
        except Exception as e:
            st.error(f"Signal evaluation failed: {e}")
            st.stop()

        # Big signal banner
        sig_color = {
            "BUY": "#2E8B57", "SELL": "#B22222",
            "WAITBUY": "#90EE90", "WAITSELL": "#FFA07A", "NOACTION": "#888",
        }.get(res.signal, "#888")

        st.markdown(
            f"""
            <div style="background:{sig_color}22;border-left:8px solid {sig_color};
                        padding:18px 20px;border-radius:8px;margin:12px 0">
              <div style="font-size:0.85rem;color:#666;letter-spacing:0.06em">CURRENT SIGNAL</div>
              <div style="font-size:2.2rem;font-weight:800;color:{sig_color}">
                {res.signal} &nbsp;·&nbsp; p={res.probability*100:.0f}%
              </div>
              <div style="font-size:0.95rem;color:#444;margin-top:4px">
                Gold last <b>{res.last_price:,.2f}</b> &nbsp;·&nbsp;
                15m composite <b>{res.composite_15m:+.1f}</b> &nbsp;·&nbsp;
                1h composite <b>{res.composite_1h:+.1f}</b> &nbsp;·&nbsp;
                Momentum: <b>{'building' if res.momentum_ok else 'fading'}</b>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Entry/SL/TP card (only when BUY or SELL)
        if res.entry is not None:
            tc1, tc2, tc3, tc4 = st.columns(4)
            tc1.metric("Entry",   f"{res.entry:,.2f}")
            tc2.metric("Stop",    f"{res.stop:,.2f}", delta=f"-{res.entry - res.stop:.1f} pts", delta_color="inverse")
            tc3.metric("Target",  f"{res.target:,.2f}", delta=f"+{res.target - res.entry:.1f} pts")
            tc4.metric("R:R",     f"{res.risk_reward:.2f}")

        # ---- Per-horizon entry setup (Signals tab) ---------------------
        st.markdown("##### 🎯 Forecast horizons — pick one for the entry setup")

        if "active_horizon" not in st.session_state:
            st.session_state.active_horizon = "medium"

        base_gold_sig = data.get("gold") if "gold" in data else None
        interval_label_sig = config.interval
        sig_horizon_meta = {
            "short":  {"tf": interval_label_sig, "data": base_gold_sig,
                       "label": f"SHORT · EMA 10/20 · {interval_label_sig}",  "use": "Use short for entry"},
            "medium": {"tf": interval_label_sig, "data": base_gold_sig,
                       "label": f"MEDIUM · EMA 20/50 · {interval_label_sig}", "use": "Use medium for entry"},
            "long":   {"tf": interval_label_sig, "data": base_gold_sig,
                       "label": f"LONG · EMA 50/200 · {interval_label_sig}",   "use": "Use long for entry"},
        }

        sig_side = 1 if res.signal == "BUY" else (-1 if res.signal == "SELL" else 0)

        # 3 cards: short / medium / long
        sig_hcols = st.columns(3)
        for i, h in enumerate(("short", "medium", "long")):
            with sig_hcols[i]:
                meta = sig_horizon_meta[h]
                gold_df = meta["data"]
                if gold_df is None or len(gold_df) < 30:
                    st.markdown(
                        f"""<div style="border:1px dashed #aaa;border-radius:8px;
                                    padding:10px;height:160px;background:#f5f5f5">
                              <div style="font-weight:700">{meta['label']}</div>
                              <div style="color:#999;font-size:0.85rem;margin-top:6px">
                                Data unavailable
                              </div>
                            </div>""",
                        unsafe_allow_html=True,
                    )
                    continue
                setup = horizon_entry_sltp(gold_df, sig_side if sig_side else 1, sig_cfg, h)
                fc_label = "—"
                fc_score = 0.0
                fc_conf  = 0.0
                try:
                    fc_label = str(f_labels[h].iloc[-1])
                    fc_score = float(fcasts[h].iloc[-1])
                    fc_conf  = float(f_conf[h].iloc[-1])
                except Exception:
                    pass
                bg     = "#d4edda" if fc_label == "Bullish" else ("#f8d7da" if fc_label == "Bearish" else "#fff3cd")
                border = "#2E8B57" if fc_label == "Bullish" else ("#B22222" if fc_label == "Bearish" else "#DAA520")
                is_active = (st.session_state.active_horizon == h)
                active_badge = (
                    f'<div style="margin-top:6px;font-size:0.7rem;color:#fff;background:{border};'
                    f'padding:2px 6px;border-radius:3px;display:inline-block">✓ Active</div>'
                    if is_active else
                    f'<div style="margin-top:6px;font-size:0.7rem;color:#666">—</div>'
                )
                ema_part = " — "
                if setup["ema_fast"] is not None and setup["ema_slow"] is not None:
                    ema_part = f"{setup['ema_fast']:.1f} → {setup['ema_slow']:.1f}"
                fc_score_str = f"+{fc_score:.1f}" if fc_score >= 0 else f"{fc_score:.1f}"
                st.markdown(
                    f"""<div style="border:2px solid {border};background:{bg};
                                border-radius:8px;padding:10px;min-height:170px">
                          <div style="font-size:0.7rem;font-weight:700;color:#444;letter-spacing:0.05em">
                            {meta['label'].split(' · ')[0]}
                          </div>
                          <div style="font-size:0.7rem;color:#666;margin-top:2px">{ema_part}</div>
                          <div style="font-size:1.2rem;font-weight:800;color:{border};margin-top:6px">
                            {fc_label}
                          </div>
                          <div style="font-size:0.78rem;color:#444;margin-top:4px">
                            Score {fc_score_str} · Conf {int(round(fc_conf*100))}%
                          </div>
                          {active_badge}
                        </div>""",
                    unsafe_allow_html=True,
                )
                if not is_active:
                    if st.button(meta["use"], key=f"sig_use_{h}", use_container_width=True):
                        st.session_state.active_horizon = h
                        st.rerun()
                else:
                    st.button("✓ Selected", key=f"sig_sel_{h}", use_container_width=True, disabled=True)

        # Reference table
        st.markdown("")
        try:
            ref_rows = []
            for h in ("short", "medium", "long"):
                gold_df = sig_horizon_meta[h]["data"]
                if gold_df is None or len(gold_df) < 30:
                    ref_rows.append({"Horizon": h.upper(), "Forecast": "—", "EMA fast": "—", "EMA slow": "—", "ATR": "—"})
                    continue
                setup = horizon_entry_sltp(gold_df, sig_side if sig_side else 1, sig_cfg, h)
                fc_label = "—"
                try:
                    fc_label = str(f_labels[h].iloc[-1])
                except Exception:
                    pass
                ref_rows.append({
                    "Horizon":   h.upper(),
                    "Forecast":  fc_label,
                    "EMA fast":  f"{setup['ema_fast']:.1f}" if setup['ema_fast'] is not None else "—",
                    "EMA slow":  f"{setup['ema_slow']:.1f}" if setup['ema_slow'] is not None else "—",
                    "ATR":       f"{setup['atr']:.2f}",
                })
            rdf = pd.DataFrame(ref_rows).set_index("Horizon")
            def _color_sig(val):
                if val == "Bullish":  return "background-color: #d4edda"
                if val == "Bearish":  return "background-color: #f8d7da"
                if val == "Neutral":  return "background-color: #fff3cd"
                return ""
            st.dataframe(rdf.style.map(_color_sig, subset=["Forecast"]), use_container_width=True, height=110)
        except Exception as e:
            st.warning(f"Reference table failed: {e}")

        # Active horizon entry-setup chart
        sig_active = st.session_state.active_horizon
        sig_active_meta = sig_horizon_meta[sig_active]
        sig_active_df = sig_active_meta["data"]
        if sig_active_df is None or len(sig_active_df) < 5:
            st.info(f"Gold chart for {sig_active_meta['tf']} is not available.")
        else:
            side_label = "Long" if sig_side == 1 else ("Short" if sig_side == -1 else "—")
            st.markdown(f"##### 🎯 Live entry setup — {sig_active} horizon")
            try:
                setup = horizon_entry_sltp(sig_active_df, sig_side if sig_side else 1, sig_cfg, sig_active)
                am1, am2, am3, am4, am5 = st.columns(5)
                am1.metric("Action", side_label, delta=("↑ Long" if sig_side == 1 else ("↓ Short" if sig_side == -1 else "—")))
                if setup["entry"] is not None:
                    am2.metric("Entry",   f"{setup['entry']:,.2f}",   delta=f"{setup['entry'] - setup['last_price']:+.1f}")
                    am3.metric("Stop",    f"{setup['stop']:,.2f}",    delta=f"-{abs(setup['entry'] - setup['stop']):.1f} pts", delta_color="inverse")
                    am4.metric("Target",  f"{setup['target']:,.2f}",  delta=f"+{abs(setup['target'] - setup['entry']):.1f} pts")
                    risk   = abs(setup['entry'] - setup['stop'])
                    reward = abs(setup['target'] - setup['entry'])
                    am5.metric("R:R",     f"{reward/risk:.2f}" if risk > 0 else "—")
                fc_label = "—"
                fc_score = 0.0
                fc_conf  = 0.0
                try:
                    fc_label = str(f_labels[sig_active].iloc[-1])
                    fc_score = float(fcasts[sig_active].iloc[-1])
                    fc_conf  = float(f_conf[sig_active].iloc[-1])
                except Exception:
                    pass
                bias = "fast > slow" if (setup["ema_fast"] and setup["ema_slow"] and setup["ema_fast"] > setup["ema_slow"]) else "fast < slow"
                fast_v = f"{setup['ema_fast']:.1f}" if setup["ema_fast"] else "—"
                slow_v = f"{setup['ema_slow']:.1f}" if setup["ema_slow"] else "—"
                last_v = f"{setup['last_price']:.1f}" if setup["last_price"] else "—"
                stop_v = f"{setup['stop']:.1f}" if setup["stop"] else "—"
                tgt_v  = f"{setup['target']:.1f}" if setup["target"] else "—"
                st.caption(
                    f"**{sig_active.upper()} forecast:** {fc_label} "
                    f"(score {fc_score:+.1f}, conf {fc_conf*100:.1f}%) · "
                    f"**EMA bias:** {bias} (fast={fast_v} slow={slow_v}) · "
                    f"Last {last_v} · Stop {stop_v} · Target {tgt_v} · R:R "
                    f"{(abs(setup['target']-setup['entry'])/max(abs(setup['entry']-setup['stop']),1e-9)):.2f}"
                )
                tail = sig_active_df.tail(120).copy()
                fig_e = go.Figure()
                fig_e.add_trace(go.Candlestick(
                    x=tail.index,
                    open=tail["Open"], high=tail["High"],
                    low=tail["Low"], close=tail["Close"],
                    name="Gold", increasing_line_color="#2E8B57",
                    decreasing_line_color="#B22222",
                ))
                from indicators import ema as _ema_fn_sig
                ef_len = {"short": 10, "medium": 20, "long": 50}.get(sig_active, 20)
                es_len = {"short": 20, "medium": 50, "long": 200}.get(sig_active, 50)
                ef_full = _ema_fn_sig(tail["Close"], ef_len)
                es_full = _ema_fn_sig(tail["Close"], es_len)
                fig_e.add_trace(go.Scatter(
                    x=tail.index, y=ef_full.values,
                    name="EMA fast", line=dict(color="#1f77b4", width=1.4, dash="dot"),
                ))
                fig_e.add_trace(go.Scatter(
                    x=tail.index, y=es_full.values,
                    name="EMA slow", line=dict(color="#ff7f0e", width=1.6),
                ))
                if setup["entry"] is not None:
                    line_color = {"BUY": "#2E8B57", "SELL": "#B22222"}.get(res.signal, "#DAA520")
                    fig_e.add_hline(
                        y=setup["entry"], line_dash="solid", line_color=line_color, line_width=2,
                        annotation_text=f"Entry {setup['entry']:,.2f}",
                        annotation_position="right", annotation_font_color=line_color,
                    )
                    fig_e.add_hrect(
                        y0=min(setup["entry"], setup["stop"]), y1=max(setup["entry"], setup["stop"]),
                        fillcolor="rgba(178,34,34,0.12)", line_width=0,
                    )
                    fig_e.add_hline(
                        y=setup["stop"], line_dash="dash", line_color="#B22222", line_width=1.5,
                        annotation_text=f"Stop {setup['stop']:,.2f}",
                        annotation_position="right", annotation_font_color="#B22222",
                    )
                    fig_e.add_hrect(
                        y0=min(setup["entry"], setup["target"]), y1=max(setup["entry"], setup["target"]),
                        fillcolor="rgba(46,139,87,0.12)", line_width=0,
                    )
                    fig_e.add_hline(
                        y=setup["target"], line_dash="dash", line_color="#2E8B57", line_width=1.5,
                        annotation_text=f"Target {setup['target']:,.2f}",
                        annotation_position="right", annotation_font_color="#2E8B57",
                    )
                fig_e.update_layout(
                    height=460, template="plotly_white",
                    yaxis_title="Gold price",
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=10, r=10, t=20, b=10),
                    hovermode="x unified",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left"),
                )
                st.plotly_chart(fig_e, use_container_width=True)
            except Exception as e:
                st.warning(f"Entry-setup chart failed: {e}")

        # Reasons
        with st.expander("Why this signal?", expanded=(res.signal in ("BUY", "SELL"))):
            st.markdown(f"**Signal:** {res.signal}")
            st.markdown(f"**15m composite:** {res.composite_15m:+.1f}")
            st.markdown(f"**1h  composite:** {res.composite_1h:+.1f}")
            st.markdown(f"**Momentum:** {'building' if res.momentum_ok else 'fading'}")
            st.markdown(f"**15m forecasts:** {res.forecast_15m}")
            st.markdown(f"**Forecast agreement:** {res.forecast_agreement}/3 bullish")
            st.markdown(f"**Reasons:** {', '.join(res.reasons)}")

        # Send buttons
        st.markdown("##### Send to Telegram")
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            if st.button("📨 Send signal text", use_container_width=True, disabled=not tg_is_configured()):
                caption, _ = format_signal_message(res)
                r = tg_send(caption)
                st.success("Sent.") if r["ok"] else st.error(f"Failed: {r.get('error')}")
        with ac2:
            if st.button("🖼  Send signal + chart", use_container_width=True, disabled=not tg_is_configured()):
                caption, _ = format_signal_message(res)
                png = render_chart_snapshot(multi_tf_data["1h"], multi_tf_data["15m"], res)
                if png is None:
                    st.warning("Chart unavailable. Sending text only.")
                    r = tg_send(caption)
                else:
                    r = tg_send_photo(png, caption=caption[:1024])
                st.success("Sent.") if r["ok"] else st.error(f"Failed: {r.get('error')}")
        with ac3:
            if st.button("🔄 Refresh signal", use_container_width=True):
                st.cache_data.clear()
                st.rerun()


# =============================================================================
# SIGNAL-BT TAB
# =============================================================================
with tab_sigbt:
    st.subheader("📈 Signal-engine backtest")
    st.caption("Walks historical 15m / 1h bars, fires the signal engine, measures forward return.")

    if not multi_tf_ready:
        st.error("Multi-timeframe data not ready.")
    else:
        if "sigbt_cfg" not in st.session_state:
            st.session_state.sigbt_cfg = SignalBacktestConfig()
        bt_cfg = st.session_state.sigbt_cfg
        sig_cfg = st.session_state.get("sig_cfg", SignalConfig())

        with st.expander("⚙️  Backtest parameters", expanded=True):
            cc1, cc2, cc3 = st.columns(3)
            today = pd.Timestamp.now().date()
            earliest = pd.Timestamp("2020-01-01").date()
            # Defensive: clamp any stored value into [earliest, today]
            try:
                cur_start = pd.Timestamp(bt_cfg.start).date()
                if cur_start < earliest: cur_start = earliest
                if cur_start > today:    cur_start = today
            except Exception:
                cur_start = earliest
            try:
                cur_end = pd.Timestamp(bt_cfg.end).date()
                if cur_end < earliest: cur_end = earliest
                if cur_end > today:    cur_end = today
            except Exception:
                cur_end = today
            with cc1:
                bt_cfg.start = st.date_input("Start", value=cur_start,
                                             min_value=earliest, max_value=today,
                                             key="sigbt_start").isoformat()
                bt_cfg.end = st.date_input("End", value=cur_end,
                                           min_value=earliest, max_value=today,
                                           key="sigbt_end").isoformat()
            with cc2:
                bt_cfg.holding_bars = st.number_input("Holding window (1h bars)", 1, 96, bt_cfg.holding_bars, 1)
                bt_cfg.cooldown_bars = st.number_input("Cooldown (1h bars)", 0, 96, bt_cfg.cooldown_bars, 1)
            with cc3:
                bt_cfg.step_bars = st.number_input("Step (1h bars)", 1, 24, bt_cfg.step_bars, 1)
                bt_cfg.max_triggers = st.number_input("Max triggers", 100, 50_000, bt_cfg.max_triggers, 100)

        if st.button("▶ Run signal backtest", type="primary"):
            with st.spinner("Running signal backtest…"):
                try:
                    st.session_state.sigbt_result = run_signal_backtest(
                        data_15m=multi_tf_data["15m"],
                        data_1h =multi_tf_data["1h"],
                        config  =config,
                        sig_cfg =sig_cfg,
                        bt_cfg  =bt_cfg,
                        initial_capital=10_000.0,
                    )
                except Exception as e:
                    st.error(f"Backtest failed: {e}")

        res = st.session_state.get("sigbt_result")
        if res is None:
            st.info("Click ▶ Run signal backtest to see results.")
        elif getattr(res, "coverage_warning", None):
            st.warning(f"⚠️ {res.coverage_warning}")
        elif res.triggers.empty:
            st.warning("No BUY/SELL triggers fired. Try loosening the thresholds.")
        else:
            m = res.metrics
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Triggers",     f"{m['n_triggers']}", delta=f"hit {m['hit_rate_pct']:.1f}%")
            mc2.metric("Expectancy",   f"{m['expectancy']:+.2f}%/trade")
            mc3.metric("Profit Factor",f"{m['profit_factor']:.2f}", delta=f"max DD {m['max_dd_pct']:.1f}%")
            mc4.metric("Total Return", f"{m['total_return_pct']:+.2f}%", delta=f"annualized {m['annualized_pct']:+.1f}%")

            if not res.equity_curve.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res.equity_curve.index, y=res.equity_curve.values,
                                         name="Strategy (signals)", line=dict(color="#DAA520", width=2)))
                fig.add_trace(go.Scatter(x=res.benchmark.index, y=res.benchmark.values,
                                         name="Buy & Hold Gold", line=dict(color="#888", width=1.5, dash="dot")))
                fig.update_layout(height=380, template="plotly_white",
                                  yaxis_title="Equity ($)", margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(res.bucket_stats, use_container_width=True)
            with st.expander("Full trigger log"):
                st.dataframe(res.triggers, use_container_width=True, height=300)


# =============================================================================
# ALERTS TAB
# =============================================================================
with tab_alerts:
    st.subheader("🔔 Telegram alerts")
    st.caption("Send Telegram messages on signal events, and respond to /commands.")

    configured = tg_is_configured()
    if configured:
        st.success("✅ Telegram bot is configured and ready.")
    else:
        st.error("❌ Telegram bot is NOT configured. Add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` to Streamlit Cloud secrets.")

    if "tg_cfg" not in st.session_state:
        st.session_state.tg_cfg = TgConfig(enabled=False)
    cfg = st.session_state.tg_cfg
    cfg.enabled = st.checkbox("Enable Telegram alerts", value=cfg.enabled,
                                key="tg_cfg_enabled", disabled=not configured)
    # The "open signal" checkbox: Streamlit stores its value in
    # session_state via the key="tg_open_signal_enabled", so we just
    # read st.session_state.tg_open_signal_enabled wherever we need it.
    st.checkbox(
        "📲 Send current signal on app open",
        value=st.session_state.get("tg_open_signal_enabled", True),
        key="tg_open_signal_enabled", disabled=not configured,
    )

    st.markdown("##### Alert conditions")
    c1, c2 = st.columns(2)
    with c1:
        cfg.alert_on_unanimous = st.checkbox("📣 All 3 forecasts agree", value=cfg.alert_on_unanimous,
                                              disabled=not cfg.enabled)
        cfg.alert_on_flip = st.checkbox("⚡ Forecast flip", value=cfg.alert_on_flip,
                                         disabled=not cfg.enabled)
        cfg.alert_on_composite_threshold = st.checkbox("📊 Composite enters non-Neutral",
                                                       value=cfg.alert_on_composite_threshold,
                                                       disabled=not cfg.enabled)
    with c2:
        cfg.min_confidence = st.slider("Min confidence", 50.0, 95.0, cfg.min_confidence, 1.0,
                                       disabled=not cfg.enabled or not cfg.alert_on_unanimous)
        cfg.composite_threshold = st.slider("Composite threshold", 20.0, 80.0, cfg.composite_threshold, 1.0,
                                            disabled=not cfg.enabled or not cfg.alert_on_composite_threshold)
        cfg.cooldown_seconds = st.number_input("Cooldown (seconds)", 60, 86_400, cfg.cooldown_seconds, 60,
                                                disabled=not cfg.enabled)

    st.markdown("##### 📲 Two-way commands")
    st.markdown("""
Send these from your Telegram chat. The app picks them up on every refresh.

| Command | What it does |
|---|---|
| `/status`  | Composite, regime, flow, gold price |
| `/signal`  | Multi-TF signal + probability |
| `/chart`   | Fresh chart snapshot (photo) |
| `/weights` | Current weight settings |
| `/help`    | List of commands |

Per-command cooldowns: `/status` 60s, `/signal` 60s, `/chart` 120s, `/weights` 30s, `/help` 10s.
""")

    # Test buttons
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        if st.button("📨 Send test alert", disabled=not configured):
            r = tg_send("🧪 Gold Scalper test alert — bot is online.")
            st.success("Sent.") if r["ok"] else st.error(f"Failed: {r.get('error')}")
    with tc2:
        if st.button("🔄 Process pending commands", disabled=not configured):
            st.session_state.tg_history.add("Manual command poll", "ok")
            st.rerun()
    with tc3:
        st.caption("On next refresh the bot will pick up any messages you sent.")

    # History
    st.markdown("##### Recent alerts (this session)")
    hist_df = st.session_state.tg_history.to_dataframe()
    if hist_df.empty:
        st.caption("No alerts fired yet.")
    else:
        st.dataframe(hist_df, use_container_width=True, height=300)


# =============================================================================
# ABOUT TAB
# =============================================================================
with tab_about:
    st.markdown(
        """
        ### What this is
        A multi-market, weighted macro engine for forecasting the next
        directional bias of **Gold (XAU/USD)**, with a Telegram bot
        for alerts and two-way commands.

        ### Market set
        | Market | Weight | Why |
        |---|---|---|
        | **DXY** | 25% | #1 driver of gold (negative correlation) |
        | **IEF / US Treasuries** | 20% | Yield / risk-off proxy |
        | **Silver** | 15% | Sister precious metal confirmation |
        | **S&P 500** | 15% | Equities risk-on/off |
        | **EUR/USD** | 10% | Dollar strength cross-check |
        | **VIX** | 10% | Volatility / fear gauge |
        | **Gold** | 5% | Own-momentum confirmation |

        ### How the signal engine works
        - Computes the **macro composite** (7-market weighted score) on both
          15m and 1h bars.
        - **BUY** when 15m composite > 1h composite (momentum building)
          AND all 3 of {short, medium, long} 15m forecasts = Bullish.
        - **SELL** is the mirror.
        - **Entry / SL / TP** are computed from the 15m chart's slow EMA + ATR.

        ### Telegram commands
        Send these from your Telegram chat:
        - `/status` — composite, regime, flow, gold price
        - `/signal` — current multi-TF signal + probability
        - `/chart` — fresh chart snapshot
        - `/weights` — current weight settings
        - `/help` — list of commands

        ### Deploy
        1. Push all `.py` files to the **root** of your GitHub repo.
        2. Connect on share.streamlit.io (entry point: `app.py`).
        3. Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in the secrets panel.
        4. The app will auto-deploy.
        """
    )

    st.markdown("#### Engine modules (flat layout)")
    st.code(
        """
your-repo/
├── app.py              # Streamlit UI (7 tabs: Live / Backtest / Trades / Signals / Signal-BT / Alerts / About)
├── config.py           # Weights, periods, data source, sidebar
├── data.py             # yfinance + Twelve Data, gold as master
├── indicators.py       # EMA, RSI, ROC, ATR
├── scoring.py          # Composite + 3 forecasts + regime + flow
├── backtest.py         # Long-flat backtest
├── trades.py           # R-based trade engine
├── trade_ui.py         # Trade dashboard
├── telegram_alerts.py  # Telegram send_message / send_photo
├── telegram_commands.py # Two-way commands /status /signal /chart
├── signals.py          # Macro-composite signal engine
├── signal_backtest.py  # Signal backtest
├── auto_refresh.py     # 15-min auto-refresh helper
├── requirements.txt
└── .gitignore
        """,
        language="text",
    )
