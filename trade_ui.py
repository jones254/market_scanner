"""
Streamlit UI helpers for the trade engine output.
"""
from __future__ import annotations
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


def render_trade_dashboard(trade_log, target_df, config) -> None:
    """
    Render a complete trade-engine dashboard inside the current Streamlit tab.

    `target_df` is the OHLCV dataframe of the target instrument (Gold).
    """
    st.subheader("🎯 Trade Engine — R-based performance")

    if trade_log.trades.empty:
        st.info("No trades generated in this window. Try widening the date range, "
                "lowering `min_rr`, or running on intraday data for more signals.")
        if not trade_log.skipped.empty:
            with st.expander(f"Skip reasons ({len(trade_log.skipped)} total)"):
                st.dataframe(trade_log.skipped["reason"].value_counts(),
                             use_container_width=True)
        return

    m = trade_log.metrics

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades",        f"{m['n_trades']}",
              delta=f"Win rate {m['hit_rate_pct']:.1f}%")
    c2.metric("Expectancy",    f"{m['expectancy_r']:+.2f}R",
              delta="per trade")
    c3.metric("Profit Factor", f"{m['profit_factor']:.2f}",
              delta=f"Max consec W/L: {m['max_consec_wins']}/{m['max_consec_losses']}")
    c4.metric("Total Return",  f"{m['return_pct']:+.2f}%",
              delta=f"P/L ${m['total_pnl']:,.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg Win",   f"{m['avg_win_r']:+.2f}R")
    c6.metric("Avg Loss",  f"{m['avg_loss_r']:+.2f}R")
    c7.metric("Final Equity", f"${m['final_equity']:,.2f}")
    c8.metric("Max Consec Losses", f"{m['max_consec_losses']}")

    st.markdown("")

    # Equity curve vs buy & hold
    eq = trade_log.equity_curve
    bh = (1 + target_df["Close"].pct_change().fillna(0)).cumprod() * eq.iloc[0]
    bh = bh.reindex(eq.index).ffill()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Strategy",
                             line=dict(color="#DAA520", width=2)))
    fig.add_trace(go.Scatter(x=bh.index, y=bh.values, name="Buy & Hold",
                             line=dict(color="#888", width=1.5, dash="dot")))
    fig.update_layout(height=350, template="plotly_white",
                      yaxis_title="Equity ($)", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Trade log
    st.markdown("##### Closed trades")
    display_cols = ["entry_time", "side", "signal", "entry", "stop", "target",
                    "exit", "size", "r_multiple", "pnl", "exit_reason", "bars_held"]
    available = [c for c in display_cols if c in trade_log.trades.columns]
    tbl = trade_log.trades[available].iloc[::-1]
    st.dataframe(tbl, use_container_width=True, height=400)

    if "signal" in trade_log.trades.columns:
        st.markdown("##### Performance by signal bucket")
        grp = trade_log.trades.groupby("signal")["r_multiple"]
        bucket_stats = pd.DataFrame({
            "n":        grp.count(),
            "hit_rate": (grp.apply(lambda x: (x > 0).mean()) * 100).round(1),
            "avg_R":    grp.mean().round(3),
            "total_R":  grp.sum().round(2),
        })
        order = ["Strong Bullish", "Bullish", "Neutral", "Bearish", "Strong Bearish"]
        bucket_stats = bucket_stats.reindex([o for o in order if o in bucket_stats.index])
        st.dataframe(bucket_stats, use_container_width=True)

    if not trade_log.skipped.empty:
        with st.expander(f"⚠️  Filtered signals ({len(trade_log.skipped)})"):
            st.caption("Signals that didn't become trades. Most are 'entry_expired' "
                       "(pullback limit didn't fill in time) or 'rr_below_min' (R:R too low).")
            vc = trade_log.skipped["reason"].value_counts().head(10)
            st.dataframe(vc, use_container_width=True)
