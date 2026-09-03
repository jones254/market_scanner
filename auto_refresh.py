"""
Auto-refresh helpers for the Gold Scalper engine.

Wraps the `streamlit-autorefresh` package if available, and falls back
to a manual refresh button if it isn't installed.  Also tracks a
"last refreshed" timestamp in session state so the UI can show when
the data was last fetched.
"""

from __future__ import annotations
import time
from datetime import datetime

try:
    import streamlit as st
except ImportError:
    # Allows this module to be imported in dev environments without
    # streamlit installed.  The functions will raise a clear error if
    # actually called.
    st = None  # type: ignore


def setup_auto_refresh(interval_seconds: int = 900) -> None:
    """
    Set up auto-refresh on the page.  Default 15 min (900 s).

    Uses streamlit-autorefresh if installed; otherwise the user has to
    click the "Refresh now" button.  Either way, when a refresh happens
    the Streamlit cache_data decorator's TTL will determine whether the
    underlying yfinance / Twelve Data call actually re-fires.
    """
    # Track last refresh in session state
    if "last_refresh_ts" not in st.session_state:
        st.session_state.last_refresh_ts = time.time()

    # Try the package
    auto_used = False
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=interval_seconds * 1000, key="auto_refresh")
        st.session_state.last_refresh_ts = time.time()
        auto_used = True
    except ImportError:
        pass

    # Show the status line
    last_str = datetime.fromtimestamp(st.session_state.last_refresh_ts).strftime("%H:%M:%S")
    if auto_used:
        st.caption(
            f"🔄 Auto-refresh every {interval_seconds // 60} min · "
            f"Last refresh: **{last_str}**"
        )
    else:
        st.caption(
            "ℹ️ Install `streamlit-autorefresh` for auto-refresh. "
            f"Last refresh: **{last_str}** (click the button below to refresh now)"
        )
        if st.button("🔄 Refresh now", key="manual_refresh"):
            st.cache_data.clear()
            st.session_state.last_refresh_ts = time.time()
            st.rerun()
