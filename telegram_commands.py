"""
Two-way Telegram integration: command polling + replies.

Reads pending messages from your Telegram chat (via getUpdates) on every
Streamlit refresh, parses the command, runs the appropriate handler,
and replies via sendMessage.

Default commands
----------------
/status  - Composite + regime + flow + last price
/signal  - Current multi-TF signal (BUY/SELL/NOACTION) + probability
/chart   - Sends a fresh chart snapshot (photo + caption)
/weights - Current weight settings
/help    - List of commands

Per-command cooldowns prevent accidental spam.

This module is a pure helper — it doesn't import Streamlit.  The
Streamlit app calls `process_pending_commands(...)` on each rerun.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, List, Tuple

import requests

# Re-use the secrets handling from telegram_alerts
try:
    from .telegram_alerts import _get_secrets, send_message, send_photo
except ImportError:
    from telegram_alerts import _get_secrets, send_message, send_photo


# -----------------------------------------------------------------------------
# Cooldown tracking
# -----------------------------------------------------------------------------
@dataclass
class CommandState:
    """Persists between Streamlit reruns (in session_state)."""
    last_update_id: int = 0       # highest Telegram update_id we've seen
    last_fired: Dict[str, float] = field(default_factory=dict)  # cmd -> ts
    # What the bot sent the last time /signal was triggered from app open
    last_open_signal_sent_ts: float = 0.0


# Per-command cooldowns (seconds)
COOLDOWNS = {
    "/status":   60,
    "/signal":   60,
    "/chart":   120,
    "/weights":  30,
    "/help":     10,
}


# -----------------------------------------------------------------------------
# Low-level: getUpdates
# -----------------------------------------------------------------------------
def fetch_updates(timeout: int = 5, offset: Optional[int] = None) -> List[Dict]:
    """
    Fetch new messages from Telegram.  `offset` = last update_id + 1.

    Returns a list of update dicts (each has 'update_id', possibly 'message').
    """
    s = _get_secrets()
    if not s["token"]:
        return []
    url = f"https://api.telegram.org/bot{s['token']}/getUpdates"
    params = {
        "timeout": timeout,
        "allowed_updates": '["message"]',
    }
    if offset is not None:
        params["offset"] = offset
    try:
        r = requests.get(url, params=params, timeout=timeout + 5)
        js = r.json()
        if r.status_code == 200 and js.get("ok"):
            return js.get("result", [])
    except Exception:
        pass
    return []


# -----------------------------------------------------------------------------
# Reply helpers
# -----------------------------------------------------------------------------
def reply(text: str, parse_mode: str = "Markdown") -> bool:
    return send_message(text, parse_mode=parse_mode).get("ok", False)


def reply_photo(photo_bytes: bytes, caption: str = "") -> bool:
    return send_photo(photo_bytes, caption=caption).get("ok", False)


# -----------------------------------------------------------------------------
# Command handler type
# Each handler receives (args: str, context: dict) and returns the reply
# text, or None to send no reply.  context has whatever the caller passes.
# -----------------------------------------------------------------------------
CommandHandler = Callable[[str, dict], Optional[str]]


HELP_TEXT = (
    "🤖 *Gold Scalper Bot — Commands*\n\n"
    "/status  — Composite, regime, flow, last price\n"
    "/signal  — Current multi-TF signal + probability\n"
    "/chart   — Send a fresh chart snapshot\n"
    "/weights — Current weight settings\n"
    "/help    — This list\n"
)


# -----------------------------------------------------------------------------
# Built-in handlers — the "data provider" is passed in via context
# -----------------------------------------------------------------------------
def _handler_help(args: str, ctx: dict) -> Optional[str]:
    return HELP_TEXT


def _handler_status(args: str, ctx: dict) -> Optional[str]:
    """Composite / regime / flow / last price."""
    data_getter = ctx.get("data_getter")
    score_res   = ctx.get("score_result")
    regime      = ctx.get("regime")
    flow        = ctx.get("flow")
    data        = ctx.get("data")
    if not (data_getter and score_res is not None and regime is not None and flow is not None and data is not None):
        return "❌ Status data not available right now. Try again after the next refresh."

    latest = score_res.composite.index[-1]
    last_score = float(score_res.composite.iloc[-1])
    last_label = str(score_res.label.iloc[-1])
    last_conf  = float(score_res.confidence.iloc[-1])
    last_price = float(data["gold"]["Close"].iloc[-1])
    flow_now   = float(flow.iloc[-1])
    reg_now    = str(regime.iloc[-1])

    return (
        f"📊 *Status*\n\n"
        f"• Gold last: `{last_price:,.2f}`\n"
        f"• Composite: *{last_label}* (score `{last_score:+.1f}`, conf {last_conf:.0f}%)\n"
        f"• Regime: *{reg_now}*\n"
        f"• Flow: `{flow_now:.0f}/100`\n"
        f"• Last bar: {latest.strftime('%Y-%m-%d %H:%M')}\n"
    )


def _handler_signal(args: str, ctx: dict) -> Optional[str]:
    """Current multi-TF signal + probability."""
    get_signal = ctx.get("get_signal_result")
    if get_signal is None:
        return "❌ Signal engine not available. Make sure 5m/15m/1h data is ready."
    res = get_signal()
    if res is None:
        return "❌ Signal engine returned no result (data not ready?)."
    # Reuse the existing formatter
    from signals import format_signal_message
    msg, _ = format_signal_message(res)
    return msg


def _handler_chart(args: str, ctx: dict) -> Optional[str]:
    """Send a fresh chart snapshot."""
    get_signal = ctx.get("get_signal_result")
    render_chart = ctx.get("render_chart")
    multi_tf_data = ctx.get("multi_tf_data")
    if not (get_signal and render_chart and multi_tf_data):
        return "❌ Chart data not available."

    res = get_signal()
    if res is None:
        return "❌ No signal to chart."

    try:
        png = render_chart(multi_tf_data["1h"], multi_tf_data["15m"], res)
    except Exception as e:
        return f"❌ Chart render failed: {e}"

    if png is None:
        return "❌ Chart rendering unavailable (kaleido not installed)."

    from signals import format_signal_message
    caption, _ = format_signal_message(res)
    ok = reply_photo(png, caption=caption[:1024])
    return "📊 Chart sent." if ok else "❌ Failed to send chart."


def _handler_weights(args: str, ctx: dict) -> Optional[str]:
    config = ctx.get("config")
    if config is None:
        return "❌ Config not available."
    try:
        from config import MARKET_LABELS
    except ImportError:
        try:
            from .config import MARKET_LABELS
        except ImportError:
            MARKET_LABELS = {}
    lines = ["⚙️ *Current weights (% — auto-normalized)*\n"]
    for mkt, label in MARKET_LABELS.items():
        w = config.weights.get(mkt, 0)
        lines.append(f"• {label}: {w:.0f}%")
    return "\n".join(lines)


BUILTIN_HANDLERS: Dict[str, CommandHandler] = {
    "/help":    _handler_status if False else _handler_help,   # see below
    "/status":  _handler_status,
    "/signal":  _handler_signal,
    "/chart":   _handler_chart,
    "/weights": _handler_weights,
}
# Re-map /help to the real handler (above line was a typo guard)
BUILTIN_HANDLERS["/help"] = _handler_help


# -----------------------------------------------------------------------------
# Process pending updates — main entry point
# -----------------------------------------------------------------------------
def process_pending_commands(
    state: CommandState,
    context: dict,
    max_updates: int = 20,
) -> List[Tuple[str, bool]]:
    """
    Poll Telegram for new messages, dispatch commands, send replies.

    Returns a list of (command_text, success) for each processed message.
    The caller (Streamlit app) can display these in the UI.

    `context` is a dict passed to every command handler.  Typical keys:
        - config          : the engine Config
        - data            : the main multi-market data dict
        - score_result    : composite ScoreResult
        - regime          : market_regime series
        - flow            : flow_meter series
        - multi_tf_data   : dict of {interval: {market: df}}
        - get_signal_result : callable returning a SignalResult or None
        - render_chart    : render_chart_snapshot function
    """
    updates = fetch_updates(timeout=2, offset=state.last_update_id + 1 or None)
    results: List[Tuple[str, bool]] = []

    for upd in updates[:max_updates]:
        # Track the latest update_id we've seen
        uid = upd.get("update_id", state.last_update_id)
        if uid > state.last_update_id:
            state.last_update_id = uid

        msg = upd.get("message") or upd.get("edited_message")
        if not msg:
            continue

        # Auth: only respond to messages from the configured chat
        s = _get_secrets()
        chat_id = str(msg.get("chat", {}).get("id", ""))
        if s["chat_id"] and chat_id != str(s["chat_id"]):
            # Silently skip messages from other chats
            continue

        text = (msg.get("text") or "").strip()
        if not text:
            continue

        # Parse command (first token, case-insensitive)
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        handler = BUILTIN_HANDLERS.get(cmd)
        if handler is None:
            # Unknown command — send help, but only if /help isn't on cooldown
            if _cooldown_ok(state, "/help"):
                reply(f"Unknown command `{cmd}`.  Try /help.")
                _mark_cooldown(state, "/help")
            results.append((cmd, False))
            continue

        # Cooldown
        if not _cooldown_ok(state, cmd):
            wait = int(COOLDOWNS.get(cmd, 60) - (time.time() - state.last_fired.get(cmd, 0)))
            reply(f"⏳ `{cmd}` is on cooldown. Try again in {wait}s.")
            results.append((cmd, False))
            continue

        # Dispatch
        try:
            response = handler(args, context)
            if response:
                ok = reply(response)
                results.append((cmd, ok))
            else:
                results.append((cmd, True))
            _mark_cooldown(state, cmd)
        except Exception as e:
            reply(f"❌ `{cmd}` failed: {e}")
            results.append((cmd, False))

    return results


# -----------------------------------------------------------------------------
# Cooldown helpers
# -----------------------------------------------------------------------------
def _cooldown_ok(state: CommandState, cmd: str) -> bool:
    cd = COOLDOWNS.get(cmd, 60)
    return (time.time() - state.last_fired.get(cmd, 0)) >= cd


def _mark_cooldown(state: CommandState, cmd: str) -> None:
    state.last_fired[cmd] = time.time()


# -----------------------------------------------------------------------------
# "Send current signal on app open" — fires once when the user opens the app
# -----------------------------------------------------------------------------
def send_open_signal(
    state: CommandState,
    get_signal_result: Callable,
    format_signal_message_fn: Callable,
    render_chart_fn: Callable,
    multi_tf_data: dict,
    cooldown_seconds: int = 1800,  # max once per 30 min
) -> Optional[str]:
    """
    Send a one-off Telegram message with the current signal when the app
    is opened.  Gated by a long cooldown so it doesn't spam if the user
    leaves the tab open with frequent refreshes.
    """
    if (time.time() - state.last_open_signal_sent_ts) < cooldown_seconds:
        return None
    res = get_signal_result()
    if res is None:
        return None
    caption, _ = format_signal_message_fn(res)
    png = None
    try:
        png = render_chart_fn(multi_tf_data["1h"], multi_tf_data["15m"], res)
    except Exception:
        png = None
    if png is not None:
        ok = reply_photo(png, caption=caption[:1024])
    else:
        ok = reply(caption)
    if ok:
        state.last_open_signal_sent_ts = time.time()
        return caption
    return None
