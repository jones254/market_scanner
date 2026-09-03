"""
Telegram alerts for the Gold Scalper engine.

Sends a Telegram message via the official Bot API when a configured
condition is met (multi-horizon agreement, forecast flip, etc.).

Configuration
-------------
The bot token and chat ID are read from Streamlit secrets:

    TELEGRAM_BOT_TOKEN = "123456:ABC..."
    TELEGRAM_CHAT_ID   = "987654321"

Set these in:
  - Local dev:    .streamlit/secrets.toml  (gitignored)
  - Streamlit Cloud:  App settings -> Secrets

The module also works in a headless / CLI mode (e.g. for use with
external cron jobs) by reading from environment variables as a fallback.
"""

from __future__ import annotations
import os
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import requests


# -----------------------------------------------------------------------------
# Configuration container
# -----------------------------------------------------------------------------
@dataclass
class TelegramConfig:
    """User-tunable alert conditions."""
    enabled: bool = False

    # Alert when all 3 forecasts agree on direction (and aren't Neutral)
    alert_on_unanimous: bool = True
    # Minimum confidence on the active forecast to fire
    min_confidence: float = 70.0

    # Alert when a single forecast flips to a new direction
    # (e.g. Short forecast was Neutral, now Bullish)
    alert_on_flip: bool = True

    # Alert when composite enters / leaves a non-Neutral bucket
    alert_on_composite_threshold: bool = True
    composite_threshold: float = 40.0

    # Cool-down between alerts (seconds) to avoid spam
    cooldown_seconds: int = 1800   # 30 min

    # Quiet hours (UTC, 24h clock) — no alerts in this window
    quiet_start_utc: int = 22      # 22:00
    quiet_end_utc:   int = 6       # 06:00


# -----------------------------------------------------------------------------
# Secrets handling — works in Streamlit, headless, and CLI
# -----------------------------------------------------------------------------
def _get_secrets() -> Dict[str, str]:
    """
    Return Telegram bot token + chat ID from any of:
      1. Streamlit secrets (when running inside Streamlit)
      2. Environment variables TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
      3. .streamlit/secrets.toml file (read directly, no Streamlit required)
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat  = os.environ.get("TELEGRAM_CHAT_ID")

    # Try Streamlit secrets (lazy import — works in either mode)
    if not token or not chat:
        try:
            import streamlit as st
            token = token or st.secrets.get("TELEGRAM_BOT_TOKEN", "")
            chat  = chat  or st.secrets.get("TELEGRAM_CHAT_ID", "")
        except Exception:
            pass

    # Try reading the secrets file directly
    if not token or not chat:
        try:
            from pathlib import Path
            secrets_path = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
            if secrets_path.exists():
                import re
                content = secrets_path.read_text()
                m = re.search(r'TELEGRAM_BOT_TOKEN\s*=\s*"([^"]+)"', content)
                if m:
                    token = m.group(1)
                m = re.search(r'TELEGRAM_CHAT_ID\s*=\s*"([^"]+)"', content)
                if m:
                    chat = m.group(1)
        except Exception:
            pass

    return {"token": token or "", "chat_id": chat or ""}


def is_configured() -> bool:
    """True if both token and chat_id are present."""
    s = _get_secrets()
    return bool(s["token"]) and bool(s["chat_id"])


# -----------------------------------------------------------------------------
# Core send
# -----------------------------------------------------------------------------
def send_message(text: str, parse_mode: str = "Markdown") -> Dict:
    """
    Send a message via the Telegram Bot API.

    Returns a dict like:
        {"ok": True,  "message_id": 123}
        {"ok": False, "error": "..."}
    """
    s = _get_secrets()
    if not s["token"] or not s["chat_id"]:
        return {"ok": False, "error": "Telegram not configured (token / chat_id missing)"}

    url = f"https://api.telegram.org/bot{s['token']}/sendMessage"
    payload = {
        "chat_id":    s["chat_id"],
        "text":       text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }

    try:
        r = requests.post(url, json=payload, timeout=10)
        js = r.json()
        if r.status_code == 200 and js.get("ok"):
            return {"ok": True, "message_id": js.get("result", {}).get("message_id")}
        return {"ok": False, "error": js.get("description", f"HTTP {r.status_code}")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def send_photo(photo_bytes: bytes, caption: str = "", parse_mode: str = "Markdown") -> Dict:
    """
    Send a photo (PNG bytes) to the configured Telegram chat.
    Caption is shown under the image, supports Markdown.
    """
    s = _get_secrets()
    if not s["token"] or not s["chat_id"]:
        return {"ok": False, "error": "Telegram not configured (token / chat_id missing)"}

    url = f"https://api.telegram.org/bot{s['token']}/sendPhoto"
    try:
        files = {"photo": ("chart.png", photo_bytes, "image/png")}
        data  = {"chat_id": s["chat_id"], "caption": caption, "parse_mode": parse_mode}
        r = requests.post(url, files=files, data=data, timeout=20)
        js = r.json()
        if r.status_code == 200 and js.get("ok"):
            return {"ok": True, "message_id": js.get("result", {}).get("message_id")}
        return {"ok": False, "error": js.get("description", f"HTTP {r.status_code}")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# -----------------------------------------------------------------------------
# Cool-down + quiet hours
# -----------------------------------------------------------------------------
@dataclass
class _State:
    last_alert_ts: float = 0.0
    last_forecast_labels: Dict[str, str] = field(default_factory=dict)
    last_composite_label: str = "Neutral"


def _in_quiet_hours(cfg: TelegramConfig) -> bool:
    """Return True if current UTC time is in the quiet window."""
    from datetime import datetime, timezone
    now_h = datetime.now(timezone.utc).hour
    start, end = cfg.quiet_start_utc, cfg.quiet_end_utc
    if start < end:
        return start <= now_h < end
    # Window crosses midnight (e.g. 22 -> 6)
    return now_h >= start or now_h < end


def _cooldown_active(state: _State, cfg: TelegramConfig) -> bool:
    return (time.time() - state.last_alert_ts) < cfg.cooldown_seconds


# -----------------------------------------------------------------------------
# Condition checkers
# -----------------------------------------------------------------------------
def _fmt_horizon_state(
    f_labels: Dict[str, str],
    f_conf:   Dict[str, float],
    f_scores: Dict[str, float],
    f_emas:   Dict[str, tuple],
) -> str:
    """Compact horizon summary for a Telegram message."""
    lines = []
    for h in ("short", "medium", "long"):
        lines.append(
            f"• *{h.upper()}* (EMA {f_emas[h][0]}/{f_emas[h][1]}): "
            f"{f_labels[h]} — score `{f_scores[h]:+.1f}`, conf {f_conf[h]:.0f}%"
        )
    return "\n".join(lines)


def check_and_alert(
    cfg: TelegramConfig,
    state: _State,
    f_labels: Dict[str, str],
    f_scores: Dict[str, float],
    f_conf:   Dict[str, float],
    f_emas:   Dict[str, tuple],
    composite_label: str,
    composite_score: float,
    last_price: float,
) -> Optional[str]:
    """
    Check alert conditions. If one is met, send a Telegram message
    and return the message that was sent.  Otherwise return None.

    The `state` argument is mutated to track the last forecast / composite
    labels and the last alert timestamp (for cool-down).
    """
    if not cfg.enabled:
        return None
    if not is_configured():
        return None
    if _in_quiet_hours(cfg):
        return None
    if _cooldown_active(state, cfg):
        return None

    # ---- 1) Multi-horizon unanimity ---------------------------------------
    if cfg.alert_on_unanimous:
        non_neutral = [l for l in f_labels.values() if l in ("Bullish", "Bearish")]
        if len(non_neutral) == 3 and len(set(non_neutral)) == 1:
            direction = non_neutral[0]
            min_conf = min(f_conf[h] for h in f_labels)
            if min_conf >= cfg.min_confidence:
                msg = (
                    f"🚨 *Gold Scalper — UNANIMOUS {direction.upper()}*\n\n"
                    f"All 3 forecasts just agreed on *{direction}* "
                    f"(min confidence {min_conf:.0f}%).\n\n"
                    f"{_fmt_horizon_state(f_labels, f_conf, f_scores, f_emas)}\n\n"
                    f"Gold last: `{last_price:,.2f}`\n"
                    f"Composite: `{composite_label}` (score `{composite_score:+.1f}`)"
                )
                if send_message(msg)["ok"]:
                    state.last_alert_ts = time.time()
                    state.last_forecast_labels = dict(f_labels)
                    state.last_composite_label = composite_label
                    return msg

    # ---- 2) Single-forecast flip ------------------------------------------
    if cfg.alert_on_flip and state.last_forecast_labels:
        for h in ("short", "medium", "long"):
            new_lbl = f_labels.get(h, "Neutral")
            old_lbl = state.last_forecast_labels.get(h, "Neutral")
            if new_lbl != old_lbl and new_lbl in ("Bullish", "Bearish"):
                msg = (
                    f"⚡ *{h.upper()} forecast flipped* → *{new_lbl}*\n\n"
                    f"Was `{old_lbl}`, now `{new_lbl}` "
                    f"(conf {f_conf[h]:.0f}%, score `{f_scores[h]:+.1f}`)\n\n"
                    f"Gold last: `{last_price:,.2f}`\n"
                    f"{_fmt_horizon_state(f_labels, f_conf, f_scores, f_emas)}"
                )
                if send_message(msg)["ok"]:
                    state.last_alert_ts = time.time()
                    state.last_forecast_labels = dict(f_labels)
                    state.last_composite_label = composite_label
                    return msg
    elif cfg.alert_on_flip and not state.last_forecast_labels:
        # First run — seed the state without alerting
        state.last_forecast_labels = dict(f_labels)
        state.last_composite_label = composite_label

    # ---- 3) Composite threshold cross -------------------------------------
    if cfg.alert_on_composite_threshold:
        was_abs = abs(_composite_to_num(state.last_composite_label))
        now_abs = abs(_composite_to_num(composite_label))
        if was_abs < cfg.composite_threshold <= now_abs:
            msg = (
                f"📊 *Composite entered {composite_label}*\n\n"
                f"Score: `{composite_score:+.1f}` (threshold ±{cfg.composite_threshold:.0f})\n"
                f"Gold last: `{last_price:,.2f}`\n\n"
                f"{_fmt_horizon_state(f_labels, f_conf, f_scores, f_emas)}"
            )
            if send_message(msg)["ok"]:
                state.last_alert_ts = time.time()
                state.last_forecast_labels = dict(f_labels)
                state.last_composite_label = composite_label
                return msg
        elif was_abs >= cfg.composite_threshold > now_abs:
            msg = (
                f"📊 *Composite returned to Neutral*\n\n"
                f"Score: `{composite_score:+.1f}` (was {state.last_composite_label})\n"
                f"Gold last: `{last_price:,.2f}`"
            )
            if send_message(msg)["ok"]:
                state.last_alert_ts = time.time()
                state.last_forecast_labels = dict(f_labels)
                state.last_composite_label = composite_label
                return msg

    # Always update state (even if no alert fired)
    state.last_forecast_labels = dict(f_labels)
    state.last_composite_label = composite_label
    return None


def _composite_to_num(label: str) -> float:
    """Map composite label to a numeric for threshold comparison."""
    return {
        "Strong Bullish":  90, "Bullish": 55, "Neutral": 0,
        "Bearish": -55, "Strong Bearish": -90,
    }.get(label, 0)


# -----------------------------------------------------------------------------
# Alert history (in-memory, for the UI panel)
# -----------------------------------------------------------------------------
@dataclass
class AlertHistory:
    entries: List[Dict] = field(default_factory=list)
    max_entries: int = 50

    def add(self, message: str, status: str, error: Optional[str] = None) -> None:
        self.entries.insert(0, {
            "time":     time.strftime("%Y-%m-%d %H:%M:%S"),
            "message":  message,
            "status":   status,   # "sent" | "skipped" | "failed"
            "error":    error,
        })
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[:self.max_entries]

    def to_dataframe(self):
        import pandas as pd
        if not self.entries:
            return pd.DataFrame(columns=["time", "status", "message", "error"])
        return pd.DataFrame(self.entries)
