"""Notification hook (week 8) — structurally complete.

Posts high-confidence deals or escalated complaints to a webhook (Slack/Discord/
Resend-compatible JSON). No webhook credential is available here, so when
SIGNALFORGE_WEBHOOK_URL is unset the payload is logged to data/notifications.log.
RUNBOOK.md explains how to wire a real webhook.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from app import config

LOG = config.DATA_DIR / "notifications.log"


def _should_notify(brief: dict) -> tuple[bool, str]:
    d = brief.get("decision") or {}
    kind = d.get("kind")
    if kind == "price_forecast" and d.get("recommendation") == "buy_now" \
            and brief.get("trend_strength") in ("medium", "high"):
        return True, f"BUY NOW: {d.get('title')} ~{config.CURRENCY_SYMBOL}{d.get('point_inr')}"
    if kind == "resolution" and d.get("escalate"):
        return True, f"ESCALATE complaint: {d.get('complaint_type')} ({d.get('severity')})"
    return False, "no high-confidence deal or escalation"


def notify(brief: dict) -> dict:
    ok, text = _should_notify(brief)
    if not ok:
        return {"sent": False, "reason": text}
    payload = {"text": f"India Commerce SignalForge — {text}",
               "intent": brief.get("intent"), "query": brief.get("query"),
               "ts": datetime.now(timezone.utc).isoformat()}
    url = os.environ.get("SIGNALFORGE_WEBHOOK_URL", "")
    if url:
        import requests
        try:
            r = requests.post(url, json=payload, timeout=10)
            return {"sent": 200 <= r.status_code < 300, "status": r.status_code}
        except Exception as e:  # noqa: BLE001
            return {"sent": False, "error": str(e)}
    with LOG.open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return {"sent": False, "logged": True, "reason": "no SIGNALFORGE_WEBHOOK_URL; logged locally"}
