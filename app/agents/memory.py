"""Agent memory / persistence (week 8) — SQLite-backed brief history.

Remembers analysed SKUs, complaints, and alerted opportunities so the engine can
recall prior decisions and avoid duplicate alerts.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from app import config


def _conn():
    conn = sqlite3.connect(config.MEMORY_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS briefs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, intent TEXT, query TEXT, summary TEXT, payload TEXT)""")
    return conn


def _summarize(brief: dict) -> str:
    d = brief.get("decision") or {}
    kind = d.get("kind", "")
    if kind == "price_forecast":
        return f"{d.get('title', '')}: {d.get('recommendation', '')} @ \u20b9{d.get('point_inr', 0)}"
    if kind == "substitution_set":
        return f"sub for {d.get('original_title', '')}: {len(d.get('candidates', []))} candidates"
    if kind == "resolution":
        return f"{d.get('complaint_type', '')} (escalate={d.get('escalate')})"
    return brief.get("intent", "")


def remember(brief: dict) -> int:
    conn = _conn()
    cur = conn.execute(
        "INSERT INTO briefs (ts,intent,query,summary,payload) VALUES (?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), brief.get("intent", ""),
         brief.get("query", ""), _summarize(brief),
         json.dumps(brief, ensure_ascii=False)))
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid


def recall(intent: str | None = None, limit: int = 10) -> list[dict]:
    conn = _conn()
    if intent:
        rows = conn.execute("SELECT payload FROM briefs WHERE intent=? ORDER BY id DESC LIMIT ?",
                            (intent, limit)).fetchall()
    else:
        rows = conn.execute("SELECT payload FROM briefs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [json.loads(r[0]) for r in rows]


def stats() -> dict:
    conn = _conn()
    rows = conn.execute("SELECT intent, COUNT(*) FROM briefs GROUP BY intent").fetchall()
    total = conn.execute("SELECT COUNT(*) FROM briefs").fetchone()[0]
    conn.close()
    return {"total": total, "by_intent": {r[0]: r[1] for r in rows}}
