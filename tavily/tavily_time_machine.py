#!/usr/bin/env python3
"""Tavily Time Machine — Temporal Web Intelligence Engine.

Tavily is stateless: every call sees the web frozen at one instant. It has no
memory, no concept of "before," no ability to detect change over time.

This module wraps Tavily Extract with a persistent SQLite snapshot store and
uses a Groq LLM to perform **semantic diffing** — not character-level diffs,
but understanding *what changed in meaning*:
  - A price dropped from ₹999 to ₹749
  - A "Buy Now" button changed to "Out of Stock"
  - A refund policy paragraph was silently removed
  - A news article headline was edited after publication

This is something Tavily literally cannot do alone. We add the dimension of
**time** to a tool that only knows about the present.

Usage:
    python -m app.ingest.tavily_time_machine snapshot https://example.com
    python -m app.ingest.tavily_time_machine snapshot https://example.com
    python -m app.ingest.tavily_time_machine diff https://example.com
    python -m app.ingest.tavily_time_machine history https://example.com
    python -m app.ingest.tavily_time_machine watch https://example.com --interval 60
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app import config
from app.llm import groq_client

# ---------------------------------------------------------------------------
# Database Layer
# ---------------------------------------------------------------------------

DB_PATH = config.DATA_DIR / "time_machine.sqlite"


def _db() -> sqlite3.Connection:
    """Open (and initialize) the snapshot database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            url         TEXT    NOT NULL,
            timestamp   TEXT    NOT NULL,
            title       TEXT,
            raw_content TEXT,
            content_hash TEXT,
            word_count  INTEGER,
            byte_count  INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_snap_url ON snapshots(url);

        CREATE TABLE IF NOT EXISTS diffs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            url             TEXT    NOT NULL,
            old_snapshot_id INTEGER REFERENCES snapshots(id),
            new_snapshot_id INTEGER REFERENCES snapshots(id),
            timestamp       TEXT    NOT NULL,
            content_changed INTEGER NOT NULL DEFAULT 0,
            change_summary  TEXT,
            changes_json    TEXT,
            significance    TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_diff_url ON diffs(url);
    """)
    return conn


# ---------------------------------------------------------------------------
# Snapshot: Tavily Extract → SQLite
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    id: int
    url: str
    timestamp: str
    title: str
    raw_content: str
    content_hash: str
    word_count: int
    byte_count: int


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def take_snapshot(url: str) -> Snapshot:
    """Extract a URL via Tavily and store a timestamped snapshot."""
    from tavily import TavilyClient

    if not config.TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not set.")

    client = TavilyClient(api_key=config.TAVILY_API_KEY)
    print(f"  ⏳ Extracting {url} via Tavily...")
    res = client.extract(urls=[url])

    results = res.get("results", [])
    if not results:
        raise RuntimeError(f"Tavily returned no results for {url}")

    page = results[0]
    raw = page.get("raw_content", "") or ""
    title = page.get("title", "") or url

    now = datetime.now(timezone.utc).isoformat()
    h = _hash(raw)
    wc = len(raw.split())
    bc = len(raw.encode("utf-8"))

    conn = _db()
    cur = conn.execute(
        "INSERT INTO snapshots (url, timestamp, title, raw_content, content_hash, word_count, byte_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (url, now, title, raw, h, wc, bc),
    )
    conn.commit()
    snap_id = cur.lastrowid

    snap = Snapshot(id=snap_id, url=url, timestamp=now, title=title,
                    raw_content=raw, content_hash=h, word_count=wc, byte_count=bc)
    print(f"  ✅ Snapshot #{snap_id} saved  ({wc} words, hash={h})")
    return snap


# ---------------------------------------------------------------------------
# Semantic Diff via Groq LLM
# ---------------------------------------------------------------------------

@dataclass
class Change:
    category: str       # e.g. "price_change", "stock_change", "content_added", "content_removed", "wording_edit"
    description: str    # human-readable explanation
    severity: str       # "critical", "major", "minor", "cosmetic"
    old_value: str | None = None
    new_value: str | None = None


@dataclass
class DiffResult:
    url: str
    old_snapshot_id: int
    new_snapshot_id: int
    old_timestamp: str
    new_timestamp: str
    content_changed: bool
    hash_changed: bool
    word_delta: int
    significance: str   # overall: "critical", "major", "minor", "identical"
    summary: str        # one-paragraph LLM summary
    changes: list[Change] = field(default_factory=list)


def _llm_semantic_diff(old_text: str, new_text: str, url: str) -> dict:
    """Use Groq LLM to semantically diff two page snapshots of flight searches."""
    system = (
        "You are a travel analyst and flight price change specialist. You compare two versions "
        "of a flight search results page (e.g. Bangalore to Hyderabad flights) captured at different times.\n"
        "Your job is to identify MEANINGFUL changes that a traveler would care about, such as price drops, "
        "schedule changes, airline carrier additions/removals, or seat availability updates.\n\n"
        "Change categories you should detect:\n"
        "  • price_change — ticket fare dropped or increased (specify amount in INR or USD)\n"
        "  • schedule_change — departure/arrival times, layovers, stops, or flight duration changed\n"
        "  • carrier_added — a new airline option or flight number appeared on this route\n"
        "  • carrier_removed — a previously available flight option is no longer shown\n"
        "  • availability_alert — a seat availability status changed (e.g. 'only 1 seat left' or sold out)\n"
        "  • metadata_change — search route, date, or cabins changed\n\n"
        "Severity levels:\n"
        "  • critical — significant price drops (>5% or >₹500 drop) or flight cancellation\n"
        "  • major — price increases, schedule/stop changes, or new budget carrier added\n"
        "  • minor — small schedule adjustments (<15 mins) or small price tweaks\n"
        "  • cosmetic — formatting, currency symbol display, or layout shifts\n\n"
        "Output STRICT JSON only."
    )

    # Truncate smartly for token budget
    old_trunc = old_text[:6000]
    new_trunc = new_text[:6000]

    user = (
        f"Compare these two flight search snapshots of {url}.\n\n"
        f"=== OLD SNAPSHOT ===\n{old_trunc}\n\n"
        f"=== NEW SNAPSHOT ===\n{new_trunc}\n\n"
        "Output a JSON object with:\n"
        '  "summary": a 2-3 sentence summary of the airfare changes and price drops (or "No airfare changes detected" if identical),\n'
        '  "significance": "critical" | "major" | "minor" | "identical",\n'
        '  "changes": [ { "category": "...", "description": "...", "severity": "...", "old_value": "..." or null, "new_value": "..." or null } ]\n'
        "If nothing meaningful changed, return an empty changes array and significance=identical."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return groq_client.chat_json(messages, model=config.GROQ_MODELS["strong"])


def compute_diff(url: str) -> DiffResult | None:
    """Compare the two most recent snapshots of a URL using LLM semantic diffing."""
    conn = _db()
    rows = conn.execute(
        "SELECT * FROM snapshots WHERE url = ? ORDER BY timestamp DESC LIMIT 2", (url,)
    ).fetchall()

    if len(rows) < 2:
        print(f"  ⚠️  Need at least 2 snapshots of {url} to diff. Found {len(rows)}.")
        return None

    new_snap = dict(rows[0])
    old_snap = dict(rows[1])

    hash_changed = old_snap["content_hash"] != new_snap["content_hash"]
    word_delta = new_snap["word_count"] - old_snap["word_count"]

    if not hash_changed:
        result = DiffResult(
            url=url,
            old_snapshot_id=old_snap["id"],
            new_snapshot_id=new_snap["id"],
            old_timestamp=old_snap["timestamp"],
            new_timestamp=new_snap["timestamp"],
            content_changed=False,
            hash_changed=False,
            word_delta=0,
            significance="identical",
            summary="No changes detected — the page content is identical between snapshots.",
            changes=[],
        )
        _save_diff(result)
        return result

    print(f"  🔍 Content hash changed. Running LLM semantic diff...")
    llm_result = _llm_semantic_diff(
        old_snap["raw_content"], new_snap["raw_content"], url
    )

    changes = []
    for c in llm_result.get("changes", []):
        changes.append(Change(
            category=c.get("category", "unknown"),
            description=c.get("description", ""),
            severity=c.get("severity", "minor"),
            old_value=c.get("old_value"),
            new_value=c.get("new_value"),
        ))

    result = DiffResult(
        url=url,
        old_snapshot_id=old_snap["id"],
        new_snapshot_id=new_snap["id"],
        old_timestamp=old_snap["timestamp"],
        new_timestamp=new_snap["timestamp"],
        content_changed=True,
        hash_changed=True,
        word_delta=word_delta,
        significance=llm_result.get("significance", "minor"),
        summary=llm_result.get("summary", "Changes detected."),
        changes=changes,
    )
    _save_diff(result)
    return result


def _save_diff(result: DiffResult) -> None:
    conn = _db()
    changes_json = json.dumps([asdict(c) for c in result.changes], ensure_ascii=False)
    conn.execute(
        "INSERT INTO diffs (url, old_snapshot_id, new_snapshot_id, timestamp, "
        "content_changed, change_summary, changes_json, significance) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (result.url, result.old_snapshot_id, result.new_snapshot_id,
         datetime.now(timezone.utc).isoformat(),
         int(result.content_changed), result.summary, changes_json,
         result.significance),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# History & Watch
# ---------------------------------------------------------------------------

def get_history(url: str) -> list[dict]:
    """Return all snapshots and diffs for a URL, ordered by time."""
    conn = _db()
    snaps = conn.execute(
        "SELECT id, url, timestamp, title, content_hash, word_count, byte_count "
        "FROM snapshots WHERE url = ? ORDER BY timestamp ASC", (url,)
    ).fetchall()

    diffs = conn.execute(
        "SELECT * FROM diffs WHERE url = ? ORDER BY timestamp ASC", (url,)
    ).fetchall()

    return {
        "url": url,
        "snapshot_count": len(snaps),
        "diff_count": len(diffs),
        "snapshots": [dict(s) for s in snaps],
        "diffs": [dict(d) for d in diffs],
    }


def get_all_watched_urls() -> list[str]:
    """Return all unique URLs that have been snapshotted."""
    conn = _db()
    rows = conn.execute("SELECT DISTINCT url FROM snapshots ORDER BY url").fetchall()
    return [r["url"] for r in rows]


def watch(url: str, interval_seconds: int = 300, max_iterations: int = 5) -> None:
    """Periodically snapshot a URL and report changes."""
    print(f"👁️  Watching {url} every {interval_seconds}s (max {max_iterations} iterations)\n")
    for i in range(max_iterations):
        print(f"─── Iteration {i + 1}/{max_iterations} ───")
        take_snapshot(url)

        if i > 0:
            diff = compute_diff(url)
            if diff:
                _print_diff(diff)
        else:
            print("  (First snapshot — nothing to diff yet)")

        if i < max_iterations - 1:
            print(f"  💤 Sleeping {interval_seconds}s...\n")
            time.sleep(interval_seconds)

    print(f"\n✅ Watch complete. {max_iterations} snapshots taken for {url}")


# ---------------------------------------------------------------------------
# Pretty Printing
# ---------------------------------------------------------------------------

_SEV_COLORS = {
    "critical": "\033[91m",  # red
    "major":    "\033[93m",  # yellow
    "minor":    "\033[94m",  # blue
    "cosmetic": "\033[90m",  # gray
    "identical": "\033[92m", # green
}
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _print_diff(diff: DiffResult) -> None:
    sev_color = _SEV_COLORS.get(diff.significance, "")

    print(f"\n  {_BOLD}Diff Result:{_RESET}")
    print(f"  Old: snapshot #{diff.old_snapshot_id} ({diff.old_timestamp})")
    print(f"  New: snapshot #{diff.new_snapshot_id} ({diff.new_timestamp})")
    print(f"  Hash changed: {'Yes' if diff.hash_changed else 'No'}")
    print(f"  Word delta: {diff.word_delta:+d}")
    print(f"  Significance: {sev_color}{_BOLD}{diff.significance.upper()}{_RESET}")
    print(f"  Summary: {diff.summary}")

    if diff.changes:
        print(f"\n  {_BOLD}Changes detected ({len(diff.changes)}):{_RESET}")
        for i, c in enumerate(diff.changes, 1):
            c_color = _SEV_COLORS.get(c.severity, "")
            print(f"    {i}. [{c_color}{c.severity.upper()}{_RESET}] "
                  f"{_BOLD}{c.category}{_RESET}: {c.description}")
            if c.old_value:
                print(f"       OLD → {c.old_value}")
            if c.new_value:
                print(f"       NEW → {c.new_value}")
    else:
        print("  No semantic changes detected.")


def _print_history(history: dict) -> None:
    print(f"\n{_BOLD}📜 History for {history['url']}{_RESET}")
    print(f"   Snapshots: {history['snapshot_count']}   Diffs: {history['diff_count']}\n")

    for snap in history["snapshots"]:
        print(f"   #{snap['id']:>3}  {snap['timestamp']}  "
              f"hash={snap['content_hash']}  words={snap['word_count']}")

    if history["diffs"]:
        print(f"\n   {_BOLD}Diff history:{_RESET}")
        for d in history["diffs"]:
            sig = d["significance"] or "unknown"
            s_color = _SEV_COLORS.get(sig, "")
            summary = (d["change_summary"] or "")[:120]
            print(f"   #{d['old_snapshot_id']}→#{d['new_snapshot_id']}  "
                  f"[{s_color}{sig.upper()}{_RESET}]  {summary}")


# ---------------------------------------------------------------------------
# Export for the HTML dashboard
# ---------------------------------------------------------------------------

def export_dashboard_json() -> dict:
    """Export all watched URLs, their snapshots, and diffs as JSON for the HTML dashboard."""
    urls = get_all_watched_urls()
    data = []
    for url in urls:
        h = get_history(url)
        # Strip raw_content from snapshots (too large for JSON export)
        for snap in h.get("snapshots", []):
            snap.pop("raw_content", None)
        data.append(h)
    return {"generated": datetime.now(timezone.utc).isoformat(), "urls": data}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tavily Flight Price Time Machine — Flight Fare Tracking Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Take a snapshot of a flight search page (e.g. Bangalore to Hyderabad)\n"
            "  python -m app.ingest.tavily_time_machine snapshot \"https://flights.google.com/search?q=BLR-HYD\"\n\n"
            "  # Compare the last two snapshots to identify price drops\n"
            "  python -m app.ingest.tavily_time_machine diff \"https://flights.google.com/search?q=BLR-HYD\"\n\n"
            "  # View full price alert history\n"
            "  python -m app.ingest.tavily_time_machine history \"https://flights.google.com/search?q=BLR-HYD\"\n\n"
            "  # Auto-watch a route for price drops every 5 minutes\n"
            "  python -m app.ingest.tavily_time_machine watch \"https://flights.google.com/search?q=BLR-HYD\" --interval 300\n\n"
            "  # Export flight tracking data to JSON for dashboard\n"
            "  python -m app.ingest.tavily_time_machine export\n"
        ),
    )
    sub = parser.add_subparsers(dest="command")

    p_snap = sub.add_parser("snapshot", help="Take a snapshot of a flight search URL")
    p_snap.add_argument("url", help="Flight search URL to snapshot")

    p_diff = sub.add_parser("diff", help="Semantic-diff flight fares between two recent snapshots")
    p_diff.add_argument("url", help="Flight URL to diff")

    p_hist = sub.add_parser("history", help="Show flight snapshot and price change history")
    p_hist.add_argument("url", help="Flight URL to show history for")

    p_watch = sub.add_parser("watch", help="Periodically snapshot and check for price drops")
    p_watch.add_argument("url", help="Flight URL to watch")
    p_watch.add_argument("--interval", type=int, default=300, help="Seconds between snapshots (default 300)")
    p_watch.add_argument("--iterations", type=int, default=3, help="Number of snapshots (default 3)")

    p_export = sub.add_parser("export", help="Export all data as JSON for the dashboard")

    args = parser.parse_args()

    print(f"{_BOLD}✈️ Tavily Flight Price Time Machine — Temporal Airfare Intelligence{_RESET}\n")

    if args.command == "snapshot":
        take_snapshot(args.url)

    elif args.command == "diff":
        diff = compute_diff(args.url)
        if diff:
            _print_diff(diff)

    elif args.command == "history":
        h = get_history(args.url)
        _print_history(h)

    elif args.command == "watch":
        watch(args.url, interval_seconds=args.interval, max_iterations=args.iterations)

    elif args.command == "export":
        data = export_dashboard_json()
        out_path = config.DATA_DIR / "time_machine_export.json"
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"  ✅ Exported flight data to {out_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
