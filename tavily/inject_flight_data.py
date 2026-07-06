#!/usr/bin/env python3
"""Inject mock flight search data for the BLR-HYD route.

This demonstrates the Flight Price Time Machine's capability to track airfare
drops and schedule modifications over time, using Groq LLM semantic diffing.
"""
import sqlite3
import sys
import hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app import config
import tavily_time_machine

DB_PATH = config.DATA_DIR / "time_machine.sqlite"

URL = "https://flights.google.com/search?q=BLR-HYD"
TITLE = "Google Flights - Bangalore (BLR) to Hyderabad (HYD)"

SNAPSHOTS = [
    # Snapshot 1: Baseline fares
    """Google Flights search results for Bangalore (BLR) to Hyderabad (HYD) on Wednesday, Jul 15, 2026.
Currency: INR (₹)

Available flight options:
1. IndiGo 6E-5312 (Non-stop)
   Departs: 06:15 BLR | Arrives: 07:20 HYD
   Duration: 1h 05m
   Price: ₹4,800
   Status: Cabin crew speaking Hindi and English. Seats available.

2. Akasa Air QP-1324 (Non-stop)
   Departs: 08:30 BLR | Arrives: 09:40 HYD
   Duration: 1h 10m
   Price: ₹4,200
   Status: Economy cabin. Only 3 seats left at this price!

3. Air India AI-615 (Non-stop)
   Departs: 12:45 BLR | Arrives: 13:55 HYD
   Duration: 1h 10m
   Price: ₹5,500
   Status: Full service carrier. Meal included.

4. IndiGo 6E-6215 (Non-stop)
   Departs: 18:30 BLR | Arrives: 19:40 HYD
   Duration: 1h 10m
   Price: ₹4,500
   Status: Standard economy.
""",

    # Snapshot 2: Price drop on IndiGo and Akasa
    """Google Flights search results for Bangalore (BLR) to Hyderabad (HYD) on Wednesday, Jul 15, 2026.
Currency: INR (₹)

Available flight options:
1. IndiGo 6E-5312 (Non-stop)
   Departs: 06:15 BLR | Arrives: 07:20 HYD
   Duration: 1h 05m
   Price: ₹4,100 (Price dropped by ₹700!)
   Status: Cabin crew speaking Hindi and English. Seats available.

2. Akasa Air QP-1324 (Non-stop)
   Departs: 08:30 BLR | Arrives: 09:40 HYD
   Duration: 1h 10m
   Price: ₹3,800 (Price dropped by ₹400!)
   Status: Economy cabin. Only 1 seat left at this price!

3. Air India AI-615 (Non-stop)
   Departs: 12:45 BLR | Arrives: 13:55 HYD
   Duration: 1h 10m
   Price: ₹5,500
   Status: Full service carrier. Meal included.

4. IndiGo 6E-6215 (Non-stop)
   Departs: 18:30 BLR | Arrives: 19:40 HYD
   Duration: 1h 10m
   Price: ₹4,500
   Status: Standard economy.
""",

    # Snapshot 3: Further price drops and new airline Star Air
    """Google Flights search results for Bangalore (BLR) to Hyderabad (HYD) on Wednesday, Jul 15, 2026.
Currency: INR (₹)

Available flight options:
1. IndiGo 6E-5312 (Non-stop)
   Departs: 06:15 BLR | Arrives: 07:20 HYD
   Duration: 1h 05m
   Price: ₹3,500 (Massive drop! Saved ₹1,300 compared to initial price)
   Status: Cabin crew speaking Hindi and English. Seats available.

2. Akasa Air QP-1324 (Non-stop)
   Departs: 08:30 BLR | Arrives: 09:40 HYD
   Duration: 1h 10m
   Price: ₹3,800
   Status: Economy cabin. Seats available.

3. Air India AI-615 (Non-stop)
   Departs: 12:45 BLR | Arrives: 13:55 HYD
   Duration: 1h 10m
   Price: ₹4,900 (Price dropped by ₹600!)
   Status: Full service carrier. Meal included.

4. IndiGo 6E-6215 (Non-stop)
   Departs: 18:30 BLR | Arrives: 19:40 HYD
   Duration: 1h 10m
   Price: ₹4,500
   Status: Standard economy.

5. Star Air S5-201 (Non-stop)
   Departs: 20:15 BLR | Arrives: 21:25 HYD
   Duration: 1h 10m
   Price: ₹2,900 (New regional option available!)
   Status: Embraer regional jet. Only 2 seats left!
"""
]


def clear_existing_data():
    """Clear any previous snapshots or diffs for the BLR-HYD route to avoid duplicates."""
    conn = sqlite3.connect(str(DB_PATH))
    print(f"🧹 Clearing existing flight data in {DB_PATH} for {URL}...")
    conn.execute("DELETE FROM diffs WHERE url = ?", (URL,))
    conn.execute("DELETE FROM snapshots WHERE url = ?", (URL,))
    conn.commit()
    conn.close()


def inject_and_diff():
    conn = sqlite3.connect(str(DB_PATH))
    base_time = datetime.now(timezone.utc) - timedelta(days=1)
    
    snapshot_ids = []
    
    for i, content in enumerate(SNAPSHOTS):
        timestamp = (base_time + timedelta(hours=i * 6)).isoformat()
        wc = len(content.split())
        bc = len(content.encode("utf-8"))
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        
        cur = conn.execute(
            "INSERT INTO snapshots (url, timestamp, title, raw_content, content_hash, word_count, byte_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (URL, timestamp, TITLE, content, h, wc, bc)
        )
        conn.commit()
        snap_id = cur.lastrowid
        snapshot_ids.append(snap_id)
        print(f"📈 Injected Flight Snapshot #{snap_id} (timestamp: {timestamp})")
        
        # If we have at least 2 snapshots, we can diff them
        if len(snapshot_ids) >= 2:
            print(f"🔍 Running Groq LLM diff between snapshot #{snapshot_ids[-2]} and #{snap_id}...")
            # We call the compute_diff function directly
            # This triggers Groq LLM over the inserted snapshots
            diff_res = tavily_time_machine.compute_diff(URL)
            if diff_res:
                print(f"   ✅ Diff computed! Significance: {diff_res.significance.upper()}")
                print(f"   Summary: {diff_res.summary}")
    
    conn.close()


if __name__ == "__main__":
    clear_existing_data()
    inject_and_diff()
    print("\n✈️ Flight data injection and diffing complete!")
