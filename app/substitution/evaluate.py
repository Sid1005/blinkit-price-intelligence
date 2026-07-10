"""RAG-vs-no-RAG substitution eval: samples real Blinkit SKUs that plausibly
need a substitute (out-of-stock, or near-MRP/poor-value), runs both arms
(``app.substitution.compare``), and scores each recommendation against the
real scraped catalog — not against a human-written answer key, since the
whole point is "did the model recommend something that actually exists on
Blinkit, in the right aisle, at roughly the right price."

Metrics per arm:
  * error_rate            — arm failed outright (API error / unparseable JSON)
  * exists_in_catalog_rate — the recommended product fuzzy-matches a REAL
    Blinkit SKU (name similarity >= MATCH_THRESHOLD). This is the headline
    number: Claude zero-shot has no way to know Blinkit's real catalog, so it
    mostly invents plausible-sounding but nonexistent products; the RAG arm
    is constrained to retrieved real text so it should score close to 100%.
  * same_aisle_rate        — of matches, is it actually in the same
    canonical aisle as the original (a sane substitute, not just any real SKU)
  * price_accuracy_rate    — of matches where the arm's reasoning stated a
    price, is it within PRICE_TOLERANCE of the real scraped price
  * cost_usd / latency_s   — averaged and totalled per arm (see
    app.substitution.compare.PRICING_PER_MTOK for the rate table)
"""
from __future__ import annotations

import difflib
import random
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.substitution import blinkit_catalog, compare

MATCH_THRESHOLD = 0.60
PRICE_TOLERANCE = 0.15  # +/- 15% of the real scraped price counts as "accurate"


def build_eval_sample(n: int = 40, seed: int = 0) -> list[dict]:
    """Real SKUs that plausibly need a substitute: every out-of-stock SKU
    first (the clearest real-world trigger), then one poor-value (lowest
    discount) SKU per remaining canonical aisle, up to n total. Aisles with
    only one SKU are excluded — there is no real in-catalog ground truth to
    score against."""
    rows = blinkit_catalog.load_products()
    buckets = blinkit_catalog.bucket_map(rows)
    eligible = {b: rs for b, rs in buckets.items() if len(rs) >= 2}
    rng = random.Random(seed)

    sample, seen = [], set()
    for r in rows:
        if r.get("in_stock") is False and blinkit_catalog.bucket_for(r.get("category")) in eligible:
            sample.append(r)
            seen.add(r["product_name"])

    bucket_names = sorted(eligible)
    rng.shuffle(bucket_names)
    for b in bucket_names:
        if len(sample) >= n:
            break
        candidates = [r for r in eligible[b] if r["product_name"] not in seen]
        if not candidates:
            continue
        candidates.sort(key=lambda r: (r.get("discount_percent") or 0, rng.random()))
        pick = candidates[0]
        sample.append(pick)
        seen.add(pick["product_name"])
    return sample[:n]


def _norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


def _best_catalog_match(name: str, catalog_rows: list[dict]) -> tuple[dict | None, float]:
    if not name:
        return None, 0.0
    target = _norm(name)
    best, best_score = None, 0.0
    for row in catalog_rows:
        score = difflib.SequenceMatcher(None, target, _norm(row["product_name"])).ratio()
        if score > best_score:
            best, best_score = row, score
    return best, best_score


def score_arm_result(original: dict, arm_result: dict, catalog_rows: list[dict]) -> dict:
    name = arm_result.get("substitute") or ""
    match, match_score = _best_catalog_match(name, catalog_rows)
    exists_in_catalog = bool(
        match and match_score >= MATCH_THRESHOLD
        and match["product_name"] != original["product_name"])
    same_aisle = bool(
        exists_in_catalog
        and blinkit_catalog.bucket_for(match.get("category")) == blinkit_catalog.bucket_for(original.get("category")))

    price_stated, price_accurate = None, None
    m = re.search(r'(?:₹|rs\.?\s*)\s*(\d+(?:\.\d+)?)', arm_result.get("reasoning") or "", re.I)
    if m and exists_in_catalog:
        price_stated = float(m.group(1))
        real_price = match["price_inr"]
        price_accurate = abs(price_stated - real_price) <= PRICE_TOLERANCE * max(real_price, 1)

    return {
        "candidate_name": name,
        "match_score": round(match_score, 3),
        "matched_product": match["product_name"] if match else None,
        "exists_in_catalog": exists_in_catalog,
        "same_aisle": same_aisle,
        "price_stated": price_stated,
        "price_accurate": price_accurate,
        "had_error": bool(arm_result.get("error")),
    }


def _aggregate(records: list[dict], arm: str) -> dict:
    scores = [r[arm]["score"] for r in records]
    results = [r[arm]["result"] for r in records]
    n = len(records)
    errors = sum(1 for s in scores if s["had_error"])
    grounded = [s for s in scores if s["exists_in_catalog"]]
    priced = [s for s in grounded if s["price_stated"] is not None]
    return {
        "n": n,
        "error_rate": round(errors / n, 4) if n else 0,
        "exists_in_catalog_rate": round(len(grounded) / n, 4) if n else 0,
        "same_aisle_rate": round(sum(1 for s in grounded if s["same_aisle"]) / len(grounded), 4) if grounded else None,
        "price_stated_count": len(priced),
        "price_accuracy_rate": round(sum(1 for s in priced if s["price_accurate"]) / len(priced), 4) if priced else None,
        "avg_cost_usd": round(sum(r["cost_usd"] for r in results) / n, 6) if n else 0,
        "total_cost_usd": round(sum(r["cost_usd"] for r in results), 6),
        "avg_latency_s": round(sum(r["latency_s"] for r in results) / n, 3) if n else 0,
        "total_latency_s": round(sum(r["latency_s"] for r in results), 3),
    }


def run_eval(sample: list[dict], k_retrieve: int = 5, verbose: bool = True) -> dict:
    catalog_rows = blinkit_catalog.load_products()
    records = []
    t0 = time.perf_counter()
    for i, item in enumerate(sample):
        if verbose:
            print(f"[{i + 1}/{len(sample)}] {item['product_name'][:60]}", flush=True)
        no_rag = compare.no_rag_substitute(item["product_name"])
        rag = compare.rag_substitute(item["product_name"], k=k_retrieve)
        records.append({
            "product_name": item["product_name"],
            "category": item.get("category"),
            "bucket": blinkit_catalog.bucket_for(item.get("category")),
            "in_stock": item.get("in_stock", True),
            "price_inr": item.get("price_inr"),
            "no_rag": {"result": no_rag, "score": score_arm_result(item, no_rag, catalog_rows)},
            "rag": {"result": rag, "score": score_arm_result(item, rag, catalog_rows)},
        })
    return {
        "n_examples": len(sample),
        "wall_clock_s": round(time.perf_counter() - t0, 2),
        "no_rag_summary": _aggregate(records, "no_rag"),
        "rag_summary": _aggregate(records, "rag"),
        "records": records,
    }
