#!/usr/bin/env python3
"""Generate the Blinkit substitution knowledge-base markdown files from the real
scraped catalog (``data/blinkit/blinkit_products.json`` — 921 real SKUs, no
synthetic rows).

Blinkit's own site taxonomy is noisy: the same shelf gets scraped under
slightly different labels depending on which listing page it came from (e.g.
``cheese & butter`` vs ``butter and cheese``, ``audio accessories`` vs
``audio & accessories``). ``CATEGORY_MERGE`` folds those textual duplicates
into one canonical bucket so a substitution cluster represents one real aisle,
not a scrape artifact — it does *not* invent new thematic groupings beyond
what Blinkit's own categories already imply.

For each canonical bucket with >= 2 items we rank every item's top-3
in-bucket substitutes with the same fit/value/availability weighting as
``app/agents/tools.py::_substitute_score`` (adapted: no ``substitute_group``
field on real Blinkit rows, and every scraped rating came back null, so
quality falls back to brand-diversity as the tie-break instead of a star
rating). Buckets with a single SKU have no in-category substitute — they get
one line in ``_no_substitute.md`` instead of a padded ranking.

Output: one markdown file per canonical bucket in
``data/blinkit/substitutions/<slug>.md``, plus ``index.md`` (bucket directory)
and ``_no_substitute.md`` (singleton SKUs). These are the real-data documents
the Part B RAG index (``app/rag/store.py``) chunks and embeds alongside
``app/knowledge_base/substitution_guide.md`` for the with-RAG substitution arm.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.substitution.blinkit_catalog import (bucket_for, rank_substitutes,
                                              slugify, unit_price_basis)

BLINKIT_JSON = Path("data/blinkit/blinkit_products.json")
OUT_DIR = Path("data/blinkit/substitutions")


def render_item_row(row: dict) -> str:
    up, basis = unit_price_basis(row)
    brand = row.get("brand") or "—"
    stock = "in stock" if row.get("in_stock", True) else "OUT OF STOCK"
    disc = f"{row['discount_percent']}%" if row.get("discount_percent") else "—"
    return (f"| {row['product_name']} | {brand} | {row.get('unit') or '—'} | "
            f"₹{row['price_inr']} | ₹{row.get('mrp_inr') or row['price_inr']} | {disc} | "
            f"₹{up} ({basis}) | {stock} |")


def render_bucket_doc(bucket: str, rows: list[dict]) -> str:
    prices = [r["price_inr"] for r in rows]
    brands = sorted({r["brand"] for r in rows if r.get("brand")})
    lines = [
        f"# Substitution Guide — {bucket}",
        "",
        f"{len(rows)} real Blinkit SKUs in this aisle "
        f"(price range ₹{min(prices)}–₹{max(prices)}, {len(brands)} distinct brands: "
        f"{', '.join(brands[:12])}{', ...' if len(brands) > 12 else ''}).",
        "",
        "## Catalog",
        "",
        "| Product | Brand | Unit | Price | MRP | Discount | Unit price | Stock |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: r["price_inr"]):
        lines.append(render_item_row(r))

    lines += ["", "## Ranked substitutes", "",
              "For each SKU, the top-3 in-aisle alternatives ranked by unit-price "
              "value, stock availability, and brand diversity (see "
              "`app/knowledge_base/substitution_guide.md` for the scoring policy; "
              "ratings are unavailable for every SKU in this scrape, so quality is "
              "inferred from brand diversity and discount depth instead of a star "
              "rating)."]
    for r in sorted(rows, key=lambda r: r["product_name"]):
        subs = rank_substitutes(r, rows)
        if not subs:
            continue
        up, basis = unit_price_basis(r)
        lines.append(f"\n**{r['product_name']}** ({r.get('brand') or 'unbranded'}, "
                     f"₹{r['price_inr']}, ₹{up} {basis})")
        for cand, score in subs:
            c_up, c_basis = unit_price_basis(cand)
            reasons = []
            if c_basis == basis:
                if c_up < up:
                    pct = round((up - c_up) / up * 100, 1) if up else 0
                    reasons.append(f"{pct}% cheaper per {('kg/L' if c_basis == 'per kg/L' else 'unit')}")
                elif c_up > up:
                    reasons.append("similar aisle, slightly pricier per "
                                    f"{'kg/L' if c_basis == 'per kg/L' else 'unit'}")
                else:
                    reasons.append("same unit price")
            else:
                # Different pricing basis (e.g. loose-weight vs. piece-priced) —
                # compare sticker price only, not a per-kg/L claim.
                if cand["price_inr"] < r["price_inr"]:
                    reasons.append("cheaper sticker price (different pack basis, not directly unit-comparable)")
                elif cand["price_inr"] > r["price_inr"]:
                    reasons.append("similar aisle, pricier sticker price (different pack basis)")
                else:
                    reasons.append("same sticker price (different pack basis)")
            if not cand.get("in_stock", True):
                reasons.append("currently OUT OF STOCK")
            if cand.get("brand") and cand.get("brand") != r.get("brand"):
                reasons.append(f"alternative brand ({cand['brand']})")
            if cand.get("discount_percent"):
                reasons.append(f"{cand['discount_percent']}% off")
            lines.append(f"- {cand['product_name']} — ₹{cand['price_inr']} "
                        f"(₹{c_up} {c_basis}), score {score:.2f} — {'; '.join(reasons)}")
    lines.append("")
    return "\n".join(lines)


def render_index(bucket_rows: dict[str, list[dict]], singleton_rows: list[dict]) -> str:
    lines = ["# Blinkit Substitution Knowledge Base — Index", "",
              f"Generated from {sum(len(v) for v in bucket_rows.values()) + len(singleton_rows)} "
              "real scraped Blinkit SKUs (`data/blinkit/blinkit_products.json`, no synthetic rows). "
              "Each bucket file below ranks in-aisle substitutes for every SKU in that aisle; "
              "singleton SKUs (no in-aisle alternative in this scrape) are listed in "
              "`_no_substitute.md`.", "",
              "| Aisle | SKUs | File |", "|---|---|---|"]
    for bucket in sorted(bucket_rows):
        rows = bucket_rows[bucket]
        lines.append(f"| {bucket} | {len(rows)} | [{slugify(bucket)}.md]({slugify(bucket)}.md) |")
    lines.append(f"| *(singletons)* | {len(singleton_rows)} | [_no_substitute.md](_no_substitute.md) |")
    lines.append("")
    return "\n".join(lines)


def render_no_substitute(rows: list[dict]) -> str:
    lines = ["# Singleton SKUs — No In-Aisle Substitute In This Scrape", "",
              "Each of these aisles yielded exactly one Blinkit SKU in the scrape, so no "
              "in-category substitute can be ranked from real data. Per "
              "`app/knowledge_base/substitution_guide.md`'s guardrails, the substitute "
              "agent should say so rather than forcing a weak cross-category recommendation.",
              "", "| Product | Aisle | Brand | Price | Stock |", "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["category"] or ""):
        brand = r.get("brand") or "—"
        stock = "in stock" if r.get("in_stock", True) else "OUT OF STOCK"
        lines.append(f"| {r['product_name']} | {r['category']} | {brand} | ₹{r['price_inr']} | {stock} |")
    lines.append("")
    return "\n".join(lines)


def main():
    rows = json.loads(BLINKIT_JSON.read_text(encoding="utf-8"))
    print(f"Loaded {len(rows):,} scraped Blinkit rows")

    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_bucket[bucket_for(r.get("category"))].append(r)

    bucket_rows = {b: rs for b, rs in by_bucket.items() if len(rs) >= 2}
    singleton_rows = [r for rs in by_bucket.values() if len(rs) < 2 for r in rs]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for bucket, rs in bucket_rows.items():
        path = OUT_DIR / f"{slugify(bucket)}.md"
        path.write_text(render_bucket_doc(bucket, rs), encoding="utf-8")
    (OUT_DIR / "_no_substitute.md").write_text(render_no_substitute(singleton_rows), encoding="utf-8")
    (OUT_DIR / "index.md").write_text(render_index(bucket_rows, singleton_rows), encoding="utf-8")

    print(f"Wrote {len(bucket_rows)} bucket docs + index.md + _no_substitute.md "
          f"({len(singleton_rows)} singleton SKUs) to {OUT_DIR}/")


if __name__ == "__main__":
    main()
