"""Curated Indian commerce world-data (week 6 data curation).

Generates and loads the two datasets the engine reasons over:
  * catalog   — SKUs with pack size, unit, platform, MRP, price, stock, rating.
  * prices    — price observations across festivals (for deal forecasting).

Data is curated + synthetically expanded with a fixed seed so it is reproducible
and clearly demo data — we never claim live Blinkit/Amazon prices.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from app import config

CATALOG_PATH = config.DATA_DIR / "catalog" / "products.json"
PRICES_PATH = config.DATA_DIR / "prices" / "price_observations.jsonl"
BLINKIT_PATH = config.DATA_DIR / "blinkit_products.json"

# --- Seed catalog: realistic Indian SKUs across grocery + electronics. ----------
# (title, category, pack_size, unit, platform, mrp_inr, price_inr, stock, rating, reviews)
_SEED_CATALOG = [
    # Grocery / staples
    ("Tata Salt Iodised", "grocery", 1.0, "kg", "Blinkit", 28, 26, 120, 4.4, 5200),
    ("Aashirvaad Atta Whole Wheat", "grocery", 5.0, "kg", "BigBasket", 285, 255, 60, 4.5, 8800),
    ("Fortune Sunflower Oil", "grocery", 1.0, "litre", "Blinkit", 165, 142, 0, 4.2, 3100),
    ("Saffola Gold Oil", "grocery", 1.0, "litre", "BigBasket", 185, 169, 40, 4.3, 2700),
    ("Amul Gold Milk", "grocery", 0.5, "litre", "Zepto", 35, 34, 200, 4.6, 9100),
    ("Maggi 2-Minute Noodles", "grocery", 0.42, "kg", "Blinkit", 96, 84, 150, 4.5, 15000),
    ("Tata Tea Premium", "grocery", 0.5, "kg", "BigBasket", 290, 255, 35, 4.4, 4200),
    ("Red Label Tea", "grocery", 0.5, "kg", "Zepto", 270, 245, 0, 4.3, 3900),
    ("Surf Excel Matic Liquid", "grocery", 2.0, "litre", "Blinkit", 530, 449, 25, 4.4, 2100),
    ("Colgate MaxFresh Toothpaste", "grocery", 0.15, "kg", "Zepto", 99, 89, 80, 4.3, 6700),
    ("Dettol Handwash Refill", "grocery", 1.5, "litre", "BigBasket", 299, 219, 30, 4.5, 1800),
    ("Lays Classic Salted", "grocery", 0.1, "kg", "Blinkit", 50, 48, 300, 4.4, 12000),
    # Electronics
    ("Apple iPhone 15 128GB", "electronics", 1.0, "piece", "Amazon.in", 79900, 67999, 12, 4.6, 23000),
    ("Samsung Galaxy S24 256GB", "electronics", 1.0, "piece", "Flipkart", 89999, 71999, 8, 4.5, 14000),
    ("OnePlus Nord CE4 256GB", "electronics", 1.0, "piece", "Amazon.in", 26999, 23999, 25, 4.3, 9800),
    ("Redmi Note 13 Pro 128GB", "electronics", 1.0, "piece", "Flipkart", 25999, 21999, 40, 4.2, 18000),
    ("boAt Airdopes 161 TWS", "electronics", 1.0, "piece", "Amazon.in", 2990, 1199, 100, 4.1, 56000),
    ("Sony WH-CH520 Headphones", "electronics", 1.0, "piece", "Flipkart", 4990, 3490, 30, 4.4, 7200),
    ("Mi Smart Band 8", "electronics", 1.0, "piece", "Amazon.in", 3499, 2799, 0, 4.2, 11000),
    ("HP 15s Ryzen 5 Laptop", "electronics", 1.0, "piece", "Flipkart", 58999, 42990, 6, 4.3, 3400),
    ("Logitech M331 Silent Mouse", "electronics", 1.0, "piece", "Amazon.in", 995, 749, 150, 4.5, 21000),
    ("TP-Link Archer C6 Router", "electronics", 1.0, "piece", "Flipkart", 2999, 1699, 45, 4.3, 8900),
    # Personal care / home
    ("Nivea Body Lotion", "personal_care", 0.4, "litre", "Zepto", 425, 339, 50, 4.5, 4100),
    ("Head & Shoulders Shampoo", "personal_care", 0.65, "litre", "Blinkit", 540, 449, 0, 4.3, 3300),
    ("Pampers Diapers Medium", "baby", 0.0, "piece", "BigBasket", 799, 649, 20, 4.6, 5600),
]

# Substitution clusters: SKUs that are interchangeable for substitution ranking.
_SUBSTITUTE_GROUPS = {
    "edible_oil": ["Fortune Sunflower Oil", "Saffola Gold Oil"],
    "tea": ["Tata Tea Premium", "Red Label Tea"],
    "midrange_phone": ["OnePlus Nord CE4 256GB", "Redmi Note 13 Pro 128GB"],
    "flagship_phone": ["Apple iPhone 15 128GB", "Samsung Galaxy S24 256GB"],
    "tws_earbuds": ["boAt Airdopes 161 TWS", "Sony WH-CH520 Headphones"],
}


def _round_paise(x: float) -> int:
    return int(round(x))


def _parse_pack_size(unit_str: str) -> tuple[float, str]:
    """Parse a Blinkit unit string (e.g. '34 g', '20g', '1 kg') into (size, unit).

    Grams/millilitres are normalized to kg/litre so unit-price math matches the rest of
    the catalog. Handles a missing space between number and unit. Unrecognised strings
    fall back to a single piece.
    """
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*(kg|g|gram|grams|litre|liter|l|ml)\b",
                  (unit_str or "").strip().lower())
    if not m:
        return 1.0, "piece"
    value, unit = float(m.group(1)), m.group(2)
    if unit in ("g", "gram", "grams"):
        return round(value / 1000.0, 4), "kg"
    if unit == "ml":
        return round(value / 1000.0, 4), "litre"
    if unit in ("litre", "liter", "l"):
        return value, "litre"
    return value, "kg"  # kg


def load_blinkit_raw() -> list[dict]:
    """Load the real scraped Blinkit products; default MRP to price*1.05 when missing."""
    if not BLINKIT_PATH.exists():
        return []
    raw = json.loads(BLINKIT_PATH.read_text())
    if not isinstance(raw, list):
        return []
    out = []
    for r in raw:
        price = float(r["price_inr"])
        mrp = r.get("mrp_inr")
        mrp = float(mrp) if mrp else round(price * 1.05)
        out.append({**r, "price_inr": price, "mrp_inr": mrp})
    return out


# Map Blinkit category strings → internal category + substitute_group.
_BL_CATEGORY_MAP = {
    # Chocolates & confectionery
    "chocolates": ("grocery", "chocolate"),
    "chocolates & candies": ("grocery", "chocolate"),
    "chocolate packs": ("grocery", "chocolate"),
    # Chips & snacks
    "chips & crisps": ("grocery", "chips"),
    "chips": ("grocery", "chips"),
    "snacks": ("grocery", "chips"),
    # Dairy
    "milk": ("grocery", "milk"),
    "dairy": ("grocery", "milk"),
    # Beverages
    "tea": ("grocery", "tea"),
    "coffee": ("grocery", "coffee"),
    "fruit juice": ("grocery", "juice"),
    "beverages": ("grocery", "juice"),
    # Personal care / oral care
    "oral care": ("personal_care", "toothpaste"),
    "personal care": ("personal_care", None),
    # Electronics / audio
    "audio & accessories": ("electronics", "earbuds"),
    "audio accessories": ("electronics", "earbuds"),
    "electronics": ("electronics", None),
}


def _bl_cat_and_group(blinkit_category: str | None) -> tuple[str, str | None]:
    """Resolve internal category and substitute_group from a raw Blinkit category string."""
    key = (blinkit_category or "").strip().lower()
    return _BL_CATEGORY_MAP.get(key, ("grocery", None))


def build_real_catalog(start_index: int = 0) -> list[dict]:
    """Convert real Blinkit products to the internal catalog schema (BL-#### SKUs).

    Category and substitute_group are derived from the scraped Blinkit category field so
    the substitute surface produces cross-brand alternatives within the right product class.
    """
    catalog = []
    for i, r in enumerate(load_blinkit_raw()):
        size, unit = _parse_pack_size(r.get("unit", ""))
        price, mrp = r["price_inr"], r["mrp_inr"]
        in_stock = bool(r.get("in_stock", True))
        category, sub_group = _bl_cat_and_group(r.get("category"))
        catalog.append({
            "sku": f"BL-{1000 + start_index + i}",
            "title": r["product_name"], "category": category,
            "pack_size": size, "unit": unit, "platform": "Blinkit",
            "mrp_inr": mrp, "price_inr": price,
            "stock": 100 if in_stock else 0, "in_stock": in_stock,
            "rating": float(r.get("rating") or 4.2), "review_count": 500,
            "discount_pct": round(100 * (mrp - price) / mrp, 1) if mrp else 0.0,
            "unit_price_inr": _unit_price(price, size, unit),
            "substitute_group": sub_group,
        })
    return catalog


def _reverse_category(sub_group: str | None) -> str:
    """Reverse-map an internal substitute_group to a plausible Blinkit category string."""
    rev = {
        "chocolate": "chocolates", "chips": "chips & crisps", "milk": "milk",
        "tea": "tea", "coffee": "coffee", "juice": "fruit juice",
        "toothpaste": "oral care", "earbuds": "audio & accessories",
        "edible_oil": "grocery", "flagship_phone": "electronics",
        "midrange_phone": "electronics", "tws_earbuds": "audio & accessories",
    }
    return rev.get(sub_group or "", "grocery")


def _persist_blinkit_from_catalog(catalog_entries: list[dict]):
    """Write a blinkit_products.json seed from existing BL-* catalog entries.

    This is a data-provenance safety net: when the original blinkit_products.json
    is missing, we reconstruct it from the current catalog so that build() does
    not accidentally wipe the rich BL-* data on the next run.
    """
    raw = []
    for item in catalog_entries:
        if not item["sku"].startswith("BL-"):
            continue
        unit_str = f"{item['pack_size']} {item['unit']}" if item.get("pack_size") else "1 piece"
        raw.append({
            "product_name": item["title"],
            "category": _reverse_category(item.get("substitute_group")),
            "price_inr": item["price_inr"],
            "mrp_inr": item["mrp_inr"],
            "unit": unit_str,
            "in_stock": item.get("in_stock", True),
            "rating": item.get("rating", 4.2),
        })
    if raw:
        BLINKIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        BLINKIT_PATH.write_text(json.dumps(raw, indent=2, ensure_ascii=False))


def build(seed: int = 13) -> dict:
    """Generate catalog and price datasets and write them to data/. Returns a summary."""
    rng = random.Random(seed)
    for p in (CATALOG_PATH, PRICES_PATH):
        p.parent.mkdir(parents=True, exist_ok=True)

    # --- data provenance guard: if blinkit_products.json is missing but we have  ---
    #     a catalog with BL-* entries, reconstruct the raw seed from them so the    ---
    #     rich real-scraped data survives across rebuilds.                         ---
    if not BLINKIT_PATH.exists() and CATALOG_PATH.exists():
        try:
            existing = json.loads(CATALOG_PATH.read_text())
            bl_items = [x for x in existing if x.get("sku", "").startswith("BL-")]
            if bl_items:
                _persist_blinkit_from_catalog(bl_items)
        except Exception:
            pass

    # --- catalog: real Blinkit products (BL-*) first, then synthetic seed (SF-*) ---
    catalog = build_real_catalog()
    name_to_group = {}
    for grp, names in _SUBSTITUTE_GROUPS.items():
        for nm in names:
            name_to_group[nm] = grp
    for i, (title, cat, size, unit, plat, mrp, price, stock, rating, nrev) in enumerate(_SEED_CATALOG):
        sku = f"SF-{1000 + i}"
        unit_price = _unit_price(price, size, unit)
        catalog.append({
            "sku": sku, "title": title, "category": cat,
            "pack_size": size, "unit": unit, "platform": plat,
            "mrp_inr": mrp, "price_inr": price, "stock": stock,
            "in_stock": stock > 0, "rating": rating, "review_count": nrev,
            "discount_pct": round(100 * (mrp - price) / mrp, 1) if mrp else 0.0,
            "unit_price_inr": unit_price,
            "substitute_group": name_to_group.get(title),
        })
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2, ensure_ascii=False))

    # --- price observations across months/festivals ---
    price_rows = []
    for item in catalog:
        base = item["price_inr"]
        for month in range(1, 13):
            fest = config.festival_for_month(month)
            bias = fest["discount_bias"] if fest else 0.0
            noise = rng.uniform(-0.04, 0.04)
            factor = 1.0 - bias * rng.uniform(0.4, 1.0) + noise
            obs = max(1, _round_paise(base * factor))
            disc = round(100 * (item["mrp_inr"] - obs) / item["mrp_inr"], 1) if item["mrp_inr"] else 0.0
            price_rows.append({
                "sku": item["sku"], "title": item["title"], "platform": item["platform"],
                "month": month, "date": f"2026-{month:02d}-15",
                "price_inr": obs, "mrp_inr": item["mrp_inr"], "discount_pct": disc,
                "festival": fest["name"] if fest else None,
                "festival_key": _festival_key(month),
            })
    _write_jsonl(PRICES_PATH, price_rows)

    return {
        "catalog": {"path": str(CATALOG_PATH), "n": len(catalog)},
        "prices": {"path": str(PRICES_PATH), "n": len(price_rows)},
    }


def _festival_key(month: int) -> str | None:
    for key, f in config.FESTIVAL_CALENDAR.items():
        if f["month"] == month:
            return key
    return None


def _unit_price(price: float, size: float, unit: str) -> float | None:
    """Normalize to a comparable unit price (per kg / per litre / per piece)."""
    if not size or size <= 0:
        return None
    if unit in ("kg", "litre"):
        return round(price / size, 2)
    if unit == "piece":
        return round(price / size, 2)
    return None


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))


# --- loaders ---------------------------------------------------------------------
def _ensure() -> None:
    if not all(p.exists() for p in (CATALOG_PATH, PRICES_PATH)):
        build()


def load_catalog() -> list[dict]:
    _ensure()
    return json.loads(CATALOG_PATH.read_text())


def load_prices() -> list[dict]:
    _ensure()
    return [json.loads(l) for l in PRICES_PATH.read_text().splitlines() if l.strip()]


def find_sku(query: str) -> dict | None:
    """Lightweight title/SKU lookup used by the catalog tool."""
    q = query.strip().lower()
    catalog = load_catalog()
    for item in catalog:
        if item["sku"].lower() == q:
            return item
    best, best_overlap = None, 0
    qtokens = {t for t in q.replace("-", " ").split() if len(t) > 1}
    for item in catalog:
        title_tokens = set(item["title"].lower().split())
        overlap = len(qtokens & title_tokens)
        if overlap > best_overlap:
            best, best_overlap = item, overlap
    return best if best_overlap else None


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
