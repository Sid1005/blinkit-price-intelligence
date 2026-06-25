"""Curated Indian commerce world-data (week 6 data curation).

Generates and loads the four datasets the engine reasons over:
  * catalog   — SKUs with pack size, unit, platform, MRP, price, stock, rating.
  * prices    — price observations across festivals (for deal forecasting).
  * reviews   — Hinglish reviews with aspect labels (quality/delivery/auth/value).
  * complaints — labelled complaint text for the triage surface.

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
REVIEWS_PATH = config.DATA_DIR / "reviews" / "reviews.jsonl"
COMPLAINTS_PATH = config.DATA_DIR / "complaints" / "complaints.jsonl"

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

# Hinglish review templates per aspect-sentiment, used to expand the review set.
_REVIEW_TEMPLATES = [
    ("Delivery time pe aa gaya, bahut accha service tha.", {"delivery": "pos"}),
    ("Product theek hai par delivery 3 din late thi, bekar.", {"delivery": "neg", "quality": "neutral"}),
    ("Quality bahut achhi hai, paisa vasool product.", {"quality": "pos", "value": "pos"}),
    ("Ekdum ghatiya quality, paise barbaad ho gaye.", {"quality": "neg", "value": "neg"}),
    ("Lagta hai duplicate piece bheja hai, original nahi hai.", {"authenticity": "neg"}),
    ("100% genuine product mila, packaging sealed thi.", {"authenticity": "pos"}),
    ("Itne price me itna achha product, value for money.", {"value": "pos"}),
    ("MRP se zyada charge kiya, bilkul value nahi.", {"value": "neg"}),
    ("Bahut achhi packaging, time se delivery, quality top.", {"delivery": "pos", "quality": "pos"}),
    ("Fake product hai ye, market me sasta milta hai.", {"authenticity": "neg", "value": "neg"}),
    ("Average product hai, kuch khaas nahi.", {"quality": "neutral"}),
    ("Delivery boy ne bahut der lagayi par product sahi tha.", {"delivery": "neg", "quality": "pos"}),
]

# Complaint templates per type (mix of clean English + Hinglish), for triage labels.
_COMPLAINT_TEMPLATES = {
    "cod_dispute": [
        "Delivery boy ne {amt} rupaye extra liye COD pe, jabki app pe {price} dikha raha tha.",
        "I paid {amt} cash on delivery but the invoice says {price}, please refund the difference.",
    ],
    "refund_delay": [
        "Maine 10 din pehle return kiya tha par abhi tak refund nahi aaya.",
        "Refund of {price} initiated last week is still not credited to my account.",
    ],
    "fake_product": [
        "Ye {item} duplicate lag raha hai, original seal nahi tha.",
        "The {item} I received looks counterfeit, the logo and box are different.",
    ],
    "expiry_issue": [
        "{item} ki expiry date nikal chuki hai, kaise use karu?",
        "Received {item} that expires in 2 days, this is near-expiry stock.",
    ],
    "wrong_item": [
        "Maine {item} order kiya tha par kuch aur hi aa gaya.",
        "Wrong item delivered — I ordered {item} but got a different product.",
    ],
    "damaged_item": [
        "{item} delivery me toot gaya, box bhi damaged tha.",
        "The {item} arrived physically damaged with a cracked screen.",
    ],
}


def _round_paise(x: float) -> int:
    return int(round(x))


def build(seed: int = 13) -> dict:
    """Generate all four datasets and write them to data/. Returns a summary."""
    rng = random.Random(seed)
    for p in (CATALOG_PATH, PRICES_PATH, REVIEWS_PATH, COMPLAINTS_PATH):
        p.parent.mkdir(parents=True, exist_ok=True)

    # --- catalog ---
    catalog = []
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
            # festival months bias prices down; demand months add noise
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

    # --- Hinglish reviews with aspect labels ---
    review_rows = []
    rid = 0
    for item in catalog:
        n = rng.randint(3, 6)
        for _ in range(n):
            text, aspects = rng.choice(_REVIEW_TEMPLATES)
            rid += 1
            review_rows.append({
                "review_id": f"R{rid:04d}", "sku": item["sku"], "title": item["title"],
                "text": text, "aspects": aspects,
                "lang": "hinglish",
            })
    _write_jsonl(REVIEWS_PATH, review_rows)

    # --- complaints ---
    complaint_rows = []
    cid = 0
    items_for_complaints = [c["title"] for c in catalog]
    for ctype, templates in _COMPLAINT_TEMPLATES.items():
        for _ in range(8):  # 8 per type -> balanced
            tmpl = rng.choice(templates)
            item = rng.choice(items_for_complaints)
            price = rng.choice([26, 142, 449, 2799, 21999, 67999])
            amt = price + rng.choice([20, 50, 100])
            cid += 1
            complaint_rows.append({
                "complaint_id": f"C{cid:04d}",
                "text": tmpl.format(item=item, price=price, amt=amt),
                "complaint_type": ctype,
                "sku_title": item,
            })
    rng.shuffle(complaint_rows)
    _write_jsonl(COMPLAINTS_PATH, complaint_rows)

    return {
        "catalog": {"path": str(CATALOG_PATH), "n": len(catalog)},
        "prices": {"path": str(PRICES_PATH), "n": len(price_rows)},
        "reviews": {"path": str(REVIEWS_PATH), "n": len(review_rows)},
        "complaints": {"path": str(COMPLAINTS_PATH), "n": len(complaint_rows)},
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
    if not all(p.exists() for p in (CATALOG_PATH, PRICES_PATH, REVIEWS_PATH, COMPLAINTS_PATH)):
        build()


def load_catalog() -> list[dict]:
    _ensure()
    return json.loads(CATALOG_PATH.read_text())


def load_prices() -> list[dict]:
    _ensure()
    return [json.loads(l) for l in PRICES_PATH.read_text().splitlines() if l.strip()]


def load_reviews() -> list[dict]:
    _ensure()
    return [json.loads(l) for l in REVIEWS_PATH.read_text().splitlines() if l.strip()]


def load_complaints() -> list[dict]:
    _ensure()
    return [json.loads(l) for l in COMPLAINTS_PATH.read_text().splitlines() if l.strip()]


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
