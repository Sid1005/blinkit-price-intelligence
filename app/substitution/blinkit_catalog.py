"""Shared real-Blinkit-catalog helpers: category canonicalization, unit-price
normalisation, and substitute scoring.

Single source of truth for both ``scripts/build_blinkit_substitutions.py``
(the markdown grounding-doc generator) and ``app/substitution/evaluate.py``
(the RAG-vs-no-RAG eval, which needs the exact same in-aisle ground-truth
pools those docs were built from). Extracted rather than duplicated so a
future change to the bucket mapping or scoring weights can't silently drift
between the two.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

BLINKIT_JSON = Path("data/blinkit/blinkit_products.json")

_WEIGHT_UNIT_TO_KG = {
    "kg": 1.0, "g": 0.001, "gm": 0.001, "gms": 0.001, "gram": 0.001, "grams": 0.001,
    "mg": 0.000001, "l": 1.0, "litre": 1.0, "litres": 1.0, "ltr": 1.0,
    "ml": 0.001, "millilitre": 0.001, "millilitres": 0.001,
}
_WEIGHT_PATTERN = re.compile(r"([\d.]+)\s*(kg|gms?|grams?|g|mg|l|litres?|ltr|ml|millilitres?)", re.IGNORECASE)


def weight_kg(unit: str | None) -> float:
    """Rough kg-equivalent for a Blinkit unit string, e.g. '58 g' -> 0.058. 0.0 if unrecognized (piece-priced)."""
    if not unit:
        return 0.0
    match = _WEIGHT_PATTERN.search(unit)
    if not match:
        return 0.0
    try:
        amount = float(match.group(1))
    except ValueError:
        return 0.0
    return round(amount * _WEIGHT_UNIT_TO_KG.get(match.group(2).lower(), 0.0), 6)


# --- canonical category buckets ---------------------------------------------------
# Maps every raw (lowercased) scraped category label to a human-readable bucket
# name. Only near-duplicate wordings of the *same* aisle are merged; distinct
# aisles are kept distinct even when thematically adjacent.
CATEGORY_MERGE: dict[str, str] = {
    "vegetables & fruits": "Vegetables & Fruits",
    "atta, rice & dal": "Atta, Rice & Dal",
    "rice & rice products": "Atta, Rice & Dal",
    "toor, urad & chana": "Atta, Rice & Dal",
    "oil & ghee": "Oil, Ghee & Masala",
    "oil, ghee & masala": "Oil, Ghee & Masala",
    "powdered spice": "Oil, Ghee & Masala",
    "salt, sugar & jaggery": "Oil, Ghee & Masala",
    "paneer and curd": "Paneer & Curd",
    "dairy, bread & eggs": "Dairy, Bread & Eggs",
    "bread & pav": "Bread & Pav",
    "butter and cheese": "Cheese & Butter",
    "cheese & butter": "Cheese & Butter",
    "bakery & biscuits": "Bakery & Biscuits",
    "cream biscuits": "Bakery & Biscuits",
    "glucose & marie": "Bakery & Biscuits",
    "cakes & others": "Cakes & Pastries",
    "cakes & rolls": "Cakes & Pastries",
    "cakes and pastries": "Cakes & Pastries",
    "chips & namkeen": "Chips & Namkeen",
    "namkeen snacks": "Chips & Namkeen",
    "namkeen and snacks": "Chips & Namkeen",
    "dry fruits": "Dry Fruits, Nuts & Seeds",
    "dry fruits snacks": "Dry Fruits, Nuts & Seeds",
    "dry fruits, nuts & seeds": "Dry Fruits, Nuts & Seeds",
    "chocolates": "Chocolates",
    "wafer chocolates": "Chocolates",
    "candies & gum": "Candies & Gum",
    "indian sweets": "Indian Sweets & Mithai",
    "sweets and mithai": "Indian Sweets & Mithai",
    "ice cream and desserts": "Ice Cream & Desserts",
    "cooking sauces": "Cooking Sauces & Ketchup",
    "cooking sauces and ketchup": "Cooking Sauces & Ketchup",
    "sauces & spreads": "Sauces & Spreads",
    "indian chutney & pickle": "Sauces & Spreads",
    "ginger garlic paste": "Sauces & Spreads",
    "tomato puree": "Sauces & Spreads",
    "schezwan chutney": "Sauces & Spreads",
    "dark soy sauce": "Sauces & Spreads",
    "pizza and pasta sauces": "Sauces & Spreads",
    "imported noodles & pasta": "Noodles & Pasta",
    "noodles": "Noodles & Pasta",
    "pasta": "Noodles & Pasta",
    "instant & frozen food": "Frozen & Instant Food",
    "frozen non-veg snacks": "Frozen & Instant Food",
    "frozen veg": "Frozen & Instant Food",
    "frozen veg snacks": "Frozen & Instant Food",
    "ready to eat": "Frozen & Instant Food",
    "gourmet and world food": "Gourmet & World Food",
    "gourmet bakery": "Gourmet & World Food",
    "organic and healthy food": "Gourmet & World Food",
    "tea, coffee & milk drinks": "Tea, Coffee & Milk Drinks",
    "tea and coffee": "Tea, Coffee & Milk Drinks",
    "milk drinks": "Tea, Coffee & Milk Drinks",
    "lassi & milkshakes": "Tea, Coffee & Milk Drinks",
    "cold drinks & juices": "Cold Drinks, Juices & Energy Drinks",
    "fruit juices and energy drinks": "Cold Drinks, Juices & Energy Drinks",
    "energy drinks": "Cold Drinks, Juices & Energy Drinks",
    "soft drinks": "Cold Drinks, Juices & Energy Drinks",
    "lemoneez syrup": "Cold Drinks, Juices & Energy Drinks",
    "chicken": "Chicken, Meat & Seafood",
    "chicken, meat & fish": "Chicken, Meat & Seafood",
    "fish & seafood": "Chicken, Meat & Seafood",
    "hair care": "Hair Care",
    "shampoo": "Hair Care",
    "men's grooming": "Men's Grooming",
    "men's grooming products": "Men's Grooming",
    "perfumes and deodorants": "Perfumes & Deodorants",
    "body & skin care": "Body & Skin Care",
    "foundation": "Cosmetics & Lip Care",
    "lip cosmetics": "Cosmetics & Lip Care",
    "lip scrubs, masks & serums": "Cosmetics & Lip Care",
    "oral care": "Oral Care",
    "feminine care": "Feminine Hygiene",
    "feminine hygiene": "Feminine Hygiene",
    "tampons & menstrual cups": "Feminine Hygiene",
    "bandaid & wound care": "Health & Hygiene Devices",
    "health & hygiene": "Health & Hygiene Devices",
    "health devices and monitors": "Health & Hygiene Devices",
    "pharma and wellness products": "Pharma, Wellness & Ayurveda",
    "ayurveda and herbal products": "Pharma, Wellness & Ayurveda",
    "chyawanprash": "Pharma, Wellness & Ayurveda",
    "baby food": "Baby Food & Diapers",
    "baby food and diapers": "Baby Food & Diapers",
    "baby care": "Baby Care",
    "baby skin & hair care": "Baby Skin & Hair Care",
    "detergent powder & bars": "Detergents & Dishwash",
    "detergents and dishwash": "Detergents & Dishwash",
    "dishwashing accessories": "Dishwashing Accessories",
    "air fresheners": "Air & Car Fresheners",
    "car fresheners": "Air & Car Fresheners",
    "fresheners": "Air & Car Fresheners",
    "disposable and party supplies": "Disposable & Party Supplies",
    "pooja needs": "Pooja Needs",
    "paan corner": "Paan Corner",
    "pet care": "Pet Care",
    "audio & accessories": "Audio Devices & Accessories",
    "audio accessories": "Audio Devices & Accessories",
    "audio devices earbuds": "Audio Devices & Accessories",
    "electronic accessories": "Electronic Accessories",
    "home and kitchen appliances": "Home & Kitchen Appliances",
    "home appliances": "Home & Kitchen Appliances",
    "bottles & flasks": "Bottles & Flasks",
    "toys & games": "Toys & Games",
    "beauty e-card": "Gift Cards",
    "bath & beauty gifts": "Gift Cards",
}

# Raw categories intentionally left unmapped (each has exactly one scraped SKU
# and therefore no in-category substitute) fall through to "singleton" via
# `bucket_for()` below rather than being force-merged into an unrelated aisle.


def bucket_for(raw_category: str | None) -> str:
    key = (raw_category or "uncategorized").strip().lower()
    return CATEGORY_MERGE.get(key, key.title())


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "uncategorized"


def unit_price_basis(row: dict) -> tuple[float, str]:
    """(comparable_price, basis_label) — per-kg/litre if a weight/volume unit is
    recognized, else the raw piece price ('each')."""
    w = weight_kg(row.get("unit"))
    price = row.get("price_inr") or 0.0
    if w > 0:
        return round(price / w, 2), "per kg/L"
    return round(price, 2), "each"


def score_candidate(original: dict, cand: dict) -> float:
    """Value/availability/brand-diversity blend — same spirit as
    app/agents/tools.py::_substitute_score, adapted for real Blinkit fields
    (no substitute_group, ratings are universally null in this scrape).

    Value is only computed from the per-kg/L unit price when *both* items
    have a recognized weight/volume unit — comparing a per-kg price against
    a piece ('each') price (e.g. a loose-weight snack vs. a boxed combo)
    isn't apples-to-apples, so that case falls back to the raw sticker price
    with a damped weight instead of a misleading percentage.
    """
    o_up, o_basis = unit_price_basis(original)
    c_up, c_basis = unit_price_basis(cand)
    if o_basis == c_basis:
        value = 1.0 if c_up <= o_up else max(0.0, 1.0 - (c_up - o_up) / max(o_up, 1))
    else:
        o_price, c_price = original.get("price_inr") or 0.0, cand.get("price_inr") or 0.0
        value = 0.5 if c_price <= o_price else max(0.0, 0.5 - (c_price - o_price) / max(o_price, 1))
    availability = 1.0 if cand.get("in_stock", True) else 0.0
    brand_diversity = 1.0 if cand.get("brand") and cand.get("brand") != original.get("brand") else 0.5
    discount_bonus = min(1.0, (cand.get("discount_percent") or 0) / 30.0)
    return round(0.45 * value + 0.30 * availability + 0.15 * brand_diversity + 0.10 * discount_bonus, 4)


def rank_substitutes(item: dict, pool: list[dict], k: int = 3) -> list[tuple[dict, float]]:
    ranked = [(c, score_candidate(item, c)) for c in pool if c is not item]
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked[:k]


def load_products(path: Path | None = None) -> list[dict]:
    return json.loads((path or BLINKIT_JSON).read_text(encoding="utf-8"))


def bucket_map(rows: list[dict] | None = None) -> dict[str, list[dict]]:
    """Canonical bucket -> rows, for every row (including singleton buckets)."""
    rows = rows if rows is not None else load_products()
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_bucket[bucket_for(r.get("category"))].append(r)
    return dict(by_bucket)
