"""Scrub/filter rules, generalized from llm_eng week6/pricer/parser.py so both
Amazon raw-jsonl rows and Blinkit scraped rows can pass through the same gate.

Note: the raw ``McAuley-Lab/Amazon-Reviews-2023`` jsonl files (read directly,
bypassing the now-unsupported trust_remote_code loading script) store
``description``/``features``/``details`` as Python-literal repr strings
(single-quoted), not JSON — so we use ``ast.literal_eval`` rather than
``json.loads`` to parse them.
"""
from __future__ import annotations

import ast
import json
import re

from app.pricer.items import Item

MIN_CHARS = 600
MIN_PRICE = 0.5
MAX_PRICE = 999.49
MAX_TEXT_EACH = 3000
MAX_TEXT_TOTAL = 4000

# Blinkit rows carry no description/features/details — just a short set of
# structured fields — so the length gate and price bounds are tuned in INR
# rather than reused from the Amazon (USD) thresholds above.
BLINKIT_MIN_CHARS = 15
BLINKIT_MIN_PRICE = 5.0
BLINKIT_MAX_PRICE = 20_000.0

_WEIGHT_UNIT_TO_KG = {
    "kg": 1.0, "g": 0.001, "gm": 0.001, "gms": 0.001, "gram": 0.001, "grams": 0.001,
    "mg": 0.000001, "l": 1.0, "litre": 1.0, "litres": 1.0, "ltr": 1.0,
    "ml": 0.001, "millilitre": 0.001, "millilitres": 0.001,
}
_WEIGHT_PATTERN = re.compile(r"([\d.]+)\s*(kg|gms?|grams?|g|mg|l|litres?|ltr|ml|millilitres?)", re.IGNORECASE)

REMOVALS = [
    "Part Number",
    "Best Sellers Rank",
    "Batteries Included?",
    "Batteries Required?",
    "Item model number",
]

_PART_NUMBER_PATTERN = re.compile(r"\b(?=[A-Z0-9]{7,}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+\b")


def simplify(text_list) -> str:
    """Collapse whitespace and cap length."""
    return (
        str(text_list)
        .replace("\n", " ")
        .replace("\r", "")
        .replace("\t", "")
        .replace("  ", " ")
        .strip()[:MAX_TEXT_EACH]
    )


def scrub(title: str, description, features, details: dict) -> str:
    """Cleansed full string with product/part numbers and boilerplate removed."""
    details = dict(details or {})
    for remove in REMOVALS:
        details.pop(remove, None)
    result = title + "\n"
    if description:
        result += simplify(description) + "\n"
    if features:
        result += simplify(features) + "\n"
    if details:
        result += json.dumps(details) + "\n"
    return _PART_NUMBER_PATTERN.sub("", result).strip()[:MAX_TEXT_TOTAL]


def get_weight(details: dict) -> float:
    weight_str = details.get("Item Weight")
    if not weight_str:
        return 0.0
    try:
        parts = str(weight_str).split(" ")
        amount = float(parts[0])
        unit = parts[1].lower()
    except (IndexError, ValueError):
        return 0.0
    if unit == "pounds":
        return amount
    if unit == "ounces":
        return amount / 16
    if unit == "grams":
        return amount / 453.592
    if unit == "milligrams":
        return amount / 453592
    if unit == "kilograms":
        return amount / 0.453592
    if unit == "hundredths" and len(parts) > 2 and parts[2].lower() == "pounds":
        return amount / 100
    return 0.0


def _literal_or_default(value, default):
    """Parse a Python-repr string field (or pass through an already-parsed value)."""
    if value is None:
        return default
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return default


def parse_amazon_row(row: dict, category: str) -> Item | None:
    """Parse one raw Amazon-Reviews-2023 meta jsonl row into an Item, or None."""
    try:
        price = float(row["price"])
    except (ValueError, TypeError):
        return None
    if not (MIN_PRICE <= price <= MAX_PRICE):
        return None

    title = row["title"]
    description = _literal_or_default(row.get("description"), [])
    features = _literal_or_default(row.get("features"), [])
    details = _literal_or_default(row.get("details"), {})
    weight = get_weight(details)
    full = scrub(title, description, features, details)
    if len(full) < MIN_CHARS:
        return None
    return Item(title=title, category=category, price=price, currency="$",
                full=full, weight=weight)


# Backwards-compatible alias matching the course's function name.
parse = parse_amazon_row


def get_weight_from_unit(unit: str | None) -> float:
    """Extract a rough kg-equivalent weight/volume from a Blinkit unit string, e.g. '58 g' -> 0.058."""
    if not unit:
        return 0.0
    match = _WEIGHT_PATTERN.search(unit)
    if not match:
        return 0.0
    try:
        amount = float(match.group(1))
    except ValueError:
        return 0.0
    factor = _WEIGHT_UNIT_TO_KG.get(match.group(2).lower(), 0.0)
    return round(amount * factor, 6)


def scrub_blinkit(product_name: str, brand: str | None, category: str | None,
                   unit: str | None, mrp_inr, discount_percent, in_stock, rating) -> str:
    """Build a compact 'full' text from Blinkit's structured scrape fields."""
    lines = [product_name]
    if brand:
        lines.append(f"Brand: {brand}")
    if category:
        lines.append(f"Category: {category}")
    if unit:
        lines.append(f"Unit: {unit}")
    if mrp_inr:
        lines.append(f"MRP: ₹{mrp_inr}")
    if discount_percent:
        lines.append(f"Discount: {discount_percent}%")
    lines.append(f"Stock: {'in stock' if in_stock else 'out of stock'}")
    if rating:
        lines.append(f"Rating: {rating}/5")
    return "\n".join(lines)[:MAX_TEXT_TOTAL]


def parse_blinkit_row(row: dict) -> Item | None:
    """Parse one scraped Blinkit product dict into an Item, or None."""
    try:
        price = float(row["price_inr"])
    except (KeyError, ValueError, TypeError):
        return None
    if not (BLINKIT_MIN_PRICE <= price <= BLINKIT_MAX_PRICE):
        return None

    title = (row.get("product_name") or "").strip()
    if not title:
        return None

    category = (row.get("category") or "grocery").strip()
    unit = row.get("unit")
    full = scrub_blinkit(title, row.get("brand"), category, unit,
                        row.get("mrp_inr"), row.get("discount_percent"),
                        row.get("in_stock", True), row.get("rating"))
    if len(full) < BLINKIT_MIN_CHARS:
        return None

    weight = get_weight_from_unit(unit)
    return Item(title=title, category=category, price=price, currency="₹",
                full=full, weight=weight)
