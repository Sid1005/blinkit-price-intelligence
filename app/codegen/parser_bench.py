"""Offline-safe code-generation and benchmarking (Week 4).

Parser candidates: deterministic product-listing parsers benchmarked against
golden listings. Optional Groq-synthesised parser generation if key present.

Run:  python -m app.codegen.parser_bench
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_LISTINGS = [
    {"raw": "Cadbury Dairy Milk Silk 150g @ INR 180", "expected": {"product": "Cadbury Dairy Milk Silk", "weight": 0.150, "unit": "kg", "price_inr": 180}},
    {"raw": "Amul Gold Milk 500ml Rs 34", "expected": {"product": "Amul Gold Milk", "weight": 0.500, "unit": "litre", "price_inr": 34}},
    {"raw": "Maggi 2-Min Noodles 420g - 96 INR", "expected": {"product": "Maggi 2-Min Noodles", "weight": 0.420, "unit": "kg", "price_inr": 96}},
    {"raw": "boAt Airdopes 161 TWS @ 1199", "expected": {"product": "boAt Airdopes 161 TWS", "weight": 1.0, "unit": "piece", "price_inr": 1199}},
    {"raw": "Tata Salt 1 kg 26 rupees", "expected": {"product": "Tata Salt", "weight": 1.0, "unit": "kg", "price_inr": 26}},
    {"raw": "iPhone 15 128GB - Rs.67999", "expected": {"product": "iPhone 15 128GB", "weight": 1.0, "unit": "piece", "price_inr": 67999}},
    {"raw": "Fortune Sunflower Oil 1L INR 142", "expected": {"product": "Fortune Sunflower Oil", "weight": 1.0, "unit": "litre", "price_inr": 142}},
    {"raw": "Surf Excel Matic 2 litre @ 449", "expected": {"product": "Surf Excel Matic", "weight": 2.0, "unit": "litre", "price_inr": 449}},
    {"raw": "KitKat 4 Finger 38.5g Rs.30", "expected": {"product": "KitKat 4 Finger", "weight": 0.0385, "unit": "kg", "price_inr": 30}},
    {"raw": "Dettol Handwash 1.5L @219", "expected": {"product": "Dettol Handwash", "weight": 1.5, "unit": "litre", "price_inr": 219}},
]


# --- Candidate 1: regex-based deterministic parser ---
def _parser_regex(raw: str) -> dict | None:
    raw_lower = raw.lower()
    price = None
    for pat, mult in [
        (r"@\s*(\d{2,6})\s*$", 1),
        (r"in?r\s*(\d{2,6})", 1),
        (r"rs\.?\s*(\d{2,6})", 1),
        (r"rupees\s*(\d{2,6})", 1),
        (r"@\s*(\d{2,6})(?:\s|$)", 1),
    ]:
        m = re.search(pat, raw_lower)
        if m:
            price = float(m.group(1)) * mult
            break

    weight = 1.0
    unit = "piece"
    for pat in [
        r"(\d+(?:\.\d+)?)\s*([kK][gG])\b",
        r"(\d+(?:\.\d+)?)\s*([gG])\b",
        r"(\d+(?:\.\d+)?)\s*([lL](?:itre)?)\b",
        r"(\d+(?:\.\d+)?)\s*([mM][lL])\b",
    ]:
        m = re.search(pat, raw)
        if m:
            val = float(m.group(1))
            u = m.group(2).lower()
            if u == "g":
                weight, unit = val / 1000, "kg"
            elif u == "kg":
                weight, unit = val, "kg"
            elif u == "ml":
                weight, unit = val / 1000, "litre"
            elif u in ("l", "litre"):
                weight, unit = val, "litre"
            break

    name_pat = re.match(r"^([A-Za-z0-9\s\-]+?)(?:\s+\d|$)", raw)
    product = name_pat.group(1).strip() if name_pat else raw

    if price is None:
        return None
    return {"product": product, "weight": weight, "unit": unit, "price_inr": price}


# --- Candidate 2: Groq-synthesised parser (opt-in, offline-safe fallback) ---
def _parser_optional_groq(raw: str) -> dict | None:
    if os.environ.get("RUN_LIVE_CODEGEN") != "1":
        return _parser_regex(raw)
    try:
        from app import config
        from app.llm import groq_client
        if not config.GROQ_API_KEY:
            return _parser_regex(raw)
        return groq_client.chat_json([
            {"role": "system", "content": "Extract product name, numeric weight in kg/litre/piece, unit (kg/litre/piece), and price_inr. "
             "Convert grams to kg and ml to litre. Output strict JSON."},
            {"role": "user", "content": raw},
        ], model=config.GROQ_MODELS["fast"])
    except Exception:
        return _parser_regex(raw)


# --- Candidate 3: simple heuristic with greedy extraction ---
def _parser_heuristic(raw: str) -> dict | None:
    tokens = raw.replace("@", " ").replace("-", " ").split()
    price = None
    price_idx = None
    for i, tok in enumerate(tokens):
        tok = tok.replace(",", "").replace("Rs.", "").replace("rs.", "").replace("INR", "")
        try:
            val = float(tok)
            if 1 <= val <= 200000:
                price = val
                price_idx = i
                break
        except ValueError:
            continue

    if price is None:
        return None

    relevant = tokens[:price_idx] if price_idx is not None else tokens[:-1]
    numbers_before = []
    weight_g = 1.0
    unit_g = "piece"
    for tok in relevant:
        m = re.match(r"(\d+(?:\.\d+)?)([gGkKmMlL]+)?", tok)
        if m:
            num = float(m.group(1))
            suffix = (m.group(2) or "").lower()
            if suffix == "g":
                weight_g, unit_g = num / 1000, "kg"
            elif suffix == "kg":
                weight_g, unit_g = num, "kg"
            elif suffix == "ml":
                weight_g, unit_g = num / 1000, "litre"
            elif suffix in ("l", "litre"):
                weight_g, unit_g = num, "litre"

    product = " ".join(t for t in relevant if not re.match(r"^\d", t))
    if not product:
        product = raw

    return {"product": product, "weight": weight_g, "unit": unit_g, "price_inr": price}


PARSERS = {
    "regex_deterministic": _parser_regex,
    "heuristic_greedy": _parser_heuristic,
    "optional_groq_or_fallback": _parser_optional_groq,
}


def _score(parsed: dict | None, expected: dict) -> dict:
    if parsed is None:
        return {"price_ok": False, "weight_ok": False, "unit_ok": False, "overall_ok": False}
    price_ok = abs(parsed["price_inr"] - expected["price_inr"]) < 0.5
    weight_ok = abs(parsed.get("weight", 0) - expected.get("weight", 0)) < 0.001
    unit_ok = parsed.get("unit") == expected.get("unit")
    return {
        "price_ok": price_ok,
        "weight_ok": weight_ok,
        "unit_ok": unit_ok,
        "overall_ok": price_ok and weight_ok and unit_ok,
    }


def benchmark() -> dict:
    results = {}
    for parser_name, parser_fn in PARSERS.items():
        correct = 0
        errors = []
        t0 = time.perf_counter()
        for listing in GOLDEN_LISTINGS:
            try:
                parsed = parser_fn(listing["raw"])
                score = _score(parsed, listing["expected"])
                if score["overall_ok"]:
                    correct += 1
                else:
                    errors.append({"raw": listing["raw"][:50], "parsed": parsed, "expected": listing["expected"]})
            except Exception as e:
                errors.append({"raw": listing["raw"][:50], "error": str(e)})
        elapsed = round(time.perf_counter() - t0, 4)
        n = len(GOLDEN_LISTINGS)
        results[parser_name] = {
            "accuracy": round(correct / n, 4) if n else 0,
            "correct": correct,
            "total": n,
            "elapsed_s": elapsed,
            "errors": len(errors),
        }
    return results


if __name__ == "__main__":
    results = benchmark()
    print(json.dumps({"parser_benchmark": results}, indent=2))
