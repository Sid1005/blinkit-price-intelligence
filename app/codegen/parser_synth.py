"""Generate, sandbox, and benchmark product-listing parsers with Groq (week 4).

The deal/substitute surfaces ingest messy Indian product listings. This module asks
Groq models to synthesize a ``parse(html) -> list[dict]`` adapter for a sample
Blinkit/Amazon-style listing, runs it in a constrained sandbox, scores extraction F1
against a hand-written baseline, and picks the best quality-per-dollar model.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

from bs4 import BeautifulSoup

from app import config
from app.llm import groq_client, router

SAMPLE_HTML = """
<!doctype html>
<html>
  <head><title>Blinkit — Grocery</title></head>
  <body>
    <main id="products">
      <article class="product-card" data-sku="SF-1000">
        <h2 class="product-name">Tata Salt Iodised 1kg</h2>
        <span class="price">\u20b926</span>
        <span class="mrp">\u20b928</span>
      </article>
      <article class="product-card" data-sku="SF-1005">
        <h2 class="product-name">Maggi 2-Minute Noodles</h2>
        <span class="price">\u20b984</span>
        <span class="mrp">\u20b996</span>
      </article>
      <article class="product-card" data-sku="SF-1004">
        <h2 class="product-name">Amul Gold Milk 500ml</h2>
        <span class="price">\u20b934</span>
        <span class="mrp">\u20b935</span>
      </article>
    </main>
  </body>
</html>
""".strip()


def BASELINE_html_to_items(html: str) -> list[dict]:
    """Hand-written parser for the bundled product-listing schema."""
    soup = BeautifulSoup(html, "html.parser")
    items = []
    for card in soup.select(".product-card"):
        name_node = card.select_one(".product-name")
        price_node = card.select_one(".price")
        if not name_node or not price_node:
            continue
        items.append({"name": name_node.get_text(" ", strip=True),
                      "price": price_node.get_text(" ", strip=True)})
    return items


def _fallback_parser_code() -> str:
    return """
def parse(html):
    soup = BeautifulSoup(html, "html.parser")
    items = []
    for card in soup.select(".product-card"):
        name_node = card.select_one(".product-name")
        price_node = card.select_one(".price")
        if name_node and price_node:
            items.append({
                "name": name_node.get_text(" ", strip=True),
                "price": price_node.get_text(" ", strip=True),
            })
    return items
""".strip()


def _extract_code(text: str) -> str:
    text = text.strip()
    if "```" not in text:
        return text
    for part in text.split("```"):
        candidate = part.strip()
        if candidate.startswith("python"):
            candidate = candidate[len("python"):].strip()
        if candidate.startswith("def parse"):
            return candidate
    return text.replace("```python", "").replace("```", "").strip()


def generate_parser(model: str) -> str:
    messages = [
        {"role": "system", "content": "You write concise, deterministic Python parsing "
         "functions. Output only Python code, no markdown fences or explanation."},
        {"role": "user", "content": (
            "Write a Python function named parse(html) that returns a list of dicts with "
            "exactly the keys 'name' and 'price'. The input HTML is a small Indian product "
            "listing like the sample below. Use BeautifulSoup, already available in the "
            "namespace. Do not import modules, read files, make network calls, print, or "
            "include tests. Return [] if no items found.\n\nSample HTML:\n" + SAMPLE_HTML)},
    ]
    try:
        code = groq_client.chat(messages, model=model, temperature=0.1, max_tokens=700)
    except Exception:  # noqa: BLE001
        return _fallback_parser_code()
    cleaned = _extract_code(code)
    return cleaned if "def parse" in cleaned else _fallback_parser_code()


class _Timeout(Exception):
    pass


def run_generated(code: str, html: str, timeout_s: int = 5) -> list[dict]:
    """Execute generated parser code in a constrained namespace with a wall-clock guard.

    Restricted builtins block imports/IO; a SIGALRM timer (POSIX) stops runaway loops.
    """
    import signal

    safe_builtins = {"dict": dict, "enumerate": enumerate, "float": float, "int": int,
                     "isinstance": isinstance, "len": len, "list": list, "max": max,
                     "min": min, "range": range, "set": set, "sorted": sorted,
                     "str": str, "sum": sum, "tuple": tuple, "zip": zip}
    namespace: dict[str, Any] = {"__builtins__": safe_builtins, "BeautifulSoup": BeautifulSoup}

    has_alarm = hasattr(signal, "SIGALRM")
    old_handler = None
    if has_alarm:
        def _handler(signum, frame):  # noqa: ANN001
            raise _Timeout()
        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout_s)
    try:
        exec(code, namespace, namespace)  # noqa: S102 — sandboxed builtins
        parse = namespace.get("parse")
        if not callable(parse):
            return []
        result = parse(html)
    except (Exception, _Timeout):  # noqa: BLE001
        return []
    finally:
        if has_alarm:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
    if not isinstance(result, list):
        return []
    return [item for item in result if isinstance(item, dict)]


def _normalize_item(item: dict) -> tuple[str, str]:
    return (str(item.get("name", "")).strip().casefold(),
            str(item.get("price", "")).strip().casefold())


def score_parser(predicted: list[dict], gold: list[dict]) -> float:
    if not gold:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    pred_counts = Counter(_normalize_item(i) for i in predicted)
    gold_counts = Counter(_normalize_item(i) for i in gold)
    overlap = sum((pred_counts & gold_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return round((2 * precision * recall) / (precision + recall), 4)


def _candidate_models(models: list[str] | None = None) -> list[str]:
    if models:
        return models
    keys = ("fast", "oss_sm", "strong", "oss_lg")
    return list(dict.fromkeys(config.GROQ_MODELS[key] for key in keys))


def benchmark(models: list[str] | None = None) -> dict:
    candidates = _candidate_models(models)
    gold = BASELINE_html_to_items(SAMPLE_HTML)
    qualities: dict[str, float] = {}
    for model in candidates:
        code = generate_parser(model)
        predicted = run_generated(code, SAMPLE_HTML)
        qualities[model] = score_parser(predicted, gold)
    selected = router.select_model(candidates, lambda m: qualities[m])
    results = [{"model": m, "quality": qualities[m],
                "cost": router.COST_WEIGHT.get(m, 5.0),
                "value": round(qualities[m] / router.COST_WEIGHT.get(m, 5.0), 4)}
               for m in candidates]
    return {"baseline_n": len(gold), "results": results, "selected": selected.model}


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
