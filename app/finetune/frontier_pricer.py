"""Claude frontier pricing arm (Arm 3 of the comparison).

A pure zero-shot estimate from a large model with no Blinkit training data and no RAG
context. This is the baseline the domain-adapted arms (RF, LoRA, Groq+RAG) are meant to
beat. Returns ``None`` when ``ANTHROPIC_API_KEY`` is unset or the call fails, so the
comparison degrades gracefully to the remaining arms.
"""
from __future__ import annotations

import json
import re

from app import config


def estimate(product_name: str, festival: str | None = None) -> float | None:
    """Zero-shot fair price in INR from Claude, or None if unavailable."""
    if not config.ANTHROPIC_API_KEY:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    when = f" during the {festival} sale" if festival and festival != "No Festival" else ""
    prompt = (f"Give your single best estimate of the fair retail price in Indian Rupees "
              f"for '{product_name}' on Blinkit{when}. Even without live data, estimate a "
              f"specific number from your general knowledge. Reply with strict JSON only "
              f"and nothing else: {{\"price_inr\": <number>}}.")
    try:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=config.ANTHROPIC_PRICE_MODEL, max_tokens=128,
            messages=[{"role": "user", "content": prompt}])
        text = "".join(block.text for block in resp.content
                       if getattr(block, "type", None) == "text")
        return _parse_price(text)
    except Exception:  # noqa: BLE001 — frontier arm is best-effort
        return None


def _parse_price(text: str) -> float | None:
    # Prefer a clean JSON object, including one wrapped in prose/markdown.
    obj = re.search(r"\{[^{}]*\}", text)
    for candidate in ([obj.group()] if obj else []) + [text]:
        try:
            return float(json.loads(candidate)["price_inr"])
        except (ValueError, KeyError, TypeError):
            continue
    # Targeted fallback: a number attached to a price_inr key or a ₹/Rs marker only.
    match = re.search(r'(?:price_inr"?\s*[:=]\s*|₹|rs\.?\s*)(\d+(?:\.\d+)?)', text, re.I)
    return float(match.group(1)) if match else None


if __name__ == "__main__":
    print(estimate("Cadbury Dairy Milk Silk Milk Chocolate Bar", "Diwali"))
