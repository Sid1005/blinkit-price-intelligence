"""Intent router (week 8) — dispatch a query to one of two decision surfaces.

predictor  -> "what will this cost / is this a good price / festival pricing?"
substitute -> "what's a cheaper/in-stock/better alternative to X?"

Uses a fast Groq classifier with a deterministic keyword heuristic as a cheap,
offline-friendly fallback and tie-breaker.
"""
from __future__ import annotations

import re

from app import config
from app.llm import groq_client, router

_PREDICTOR_KW = re.compile(r"\b(price|deal|buy|cheap|discount|sale|festival|diwali|"
                           r"worth it|mrp|drop|wait|offer|kitne|kitna|sasta|daam|predict)\b", re.I)
_SUB_KW = re.compile(r"\b(substitut|alternativ|instead|replace|out of stock|"
                     r"unavailable|similar|other option|swap|badal|dusra|vikalp)\b", re.I)


def heuristic_intent(query: str) -> str:
    scores = {
        "substitute": len(_SUB_KW.findall(query)) * 1.1,
        "predictor": len(_PREDICTOR_KW.findall(query)) * 1.0,
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "predictor"


def route_intent(query: str, model: str | None = None) -> dict:
    """Return {'intent': ..., 'method': 'llm'|'heuristic', 'confidence': float}."""
    try:
        data = groq_client.chat_json(
            [{"role": "system", "content":
              "Classify the Indian shopper's message into one intent. Output JSON "
              '{"intent": "predictor"|"substitute", "confidence": 0..1}. '
              "predictor = price prediction / festival pricing / is-this-a-good-price. "
              "substitute = wants an alternative product. "
              "Input may be in English or Hinglish."},
             {"role": "user", "content": query}],
            model=model or router.route("intent"))
        intent = data.get("intent")
        if intent in config.INTENTS:
            conf = float(data.get("confidence", 0.6) or 0.6)
            return {"intent": intent, "method": "llm",
                    "confidence": max(0.0, min(1.0, conf))}
    except Exception:  # noqa: BLE001 — LLM unreachable, fall back to keyword heuristic
        pass
    heur = heuristic_intent(query)
    return {"intent": heur, "method": "heuristic", "confidence": 0.5}
