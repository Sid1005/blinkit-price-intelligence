"""Intent router (week 8) — dispatch a query to one of three decision surfaces.

deal       -> "should I buy this now / is this a good price / festival deal?"
substitute -> "what's a cheaper/in-stock/better alternative to X?"
triage     -> "complaint: refund/COD/fake/expiry/wrong/damaged item".

Uses a fast Groq classifier with a deterministic keyword heuristic as a cheap,
offline-friendly fallback and tie-breaker.
"""
from __future__ import annotations

import re

from app import config
from app.llm import groq_client, router

_DEAL_KW = re.compile(r"\b(price|deal|buy|cheap|discount|sale|festival|diwali|"
                      r"worth it|mrp|drop|wait|offer|kitne|kitna|sasta|daam)\b", re.I)
_SUB_KW = re.compile(r"\b(substitut|alternativ|instead|replace|out of stock|"
                     r"unavailable|similar|other option|swap|badal|dusra|vikalp)\b", re.I)
_TRIAGE_KW = re.compile(r"\b(complaint|refund|return|cod|fake|duplicate|expir|"
                        r"wrong item|damaged|broken|missing|charged|paisa wapas|"
                        r"shikayat|toot|kharab)\b", re.I)


def heuristic_intent(query: str) -> str:
    scores = {
        "triage": len(_TRIAGE_KW.findall(query)) * 1.2,   # complaints win ties
        "substitute": len(_SUB_KW.findall(query)) * 1.1,
        "deal": len(_DEAL_KW.findall(query)) * 1.0,
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "deal"


def route_intent(query: str, model: str | None = None) -> dict:
    """Return {'intent': ..., 'method': 'llm'|'heuristic', 'confidence': float}."""
    heur = heuristic_intent(query)
    try:
        data = groq_client.chat_json(
            [{"role": "system", "content":
              "Classify the Indian shopper's message into one intent. Output JSON "
              '{"intent": "deal"|"substitute"|"triage", "confidence": 0..1}. '
              "deal = pricing/buy-timing/festival question. "
              "substitute = wants an alternative product. "
              "triage = a complaint (refund/COD/fake/expiry/wrong/damaged)."},
             {"role": "user", "content": query}],
            model=model or router.route("intent"))
        intent = data.get("intent")
        if intent in config.INTENTS:
            conf = float(data.get("confidence", 0.6) or 0.6)
            return {"intent": intent, "method": "llm",
                    "confidence": max(0.0, min(1.0, conf)), "heuristic": heur}
    except Exception:  # noqa: BLE001 — fall back to heuristic
        pass
    return {"intent": heur, "method": "heuristic", "confidence": 0.5, "heuristic": heur}
