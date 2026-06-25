"""Translation layer — Gemini live translate with a Groq fallback.

Indian commerce text is heavily Hinglish + regional. This module translates between
Hinglish/regional languages and English so reviews, complaints, and voice notes are
normalised before NLP/agent processing, and decisions can be read back in the user's
language.

Provider order:
  1. Gemini (GEMINI_API_KEY) via the Generative Language REST API — preferred.
  2. Groq (open-weight tier) — robust Hinglish fallback, always available here.

The Gemini integration is finalised/verified in the dedicated build phase; the Groq
fallback guarantees translation works today.
"""
from __future__ import annotations

import json
from typing import Iterator

import requests

from app import config

_GEMINI_MODELS = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash"]
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# Common Indian languages offered in the UI translate surface.
LANGUAGES = ["English", "Hindi", "Hinglish", "Tamil", "Telugu", "Bengali",
             "Marathi", "Kannada", "Gujarati", "Malayalam", "Punjabi"]


def _gemini_translate(text: str, target: str, source: str | None) -> str | None:
    if not config.GEMINI_API_KEY:
        return None
    src = f" from {source}" if source else ""
    prompt = (f"Translate the following text{src} to {target}. Preserve product and brand "
              f"names and any \u20b9 INR amounts. Output only the translation:\n\n{text}")
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json", "x-goog-api-key": config.GEMINI_API_KEY}
    for model in _GEMINI_MODELS:
        url = f"{_GEMINI_BASE}/{model}:generateContent"
        try:
            r = requests.post(url, headers=headers, data=json.dumps(body), timeout=20)
            if r.status_code != 200:
                continue
            data = r.json()
            parts = data["candidates"][0]["content"]["parts"]
            out = "".join(p.get("text", "") for p in parts).strip()
            if out:
                return out
        except Exception:  # noqa: BLE001 — try next model / fall back
            continue
    return None


def _groq_translate(text: str, target: str, source: str | None) -> str:
    from app.llm import groq_client
    src = f" from {source}" if source else ""
    try:
        return groq_client.chat(
            [{"role": "system", "content": f"Translate the user's text{src} to {target}. "
              "Output only the translation, preserving product/brand names and \u20b9 amounts."},
             {"role": "user", "content": text}],
            model=config.GROQ_MODELS["oss_sm"], temperature=0.0, max_tokens=400)
    except Exception:  # noqa: BLE001 — both providers down; return original text
        return text


def translate(text: str, target: str = "English", source: str | None = None) -> dict:
    """Translate text, preferring Gemini, falling back to Groq."""
    if not text or not text.strip():
        return {"text": "", "provider": "none", "target": target}
    out = _gemini_translate(text, target, source)
    if out:
        return {"text": out, "provider": "gemini", "target": target}
    return {"text": _groq_translate(text, target, source), "provider": "groq", "target": target}


def _gemini_translate_stream(text: str, target: str, source: str | None) -> Iterator[str]:
    """Stream a Gemini translation token-by-token (live translate). Yields nothing on failure."""
    if not config.GEMINI_API_KEY:
        return
    src = f" from {source}" if source else ""
    prompt = (f"Translate the following text{src} to {target}. Preserve product/brand names "
              f"and \u20b9 INR amounts. Output only the translation:\n\n{text}")
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json", "x-goog-api-key": config.GEMINI_API_KEY}
    for model in _GEMINI_MODELS:
        url = f"{_GEMINI_BASE}/{model}:streamGenerateContent?alt=sse"
        try:
            with requests.post(url, headers=headers, data=json.dumps(body),
                               timeout=30, stream=True) as r:
                if r.status_code != 200:
                    continue
                got = False
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[len("data:"):].strip()
                    if chunk in ("", "[DONE]"):
                        continue
                    try:
                        data = json.loads(chunk)
                        parts = data["candidates"][0]["content"]["parts"]
                        piece = "".join(p.get("text", "") for p in parts)
                        if piece:
                            got = True
                            yield piece
                    except Exception:  # noqa: BLE001 — skip malformed SSE frames
                        continue
                if got:
                    return
        except Exception:  # noqa: BLE001 — try next model
            continue


def translate_stream(text: str, target: str = "English", source: str | None = None) -> Iterator[str]:
    """Live streaming translation: Gemini SSE first, else a single Groq chunk."""
    if not text or not text.strip():
        return
    streamed = False
    for piece in _gemini_translate_stream(text, target, source):
        streamed = True
        yield piece
    if not streamed:
        yield _groq_translate(text, target, source)


def to_english(text: str) -> str:
    return translate(text, target="English")["text"]


def normalize_to_english(text: str) -> dict:
    """Translate a review/complaint to English for downstream NLP, keeping the original."""
    res = translate(text, target="English")
    return {"original": text, "english": res["text"], "provider": res["provider"]}


def gemini_available() -> bool:
    """Cheap health check for the Gemini key (used by the UI/dashboard)."""
    return bool(_gemini_translate("namaste", "English", "Hindi"))


if __name__ == "__main__":
    for t in ["Delivery boy ne 100 rupaye extra liye", "Quality bahut achhi hai, paisa vasool"]:
        print(t, "->", json.dumps(translate(t), ensure_ascii=False))
