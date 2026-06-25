"""Gemini Live API (realtime, bidirectional) translation — best-effort.

This uses the *real* Live API (``client.aio.live.connect`` -> bidiGenerateContent over
WebSockets) via the ``google-genai`` SDK, for low-latency realtime translation of text
and audio (voice complaints / shopping voice-notes).

Access note: the Live API requires a Google project with Live models enabled. The
``GEMINI_API_KEY`` in this environment works for the REST ``generateContent`` path
(see app/i18n.py) but, at build time, the Live (bidiGenerateContent) models returned
"not found / not supported" for this key. So every entry point here health-checks the
Live API first and transparently falls back to the streaming REST translator. Once the
user enables Live API access (RUNBOOK §14), the realtime path activates with no code
change.
"""
from __future__ import annotations

import asyncio
import threading

from app import config, i18n

# Candidate Live models, in preference order (availability varies by project).
LIVE_MODELS = [
    "gemini-2.0-flash-live-001",
    "gemini-2.5-flash-live-preview",
    "gemini-live-2.5-flash-preview",
]
_TIMEOUT_S = 20

# Cache only a confirmed-working model (never cache a transient failure as permanent).
_working_model: str | None = None


def _client(api_version: str = "v1beta"):
    from google import genai
    return genai.Client(api_key=config.GEMINI_API_KEY,
                        http_options={"api_version": api_version})


async def _live_text(model: str, prompt: str) -> str | None:
    """Open a Live session, send one text turn, collect the streamed text reply."""
    client = _client()
    async with client.aio.live.connect(model=model, config={"response_modalities": ["TEXT"]}) as session:
        await session.send_client_content(
            turns={"role": "user", "parts": [{"text": prompt}]}, turn_complete=True)
        out = ""
        async for resp in session.receive():
            if getattr(resp, "text", None):
                out += resp.text
        return out.strip() or None


def _run(coro):
    """Run a coroutine to completion in a dedicated worker thread + event loop.

    A fresh loop in its own thread keeps this safe whether or not the caller (e.g. a
    Gradio handler) is itself running inside an event loop.
    """
    result: dict = {}

    def _worker():
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result["value"] = loop.run_until_complete(asyncio.wait_for(coro, timeout=_TIMEOUT_S))
        except Exception:  # noqa: BLE001 — any Live API/network failure -> caller falls back
            result["value"] = None
        finally:
            loop.close()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=_TIMEOUT_S + 5)
    return result.get("value")


def _probe_model() -> str | None:
    """Return the first Live model that accepts a bidi session, caching success only."""
    global _working_model
    if _working_model:
        return _working_model
    if not config.GEMINI_API_KEY:
        return None
    for model in LIVE_MODELS:
        if _run(_live_text(model, "Reply with: OK")):
            _working_model = model
            return model
    return None


def live_available() -> bool:
    """True if a Live model accepts a bidi session with this key. Caches only success."""
    return _probe_model() is not None


def live_translate(text: str, target: str = "English", source: str | None = None) -> dict:
    """Realtime translate via the Live API; fall back to streaming REST (Gemini/Groq)."""
    if not text or not text.strip():
        return {"text": "", "provider": "none", "target": target}
    model = _probe_model()  # skips Live entirely (single probe) when unavailable
    if model:
        src = f" from {source}" if source else ""
        prompt = (f"Translate the following text{src} to {target}. Preserve product/brand "
                  f"names and \u20b9 INR amounts. Output only the translation:\n\n{text}")
        out = _run(_live_text(model, prompt))
        if out:
            return {"text": out, "provider": f"gemini_live:{model}", "target": target}
    # Fallback: the working streaming REST translator (Gemini generateContent / Groq).
    res = i18n.translate(text, target=target, source=source)
    res["provider"] = f"fallback:{res['provider']}"
    return res


if __name__ == "__main__":
    print("live_available:", live_available())
    print(live_translate("Delivery boy ne 100 rupaye extra liye", target="English"))
