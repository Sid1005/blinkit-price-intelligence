"""Groq chat wrapper — completion, streaming, JSON-mode, vision, and audio.

Week 1: frontier chat + streaming + system/user prompts + markdown output.
Week 2: multimodal vision hook (product screenshots / damaged-item photos).
Week 3: audio transcription (voice complaints / shopping voice-notes) via Whisper.
Week 8: JSON-mode structured outputs feed the agent layer.
"""
from __future__ import annotations

import base64
import json
import mimetypes
from typing import Iterator

from groq import Groq

from app import config

_client: Groq | None = None


def client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=config.require_groq())
    return _client


def chat(messages: list[dict], model: str | None = None, temperature: float = 0.3,
         max_tokens: int = 1024, json_mode: bool = False, reasoning_effort: str | None = None) -> str:
    """Single-shot completion. Returns the assistant text."""
    kwargs = dict(model=model or config.DEFAULT_MODEL, messages=messages,
                  temperature=temperature, max_tokens=max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    resp = client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def chat_usage(messages: list[dict], model: str | None = None, temperature: float = 0.3,
               max_tokens: int = 1024, json_mode: bool = False) -> tuple[str, dict]:
    """Completion that also returns token usage (for cost/eval telemetry)."""
    kwargs = dict(model=model or config.DEFAULT_MODEL, messages=messages,
                  temperature=temperature, max_tokens=max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client().chat.completions.create(**kwargs)
    u = resp.usage
    return resp.choices[0].message.content or "", {
        "prompt_tokens": getattr(u, "prompt_tokens", 0),
        "completion_tokens": getattr(u, "completion_tokens", 0),
        "total_tokens": getattr(u, "total_tokens", 0),
    }


def stream(messages: list[dict], model: str | None = None,
           temperature: float = 0.3, max_tokens: int = 1024) -> Iterator[str]:
    """Yield token deltas (week 1 streaming / week 2 streaming chat)."""
    s = client().chat.completions.create(
        model=model or config.DEFAULT_MODEL, messages=messages,
        temperature=temperature, max_tokens=max_tokens, stream=True)
    for chunk in s:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def chat_json(messages: list[dict], model: str | None = None,
              temperature: float = 0.0) -> dict:
    """JSON-mode completion parsed to a dict (best-effort)."""
    raw = chat(messages, model=model, temperature=temperature, json_mode=True)
    parsed = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            try:
                parsed = json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                parsed = None
    # Callers use dict .get(); a bare list/str/number would crash them.
    if isinstance(parsed, dict):
        return parsed
    return {"_raw": raw, "_parse_error": True, "_value": parsed}


def _data_url(image_path: str) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def vision(prompt: str, image_path: str | None = None, image_url: str | None = None,
           model: str | None = None, max_tokens: int = 700) -> str:
    """Multimodal analysis of an image (week 2). Accepts a local path or URL.

    Used by the Substitute/Triage surfaces to read product screenshots, receipts,
    or damaged-item photos.
    """
    if image_path:
        url = _data_url(image_path)
    elif image_url:
        url = image_url
    else:
        raise ValueError("provide image_path or image_url")
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": url}},
    ]}]
    return chat(messages, model=model or config.GROQ_MODELS["vision"], max_tokens=max_tokens)


def transcribe(audio_path: str, model: str | None = None, language: str | None = None,
               prompt: str | None = None) -> str:
    """Transcribe a voice complaint / shopping voice-note with Groq Whisper (week 3).

    Adapts the course meeting-minutes concept to commerce. Returns the transcript text.
    """
    with open(audio_path, "rb") as f:
        resp = client().audio.transcriptions.create(
            file=(audio_path, f.read()),
            model=model or config.GROQ_MODELS["audio"],
            language=language, prompt=prompt or "",
            response_format="text")
    return resp if isinstance(resp, str) else getattr(resp, "text", str(resp))
