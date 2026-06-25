"""Multimodal generation + audio (weeks 2 & 3).

Three capabilities over a Brief:
  * generate_deal_card  — a festival deal-card image. Tries HF Inference text-to-image
    (with HF_TOKEN); always falls back to a dependency-free SVG card so an artifact is
    produced even offline.
  * tts_brief           — a spoken summary. Tries HF Inference text-to-speech; otherwise
    writes the narration text + a runbook note (external provider step).
  * transcribe_voice    — voice complaint / shopping voice-note -> text via Groq Whisper.

Every generation returns an artifact dict: {kind, path, provider, status, ...}.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from xml.sax.saxutils import escape as _xml_escape

from app import config
from app.llm import groq_client

MEDIA_DIR = config.DATA_DIR / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

HF_IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
HF_TTS_MODEL = "espnet/kan-bayashi_ljspeech_vits"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def deal_card_caption(brief: dict) -> str:
    """Compose the marketing caption for the deal card from a deal Brief."""
    d = brief.get("decision") or {}
    sym = config.CURRENCY_SYMBOL
    title = d.get("title", brief.get("query", "Product"))
    point = d.get("point_inr", 0)
    rec = (d.get("recommendation") or "").replace("_", " ").upper()
    fest = d.get("festival_context") or ""
    return f"{title} — {sym}{point:g} | {rec}{(' | ' + fest) if fest else ''}"


def _svg_card(brief: dict) -> str:
    d = brief.get("decision") or {}
    sym = config.CURRENCY_SYMBOL
    title = _xml_escape((d.get("title") or brief.get("query", "Product"))[:40])
    point = d.get("point_inr", 0)
    low, high = d.get("low_inr", 0), d.get("high_inr", 0)
    rec = _xml_escape((d.get("recommendation") or "wait").replace("_", " ").upper())
    fest = _xml_escape(d.get("festival_context") or "Festival deal")
    color = {"BUY NOW": "#16a34a", "WAIT": "#d97706", "AVOID": "#dc2626"}.get(rec, "#2563eb")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="800" height="418" viewBox="0 0 800 418">
  <defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#0f172a"/><stop offset="1" stop-color="#1e293b"/></linearGradient></defs>
  <rect width="800" height="418" fill="url(#bg)"/>
  <text x="40" y="70" fill="#f8fafc" font-family="Inter,Arial" font-size="26" font-weight="700">India Commerce SignalForge</text>
  <text x="40" y="100" fill="#94a3b8" font-family="Inter,Arial" font-size="16">{fest}</text>
  <text x="40" y="190" fill="#f1f5f9" font-family="Inter,Arial" font-size="34" font-weight="700">{title}</text>
  <text x="40" y="260" fill="#e2e8f0" font-family="Inter,Arial" font-size="48" font-weight="800">{sym}{point:g}</text>
  <text x="40" y="300" fill="#94a3b8" font-family="Inter,Arial" font-size="18">band {sym}{low:g} – {sym}{high:g}</text>
  <rect x="40" y="330" rx="10" width="220" height="56" fill="{color}"/>
  <text x="150" y="367" fill="#ffffff" font-family="Inter,Arial" font-size="24" font-weight="700" text-anchor="middle">{rec}</text>
  <text x="760" y="400" fill="#475569" font-family="Inter,Arial" font-size="12" text-anchor="end">demo data — not a live quote</text>
</svg>"""


def generate_deal_card(brief: dict, use_hf: bool = True) -> dict:
    """Produce a festival deal-card. HF text-to-image if available, else SVG fallback."""
    caption = deal_card_caption(brief)
    stamp = _ts()
    if use_hf and config.HF_TOKEN:
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=config.HF_TOKEN)
            prompt = (f"Vibrant Indian festival e-commerce sale banner, diyas and marigold, "
                      f"product hero shot, bold price tag, clean modern layout. {caption}")
            image = client.text_to_image(prompt, model=HF_IMAGE_MODEL)
            path = MEDIA_DIR / f"deal_card_{stamp}.png"
            image.save(path)
            return {"kind": "deal_card", "path": str(path), "provider": "hf_inference",
                    "model": HF_IMAGE_MODEL, "status": "ok", "caption": caption}
        except Exception as e:  # noqa: BLE001 — provider/model may be gated or rate-limited
            fallback_note = str(e)[:200]
        else:
            fallback_note = ""
    else:
        fallback_note = "HF token unavailable or use_hf=False"
    path = MEDIA_DIR / f"deal_card_{stamp}.svg"
    path.write_text(_svg_card(brief))
    return {"kind": "deal_card", "path": str(path), "provider": "svg_fallback",
            "status": "fallback", "caption": caption, "note": fallback_note,
            "runbook": "For raster images set HF_TOKEN and ensure the HF Inference "
                       "text-to-image model is available, or wire an external image API."}


def _deterministic_narration(brief: dict) -> str:
    d = brief.get("decision") or {}
    kind = d.get("kind")
    sym = config.CURRENCY_SYMBOL
    if kind == "price_forecast":
        return (f"{d.get('title') or brief.get('query', 'This product')}: our recommendation is "
                f"{(d.get('recommendation') or '').replace('_', ' ')}, around {sym}{d.get('point_inr', 0)}.")
    if kind == "substitution_set":
        n = len(d.get("candidates", []))
        return f"Found {n} substitute options for {d.get('original_title') or brief.get('query', 'this item')}."
    if kind == "resolution":
        return (f"This looks like a {str(d.get('complaint_type', '')).replace('_', ' ')} complaint. "
                "We have drafted policy-grounded next steps.")
    return "Here is your India Commerce SignalForge summary."


def narration_text(brief: dict) -> str:
    """LLM-written spoken summary script for a Brief; deterministic fallback if Groq fails."""
    try:
        return groq_client.chat(
            [{"role": "system", "content": "Write a 2-3 sentence spoken summary of this shopping "
              "decision for text-to-speech. Natural, friendly, INR amounts spoken in full. No markdown."},
             {"role": "user", "content": json.dumps(brief.get("decision") or brief, ensure_ascii=False)[:1200]}],
            model=config.GROQ_MODELS["fast"], max_tokens=160, temperature=0.4)
    except Exception:  # noqa: BLE001
        return _deterministic_narration(brief)


def tts_brief(brief: dict, use_hf: bool = True) -> dict:
    """Spoken summary of a Brief. HF TTS if available, else narration text + runbook."""
    text = narration_text(brief)
    stamp = _ts()
    if use_hf and config.HF_TOKEN:
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=config.HF_TOKEN)
            audio = client.text_to_speech(text, model=HF_TTS_MODEL)
            path = MEDIA_DIR / f"brief_tts_{stamp}.flac"
            with open(path, "wb") as f:
                f.write(audio)
            return {"kind": "tts", "path": str(path), "provider": "hf_inference",
                    "model": HF_TTS_MODEL, "status": "ok", "text": text}
        except Exception as e:  # noqa: BLE001
            note = str(e)[:200]
    else:
        note = "HF token unavailable or use_hf=False"
    path = MEDIA_DIR / f"brief_tts_{stamp}.txt"
    path.write_text(text)
    return {"kind": "tts", "path": str(path), "provider": "text_fallback",
            "status": "fallback", "text": text, "note": note,
            "runbook": "For audio, enable an HF Inference TTS model (set HF_TOKEN) or wire "
                       "an external TTS provider (e.g. ElevenLabs/Google) per RUNBOOK.md."}


def transcribe_voice(audio_path: str, language: str | None = None) -> dict:
    """Transcribe a voice complaint / shopping voice-note via Groq Whisper (week 3)."""
    try:
        text = groq_client.transcribe(audio_path, language=language,
                                      prompt="Indian shopping or complaint voice note, Hinglish expected.")
        return {"kind": "transcript", "provider": "groq_whisper", "status": "ok", "text": text}
    except Exception as e:  # noqa: BLE001
        return {"kind": "transcript", "provider": "groq_whisper", "status": "error",
                "error": str(e)[:200],
                "runbook": "Provide a valid audio file (wav/mp3/m4a). Groq Whisper handles "
                           "the transcription; see RUNBOOK.md for supported formats."}


if __name__ == "__main__":
    from app.agents import ensemble
    brief = ensemble.run("Is iPhone 15 a good deal right now?", verify=False, persist=False)
    print(json.dumps(generate_deal_card(brief, use_hf=False), indent=2))
