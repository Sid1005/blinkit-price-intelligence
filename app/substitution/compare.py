"""Blinkit substitution: RAG-vs-no-RAG comparison arms.

  * ``no_rag_substitute`` — Claude Sonnet 5, zero-shot. No retrieval, no
    catalog access: only the product name and the model's general knowledge.
    This is the baseline the RAG arm is meant to beat.
  * ``rag_substitute`` — Groq, grounded in real retrieved context from the
    dedicated ``blinkit_substitutions`` Chroma index (``app/substitution/
    rag_index.py``), which is built only from the real-scraped-data markdown
    docs in ``data/blinkit/substitutions/``.

Both arms return the same shape so the eval can score them identically:
``{substitute, reasoning, model, input_tokens, output_tokens, cost_usd,
latency_s, sources, error}``. ``sources`` is empty for the no-RAG arm (by
construction — nothing was retrieved) and lists the aisle doc(s) used for the
RAG arm.
"""
from __future__ import annotations

import json
import re
import time

from app import config
from app.llm import groq_client, router
from app.substitution import rag_index

# Public list price per million tokens (input, output), USD. Approximate,
# correct as of 2026-07 — used only for *relative* cost telemetry in the
# substitution_rag_ablation eval, not for billing. Update if rates change.
PRICING_PER_MTOK = {
    "claude-sonnet-5": (3.00, 15.00),
    "llama-3.3-70b-versatile": (0.59, 0.79),
}


def _cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    rate_in, rate_out = PRICING_PER_MTOK.get(model, (0.0, 0.0))
    return round(input_tokens / 1_000_000 * rate_in + output_tokens / 1_000_000 * rate_out, 6)


def _extract_json(text: str) -> dict:
    """Best-effort JSON extraction, mirroring app/finetune/frontier_pricer.py's
    approach — Claude sometimes wraps JSON in prose even when asked not to."""
    obj = re.search(r"\{.*\}", text, re.DOTALL)
    for candidate in ([obj.group()] if obj else []) + [text]:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, TypeError):
            continue
    return {}


NO_RAG_SYSTEM = (
    "You are a shopping assistant. A shopper on Blinkit (an Indian quick-commerce "
    "app) can't get a product and wants one alternative. You have NO access to "
    "Blinkit's live catalog, stock, or prices — answer purely from your general "
    "knowledge of Indian grocery/retail products. Reply with strict JSON only, "
    "nothing else: {\"substitute\": \"<specific product name and brand>\", "
    "\"reasoning\": \"<one sentence, may state a price if you know one>\"}."
)


def no_rag_substitute(product_name: str, model: str | None = None) -> dict:
    """Arm A: Claude Sonnet 5 zero-shot, no retrieval, no catalog access."""
    model = model or config.ANTHROPIC_SUBSTITUTE_MODEL
    result = {"substitute": "", "reasoning": "", "model": model,
              "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
              "latency_s": 0.0, "sources": [], "error": None}
    if not config.ANTHROPIC_API_KEY:
        result["error"] = "ANTHROPIC_API_KEY missing"
        return result
    try:
        import anthropic
    except ImportError:
        result["error"] = "anthropic package not installed"
        return result

    t0 = time.perf_counter()
    try:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=model, max_tokens=300,
            system=NO_RAG_SYSTEM,
            messages=[{"role": "user", "content":
                      f"Product that is unavailable: {product_name}\n"
                      "Suggest one alternative."}])
        result["latency_s"] = round(time.perf_counter() - t0, 3)
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        parsed = _extract_json(text)
        result["substitute"] = str(parsed.get("substitute", ""))[:200]
        result["reasoning"] = str(parsed.get("reasoning", ""))[:400]
        result["input_tokens"] = getattr(resp.usage, "input_tokens", 0)
        result["output_tokens"] = getattr(resp.usage, "output_tokens", 0)
        result["cost_usd"] = _cost_usd(model, result["input_tokens"], result["output_tokens"])
        if not result["substitute"]:
            result["error"] = f"unparseable response: {text[:200]!r}"
    except Exception as exc:  # noqa: BLE001 — arm is best-effort, eval records the failure
        result["latency_s"] = round(time.perf_counter() - t0, 3)
        result["error"] = str(exc)[:300]
    return result


RAG_SYSTEM = (
    "You are Blinkit's substitution assistant. A shopper can't get a product. "
    "You are given real retrieved excerpts from Blinkit's own scraped catalog "
    "listing real alternatives in the same aisle, with real prices and stock "
    "status. Recommend ONE substitute drawn ONLY from the retrieved context — "
    "never invent a product that is not in it. If the context genuinely has no "
    "suitable alternative, set \"substitute\" to \"\". Reply with strict JSON "
    "only, nothing else: {\"substitute\": \"<exact product name from the "
    "context>\", \"reasoning\": \"<one sentence citing price/stock/brand from "
    "the context>\"}."
)


def rag_substitute(product_name: str, k: int = 5, model: str | None = None) -> dict:
    """Arm B: Groq, grounded in the real-data blinkit_substitutions RAG index."""
    model = model or router.route("substitute")
    result = {"substitute": "", "reasoning": "", "model": model,
              "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
              "latency_s": 0.0, "sources": [], "error": None}

    t0 = time.perf_counter()
    try:
        hits = rag_index.retrieve(f"substitute alternative for {product_name}", k=k)
        result["sources"] = sorted({h["source"] for h in hits})
        if not hits:
            result["latency_s"] = round(time.perf_counter() - t0, 3)
            result["error"] = "no retrieval hits"
            return result
        context = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)
        content, usage = groq_client.chat_usage(
            [{"role": "system", "content": RAG_SYSTEM},
             {"role": "user", "content":
              f"Context:\n{context[:6000]}\n\nProduct that is unavailable: "
              f"{product_name}\n\nGrounded substitute recommendation (JSON):"}],
            model=model, temperature=0.1, max_tokens=300, json_mode=True)
        result["latency_s"] = round(time.perf_counter() - t0, 3)
        parsed = _extract_json(content)
        result["substitute"] = str(parsed.get("substitute", ""))[:200]
        result["reasoning"] = str(parsed.get("reasoning", ""))[:400]
        result["input_tokens"] = usage["prompt_tokens"]
        result["output_tokens"] = usage["completion_tokens"]
        result["cost_usd"] = _cost_usd(model, result["input_tokens"], result["output_tokens"])
        if not result["substitute"]:
            result["error"] = f"unparseable response: {content[:200]!r}"
    except Exception as exc:  # noqa: BLE001 — arm is best-effort, eval records the failure
        result["latency_s"] = round(time.perf_counter() - t0, 3)
        result["error"] = str(exc)[:300]
    return result


if __name__ == "__main__":
    for q in ["Haldiram's Fatafat Bhelpuri", "Cadbury Dairy Milk Silk"]:
        print("\n===", q)
        print("no-RAG (Claude):", json.dumps(no_rag_substitute(q), indent=2, ensure_ascii=False))
        print("with-RAG (Groq):", json.dumps(rag_substitute(q), indent=2, ensure_ascii=False))
