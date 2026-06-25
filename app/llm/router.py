"""Multi-model router + model selection (weeks 2 & 4).

Routes a task to the cheapest Groq model that clears a quality bar, and exposes a
quality-per-dollar model-selection helper used by the code-gen workstream.

Groq-only limitation: cross-vendor (OpenAI/Anthropic/Gemini) tournaments are
structurally represented, but runtime selection uses available Groq / open-weight
tiers because those are the only working runtime keys.
"""
from __future__ import annotations

from dataclasses import dataclass

from app import config
from app.llm import groq_client  # noqa: F401  (re-exported for convenience)

# Rough relative cost weights (lower = cheaper). Used for quality-per-dollar.
COST_WEIGHT = {
    "llama-3.1-8b-instant": 1.0,
    "openai/gpt-oss-20b": 2.0,
    "meta-llama/llama-4-scout-17b-16e-instruct": 3.0,
    "llama-3.3-70b-versatile": 6.0,
    "openai/gpt-oss-120b": 8.0,
}

TASK_ROUTES = {
    "triage":    config.GROQ_MODELS["fast"],
    "classify":  config.GROQ_MODELS["fast"],
    "intent":    config.GROQ_MODELS["fast"],
    "summarize": config.GROQ_MODELS["oss_sm"],
    "synthesis": config.GROQ_MODELS["strong"],
    "pricing":   config.GROQ_MODELS["strong"],
    "substitute": config.GROQ_MODELS["strong"],
    "resolution": config.GROQ_MODELS["strong"],
    "judge":     config.GROQ_MODELS["strong"],
    "codegen":   config.GROQ_MODELS["strong"],
    "vision":    config.GROQ_MODELS["vision"],
}


def route(task: str) -> str:
    return TASK_ROUTES.get(task, config.DEFAULT_MODEL)


@dataclass
class ModelScore:
    model: str
    quality: float          # 0..1 from a scorer
    cost: float
    value: float            # quality / cost


def select_model(candidates: list[str], scorer) -> ModelScore:
    """Pick the best quality-per-dollar model. `scorer(model)->float in 0..1`."""
    if not candidates:
        raise ValueError("select_model requires a non-empty candidate list")
    scored = []
    for m in candidates:
        q = scorer(m)
        c = COST_WEIGHT.get(m, 5.0)
        scored.append(ModelScore(m, q, c, round(q / c, 4)))
    scored.sort(key=lambda s: s.value, reverse=True)
    return scored[0]
