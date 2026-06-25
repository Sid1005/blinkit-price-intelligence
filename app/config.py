"""Central configuration for India Commerce SignalForge.

Loads credentials from the repo-root .env, defines the Groq model registry, the
Indian festival calendar, the commerce label taxonomy, and resolves project paths.

Provider policy:
  * Runtime LLM inference  -> Groq only.
  * Open models / datasets / Hub -> Hugging Face only.
  * Web evidence           -> Tavily.
  * Experiment tracking    -> Weights & Biases (optional; offline JSONL fallback).
  * Live translation       -> Gemini (optional; Groq fallback). See app/i18n.py.
"""
from __future__ import annotations

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
CAPSTONE_DIR = APP_DIR.parent
REPO_ROOT = CAPSTONE_DIR.parent
DATA_DIR = CAPSTONE_DIR / "data"
KB_DIR = APP_DIR / "knowledge_base"
FIXTURES_DIR = CAPSTONE_DIR / "fixtures"
CHROMA_DIR = DATA_DIR / "chroma"
MEMORY_DB = DATA_DIR / "memory.sqlite"
ADAPTER_DIR = DATA_DIR / "lora_adapter"
RUNS_DIR = DATA_DIR / "runs"          # offline experiment-tracking logs

for _d in (DATA_DIR, KB_DIR, FIXTURES_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _load_dotenv() -> None:
    """Minimal .env loader (no extra dependency); does not overwrite real env."""
    for candidate in (REPO_ROOT / ".env", CAPSTONE_DIR / ".env"):
        if not candidate.exists():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, val)


_load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGINGFACE_TOKEN", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")

# --- Groq model registry — tiers used by the multi-model router (weeks 2 & 4). ---
# Names verified against the live Groq API at build time.
GROQ_MODELS = {
    "fast":   "llama-3.1-8b-instant",                       # cheap/fast triage
    "strong": "llama-3.3-70b-versatile",                    # synthesis / judging
    "oss_sm": "openai/gpt-oss-20b",                         # open-weights small
    "oss_lg": "openai/gpt-oss-120b",                        # open-weights large
    "vision": "meta-llama/llama-4-scout-17b-16e-instruct",  # multimodal (week 2)
    "audio":  "whisper-large-v3-turbo",                     # speech-to-text (week 3)
}
DEFAULT_MODEL = GROQ_MODELS["strong"]
JUDGE_MODEL = GROQ_MODELS["strong"]

# --- Hugging Face open-source models (weeks 3 & 7). ---
HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
HF_ZEROSHOT_MODEL = "typeform/distilbert-base-uncased-mnli"
HF_NER_MODEL = "dslim/bert-base-NER"
HF_FINETUNE_BASE = "prajjwal1/bert-tiny"

# --- India commerce taxonomy ----------------------------------------------------
# Intents the router dispatches across the three decision surfaces.
INTENTS = ["deal", "substitute", "triage"]

# Commerce signal labels (the LoRA classifier target, week 7).
SIGNAL_LABELS = [
    "festival_discount",     # genuine festival/sale price cut
    "demand_spike",          # price up / low stock from demand
    "review_sentiment",      # quality/authenticity sentiment signal
    "complaint_policy",      # complaint requiring a policy-grounded resolution
    "catalog_substitution",  # SKU unavailable / poor value -> needs alternative
    "noise",                 # marketing fluff / irrelevant
]

# Complaint categories for the triage surface.
COMPLAINT_TYPES = [
    "cod_dispute",       # cash-on-delivery amount / payment dispute
    "refund_delay",      # refund not received / delayed
    "fake_product",      # counterfactual authenticity claim
    "expiry_issue",      # expired / near-expiry perishable
    "wrong_item",        # wrong / missing item delivered
    "damaged_item",      # physically damaged on delivery
    "other",
]

# Review aspects extracted by the Hinglish NLP enrichment layer (week 3).
REVIEW_ASPECTS = ["quality", "delivery", "authenticity", "value"]

# E-commerce platforms modelled in the curated catalog.
PLATFORMS = ["Blinkit", "Zepto", "Amazon.in", "Flipkart", "BigBasket"]

# --- Indian festival calendar (demo year). Dates are illustrative/curated. ------
# Used for festival-aware deal conditioning and counterfactual evals.
FESTIVAL_CALENDAR = {
    "republic_day":      {"month": 1,  "name": "Republic Day Sale",     "discount_bias": 0.18},
    "holi":              {"month": 3,  "name": "Holi",                  "discount_bias": 0.10},
    "summer_sale":       {"month": 5,  "name": "Summer Sale",          "discount_bias": 0.12},
    "independence_day":  {"month": 8,  "name": "Independence Day Sale", "discount_bias": 0.20},
    "onam":              {"month": 9,  "name": "Onam",                  "discount_bias": 0.14},
    "big_billion_days":  {"month": 10, "name": "Big Billion Days",      "discount_bias": 0.30},
    "great_indian_fest": {"month": 10, "name": "Great Indian Festival", "discount_bias": 0.30},
    "dhanteras":         {"month": 10, "name": "Dhanteras",             "discount_bias": 0.22},
    "diwali":            {"month": 11, "name": "Diwali",                "discount_bias": 0.28},
    "black_friday":      {"month": 11, "name": "Black Friday",          "discount_bias": 0.25},
    "year_end":          {"month": 12, "name": "Year End Sale",         "discount_bias": 0.16},
}

CURRENCY = "INR"
CURRENCY_SYMBOL = "\u20b9"  # ₹


def require_groq() -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY missing — set it in the repo-root .env")
    return GROQ_API_KEY


def festival_for_month(month: int) -> dict | None:
    """Return the highest-impact festival active in a given month, if any."""
    active = [f for f in FESTIVAL_CALENDAR.values() if f["month"] == month]
    if not active:
        return None
    return max(active, key=lambda f: f["discount_bias"])
