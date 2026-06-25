"""Open-source Hugging Face inference (week 3) — shared Hinglish enrichment.

Covers:
  * sentiment + aspect-based sentiment over Hinglish reviews,
  * zero-shot commerce-signal classification,
  * NER over brands / SKUs,
  * tokenizer usage / token counting for chunk-budget control,
  * optional translation (Hinglish/regional -> English) via an open model.

All pipelines are lazy-loaded so importing the package stays cheap. Models run
locally (open-weights only). Heavy models degrade gracefully if unavailable.
"""
from __future__ import annotations

from functools import lru_cache

from app import config

# Candidate phrases for zero-shot commerce-signal classification.
_SIGNAL_CANDIDATES = {
    "festival discount / sale price cut": "festival_discount",
    "price increase due to demand or low stock": "demand_spike",
    "product quality or authenticity review sentiment": "review_sentiment",
    "customer complaint needing a policy resolution": "complaint_policy",
    "product unavailable or poor value needing a substitute": "catalog_substitution",
    "irrelevant marketing or noise": "noise",
}

_ASPECT_CANDIDATES = {
    "product quality": "quality",
    "delivery speed and packaging": "delivery",
    "product authenticity / genuine vs fake": "authenticity",
    "price and value for money": "value",
}


@lru_cache(maxsize=1)
def _zeroshot():
    from transformers import pipeline
    return pipeline("zero-shot-classification", model=config.HF_ZEROSHOT_MODEL)


@lru_cache(maxsize=1)
def _sentiment():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model=config.HF_SENTIMENT_MODEL)


@lru_cache(maxsize=1)
def _ner():
    from transformers import pipeline
    return pipeline("token-classification", model=config.HF_NER_MODEL,
                    aggregation_strategy="simple")


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config.HF_FINETUNE_BASE)


def count_tokens(text: str) -> int:
    """Token count via an open-source HF tokenizer (chunk-budget control)."""
    return len(_tokenizer().encode(text, add_special_tokens=False))


def classify_signal(text: str) -> dict:
    """Zero-shot classify a snippet into the commerce-signal taxonomy."""
    labels = list(_SIGNAL_CANDIDATES.keys())
    out = _zeroshot()(text, candidate_labels=labels, multi_label=False)
    top_phrase = out["labels"][0]
    return {"label": _SIGNAL_CANDIDATES[top_phrase], "phrase": top_phrase,
            "score": round(float(out["scores"][0]), 4),
            "ranking": [(_SIGNAL_CANDIDATES[l], round(float(s), 4))
                        for l, s in zip(out["labels"], out["scores"])]}


def sentiment(text: str) -> dict:
    """Overall sentiment of a (possibly Hinglish) review."""
    out = _sentiment()(text[:512])[0]
    return {"label": out["label"].lower(), "score": round(float(out["score"]), 4)}


def aspect_sentiment(text: str) -> dict:
    """Aspect-based sentiment: which aspects the review touches + overall polarity.

    Combines zero-shot aspect detection with the sentiment head. Returns a dict of
    aspect -> 'pos'|'neg'|'neutral' for aspects clearly present in the text.
    """
    labels = list(_ASPECT_CANDIDATES.keys())
    z = _zeroshot()(text, candidate_labels=labels, multi_label=True)
    polarity = sentiment(text)
    pol = "pos" if polarity["label"] == "positive" else "neg"
    aspects = {}
    for phrase, score in zip(z["labels"], z["scores"]):
        if score >= 0.5:
            aspects[_ASPECT_CANDIDATES[phrase]] = pol
    return {"overall": polarity["label"], "overall_score": polarity["score"],
            "aspects": aspects}


def extract_entities(text: str) -> list[dict]:
    """NER over brands / products / SKUs (week 3 token classification)."""
    try:
        ents = _ner()(text[:512])
    except Exception:  # noqa: BLE001 — model download/edge failures degrade gracefully
        return []
    return [{"text": e.get("word", ""), "type": e.get("entity_group", ""),
             "score": round(float(e.get("score", 0.0)), 3)} for e in ents]


def translate(text: str, target: str = "English", model: str | None = None) -> str:
    """Translate Hinglish/regional text to a target language.

    Uses Groq (open-weight tier) for robust Hinglish handling, which transformer
    MT models handle poorly. See app/i18n.py for the Gemini live-translate path.
    """
    from app.llm import groq_client
    return groq_client.chat(
        [{"role": "system", "content": f"Translate the user's text to {target}. "
          "Output only the translation, preserving meaning and product/brand names."},
         {"role": "user", "content": text}],
        model=model or config.GROQ_MODELS["oss_sm"], temperature=0.0, max_tokens=300)
