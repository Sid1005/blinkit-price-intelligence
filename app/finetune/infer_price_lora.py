"""Inference for the LoRA price-regression adapter (Arm 2 of the comparison).

The adapter is trained on Colab (see lora.ipynb): a regression head over a base BERT
that maps ``"<product>, <festival>, <platform>"`` to a price in INR. Targets are scaled
by ``price_scale`` at train time, so predictions are multiplied back here.

If the adapter has not been trained/downloaded yet, ``predict`` returns ``None`` and the
comparison surface simply shows the arm as unavailable.
"""
from __future__ import annotations

import json
from functools import lru_cache

from app import config

META_PATH = config.PRICE_ADAPTER_DIR / "price_lora_meta.json"


def adapter_exists() -> bool:
    return (config.PRICE_ADAPTER_DIR / "adapter_config.json").exists()


def _meta() -> dict:
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {"price_scale": 10000.0, "base_model": "distilbert-base-uncased"}


@lru_cache(maxsize=1)
def _model_and_tok():
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    meta = _meta()
    base = AutoModelForSequenceClassification.from_pretrained(
        meta["base_model"], num_labels=1, problem_type="regression")
    model = PeftModel.from_pretrained(base, str(config.PRICE_ADAPTER_DIR))
    model.eval()
    try:
        tok = AutoTokenizer.from_pretrained(str(config.PRICE_ADAPTER_DIR))
    except Exception:  # noqa: BLE001 — adapter export may omit tokenizer files
        tok = AutoTokenizer.from_pretrained(meta["base_model"])
    return model, tok, float(meta.get("price_scale", 10000.0))


def predict(product_name: str, festival: str = "No Festival",
            platform: str = "Blinkit") -> float | None:
    """Predict a price in INR, or None if the adapter is unavailable / errors."""
    if not adapter_exists():
        return None
    try:
        import torch
        model, tok, price_scale = _model_and_tok()
        text = f"{product_name}, {festival}, {platform}"
        enc = tok(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            logit = float(model(**enc).logits[0, 0])
        return round(abs(logit) * price_scale, 2)
    except Exception:  # noqa: BLE001 — inference must never crash the comparison
        return None


if __name__ == "__main__":
    if adapter_exists():
        for p in ["Cadbury Dairy Milk Silk Milk Chocolate Bar", "boAt Airdopes 161 TWS"]:
            print(p, "->", predict(p, "Diwali"))
    else:
        print("No price adapter yet — train lora.ipynb on Colab and copy to "
              f"{config.PRICE_ADAPTER_DIR}")
