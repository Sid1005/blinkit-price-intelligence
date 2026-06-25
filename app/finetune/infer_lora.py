"""Load the LoRA adapter and run inference (week 7) + HF Hub publish helpers.

The adapter is produced by train_lora.py (PEFT LoRA over prajjwal1/bert-tiny). This
module loads base+adapter, classifies commerce-signal snippets, and offers Hub-publish
functions for the adapter and the curated dataset (real code; the push is also
documented in RUNBOOK.md and uses the valid HF token).
"""
from __future__ import annotations

import json
from functools import lru_cache

from app import config


def _labels() -> list[str]:
    fp = config.DATA_DIR / "labels.json"
    return json.loads(fp.read_text()) if fp.exists() else config.SIGNAL_LABELS


@lru_cache(maxsize=1)
def _model_and_tok():
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    labels = _labels()
    base = AutoModelForSequenceClassification.from_pretrained(
        config.HF_FINETUNE_BASE, num_labels=len(labels))
    model = PeftModel.from_pretrained(base, str(config.ADAPTER_DIR))
    model.eval()
    tok = AutoTokenizer.from_pretrained(str(config.ADAPTER_DIR))
    return model, tok, labels


def classify(text: str) -> dict:
    import torch
    model, tok, labels = _model_and_tok()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        logits = model(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1).tolist()
    idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    return {"label": labels[idx], "score": round(probs[idx], 4)}


def adapter_exists() -> bool:
    return (config.ADAPTER_DIR / "adapter_config.json").exists()


def publish_adapter_to_hub(repo_id: str) -> str:
    """Push the LoRA adapter to the HF Hub (requires HF_TOKEN)."""
    from huggingface_hub import HfApi
    if not config.HF_TOKEN:
        raise RuntimeError("HF_TOKEN missing")
    api = HfApi(token=config.HF_TOKEN)
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(folder_path=str(config.ADAPTER_DIR), repo_id=repo_id)
    return f"https://huggingface.co/{repo_id}"


def publish_dataset_to_hub(repo_id: str) -> str:
    """Push the curated signal dataset splits to the HF Hub as a dataset repo."""
    from huggingface_hub import HfApi
    if not config.HF_TOKEN:
        raise RuntimeError("HF_TOKEN missing")
    api = HfApi(token=config.HF_TOKEN)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    for split in ("train", "val", "test", "golden"):
        fp = config.DATA_DIR / f"signals_{split}.jsonl"
        if fp.exists():
            api.upload_file(path_or_fileobj=str(fp), path_in_repo=fp.name,
                            repo_id=repo_id, repo_type="dataset")
    return f"https://huggingface.co/datasets/{repo_id}"


if __name__ == "__main__":
    if adapter_exists():
        for t in ["Diwali sale me iPhone sabse sasta", "refund nahi aaya 10 din se",
                  "ye product fake lag raha hai"]:
            print(t, "->", classify(t))
    else:
        print("No adapter yet — run: python -m app.finetune.train_lora")
