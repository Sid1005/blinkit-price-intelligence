"""Baselines the fine-tuned LoRA must beat (week 6, model- & data-centric).

Classifier baselines (commerce-signal labels):
  * classify_signal_groq  — Groq few-shot baseline.
  * SklearnBaseline       — TF-IDF + LogisticRegression (classical ML).
  * TorchMLPBaseline      — small PyTorch NN over hashed bag-of-words features.

Price baselines (deal surface):
  * random / constant price, and a rule-based unit-price model.

evaluate() reports accuracy / macro-F1 so the harness can compare members.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from app import commerce_data, config
from app.finetune import dataset
from app.llm import groq_client

_SYS = ("Classify the snippet into exactly one commerce-signal label from this set: "
        + ", ".join(config.SIGNAL_LABELS) + ". Hinglish is expected. Reply with only the label.")


# --- Groq baseline ---------------------------------------------------------------
def classify_signal_groq(text: str, model: str | None = None) -> str:
    out = groq_client.chat(
        [{"role": "system", "content": _SYS}, {"role": "user", "content": text}],
        model=model or config.GROQ_MODELS["fast"], temperature=0.0, max_tokens=12).strip().lower()
    for lab in config.SIGNAL_LABELS:
        if lab in out:
            return lab
    return "noise"


def macro_f1(preds, golds, labels) -> float:
    """Public macro-F1 helper (used by the eval harness too)."""
    f1s = []
    for lab in labels:
        tp = sum(1 for p, g in zip(preds, golds) if p == g == lab)
        fp = sum(1 for p, g in zip(preds, golds) if p == lab and g != lab)
        fn = sum(1 for p, g in zip(preds, golds) if p != lab and g == lab)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    return round(sum(f1s) / len(f1s), 4) if f1s else 0.0


# Backwards-compatible private alias.
_macro_f1 = macro_f1


def evaluate(split: str = "golden", model: str | None = None, limit: int | None = None) -> dict:
    rows = dataset.load(split)
    if limit:
        rows = rows[:limit]
    preds = [classify_signal_groq(r["text"], model=model) for r in rows]
    golds = [r["label"] for r in rows]
    acc = round(sum(int(p == g) for p, g in zip(preds, golds)) / len(rows), 4) if rows else 0.0
    return {"split": split, "n": len(rows), "method": "groq_fewshot",
            "model": model or config.GROQ_MODELS["fast"],
            "accuracy": acc, "macro_f1": _macro_f1(preds, golds, config.SIGNAL_LABELS)}


# --- sklearn baseline ------------------------------------------------------------
class SklearnBaseline:
    """TF-IDF + multinomial LogisticRegression over the signal text."""

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.clf = LogisticRegression(max_iter=500, C=4.0)
        self._fitted = False

    def fit(self, rows: list[dict]) -> "SklearnBaseline":
        X = self.vec.fit_transform([r["text"] for r in rows])
        self.clf.fit(X, [r["label"] for r in rows])
        self._fitted = True
        return self

    def predict(self, texts: list[str]) -> list[str]:
        return list(self.clf.predict(self.vec.transform(texts)))

    def evaluate(self, split: str = "golden") -> dict:
        if not self._fitted:
            self.fit(dataset.load("train"))
        rows = dataset.load(split)
        if not rows:
            return {"method": "sklearn_tfidf_logreg", "split": split, "n": 0,
                    "accuracy": 0.0, "macro_f1": 0.0}
        preds = self.predict([r["text"] for r in rows])
        golds = [r["label"] for r in rows]
        acc = round(sum(int(p == g) for p, g in zip(preds, golds)) / len(rows), 4)
        return {"method": "sklearn_tfidf_logreg", "split": split, "n": len(rows),
                "accuracy": acc, "macro_f1": macro_f1(preds, golds, config.SIGNAL_LABELS)}


# --- PyTorch NN baseline ---------------------------------------------------------
_HASH_DIM = 512


def _hash_features(text: str, dim: int = _HASH_DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in text.lower().split():
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim
        vec[h] += 1.0
    n = np.linalg.norm(vec)
    return vec / n if n else vec


class TorchMLPBaseline:
    """Small PyTorch MLP over hashed bag-of-words features (week 6 NN baseline)."""

    def __init__(self, hidden: int = 128, epochs: int = 40, lr: float = 1e-2, seed: int = 7) -> None:
        self.hidden, self.epochs, self.lr, self.seed = hidden, epochs, lr, seed
        self.labels = config.SIGNAL_LABELS
        self.model = None

    def _build(self):
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(_HASH_DIM, self.hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(self.hidden, len(self.labels)))

    def fit(self, rows: list[dict] | None = None) -> "TorchMLPBaseline":
        import torch
        torch.manual_seed(self.seed)
        rows = rows or dataset.load("train")
        l2i = {l: i for i, l in enumerate(self.labels)}
        X = torch.tensor(np.stack([_hash_features(r["text"]) for r in rows]))
        y = torch.tensor([l2i[r["label"]] for r in rows], dtype=torch.long)
        self.model = self._build()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(self.model(X), y)
            loss.backward()
            opt.step()
        return self

    def predict(self, texts: list[str]) -> list[str]:
        import torch
        if self.model is None:
            self.fit()
        X = torch.tensor(np.stack([_hash_features(t) for t in texts]))
        self.model.eval()
        with torch.no_grad():
            idx = self.model(X).argmax(dim=-1).tolist()
        return [self.labels[i] for i in idx]

    def evaluate(self, split: str = "golden") -> dict:
        if self.model is None:
            self.fit()
        rows = dataset.load(split)
        if not rows:
            return {"method": "pytorch_mlp_hashed", "split": split, "n": 0,
                    "accuracy": 0.0, "macro_f1": 0.0}
        preds = self.predict([r["text"] for r in rows])
        golds = [r["label"] for r in rows]
        acc = round(sum(int(p == g) for p, g in zip(preds, golds)) / len(rows), 4)
        return {"method": "pytorch_mlp_hashed", "split": split, "n": len(rows),
                "accuracy": acc, "macro_f1": macro_f1(preds, golds, config.SIGNAL_LABELS)}


# --- price baselines -------------------------------------------------------------
def price_baselines(seed: int = 7) -> dict:
    """random / constant / rule-based unit-price baselines vs observed prices (MAPE)."""
    rng = np.random.default_rng(seed)
    prices = commerce_data.load_prices()
    catalog = {c["sku"]: c for c in commerce_data.load_catalog()}
    y = np.array([r["price_inr"] for r in prices], dtype=float)

    rand = rng.uniform(0.5, 1.0, size=len(y)) * np.array([catalog[r["sku"]]["mrp_inr"] for r in prices])
    const = np.full_like(y, float(np.mean(y)))
    # rule: MRP * (1 - festival bias)
    rule = []
    for r in prices:
        bias = config.FESTIVAL_CALENDAR.get(r.get("festival_key") or "", {}).get("discount_bias", 0.0)
        rule.append(catalog[r["sku"]]["mrp_inr"] * (1 - bias * 0.8))
    rule = np.array(rule, dtype=float)

    def mape(pred):
        return round(float(np.mean(np.abs(pred - y) / np.maximum(y, 1))), 4)

    return {"random_mape": mape(rand), "constant_mean_mape": mape(const),
            "rule_unit_price_mape": mape(rule), "n": len(y)}


def evaluate_all_classifier_baselines() -> dict:
    return {
        "groq": evaluate("golden", limit=30),
        "sklearn": SklearnBaseline().evaluate("golden"),
        "pytorch_mlp": TorchMLPBaseline().fit().evaluate("golden"),
    }


if __name__ == "__main__":
    print(json.dumps({"classifier": {
        "sklearn": SklearnBaseline().evaluate("golden"),
        "pytorch_mlp": TorchMLPBaseline().fit().evaluate("golden")},
        "price": price_baselines()}, indent=2))
