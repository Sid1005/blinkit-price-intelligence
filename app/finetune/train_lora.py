"""Fine-tune a tiny BERT commerce-signal classifier with PEFT LoRA (week 7).

Local runnable target: LoRA over prajjwal1/bert-tiny on the Indian commerce signal
labels (festival_discount, demand_spike, review_sentiment, complaint_policy,
catalog_substitution, noise), including Hinglish examples.

Training metrics (train/eval loss, accuracy, macro-F1, LR, wall-clock, adapter path)
are logged through the shared experiment tracker (W&B or offline JSONL).

Run:  python -m app.finetune.train_lora
For the CUDA/Colab 4-bit QLoRA variant, see RUNBOOK.md (Mac/MPS cannot run bnb 4-bit).
"""
from __future__ import annotations

import json
import random
import time
from collections.abc import Iterable

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app import config
from app.finetune import dataset
from app.monitoring import experiment_tracking as tracking

MAX_LENGTH = 64
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
SEED = 7


class SignalDataset(Dataset):
    def __init__(self, encodings: dict[str, torch.Tensor]) -> None:
        self.encodings = encodings

    def __len__(self) -> int:
        return int(self.encodings["labels"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {key: value[idx] for key, value in self.encodings.items()}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _ensure_data() -> None:
    needed = [config.DATA_DIR / f"signals_{s}.jsonl" for s in ("train", "val", "test", "golden")]
    if not all(p.exists() for p in needed):
        dataset.build()


def _encode_rows(rows: list[dict], tokenizer, label2id: dict[str, int]) -> SignalDataset:
    texts = [row["text"] for row in rows]
    labels = [label2id[row["label"]] for row in rows]
    encodings = tokenizer(texts, truncation=True, padding=True,
                          max_length=MAX_LENGTH, return_tensors="pt")
    encodings["labels"] = torch.tensor(labels, dtype=torch.long)
    return SignalDataset(encodings)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _macro_f1(predictions: Iterable[int], labels: Iterable[int], num_labels: int) -> float:
    preds, golds = list(predictions), list(labels)
    scores = []
    for label_id in range(num_labels):
        tp = sum(1 for p, g in zip(preds, golds) if p == g == label_id)
        fp = sum(1 for p, g in zip(preds, golds) if p == label_id and g != label_id)
        fn = sum(1 for p, g in zip(preds, golds) if p != label_id and g == label_id)
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _evaluate(model, loader: DataLoader, device: torch.device, num_labels: int,
              loss_fn) -> tuple[float, float, float]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            out = model(**batch)
            total_loss += float(out.loss)
            n_batches += 1
            pred = out.logits.argmax(dim=-1)
            predictions.extend(pred.detach().cpu().tolist())
            labels.extend(batch["labels"].detach().cpu().tolist())
    accuracy = sum(int(p == g) for p, g in zip(predictions, labels)) / len(labels)
    return accuracy, _macro_f1(predictions, labels, num_labels), total_loss / max(1, n_batches)


def main() -> dict:
    _set_seed(SEED)
    _ensure_data()
    t0 = time.time()

    labels = list(config.SIGNAL_LABELS)
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    run = tracking.start_run(
        name="lora-commerce-signal",
        config_dict={"base_model": config.HF_FINETUNE_BASE, "epochs": EPOCHS,
                     "batch_size": BATCH_SIZE, "lr": LEARNING_RATE, "max_length": MAX_LENGTH,
                     "labels": labels, "method": "lora_peft", "seed": SEED},
        tags=["week7", "lora", "hinglish"])

    tokenizer = AutoTokenizer.from_pretrained(config.HF_FINETUNE_BASE)
    train_rows, test_rows = dataset.load("train"), dataset.load("test")
    train_data = _encode_rows(train_rows, tokenizer, label2id)
    test_data = _encode_rows(test_rows, tokenizer, label2id)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.HF_FINETUNE_BASE, num_labels=len(labels),
        id2label=id2label, label2id=label2id)
    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, target_modules=["query", "value"],
                             r=8, lora_alpha=16, lora_dropout=0.05)
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    device = _device()
    model.to(device)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss, n = 0.0, 0
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            n += 1
        acc, f1, eval_loss = _evaluate(model, test_loader, device, len(labels), loss_fn)
        tracking.log_metrics(run, {"train_loss": round(epoch_loss / max(1, n), 4),
                                   "eval_loss": round(eval_loss, 4),
                                   "eval_accuracy": round(acc, 4),
                                   "eval_macro_f1": round(f1, 4),
                                   "learning_rate": LEARNING_RATE}, step=epoch)

    accuracy, macro_f1, eval_loss = _evaluate(model, test_loader, device, len(labels), loss_fn)
    config.ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(config.ADAPTER_DIR))
    tokenizer.save_pretrained(str(config.ADAPTER_DIR))
    wall = round(time.time() - t0, 1)

    tracking.log_artifact(run, str(config.ADAPTER_DIR), type="model", name="lora-adapter")
    summary = {"test_accuracy": round(accuracy, 4), "macro_f1": round(macro_f1, 4),
               "eval_loss": round(eval_loss, 4), "wall_clock_s": wall,
               "trainable_params": trainable, "adapter_path": str(config.ADAPTER_DIR)}
    tracking.finish_run(run, summary=summary)

    result = {"adapter_dir": str(config.ADAPTER_DIR), "epochs": EPOCHS,
              "tracking_backend": run.backend, "run_id": run.run_id, **summary}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    main()
