"""Dataset curation + splits for the commerce-signal classifier (week 6).

Builds a labelled snippet dataset (text -> commerce signal label) mixing English and
Hinglish phrasings drawn from real Indian shopping/complaint language, then writes
train/validation/test/golden JSONL splits with dedup + leakage checks.

Labels (see config.SIGNAL_LABELS):
  festival_discount, demand_spike, review_sentiment,
  complaint_policy, catalog_substitution, noise
"""
from __future__ import annotations

import hashlib
import json
import random

from app import config

LABELS = config.SIGNAL_LABELS

PRODUCTS = ["iPhone 15", "Redmi Note 13", "Tata Salt", "Aashirvaad Atta", "boAt Airdopes",
            "Fortune Oil", "Amul Milk", "Samsung S24", "Mi Band", "Surf Excel",
            "Nivea lotion", "Maggi", "OnePlus Nord", "Sony headphones", "Colgate"]

TEMPLATES = {
    "festival_discount": [
        "{p} Big Billion Days me {pct}% off, abhi sabse sasta hai.",
        "Diwali sale: {p} now \u20b9{n}, down from \u20b9{m}.",
        "Great Indian Festival pe {p} ka damaka offer, {pct}% discount.",
        "{p} dhamaka deal this festive season, save \u20b9{d}.",
        "Republic Day sale me {p} ekdum sasta, \u20b9{n} only.",
    ],
    "demand_spike": [
        "{p} ka price badh gaya hai, demand zyada hai aur stock kam.",
        "Due to high demand {p} now costs \u20b9{m}, up from \u20b9{n}.",
        "{p} out of stock everywhere, price \u20b9{m} pe pahunch gaya.",
        "Limited stock: {p} prices rising fast this week.",
        "{p} mehnga ho gaya, sab jagah short supply hai.",
    ],
    "complaint_policy": [
        "Maine {p} return kiya par refund nahi aaya 10 din se.",
        "Delivery boy ne {p} pe COD me extra paise liye.",
        "{p} expired aaya, replacement chahiye.",
        "Wrong item delivered, ordered {p} got something else.",
        "{p} damaged condition me mila, box bhi toota tha.",
    ],
    "catalog_substitution": [
        "{p} out of stock hai, koi alternative batao.",
        "Is there a cheaper substitute for {p}?",
        "{p} ke jaisa dusra product chahiye, ye mehnga hai.",
        "{p} unavailable, similar option suggest karo.",
        "Need a replacement for {p}, poor value lagta hai.",
    ],
    "noise": [
        "Festive season ki shubhkamnayein from our store!",
        "Read our blog about smart shopping tips.",
        "Download the app for a better experience.",
        "Humari team se milne mall me aaiye.",
        "Happy Diwali to all our customers!",
    ],
}


def _fill(t: str, rng: random.Random) -> str:
    return t.format(p=rng.choice(PRODUCTS),
                    n=rng.choice([26, 84, 142, 449, 1199, 21999, 67999]),
                    m=rng.choice([28, 96, 165, 530, 2990, 25999, 79900]),
                    d=rng.choice([5, 20, 50, 200, 2000, 8000]),
                    pct=rng.choice([10, 15, 20, 25, 30]))


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def build(n_per_label: int = 70, seed: int = 7) -> dict:
    rng = random.Random(seed)
    seen: set[str] = set()
    rows = []
    for label, templates in TEMPLATES.items():
        made = 0
        attempts = 0
        while made < n_per_label and attempts < n_per_label * 20:
            attempts += 1
            text = _fill(rng.choice(templates), rng)
            h = _hash(text + label)
            if h in seen:  # dedup
                continue
            seen.add(h)
            rows.append({"text": text, "label": label})
            made += 1
    rng.shuffle(rows)
    n = len(rows)
    train = rows[: int(n * 0.7)]
    val = rows[int(n * 0.7): int(n * 0.8)]
    test = rows[int(n * 0.8): int(n * 0.9)]
    golden = rows[int(n * 0.9):]

    # leakage check across splits (no identical text in two splits)
    _leakage_check(train, val, test, golden)

    out = {}
    for name, split in (("train", train), ("val", val), ("test", test), ("golden", golden)):
        fp = config.DATA_DIR / f"signals_{name}.jsonl"
        fp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in split))
        out[name] = {"path": str(fp), "n": len(split)}
    (config.DATA_DIR / "labels.json").write_text(json.dumps(LABELS))
    out["n_total"] = n
    out["duplicates_removed"] = True
    return out


def _leakage_check(*splits) -> None:
    texts_per_split = [{r["text"] for r in s} for s in splits]
    for i in range(len(texts_per_split)):
        for j in range(i + 1, len(texts_per_split)):
            overlap = texts_per_split[i] & texts_per_split[j]
            if overlap:
                raise ValueError(f"Leakage: {len(overlap)} shared texts across splits")


def load(split: str) -> list[dict]:
    fp = config.DATA_DIR / f"signals_{split}.jsonl"
    return [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]


# --- price-regression dataset (the LoRA price arm + eval MAE comparison) ---------
PRICE_TRAIN_PATH = config.DATA_DIR / "price_train.jsonl"
PRICE_TEST_PATH = config.DATA_DIR / "price_test.jsonl"


def build_price_dataset(seed: int = 42) -> dict:
    """Format catalog price observations as ``{text, price}`` rows with an 80/20 split.

    Each row is ``"<title>, <festival>, <platform>" -> price_inr``. This mirrors the
    schema the Colab notebook (lora.ipynb) trains on, and is consumed by the eval
    harness MAE comparison.
    """
    from app import commerce_data

    rows = []
    for r in commerce_data.load_prices():
        festival = r.get("festival") or "No Festival"
        # `text` is what the LoRA notebook trains on; the structured fields let the eval
        # recover title/festival/platform without fragile string splitting.
        rows.append({"text": f"{r['title']}, {festival}, {r['platform']}",
                     "title": r["title"], "festival": festival,
                     "platform": r["platform"], "price": float(r["price_inr"])})
    random.Random(seed).shuffle(rows)
    split = int(len(rows) * 0.8)
    train, test = rows[:split], rows[split:]
    PRICE_TRAIN_PATH.write_text("\n".join(json.dumps(r) for r in train))
    PRICE_TEST_PATH.write_text("\n".join(json.dumps(r) for r in test))
    return {"train": {"path": str(PRICE_TRAIN_PATH), "n": len(train)},
            "test": {"path": str(PRICE_TEST_PATH), "n": len(test)}}


def load_price(split: str) -> list[dict]:
    fp = PRICE_TRAIN_PATH if split == "train" else PRICE_TEST_PATH
    if not fp.exists():
        build_price_dataset()
    return [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
    print(json.dumps(build_price_dataset(), indent=2))
