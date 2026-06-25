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
    "review_sentiment": [
        "{p} ki quality bahut achhi hai, paisa vasool.",
        "Bekar product, {p} ekdum ghatiya nikla.",
        "{p} genuine hai, sealed packaging mili, satisfied.",
        "Worst experience, {p} fake jaisa lag raha hai.",
        "{p} value for money, rating 5 star deta hu.",
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


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
