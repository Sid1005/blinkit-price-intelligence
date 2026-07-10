"""Dataset loaders — Amazon (Part A) for now; BlinkitLoader (Part B) comes later.

The upstream ``McAuley-Lab/Amazon-Reviews-2023`` dataset ships a Hub *loading
script* (``Amazon-Reviews-2023.py``); recent ``datasets`` releases (>=3.0)
dropped script support entirely ("Dataset scripts are no longer supported"),
so ``load_dataset(..., trust_remote_code=True)`` — what the original course
notebook used — no longer works. We bypass the script and stream the
underlying raw per-category jsonl files directly
(``raw/meta_categories/meta_<Category>.jsonl``), which have the identical
schema (title/description/features/details/price/...).

We read the file as plain newline-delimited JSON via ``requests`` + one
``json.loads`` per line rather than ``datasets.load_dataset("json", ...)``:
the raw file has schema drift row-to-row (some items carry extra fields like
``subtitle``/``author``, ``bought_together`` is sometimes null/sometimes a
string, etc.), which trips pyarrow's strict cross-chunk schema unification
(``CastError: column names don't match``) — a non-issue for row-independent
``json.loads``.
"""
from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import requests
from huggingface_hub import hf_hub_url
from tqdm import tqdm

from app.pricer.items import Item
from app.pricer.parser import parse_amazon_row, parse_blinkit_row

# Appliances is the same category the course itself uses for its small/lite
# illustrative example — small raw file (~285MB), representative schema.
DEFAULT_AMAZON_CATEGORY = "Appliances"

TRAIN_LITE = 20_000
VAL_LITE = 1_000
TEST_LITE = 1_000
SHUFFLE_SEED = 42

DEFAULT_BLINKIT_JSON = "data/blinkit/blinkit_products.json"
BLINKIT_TRAIN_FRAC = 0.8
BLINKIT_VAL_FRAC = 0.1
# remainder (~0.1) goes to test


class AmazonLoader:
    """Streams one Amazon-Reviews-2023 category, scrubs/filters, and curates
    train/val/test splits, then slices the llm_eng "lite" sizes off the top.
    """

    def __init__(self, category: str = DEFAULT_AMAZON_CATEGORY):
        self.category = category

    def _resolve_url(self) -> str:
        safe = self.category.replace(" ", "_")
        return hf_hub_url(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw/meta_categories/meta_{safe}.jsonl",
            repo_type="dataset",
        )

    def _iter_rows(self):
        """Yield one parsed JSON dict per line, streamed over HTTP."""
        with requests.get(self._resolve_url(), stream=True, timeout=60) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def load_qualifying(self, max_qualifying: int | None = None) -> list[Item]:
        """Stream the category, keep rows that pass the scrub/price/length gate."""
        start = datetime.now()
        print(f"Streaming {self.category} from McAuley-Lab/Amazon-Reviews-2023 ...", flush=True)

        items: list[Item] = []
        for row in tqdm(self._iter_rows(), desc=f"scrub/filter {self.category}"):
            item = parse_amazon_row(row, self.category)
            if item is not None:
                items.append(item)
                if max_qualifying and len(items) >= max_qualifying:
                    break
        finish = datetime.now()
        print(f"Kept {len(items):,} qualifying items from {self.category} "
              f"in {(finish - start).total_seconds() / 60:.1f} min", flush=True)
        return items

    def curate_lite(self, max_qualifying: int | None = None) -> tuple[list[Item], list[Item], list[Item]]:
        """Full curated split (shuffled, held-out val/test) then the lite-size prefix."""
        items = self.load_qualifying(max_qualifying=max_qualifying)
        if len(items) < VAL_LITE + TEST_LITE + 100:
            raise RuntimeError(
                f"Only {len(items)} qualifying items in '{self.category}' — "
                "too few to carve out val/test holdouts. Pick a larger category."
            )

        rng = random.Random(SHUFFLE_SEED)
        rng.shuffle(items)
        for i, item in enumerate(items):
            item.id = i

        full_test = items[-TEST_LITE:]
        full_val = items[-(TEST_LITE + VAL_LITE):-TEST_LITE]
        full_train = items[:-(TEST_LITE + VAL_LITE)]

        train_lite = full_train[:TRAIN_LITE]
        val_lite = full_val[:VAL_LITE]
        test_lite = full_test[:TEST_LITE]

        for item in train_lite + val_lite + test_lite:
            item.make_prompt(item.full or item.title)

        print(f"Lite splits: train={len(train_lite)} val={len(val_lite)} test={len(test_lite)} "
              f"(full curated pool: {len(items)})")
        return train_lite, val_lite, test_lite


class BlinkitLoader:
    """Turns the Tavily-scraped Blinkit catalog (real, no synthetic rows) into
    Items (INR) and an 80/10/10 split — the Part B counterpart to AmazonLoader.
    """

    def __init__(self, json_path: str = DEFAULT_BLINKIT_JSON):
        self.json_path = json_path

    def load_qualifying(self) -> list[Item]:
        with open(self.json_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        items: list[Item] = []
        for row in rows:
            item = parse_blinkit_row(row)
            if item is not None:
                items.append(item)
        print(f"Kept {len(items):,} qualifying Blinkit items out of {len(rows):,} scraped rows")
        return items

    def curate_split(self) -> tuple[list[Item], list[Item], list[Item]]:
        """Shuffle and carve an 80/10/10 train/val/test split (real data — no lite subsample needed)."""
        items = self.load_qualifying()

        rng = random.Random(SHUFFLE_SEED)
        rng.shuffle(items)
        for i, item in enumerate(items):
            item.id = i

        n = len(items)
        n_train = int(n * BLINKIT_TRAIN_FRAC)
        n_val = int(n * BLINKIT_VAL_FRAC)

        train = items[:n_train]
        val = items[n_train:n_train + n_val]
        test = items[n_train + n_val:]

        for item in train + val + test:
            item.make_prompt(item.full or item.title)

        print(f"Blinkit splits: train={len(train)} val={len(val)} test={len(test)} (total {n})")
        return train, val, test


def _write_jsonl(items: list[Item], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")


def main():
    import argparse

    from app.pricer import hub
    from app.pricer.preprocessor import rewrite_summaries

    parser = argparse.ArgumentParser(description="Curate the Amazon or Blinkit pricer dataset and push it to the user's HF account.")
    parser.add_argument("--source", choices=["amazon", "blinkit"], default="amazon")
    parser.add_argument("--category", default=DEFAULT_AMAZON_CATEGORY, help="Amazon-only: raw_meta category to stream")
    parser.add_argument("--max-qualifying", type=int, default=None,
                        help="Amazon-only: cap the number of qualifying rows scanned (for a quick dry run)")
    parser.add_argument("--blinkit-json", default=DEFAULT_BLINKIT_JSON)
    parser.add_argument("--dataset-name", default=None, help="Defaults to amazon-pricer-lite / blinkit-pricer")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip the Groq description rewrite (summary stays empty)")
    parser.add_argument("--no-push", action="store_true", help="Curate locally only; skip the HF Hub push")
    args = parser.parse_args()

    if args.source == "amazon":
        loader = AmazonLoader(category=args.category)
        train, val, test = loader.curate_lite(max_qualifying=args.max_qualifying)
        dataset_name = args.dataset_name or "amazon-pricer-lite"
        jsonl_dir = "data/amazon"
    else:
        loader = BlinkitLoader(json_path=args.blinkit_json)
        train, val, test = loader.curate_split()
        dataset_name = args.dataset_name or "blinkit-pricer"
        jsonl_dir = "data/blinkit"

    all_items = train + val + test
    if not args.no_preprocess:
        rewrite_summaries(all_items, max_workers=24)
        for item in all_items:
            item.make_prompt(item.summary or item.full or item.title)

    print(f"\nExample item:\n{train[0]!r}\n")
    print(train[0].prompt)

    _write_jsonl(train, f"{jsonl_dir}/price_train.jsonl")
    _write_jsonl(val, f"{jsonl_dir}/price_val.jsonl")
    _write_jsonl(test, f"{jsonl_dir}/price_test.jsonl")
    print(f"Wrote local jsonl splits to {jsonl_dir}/price_{{train,val,test}}.jsonl")

    if not args.no_push:
        repo_id = hub.push(dataset_name, train, val, test)
        print(f"\nPushed to Hugging Face Hub: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
