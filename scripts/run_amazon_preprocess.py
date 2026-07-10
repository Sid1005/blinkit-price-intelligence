#!/usr/bin/env python3
"""Pull the curated amazon-pricer-lite splits back from the Hub, run the Groq
description rewrite over all 22,000 items, rebuild prompts from the rewritten
summaries (course parity: item.summary is what feeds the prompt/classical ML/
frontier arms downstream), and push the enriched dataset back to the Hub.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.pricer import hub
from app.pricer.preprocessor import rewrite_summaries


def main():
    print("Pulling amazon-pricer-lite from the Hub...")
    train, val, test = hub.pull("amazon-pricer-lite")
    all_items = train + val + test
    print(f"Loaded {len(train)} train / {len(val)} val / {len(test)} test = {len(all_items)} total")

    rewrite_summaries(all_items, max_workers=24)

    for item in all_items:
        item.make_prompt(item.summary)

    print("\nExample after rewrite:")
    print(all_items[0].title)
    print(all_items[0].summary)
    print(all_items[0].prompt)

    repo_id = hub.push("amazon-pricer-lite", train, val, test)
    print(f"\nRe-pushed enriched dataset to https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
