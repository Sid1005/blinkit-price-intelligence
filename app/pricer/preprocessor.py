"""LLM description rewrite — llm_eng week6 preprocessor.py/batch.py parity.

The course submits this as a Groq async Batch job (up to 24h turnaround).
Our Groq account's rate limits (500k req/day, 250k tok/min) make that
unnecessary — a thread-pooled synchronous sweep finishes the same 20k+ item
set in minutes with the same model/prompt, so we use that instead.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from app import config
from app.llm import groq_client
from app.pricer.items import Item

MODEL = config.GROQ_MODELS["oss_sm"]  # "openai/gpt-oss-20b", matches course's batch.py MODEL

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


def _rewrite_one(item: Item) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item.full or item.title},
    ]
    try:
        return groq_client.chat(messages, model=MODEL, temperature=0.3, max_tokens=200,
                                 reasoning_effort="low")
    except Exception:  # noqa: BLE001 — one bad item shouldn't kill the sweep
        return f"Title: {item.title}\nCategory: {item.category}\nBrand: unknown\nDescription: {item.title}\nDetails: n/a"


def rewrite_summaries(items: list[Item], max_workers: int = 24) -> None:
    """Fill in item.summary for every item, in place, via a thread-pooled sweep."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_item = {pool.submit(_rewrite_one, item): item for item in items}
        for future in tqdm(as_completed(future_to_item), total=len(items), desc="Groq description rewrite"):
            item = future_to_item[future]
            item.summary = future.result()
