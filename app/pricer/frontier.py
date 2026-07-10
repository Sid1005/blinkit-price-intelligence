"""Frontier zero-shot pricing arms — Groq + Claude, no RAG, no fine-tuning.

Mirrors llm_eng week6/day5: given only ``item.test_prompt()`` (the same text
the baselines see, truncated right before the price), ask a frontier LLM to
estimate the price from its general knowledge alone. Used by **both** parts
(Amazon $, Blinkit ₹) — this is the "can a big model just guess it" arm that
classical ML and the later RAG / QLoRA arms are meant to beat. Scored by the
same ``evaluator.Tester`` as every other arm, so MAE/RMSE/R²/hit-rate are
directly comparable to ``baselines.py``.
"""
from __future__ import annotations

import re
from typing import Callable

from app import config
from app.pricer.items import Item

_SYSTEM = (
    "You estimate retail prices from a product description. Reply with strict "
    'JSON only, nothing else: {{"price": <number>}}. The number is your single '
    "best point estimate in {currency}, with no currency symbol, no commas, "
    "and no explanation."
)

_PRICE_RE = re.compile(r'"?price"?\s*[:=]\s*"?(-?\d+(?:\.\d+)?)', re.I)
_NUMBER_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


def _parse_price(text: str) -> float:
    match = _PRICE_RE.search(text)
    if not match:
        match = _NUMBER_RE.search(text)
    return max(0.0, float(match.group(1))) if match else 0.0


def _currency_name(item: Item) -> str:
    return "US dollars" if item.currency == "$" else "Indian rupees"


class GroqFrontier:
    """Zero-shot price estimate from Groq's strong-tier model."""

    def __init__(self, model: str | None = None):
        self.model = model or config.DEFAULT_MODEL

    def predict(self, item: Item) -> float:
        from app.llm import groq_client

        messages = [
            {"role": "system", "content": _SYSTEM.format(currency=_currency_name(item))},
            {"role": "user", "content": item.test_prompt()},
        ]
        raw = groq_client.chat(messages, model=self.model, temperature=0.0,
                                max_tokens=60, json_mode=True)
        return _parse_price(raw)

    def predictor(self) -> Callable[[Item], float]:
        """Adapter matching evaluator.Tester's predictor(item) -> float signature."""
        return self.predict


class ClaudeFrontier:
    """Zero-shot price estimate from Claude — a second, independent frontier arm."""

    def __init__(self, model: str | None = None):
        self.model = model or config.ANTHROPIC_PRICE_MODEL
        self._client = None

    def _client_(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=config.require_anthropic())
        return self._client

    def predict(self, item: Item) -> float:
        resp = self._client_().messages.create(
            model=self.model, max_tokens=60,
            system=_SYSTEM.format(currency=_currency_name(item)),
            messages=[{"role": "user", "content": item.test_prompt()}])
        text = "".join(block.text for block in resp.content
                        if getattr(block, "type", None) == "text")
        return _parse_price(text)

    def predictor(self) -> Callable[[Item], float]:
        return self.predict


_FRONTIER: dict[str, tuple[str, type]] = {
    "groq": ("Groq Frontier", GroqFrontier),
    "claude": ("Claude Frontier", ClaudeFrontier),
}


def main():
    import argparse

    from app.pricer import hub
    from app.pricer.evaluator import Tester

    parser = argparse.ArgumentParser(description="Score the zero-shot frontier LLM arms (no RAG).")
    parser.add_argument("--source", choices=["amazon", "blinkit"], default="amazon")
    parser.add_argument("--model", choices=[*_FRONTIER, "all"], default="all",
                         help="Which frontier arm to run (default: both, printed as a comparison table)")
    parser.add_argument("--dataset-name", default=None, help="Defaults to amazon-pricer-lite / blinkit-pricer")
    parser.add_argument("--size", type=int, default=None,
                         help="Cap the number of test items scored (each item is a live API call)")
    parser.add_argument("--chart-dir", default=None, help="Directory to save scatter/cumulative-error charts")
    parser.add_argument("--workers", type=int, default=8,
                         help="Concurrent API calls (frontier predict() is I/O-bound; default 8)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    dataset_name = args.dataset_name or ("amazon-pricer-lite" if args.source == "amazon" else "blinkit-pricer")
    chart_dir = args.chart_dir or f"data/{args.source}/eval_charts"

    print(f"Pulling {dataset_name} from the Hub...")
    train, val, test = hub.pull(dataset_name)
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    keys = list(_FRONTIER) if args.model == "all" else [args.model]
    results = []
    for key in keys:
        label, cls = _FRONTIER[key]
        print(f"\nScoring {label} ({args.source})...")
        arm = cls()
        results.append(Tester.test(arm.predictor(), f"{label} ({args.source})", test,
                                    size=args.size, chart_dir=chart_dir, verbose=args.verbose,
                                    max_workers=args.workers))

    if len(results) > 1:
        print(f"\n{'Model':<28}{'MAE':>10}{'RMSE':>10}{'R²':>10}{'Hit rate':>12}")
        for r in sorted(results, key=lambda r: r["mae"]):
            print(f"{r['title'].split(' (')[0]:<28}{r['mae']:>10.2f}{r['rmse']:>10.2f}"
                  f"{r['r2']:>10.3f}{r['hit_rate']:>12.1%}")


if __name__ == "__main__":
    main()
