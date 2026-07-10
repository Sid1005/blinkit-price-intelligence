#!/usr/bin/env python3
"""Run the Blinkit substitution RAG-vs-no-RAG eval and print a leaderboard.

Usage:
  python3 scripts/eval_substitution_rag.py [--n 40] [--seed 0] [--k 5]

Writes evals/results/substitution_rag_ablation.json (per-example records +
aggregate summary) and prints a comparison table to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.substitution import evaluate

RESULTS_PATH = ROOT / "evals" / "results" / "substitution_rag_ablation.json"


def _print_leaderboard(result: dict) -> None:
    no_rag, rag = result["no_rag_summary"], result["rag_summary"]
    rows = [
        ("n examples", no_rag["n"], rag["n"]),
        ("error rate", no_rag["error_rate"], rag["error_rate"]),
        ("exists in real catalog", no_rag["exists_in_catalog_rate"], rag["exists_in_catalog_rate"]),
        ("same aisle (of matched)", no_rag["same_aisle_rate"], rag["same_aisle_rate"]),
        ("price stated count", no_rag["price_stated_count"], rag["price_stated_count"]),
        ("price accuracy (of stated)", no_rag["price_accuracy_rate"], rag["price_accuracy_rate"]),
        ("avg cost/query (USD)", no_rag["avg_cost_usd"], rag["avg_cost_usd"]),
        ("total cost (USD)", no_rag["total_cost_usd"], rag["total_cost_usd"]),
        ("avg latency (s)", no_rag["avg_latency_s"], rag["avg_latency_s"]),
        ("total latency (s)", no_rag["total_latency_s"], rag["total_latency_s"]),
    ]
    name_w = max(len(r[0]) for r in rows)
    print(f"\n{'metric':<{name_w}}  {'no-RAG (Claude Sonnet 5)':>26}  {'with-RAG (Groq)':>18}")
    print("-" * (name_w + 50))
    for label, a, b in rows:
        print(f"{label:<{name_w}}  {str(a):>26}  {str(b):>18}")
    print(f"\nWall clock: {result['wall_clock_s']}s over {result['n_examples']} examples")
    print(f"Results written to {RESULTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40, help="sample size")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument("--k", type=int, default=5, help="RAG retrieval k")
    args = parser.parse_args()

    sample = evaluate.build_eval_sample(n=args.n, seed=args.seed)
    print(f"Sampled {len(sample)} real Blinkit SKUs across "
          f"{len({evaluate.blinkit_catalog.bucket_for(r.get('category')) for r in sample})} aisles "
          f"({sum(1 for r in sample if not r.get('in_stock', True))} out-of-stock).")

    result = evaluate.run_eval(sample, k_retrieve=args.k)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_leaderboard(result)


if __name__ == "__main__":
    main()
