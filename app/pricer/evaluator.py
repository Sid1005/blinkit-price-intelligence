"""Tester — shared evaluation harness for every pricing arm in both parts,
mirroring llm_eng week6/pricer/testing.py.

Any predictor with signature ``predictor(item: Item) -> float`` (baselines,
frontier, RAG, ensemble, ...) can be scored by the same ``Tester``, on either
the Amazon or Blinkit test set, so every arm in both parts is compared with
identical metrics and identical charts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from app.pricer.items import Item

GREEN = "\033[92m"
ORANGE = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
_ANSI = {"green": GREEN, "orange": ORANGE, "red": RED}
_HEX = {"green": "#2ecc71", "orange": "#f39c12", "red": "#e74c3c"}


def color_for(error: float, truth: float) -> str:
    """Course's traffic-light rule: tight errors are green, loose ones red."""
    ratio = error / truth if truth else float("inf")
    if error < 40 or ratio < 0.2:
        return "green"
    if error < 80 or ratio < 0.4:
        return "orange"
    return "red"


class Tester:
    """Runs a predictor over a (sliced) test set and reports MAE/RMSE/R² plus
    color-coded scatter and cumulative-error charts.
    """

    def __init__(self, predictor: Callable[[Item], float], title: str,
                 data: list[Item], size: int | None = None, max_workers: int = 1):
        self.predictor = predictor
        self.title = title
        self.data = data[:size] if size else data
        self.max_workers = max_workers
        self.truths: list[float] = []
        self.guesses: list[float] = []
        self.errors: list[float] = []
        self.colors: list[str] = []

    def _run_datapoint(self, item: Item, index: int, verbose: bool, guess: float | None = None) -> None:
        guess = self.predictor(item) if guess is None else guess
        truth = item.price
        error = abs(guess - truth)
        color = color_for(error, truth)
        self.truths.append(truth)
        self.guesses.append(guess)
        self.errors.append(error)
        self.colors.append(color)
        if verbose:
            snippet = (item.summary or item.title)[:40].replace("\n", " ")
            print(f"{_ANSI[color]}{index+1}: guess {item.currency}{guess:,.2f} "
                  f"truth {item.currency}{truth:,.2f} error {item.currency}{error:,.2f} "
                  f"  {snippet}...{RESET}")

    def chart(self, save_path: str | None = None) -> None:
        import matplotlib.pyplot as plt

        max_val = max(max(self.truths, default=0), max(self.guesses, default=0)) * 1.05
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(self.truths, self.guesses,
                   c=[_HEX[c] for c in self.colors], s=12, alpha=0.6)
        ax.plot([0, max_val], [0, max_val], color="#999999", linestyle="--", linewidth=1)
        ax.set_xlabel("Truth")
        ax.set_ylabel("Prediction")
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_title(self.title)
        fig.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=120)
        plt.close(fig)

    def cumulative_error_chart(self, save_path: str | None = None) -> None:
        import matplotlib.pyplot as plt

        sorted_errors = np.sort(self.errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(sorted_errors, cumulative, color="#2c7fb8")
        ax.set_xlabel("Absolute error")
        ax.set_ylabel("Cumulative fraction of test set")
        ax.set_title(f"{self.title} — cumulative error")
        fig.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=120)
        plt.close(fig)

    def report(self, chart_dir: str | None = None, verbose: bool = False) -> dict:
        if self.max_workers > 1:
            # Live LLM predictors (frontier/RAG arms) are I/O-bound network
            # calls — safe and much faster to fan out concurrently. Classical
            # baselines just run with max_workers=1 (their default) since
            # local sklearn/xgboost inference is already fast.
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                guesses = list(pool.map(self.predictor, self.data))
            for i, (item, guess) in enumerate(zip(self.data, guesses)):
                self._run_datapoint(item, i, verbose, guess=guess)
        else:
            for i, item in enumerate(self.data):
                self._run_datapoint(item, i, verbose)

        truths = np.array(self.truths)
        guesses = np.array(self.guesses)
        errors = np.array(self.errors)

        mae = float(errors.mean())
        rmse = float(np.sqrt(((guesses - truths) ** 2).mean()))
        ss_res = float(((truths - guesses) ** 2).sum())
        ss_tot = float(((truths - truths.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
        hits = {c: self.colors.count(c) for c in ("green", "orange", "red")}
        hit_rate = hits["green"] / len(self.colors) if self.colors else 0.0

        title = f"{self.title}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.3f}  hit_rate={hit_rate:.1%}"
        print(f"\n{title}")
        print(f"  green={hits['green']} orange={hits['orange']} red={hits['red']} (n={len(self.colors)})")

        if chart_dir:
            slug = self.title.lower().replace(" ", "_")
            self.chart(save_path=f"{chart_dir}/{slug}_scatter.png")
            self.cumulative_error_chart(save_path=f"{chart_dir}/{slug}_cumulative.png")

        return {
            "title": self.title,
            "n": len(self.colors),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "hit_rate": hit_rate,
            "green": hits["green"],
            "orange": hits["orange"],
            "red": hits["red"],
        }

    @classmethod
    def test(cls, predictor: Callable[[Item], float], title: str, data: list[Item],
              size: int | None = None, chart_dir: str | None = None, verbose: bool = False,
              max_workers: int = 1) -> dict:
        """Convenience one-shot: build a Tester and immediately report()."""
        return cls(predictor, title, data, size=size, max_workers=max_workers).report(
            chart_dir=chart_dir, verbose=verbose)
