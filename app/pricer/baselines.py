"""Classical ML baselines — course week6/day3-4 parity.

LinearRegression (weight/text length/word count/category), BoW+LinearRegression,
RandomForest, and XGBoost.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from app.pricer.items import Item

_WORD_RE = re.compile(r"[A-Za-z]+")

# --- category normalization -------------------------------------------------
# Blinkit's scraped category taxonomy has ~112 near-duplicate raw labels
# (HTML-entity/casing drift, e.g. "Audio &amp; Accessories" vs "Audio
# Accessories"). One-hot-encoding all of them starves most columns of
# training support. Every baseline below collapses categories to <=10
# canonical buckets via Groq the first time it's fit on a given dataset, then
# caches the mapping to disk (data/<source>/category_map.json) so later runs
# reuse it instead of re-calling Groq. Datasets that already have <=10
# categories (e.g. Amazon's single hardcoded "Appliances") skip Groq entirely
# — the map is just the identity.
_MAX_CATEGORIES = 10
_category_map_cache: dict[str, dict[str, str]] = {}


def _clean_category(raw: str) -> str:
    return " ".join(html.unescape(raw).split()).strip()


def _category_map_path(items: list[Item]) -> Path:
    currency = items[0].currency if items else "$"
    folder = "blinkit" if currency == "₹" else "amazon"
    return Path(f"data/{folder}/category_map.json")


_CLUSTER_SYSTEM = f"""You are cleaning up a noisy e-commerce category taxonomy for a
grocery-delivery pricing model. You will be given a list of raw scraped
category labels (some are near-duplicates of each other from inconsistent
scraping — different casing, "X & Y" vs "X and Y", singular vs plural, etc).

Cluster every raw label into at most {_MAX_CATEGORIES} broad canonical
categories that best preserve price-relevant grouping (e.g. "Dairy &
Bakery", "Snacks & Beverages", "Personal Care", "Household & Cleaning",
"Baby & Pet Care", "Health & Wellness", "Electronics & Accessories",
"Home & Kitchen", "Stationery & Gifts", "Fresh Produce & Meat" are the kind
of buckets that make sense for a grocery app — adapt as needed for the
actual labels given, but do not exceed {_MAX_CATEGORIES} distinct canonical
names).

Respond with strict JSON: {{"mapping": {{"<raw label exactly as given>":
"<canonical category name>", ...}}}}. Every raw label in the input must
appear as a key. Use at most {_MAX_CATEGORIES} distinct values across all
the mapping's values."""


def _cluster_categories_via_groq(raw_categories: list[str]) -> dict[str, str]:
    from app.llm import groq_client

    listing = "\n".join(f"- {c}" for c in raw_categories)
    messages = [
        {"role": "system", "content": _CLUSTER_SYSTEM},
        {"role": "user", "content": f"Raw category labels ({len(raw_categories)} total):\n{listing}"},
    ]
    # chat_json's default 1024-token cap truncates a ~100-label mapping
    # mid-JSON, so call chat() directly with room to finish.
    raw = groq_client.chat(messages, model=groq_client.config.DEFAULT_MODEL,
                            temperature=0.0, max_tokens=8000, json_mode=True)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Groq returned invalid JSON while clustering categories: {e}\n{raw[:2000]}") from e
    mapping = result.get("mapping")
    if not isinstance(mapping, dict):
        raise RuntimeError(f"Groq did not return a usable category mapping: {result}")

    # Fill in anything Groq dropped, and if it overshot the cap, greedily
    # fold the smallest canonical buckets into the largest until we're back
    # at <= _MAX_CATEGORIES.
    fixed = dict(mapping)
    for raw_label in raw_categories:
        fixed.setdefault(raw_label, "Other")
    fixed = {k: v for k, v in fixed.items() if k in set(raw_categories)}

    bucket_members: dict[str, list[str]] = {}
    for raw_label, canon in fixed.items():
        bucket_members.setdefault(canon, []).append(raw_label)
    while len(bucket_members) > _MAX_CATEGORIES:
        smallest = min(bucket_members, key=lambda k: len(bucket_members[k]))
        members = bucket_members.pop(smallest)
        target = max(bucket_members, key=lambda k: len(bucket_members[k]))
        bucket_members[target].extend(members)
        for raw_label in members:
            fixed[raw_label] = target

    return fixed


def _ensure_category_map(train_items: list[Item]) -> dict[str, str]:
    """Get (or lazily build + cache) the raw->canonical category map for
    whichever dataset ``train_items`` belongs to. Always runs as part of
    ``fit()`` — never a separate manual step.
    """
    path = _category_map_path(train_items)
    key = str(path)
    if key in _category_map_cache:
        return _category_map_cache[key]
    if path.exists():
        mapping = json.loads(path.read_text())
        _category_map_cache[key] = mapping
        return mapping

    raw_categories = sorted({_clean_category(item.category) for item in train_items if item.category})
    if len(raw_categories) <= _MAX_CATEGORIES:
        mapping = {c: c for c in raw_categories}
    else:
        mapping = _cluster_categories_via_groq(raw_categories)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True))
    _category_map_cache[key] = mapping
    return mapping


def _normalize_category(raw: str, category_map: dict[str, str]) -> str:
    return category_map.get(_clean_category(raw), _clean_category(raw))


# --- shared feature helpers --------------------------------------------------

def _text(item: Item) -> str:
    return item.summary or item.full or item.title


def _features(item: Item, category_map: dict[str, str]) -> dict:
    text = _text(item)
    return {
        "weight": item.weight or 0.0,
        "text_length": len(text),
        "word_count": len(_WORD_RE.findall(text)),
        "category": _normalize_category(item.category, category_map) if item.category else "unknown",
    }


class LinearRegressionBaseline:
    """Engineered-feature LinearRegression predicting price directly (course
    day3's first ML baseline: weight + text length/word count + category).
    """

    def __init__(self):
        self.vectorizer = DictVectorizer(sparse=False)
        self.model = LinearRegression()
        self.category_map: dict[str, str] = {}

    def fit(self, train_items: list[Item]) -> "LinearRegressionBaseline":
        self.category_map = _ensure_category_map(train_items)
        X = self.vectorizer.fit_transform([_features(item, self.category_map) for item in train_items])
        y = np.array([item.price for item in train_items])
        self.model.fit(X, y)
        return self

    def predict(self, item: Item) -> float:
        X = self.vectorizer.transform([_features(item, self.category_map)])
        return max(0.0, float(self.model.predict(X)[0]))

    def predictor(self) -> Callable[[Item], float]:
        """Adapter matching evaluator.Tester's predictor(item) -> float signature."""
        return self.predict


class BagOfWordsLinearRegressionBaseline:
    """CountVectorizer(BoW) + LinearRegression on raw text (course day3's
    second baseline: word-count features do the talking instead of
    engineered ones — no weight/category, just the text).
    """

    def __init__(self, max_features: int = 8000):
        self.vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
        self.model = LinearRegression()

    def fit(self, train_items: list[Item]) -> "BagOfWordsLinearRegressionBaseline":
        X = self.vectorizer.fit_transform([_text(item) for item in train_items])
        y = np.array([item.price for item in train_items])
        self.model.fit(X, y)
        return self

    def predict(self, item: Item) -> float:
        X = self.vectorizer.transform([_text(item)])
        return max(0.0, float(self.model.predict(X)[0]))

    def predictor(self) -> Callable[[Item], float]:
        return self.predict


class RandomForestBaseline:
    """CountVectorizer(BoW) + RandomForestRegressor — same text features as
    the BoW baseline, but a non-linear tree ensemble instead of a linear fit
    (course day4's tree-ensemble baseline).
    """

    def __init__(self, max_features: int = 8000, n_estimators: int = 100):
        self.vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)

    def fit(self, train_items: list[Item]) -> "RandomForestBaseline":
        X = self.vectorizer.fit_transform([_text(item) for item in train_items])
        y = np.array([item.price for item in train_items])
        self.model.fit(X, y)
        return self

    def predict(self, item: Item) -> float:
        X = self.vectorizer.transform([_text(item)])
        return max(0.0, float(self.model.predict(X)[0]))

    def predictor(self) -> Callable[[Item], float]:
        return self.predict


class XGBoostBaseline:
    """CountVectorizer(BoW) + XGBoost gradient-boosted trees — same text
    features again, boosted instead of bagged. The strongest classical
    baseline before the LLM arms.
    """

    def __init__(self, max_features: int = 8000, n_estimators: int = 200):
        self.vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
        self.model = XGBRegressor(n_estimators=n_estimators, random_state=42,
                                   objective="reg:squarederror", n_jobs=-1)

    def fit(self, train_items: list[Item]) -> "XGBoostBaseline":
        X = self.vectorizer.fit_transform([_text(item) for item in train_items])
        y = np.array([item.price for item in train_items])
        self.model.fit(X, y)
        return self

    def predict(self, item: Item) -> float:
        X = self.vectorizer.transform([_text(item)])
        return max(0.0, float(self.model.predict(X)[0]))

    def predictor(self) -> Callable[[Item], float]:
        return self.predict


_BASELINES: dict[str, tuple[str, type]] = {
    "linear": ("LinearRegression", LinearRegressionBaseline),
    "bow": ("BoW+LinearRegression", BagOfWordsLinearRegressionBaseline),
    "rf": ("RandomForest", RandomForestBaseline),
    "xgb": ("XGBoost", XGBoostBaseline),
}


def main():
    import argparse

    from app.pricer import hub
    from app.pricer.evaluator import Tester

    parser = argparse.ArgumentParser(description="Fit and score the classical ML baselines.")
    parser.add_argument("--source", choices=["amazon", "blinkit"], default="amazon")
    parser.add_argument("--model", choices=[*_BASELINES, "all"], default="all",
                         help="Which baseline to run (default: all four, printed as a comparison table)")
    parser.add_argument("--dataset-name", default=None, help="Defaults to amazon-pricer-lite / blinkit-pricer")
    parser.add_argument("--size", type=int, default=None, help="Cap the number of test items scored")
    parser.add_argument("--chart-dir", default=None, help="Directory to save scatter/cumulative-error charts")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    dataset_name = args.dataset_name or ("amazon-pricer-lite" if args.source == "amazon" else "blinkit-pricer")
    chart_dir = args.chart_dir or f"data/{args.source}/eval_charts"

    print(f"Pulling {dataset_name} from the Hub...")
    train, val, test = hub.pull(dataset_name)
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    keys = list(_BASELINES) if args.model == "all" else [args.model]
    results = []
    for key in keys:
        label, cls = _BASELINES[key]
        print(f"\nFitting {label} ({args.source})...")
        baseline = cls().fit(train)
        results.append(Tester.test(baseline.predictor(), f"{label} ({args.source})", test,
                                    size=args.size, chart_dir=chart_dir, verbose=args.verbose))

    if len(results) > 1:
        print(f"\n{'Model':<28}{'MAE':>10}{'RMSE':>10}{'R²':>10}{'Hit rate':>12}")
        for r in sorted(results, key=lambda r: r["mae"]):
            print(f"{r['title'].split(' (')[0]:<28}{r['mae']:>10.2f}{r['rmse']:>10.2f}"
                  f"{r['r2']:>10.3f}{r['hit_rate']:>12.1%}")


if __name__ == "__main__":
    main()
