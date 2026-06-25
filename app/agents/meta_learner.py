"""Trained meta-learner for ensemble price forecasting (weeks 6 & 8).

Instead of a hand-weighted average, the final price point is produced by a *stacked*
ensemble:

  base members (per SKU/month):
    m1 = trailing median observed price        (history model)
    m2 = MRP * (1 - festival discount bias)     (festival rule model)
    m3 = RandomForestRegressor prediction       (learned classical-ML model)

  meta-learner:
    LinearRegression([m1, m2, m3]) -> fair price point.

Both models are trained on the curated price-observation dataset and persisted to
``data/`` so inference is fast. At runtime the Groq estimate is blended in as an extra
member (see app/agents/ensemble.py::pricer_agent).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app import commerce_data, config

MODEL_PATH = config.DATA_DIR / "price_models.npz"
META_PATH = config.DATA_DIR / "price_meta.json"

_CATEGORIES = ["grocery", "electronics", "personal_care", "baby"]


def _cat_id(cat: str) -> int:
    return _CATEGORIES.index(cat) if cat in _CATEGORIES else len(_CATEGORIES)


def _features(catalog_by_sku: dict, trailing: dict, row: dict) -> list[float]:
    item = catalog_by_sku[row["sku"]]
    fest_key = row.get("festival_key")
    bias = config.FESTIVAL_CALENDAR.get(fest_key, {}).get("discount_bias", 0.0) if fest_key else 0.0
    return [
        float(row["mrp_inr"]),
        float(trailing[row["sku"]]),
        float(bias),
        float(item.get("stock", 0)),
        float(item.get("rating", 0.0)),
        float(_cat_id(item["category"])),
        float(item.get("pack_size", 0.0)),
    ]


def _trailing_median() -> dict:
    prices = commerce_data.load_prices()
    by_sku: dict[str, list[float]] = {}
    for r in prices:
        by_sku.setdefault(r["sku"], []).append(float(r["price_inr"]))
    return {sku: float(np.median(v)) for sku, v in by_sku.items()}


def train(seed: int = 7) -> dict:
    """Fit the RandomForest price model + the linear stacking meta-learner."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    catalog = {c["sku"]: c for c in commerce_data.load_catalog()}
    prices = commerce_data.load_prices()
    trailing = _trailing_median()

    X = np.array([_features(catalog, trailing, r) for r in prices], dtype=float)
    y = np.array([float(r["price_inr"]) for r in prices], dtype=float)

    rf = RandomForestRegressor(n_estimators=120, max_depth=8, random_state=seed)
    rf.fit(X, y)
    rf_pred = rf.predict(X)

    # base members for the stack
    m1 = X[:, 1]                       # trailing median
    m2 = X[:, 0] * (1.0 - X[:, 2])     # MRP * (1 - festival bias)
    m3 = rf_pred                       # RF prediction
    stack_X = np.column_stack([m1, m2, m3])
    meta = LinearRegression()
    meta.fit(stack_X, y)

    meta_pred = meta.predict(stack_X)
    mae = float(np.mean(np.abs(meta_pred - y)))
    rf_mae = float(np.mean(np.abs(rf_pred - y)))

    # Persist RF via its internal arrays is awkward; pickle it alongside numpy meta.
    import pickle
    with open(config.DATA_DIR / "rf_price_model.pkl", "wb") as f:
        pickle.dump(rf, f)
    np.savez(MODEL_PATH, coef=meta.coef_, intercept=np.array([meta.intercept_]))
    META_PATH.write_text(json.dumps({
        "meta_coef": meta.coef_.tolist(), "meta_intercept": float(meta.intercept_),
        "members": ["trailing_median", "mrp_x_(1-festival_bias)", "random_forest"],
        "train_mae_inr": round(mae, 2), "rf_mae_inr": round(rf_mae, 2),
        "n_train": len(y), "feature_order": [
            "mrp_inr", "trailing_median", "festival_bias", "stock", "rating",
            "category_id", "pack_size"]}, indent=2))
    return json.loads(META_PATH.read_text())


def _load():
    import pickle
    if not (MODEL_PATH.exists() and META_PATH.exists() and
            (config.DATA_DIR / "rf_price_model.pkl").exists()):
        train()
    with open(config.DATA_DIR / "rf_price_model.pkl", "rb") as f:
        rf = pickle.load(f)
    meta = json.loads(META_PATH.read_text())
    return rf, meta


def predict_point(sku: str, festival_key: str | None = None) -> dict | None:
    """Predict a fair price point for a known SKU using the trained stack."""
    catalog = {c["sku"]: c for c in commerce_data.load_catalog()}
    if sku not in catalog:
        return None
    trailing = _trailing_median()
    rf, meta = _load()
    row = {"sku": sku, "mrp_inr": catalog[sku]["mrp_inr"], "festival_key": festival_key}
    feats = np.array([_features(catalog, trailing, row)], dtype=float)
    rf_pred = float(rf.predict(feats)[0])
    m1 = float(feats[0, 1])
    bias = feats[0, 2]
    m2 = float(feats[0, 0] * (1.0 - bias))
    coef = meta["meta_coef"]
    point = coef[0] * m1 + coef[1] * m2 + coef[2] * rf_pred + meta["meta_intercept"]
    return {"point_inr": round(max(1.0, point), 2),
            "members": {"trailing_median": round(m1, 2),
                        "festival_rule": round(m2, 2),
                        "random_forest": round(rf_pred, 2)},
            "meta_coef": coef}


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
    print(json.dumps(predict_point("SF-1012", "diwali"), indent=2))
