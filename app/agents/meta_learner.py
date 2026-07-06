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

import numpy as np

from app import commerce_data, config

META_PATH = config.DATA_DIR / "price_meta.json"
RF_PATH = config.DATA_DIR / "rf_price_model.pkl"

_CATEGORIES = ["grocery", "electronics", "personal_care", "baby"]


def _cat_id(cat: str) -> int:
    return _CATEGORIES.index(cat) if cat in _CATEGORIES else len(_CATEGORIES)


def _features(catalog_by_sku: dict, trailing: dict, row: dict) -> list[float]:
    item = catalog_by_sku[row["sku"]]
    fest_key = row.get("festival_key")
    bias = config.FESTIVAL_CALENDAR.get(fest_key, {}).get("discount_bias", 0.0) if fest_key else 0.0
    return [
        float(row["mrp_inr"]),
        float(trailing.get(row["sku"], row["mrp_inr"])),
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


def _stack_members(X: np.ndarray, rf_pred: np.ndarray) -> np.ndarray:
    """Base members for the stacking meta-learner."""
    m1 = X[:, 1]                       # trailing median
    m2 = X[:, 0] * (1.0 - X[:, 2])     # MRP * (1 - festival bias)
    m3 = rf_pred                       # RF prediction
    return np.column_stack([m1, m2, m3])


def train(seed: int = 7) -> dict:
    """Fit the RandomForest price model + linear stacking meta-learner on an 80/20 split.

    Both train and held-out test MAE are reported so overfitting is visible: the model
    is fit only on the 80% train split and scored on the unseen 20%.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    catalog = {c["sku"]: c for c in commerce_data.load_catalog()}
    prices = commerce_data.load_prices()
    trailing = _trailing_median()

    X = np.array([_features(catalog, trailing, r) for r in prices], dtype=float)
    y = np.array([float(r["price_inr"]) for r in prices], dtype=float)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)

    rf = RandomForestRegressor(n_estimators=120, max_depth=8, random_state=seed)
    rf.fit(X_tr, y_tr)

    # Fit the stacking meta-learner on the train split only.
    meta = LinearRegression()
    meta.fit(_stack_members(X_tr, rf.predict(X_tr)), y_tr)

    def _mae(Xs, ys):
        rf_p = rf.predict(Xs)
        meta_p = meta.predict(_stack_members(Xs, rf_p))
        return (float(np.mean(np.abs(meta_p - ys))), float(np.mean(np.abs(rf_p - ys))))

    train_mae, rf_train_mae = _mae(X_tr, y_tr)
    test_mae, rf_test_mae = _mae(X_te, y_te)

    # Refit on the full dataset for the deployed predictor (after metrics are recorded).
    rf.fit(X, y)
    meta.fit(_stack_members(X, rf.predict(X)), y)

    import pickle
    with open(RF_PATH, "wb") as f:
        pickle.dump(rf, f)
    META_PATH.write_text(json.dumps({
        "meta_coef": meta.coef_.tolist(), "meta_intercept": float(meta.intercept_),
        "members": ["trailing_median", "mrp_x_(1-festival_bias)", "random_forest"],
        "train_mae_inr": round(train_mae, 2), "test_mae_inr": round(test_mae, 2),
        "rf_train_mae_inr": round(rf_train_mae, 2), "rf_test_mae_inr": round(rf_test_mae, 2),
        "n_train": len(y_tr), "n_test": len(y_te), "feature_order": [
            "mrp_inr", "trailing_median", "festival_bias", "stock", "rating",
            "category_id", "pack_size"]}, indent=2))
    return json.loads(META_PATH.read_text())


def _load():
    import pickle
    if not (META_PATH.exists() and RF_PATH.exists()):
        train()
    with open(RF_PATH, "rb") as f:
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
