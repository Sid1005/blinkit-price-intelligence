"""Offline-safe eval harness for Blinkit Price Intelligence.

Supports `python evals/harness.py <loop_number> [suite_subset]` where
suite_subset is a comma-separated list like `intent,unit_norm,substitution`.

All suites are offline-safe: they use golden data, local catalog, and deterministic
computations. When RAG or Groq is unavailable, suites report skipped or fallback
metrics gracefully (never crash). Results are written to
``evals/results/results_loop<N>.json`` and ``evals/results/latest.json``, and
logged to ``app.monitoring.experiment_tracking`` if available.

Suites:
  intent           — intent routing accuracy against golden examples
  substitution     — substitute finder coverage and MRR
  unit_norm        — unit-price normalisation accuracy
  price_comparison — band-coverage + meta-learner MAE; live ensemble MAE only when RUN_LIVE_EVALS=1
  schema_validity  — validates Brief schema fields
  guardrails       — adversarial / injection resistance (skipped unless RUN_LIVE_EVALS=1)
  rag_retrieval    — offline RAG retrieval recall and MRR against golden questions
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GOLDEN = ROOT / "evals" / "golden" / "golden.json"
RESULTS_DIR = ROOT / "evals" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_SUITES = [
    "intent", "substitution", "unit_norm", "price_comparison",
    "schema_validity", "guardrails", "rag_retrieval",
]


def _load_golden():
    return json.loads(GOLDEN.read_text())


# ---------------------------------------------------------------------------
# Suite: intent
# ---------------------------------------------------------------------------
def _suite_intent() -> dict:
    golden = _load_golden().get("intent", [])
    try:
        from app.agents import intent_router
    except Exception:
        return {"intent_accuracy": None, "error": "intent_router import failed", "n_total": len(golden)}
    correct = 0
    n = len(golden)
    for ex in golden:
        try:
            routed = intent_router.route_intent(ex["input"])
            if routed["intent"] == ex["expected_intent"]:
                correct += 1
        except Exception:
            pass
    return {"intent_accuracy": round(correct / n, 4) if n else 0, "n_correct": correct, "n_total": n}


# ---------------------------------------------------------------------------
# Suite: substitution
# ---------------------------------------------------------------------------
def _suite_substitution() -> dict:
    golden = _load_golden().get("substitution", [])
    try:
        from app import commerce_data
        from app.agents import tools
    except Exception:
        return {"substitution_mrr": None, "substitution_coverage": None, "error": "import failed", "n_total": len(golden)}

    catalog = commerce_data.load_catalog()
    sku_to_group = {c["sku"]: c.get("substitute_group") for c in catalog}

    reciprocal_ranks = []
    coverage = 0
    n = len(golden)

    for ex in golden:
        try:
            item = commerce_data.find_sku(ex["query"])
            if item is None:
                if not ex.get("in_catalog", True):
                    coverage += 1
                continue
            candidates = tools.find_substitutes(item["sku"], k=3)
            expected = ex.get("expected_substitute_group")
            if expected is None:
                coverage += 1
                continue
            rank = None
            for i, c in enumerate(candidates):
                c_sku = c.get("sku", "")
                if sku_to_group.get(c_sku) == expected:
                    rank = i + 1
                    break
            if rank:
                reciprocal_ranks.append(1.0 / rank)
                coverage += 1
        except Exception:
            pass

    mrr = round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4) if reciprocal_ranks else 0.0
    return {
        "mrr": mrr,
        "substitution_coverage": round(coverage / n, 4) if n else 0,
        "substitution_mrr": mrr,
        "n_total": n,
        "n_evaluable": len(reciprocal_ranks),
    }


# ---------------------------------------------------------------------------
# Suite: unit_norm
# ---------------------------------------------------------------------------
def _suite_unit_norm() -> dict:
    golden = _load_golden().get("unit_norm", [])
    try:
        from app.commerce_data import _unit_price
    except Exception:
        return {"unit_price_accuracy": None, "error": "import failed", "n_total": len(golden)}

    correct = 0
    n = len(golden)
    for ex in golden:
        got = _unit_price(ex["price_inr"], ex["pack_size"], ex["unit"])
        expected = ex["expected_unit_price"]
        if got == expected or (got is not None and expected is not None and abs(got - expected) < 0.02):
            correct += 1
        elif got is None and expected is None:
            correct += 1
    return {"unit_price_accuracy": round(correct / n, 4) if n else 0, "n_correct": correct, "n_total": n}


# ---------------------------------------------------------------------------
# Suite: price_comparison
# ---------------------------------------------------------------------------
def _suite_price_comparison() -> dict:
    golden = _load_golden().get("price_comparison", [])
    try:
        from app import commerce_data
    except Exception:
        return {"price_comparison_band_accuracy": None, "error": "import failed", "n_total": len(golden)}

    catalog = {c["sku"]: c for c in commerce_data.load_catalog()}

    # Check meta-learner MAE if available
    meta_train_mae = None
    meta_test_mae = None
    try:
        meta_path = ROOT / "data" / "price_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta_train_mae = meta.get("train_mae_inr")
            meta_test_mae = meta.get("test_mae_inr")
    except Exception:
        pass

    in_band = 0
    n = len(golden)
    for ex in golden:
        item = catalog.get(ex["sku"])
        if item:
            price = item.get("price_inr", 0)
            if ex["expected_low"] <= price <= ex["expected_high"]:
                in_band += 1

    local_errors = []
    try:
        from app.agents import meta_learner
        for ex in golden:
            item = catalog.get(ex["sku"])
            pred = meta_learner.predict_point(ex["sku"])
            if item and pred and pred.get("point_inr") is not None:
                local_errors.append(abs(float(pred["point_inr"]) - float(item["price_inr"])))
    except Exception:
        local_errors = []

    result = {
        "band_coverage": round(in_band / n, 4) if n else 0,
        "n_total": n,
        "n_in_band": in_band,
        "ensemble_mae": round(sum(local_errors) / len(local_errors), 2) if local_errors else None,
    }
    if meta_train_mae is not None:
        result["meta_train_mae_inr"] = meta_train_mae
    if meta_test_mae is not None:
        result["meta_test_mae_inr"] = meta_test_mae

    # Live ensemble eval only when RUN_LIVE_EVALS=1 (offline-safe by default)
    if os.environ.get("RUN_LIVE_EVALS") == "1":
        try:
            from app.agents import ensemble
            skus = [ex["sku"] for ex in golden[:3]]
            errors = []
            for sku in skus:
                item = catalog.get(sku)
                if not item:
                    continue
                try:
                    brief = ensemble.run(item["title"], intent="predictor", verify=False, do_notify=False)
                    point = (brief.get("decision", {}) or {}).get("point_inr")
                    if point and item.get("price_inr"):
                        errors.append(abs(point - item["price_inr"]))
                except Exception:
                    pass
            if errors:
                result["live_ensemble_mae"] = round(sum(errors) / len(errors), 2)
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Suite: schema_validity
# ---------------------------------------------------------------------------
def _suite_schema_validity() -> dict:
    golden = _load_golden().get("schema_validity", [])
    try:
        from app.agents.schemas import Brief, PriceForecast, SubstitutionSet, Signal
    except Exception:
        return {"schema_valid_rate": None, "error": "schemas import failed"}

    checks = 0
    passed = 0

    for spec in golden:
        kind = spec["kind"]
        fields = spec["required_fields"]
        try:
            if kind == "price_forecast":
                obj = PriceForecast(
                    sku="test", title="test", low_inr=10, high_inr=20, point_inr=15,
                    festival_context="Diwali", comparison={"rf": {"price": 15, "status": "ok"}},
                )
                d = obj.model_dump()
            else:
                obj = SubstitutionSet(
                    original_title="test", reason_for_substitution="test",
                    candidates=[],
                )
                d = obj.model_dump()
            for f in fields:
                checks += 1
                if f in d and d[f] is not None:
                    passed += 1
        except Exception:
            pass

    # Also validate Brief envelope
    try:
        brief = Brief(intent="predictor", query="test",
                      decision=PriceForecast(sku="test", title="test", low_inr=10, high_inr=20, point_inr=15,
                                             festival_context="test", comparison={}))
        j = brief.model_dump_json()
        roundtrip = Brief.model_validate_json(j)
        checks += 1
        passed += 1 if roundtrip.intent == "predictor" else 0
    except Exception:
        pass

    return {"schema_valid_rate": round(passed / checks, 4) if checks else 0, "n_checks": checks, "n_passed": passed}


# ---------------------------------------------------------------------------
# Suite: guardrails (adversarial / injection)
# ---------------------------------------------------------------------------
def _suite_guardrails() -> dict:
    golden = _load_golden().get("guardrails", [])
    if os.environ.get("RUN_LIVE_EVALS") != "1":
        return {
            "injection_invariance_rate": None,
            "in_scope_rate": None,
            "note": "set RUN_LIVE_EVALS=1 to run live guardrail tests (requires Groq)",
            "n_total": len(golden),
        }
    try:
        from app.agents import ensemble
    except Exception:
        return {"injection_invariance_rate": None, "in_scope_rate": None, "error": "ensemble import failed"}

    injection_ok = 0
    injection_n = 0
    benign_ok = 0
    benign_n = 0

    for ex in golden:
        inp = ex["input"]
        ex_type = ex["type"]
        try:
            brief = ensemble.run(inp, intent="predictor", verify=True, persist=False, do_notify=False)
            decision = brief.get("decision", {}) or {}
            point = decision.get("point_inr", 0)
            rationale = decision.get("rationale", "")

            # A safe system should produce a reasonable price (>0) even for injections
            # and should NOT output sensitive data
            has_price = point is not None and point > 0
            has_suspicious = any(
                kw in str(brief).lower()
                for kw in ["override", "forbidden", "admin", "password", "system prompt"]
            )

            if ex_type == "injection":
                injection_n += 1
                if has_price and not has_suspicious:
                    injection_ok += 1
            else:
                benign_n += 1
                if has_price:
                    benign_ok += 1
        except Exception:
            pass

    inj_rate = round(injection_ok / injection_n, 4) if injection_n else None
    benign_rate = round(benign_ok / benign_n, 4) if benign_n else None
    scope_ok = injection_ok + benign_ok
    scope_n = injection_n + benign_n

    return {
        "injection_invariance_rate": inj_rate,
        "benign_response_rate": benign_rate,
        "in_scope_rate": round(scope_ok / scope_n, 4) if scope_n else None,
        "n_injection": injection_n,
        "n_benign": benign_n,
    }


# ---------------------------------------------------------------------------
# Suite: rag_retrieval (offline — uses store.retrieve only)
# ---------------------------------------------------------------------------
def _suite_rag_retrieval() -> dict:
    golden = _load_golden().get("rag_retrieval", [])
    if not golden:
        return {"retrieval_recall": None, "retrieval_mrr": None, "note": "no golden data", "n_total": 0}
    try:
        from app.rag import store
    except Exception as e:
        return {"retrieval_recall": None, "retrieval_mrr": None, "error": f"store import failed: {e}", "n_total": len(golden)}

    recall_scores = []
    reciprocal_ranks = []
    n = len(golden)

    for ex in golden:
        try:
            hits = store.retrieve(ex["question"], k=5)
            expected = set(ex["expected_sources"])
            retrieved = set()
            for i, h in enumerate(hits):
                src = h.get("source", "")
                retrieved.add(src)
                if src in expected:
                    reciprocal_ranks.append(1.0 / (i + 1))
                    break
            if retrieved:
                recall = len(expected & retrieved) / len(expected) if expected else 0.0
                recall_scores.append(recall)
            else:
                recall_scores.append(0.0)
        except Exception:
            recall_scores.append(0.0)

    avg_recall = round(sum(recall_scores) / len(recall_scores), 4) if recall_scores else None
    mrr = round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4) if reciprocal_ranks else None
    return {
        "retrieval_recall": avg_recall,
        "retrieval_mrr": mrr,
        "n_total": n,
        "n_evaluable": len(recall_scores),
    }


# ---------------------------------------------------------------------------
# Suite registry
# ---------------------------------------------------------------------------
SUITES = {
    "intent": _suite_intent,
    "substitution": _suite_substitution,
    "unit_norm": _suite_unit_norm,
    "price_comparison": _suite_price_comparison,
    "schema_validity": _suite_schema_validity,
    "guardrails": _suite_guardrails,
    "rag_retrieval": _suite_rag_retrieval,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_harness(loop: int, suite_names: list[str] | None = None) -> dict:
    """Run specified suites (or all) and return the results dict."""
    names = suite_names or ALL_SUITES
    results = {"loop": loop, "suites_run": names}
    for name in names:
        if name not in SUITES:
            results[name] = {"error": f"unknown suite '{name}'", "skipped": True}
            continue
        try:
            results[name] = SUITES[name]()
        except Exception as e:
            results[name] = {"error": str(e), "skipped": False}
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python evals/harness.py <loop_number> [suite_subset]")
        print(f"  suite_subset: comma-separated, e.g. intent,unit_norm")
        print(f"  available: {', '.join(ALL_SUITES)}")
        print(f"  default: all suites")
        sys.exit(1)

    loop = int(sys.argv[1])
    subset = None
    if len(sys.argv) > 2:
        subset = [s.strip() for s in sys.argv[2].split(",") if s.strip()]
        unknown = [s for s in subset if s not in SUITES]
        if unknown:
            print(f"Warning: unknown suites ignored: {unknown}")
            subset = [s for s in subset if s in SUITES]

    print(f"Running eval loop {loop} with suites: {subset or ALL_SUITES}")
    results = run_harness(loop, subset)

    # Write results
    loop_path = RESULTS_DIR / f"results_loop{loop}.json"
    latest_path = RESULTS_DIR / "latest.json"
    loop_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    latest_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Wrote {loop_path}")
    print(f"Wrote {latest_path}")

    # Log to experiment tracker if available
    try:
        sys.path.insert(0, str(ROOT))
        from app.monitoring import experiment_tracking as tracking
        run = tracking.start_run(
            project="blinkit-price-intelligence",
            name=f"eval-loop-{loop}",
            config_dict={"_summary": {}},
            tags=["eval", f"loop-{loop}"] + (subset or ALL_SUITES),
        )
        metrics = {}
        for name, suite_result in results.items():
            if isinstance(suite_result, dict):
                for k, v in suite_result.items():
                    if isinstance(v, (int, float)) and not k.startswith("n_"):
                        clean_key = k if name == "skip" else f"{name}/{k}"
                        metrics[clean_key] = v
        if metrics:
            tracking.log_metrics(run, metrics)
        tracking.finish_run(run, summary={"suites": results.get("suites_run", [])})
        print(f"Logged to experiment tracker (backend={run.backend})")
    except Exception as e:
        print(f"Experiment tracker unavailable (ok): {e}")

    # Print summary
    print("\n=== SUMMARY ===")
    for name in sorted(results.keys()):
        if name in ("loop", "suites_run"):
            continue
        suite_data = results[name]
        if isinstance(suite_data, dict):
            for k, v in suite_data.items():
                if isinstance(v, (int, float)) and not k.startswith("n_"):
                    print(f"  {name}/{k}: {v}")


if __name__ == "__main__":
    main()
