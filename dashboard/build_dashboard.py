"""Build a static dashboard/index.html summarising the Blinkit Price Intelligence state.

Run:  python dashboard/build_dashboard.py  ->  dashboard/index.html
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CATALOG = DATA / "catalog" / "products.json"
PRICES = DATA / "prices" / "price_observations.jsonl"
META = DATA / "price_meta.json"
EVAL_LATEST = ROOT / "evals" / "results" / "latest.json"
RUNS_DIR = DATA / "runs"
OUT = Path(__file__).resolve().parent / "index.html"

CSS = """
:root{--bg:#0b1020;--card:#141b2e;--ink:#e6edf6;--mut:#93a1bd;--acc:#5b8cff;--ok:#16a34a;--warn:#f59e0b;--off:#94a3b8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:Inter,-apple-system,Segoe UI,Roboto,Arial,sans-serif;line-height:1.6}
.wrap{max-width:960px;margin:0 auto;padding:28px 22px 80px}
h1{font-size:28px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 22px;font-size:14px}
h2{font-size:18px;margin:22px 0 8px;color:var(--acc)}
.row{display:flex;flex-wrap:wrap;gap:12px;margin:10px 0}
.stat{background:var(--card);border:1px solid #24304d;border-radius:10px;padding:14px 18px;min-width:140px;flex:1}
.stat .val{font-size:28px;font-weight:700;color:var(--ink);margin:4px 0}
.stat .lbl{font-size:12px;color:var(--mut)}
table{border-collapse:collapse;width:100%;margin:8px 0;font-size:13px}
th,td{border:1px solid #24304d;padding:7px 10px;text-align:left}
th{background:#141b2e;color:var(--mut);font-weight:600}
td{color:#d7e0ef}
.tag{display:inline-block;border-radius:999px;padding:2px 10px;font-size:11px;font-weight:600}
.tag-ok{background:rgba(22,163,74,.18);color:var(--ok)}
.tag-off{background:rgba(148,163,184,.15);color:var(--off)}
.tag-warn{background:rgba(245,158,11,.15);color:var(--warn)}
.foot{color:var(--mut);font-size:11px;margin-top:36px}
"""

ARM_TABLE = [
    ("Arm 1: RandomForest", "Stacked classical ML (RF + linear meta-learner)", "app/agents/meta_learner.py", "Trained on curated data; 80/20 split with visible train/test MAE"),
    ("Arm 2: LoRA regression", "distilbert-base-uncased fine-tuned on Colab", "lora.ipynb + app/finetune/infer_price_lora.py", "Auto-detected from data/price_lora_adapter/"),
    ("Arm 3: Claude frontier", "Anthropic API zero-shot, no RAG, no training", "app/finetune/frontier_pricer.py", "claude-haiku-4-5; excluded if ANTHROPIC_API_KEY unset"),
    ("Arm 4: Groq + RAG", "Groq grounded in pricing playbook + festival calendar", "app/agents/ensemble.py::_groq_rag_price_estimate", "Retrieves context from Chroma; shows RAG works"),
    ("Arm 5: Ensemble arbitrator", "Groq reasons over all 4 base arms, picks final price", "app/agents/ensemble.py::_groq_ensemble_arbitrate", "Meta-reasoning: weighs strengths/weaknesses of each arm"),
]


def _env_ok(var: str) -> bool:
    val = os.environ.get(var, "")
    return bool(val and len(val) > 4)


def _adapter_ok(dir_name: str) -> bool:
    return (DATA / dir_name / "adapter_config.json").exists()


def build() -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # --- catalog stats ---
    bl_count = sf_count = 0
    categories = {}
    if CATALOG.exists():
        catalog = json.loads(CATALOG.read_text())
        bl_count = sum(1 for x in catalog if x["sku"].startswith("BL-"))
        sf_count = sum(1 for x in catalog if x["sku"].startswith("SF-"))
        for x in catalog:
            cat = x.get("category", "other")
            categories[cat] = categories.get(cat, 0) + 1

    # --- price stats ---
    price_count = 0
    if PRICES.exists():
        price_count = sum(1 for _ in open(PRICES))

    # --- meta-learner stats ---
    meta_html = ""
    if META.exists():
        meta = json.loads(META.read_text())
        meta_html = f"""<div class="stat"><div class="val">{meta.get('train_mae_inr','—')}</div><div class="lbl">Train MAE (INR)</div></div>
<div class="stat"><div class="val">{meta.get('test_mae_inr','—')}</div><div class="lbl">Test MAE (INR, held-out 20%)</div></div>
<div class="stat"><div class="val">{meta.get('n_train','—')}</div><div class="lbl">Train samples</div></div>
<div class="stat"><div class="val">{meta.get('n_test','—')}</div><div class="lbl">Test samples</div></div>"""
    else:
        meta_html = '<div class="stat"><div class="val">—</div><div class="lbl">Meta-learner not trained yet</div></div>'

    # --- eval metrics ---
    eval_rows = ""
    if EVAL_LATEST.exists():
        latest = json.loads(EVAL_LATEST.read_text())
        loop = latest.get("loop", "—")
        eval_rows += f"<tr><td colspan='3' style='color:var(--acc)'>Latest eval: loop {loop}</td></tr>"
        for suite_name, suite_data in sorted(latest.items()):
            if isinstance(suite_data, dict):
                for metric, val in sorted(suite_data.items()):
                    if isinstance(val, (int, float)) and not metric.startswith("n_"):
                        eval_rows += f"<tr><td>{suite_name}</td><td>{metric}</td><td>{val:.4f}</td></tr>"
    else:
        eval_rows = "<tr><td colspan='3'>Run <code>python evals/harness.py 1</code> to populate</td></tr>"

    # --- run summaries ---
    run_rows = ""
    if RUNS_DIR.exists():
        runs = sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
        for rp in runs:
            try:
                r = json.loads(rp.read_text())
                name = r.get("name", rp.stem)
                backend = r.get("backend", "?")
                status = r.get("status", "?")
                summary = r.get("config", {}).get("_summary", {})
                summ_str = ", ".join(f"{k}={v}" for k, v in list(summary.items())[:3])
                run_rows += f"<tr><td>{name}</td><td>{backend}</td><td>{status}</td><td>{summ_str}</td></tr>"
            except Exception:
                pass
    if not run_rows:
        run_rows = "<tr><td colspan='4'>No experiment runs recorded yet</td></tr>"

    # --- provider status ---
    providers = [
        ("Groq", _env_ok("GROQ_API_KEY")),
        ("Claude API", _env_ok("ANTHROPIC_API_KEY")),
        ("Tavily", _env_ok("TAVILY_API_KEY")),
        ("Gemini", _env_ok("GEMINI_API_KEY")),
        ("HuggingFace", _env_ok("HF_TOKEN") or _env_ok("HUGGINGFACE_TOKEN")),
        ("W&B", _env_ok("WANDB_API_KEY")),
        ("Price LoRA adapter", _adapter_ok("price_lora_adapter")),
        ("Signal LoRA adapter", _adapter_ok("lora_adapter")),
    ]
    prov_tags = " ".join(
        f'<span class="tag tag-{"ok" if ok else "off"}">{name}: {"ok" if ok else "offline"}</span>'
        for name, ok in providers
    )

    # --- arm table rows ---
    arm_rows = "".join(
        f"<tr><td>{a[0]}</td><td>{a[1]}</td><td><code>{a[2]}</code></td><td>{a[3]}</td></tr>"
        for a in ARM_TABLE
    )

    # --- category rows ---
    cat_rows = "".join(
        f"<tr><td>{cat}</td><td>{n}</td></tr>"
        for cat, n in sorted(categories.items(), key=lambda x: -x[1])
    )

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blinkit Price Intelligence — Dashboard</title><style>{CSS}</style></head><body><div class="wrap">
<h1>Blinkit Price Intelligence</h1>
<p class="sub">Dashboard snapshot · generated {now}</p>

<h2>Providers</h2>
<div>{prov_tags}</div>

<h2>Catalog</h2>
<div class="row">
<div class="stat"><div class="val">{len(catalog) if CATALOG.exists() else '—'}</div><div class="lbl">Total SKUs</div></div>
<div class="stat"><div class="val">{bl_count}</div><div class="lbl">BL-* Blinkit (real scraped)</div></div>
<div class="stat"><div class="val">{sf_count}</div><div class="lbl">SF-* Synthetic (demo expansion)</div></div>
<div class="stat"><div class="val">{price_count}</div><div class="lbl">Price observations</div></div>
</div>
<table><tr><th>Category</th><th>Count</th></tr>{cat_rows}</table>

<h2>Meta-learner (Arm 1 — RandomForest stack)</h2>
<div class="row">{meta_html}</div>

<h2>5-arm model comparison</h2>
<table><tr><th>Arm</th><th>Method</th><th>Source</th><th>Notes</th></tr>{arm_rows}</table>

<h2>Latest eval metrics</h2>
<table><tr><th>Suite</th><th>Metric</th><th>Value</th></tr>{eval_rows}</table>

<h2>Recent experiment runs</h2>
<table><tr><th>Run</th><th>Backend</th><th>Status</th><th>Summary</th></tr>{run_rows}</table>

<p class="foot">BL-* = real-scraped/curated Blinkit products &middot; SF-* = synthetic demo expansion &middot;
Prices are demo data, not live quotes &middot; Run <code>python dashboard/build_dashboard.py</code> to refresh</p>
</div></body></html>"""
    OUT.write_text(html)
    return str(OUT)


if __name__ == "__main__":
    print("wrote", build())
