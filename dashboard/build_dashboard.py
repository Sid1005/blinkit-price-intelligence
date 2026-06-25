"""Generate the India Commerce SignalForge dashboard (static HTML).

Sources:
  * orchestration/coverage_map.json  — course-concept coverage incl. eval_evidence.
  * evals/results/latest.json        — latest eval metrics.
  * data/runs/*.json                 — offline experiment-tracking runs.

Run:  python dashboard/build_dashboard.py  ->  dashboard/index.html
"""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "orchestration" / "coverage_map.json"
LATEST = ROOT / "evals" / "results" / "latest.json"
RUNS_DIR = ROOT / "data" / "runs"
OUT = Path(__file__).resolve().parent / "index.html"

CSS = """
:root{--bg:#0b1020;--card:#141b2e;--ink:#e6edf6;--mut:#93a1bd;--acc:#5b8cff;--ok:#16a34a;--rb:#d97706}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:Inter,-apple-system,Segoe UI,Roboto,Arial,sans-serif;line-height:1.5}
.wrap{max-width:1080px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:30px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 28px}
h2{font-size:20px;margin:34px 0 12px;border-bottom:1px solid #24304d;padding-bottom:6px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}
.card{background:var(--card);border:1px solid #24304d;border-radius:12px;padding:16px}
.k{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.v{font-size:24px;font-weight:700;margin-top:4px}
.week{background:var(--card);border:1px solid #24304d;border-radius:12px;padding:14px 16px;margin:10px 0}
.week h3{margin:0 0 8px;font-size:16px}
.cpt{display:flex;justify-content:space-between;gap:12px;padding:6px 0;border-top:1px solid #1d2740;font-size:13px}
.cpt:first-of-type{border-top:none}
.badge{font-size:11px;padding:2px 8px;border-radius:999px;white-space:nowrap}
.b-ok{background:rgba(22,163,74,.16);color:#5be584}.b-rb{background:rgba(217,119,6,.16);color:#fbbf24}
.ev{color:var(--mut);font-family:ui-monospace,Menlo,monospace;font-size:11px}
table{width:100%;border-collapse:collapse;font-size:13px}
td,th{text-align:left;padding:6px 8px;border-bottom:1px solid #1d2740}
.pill{font-size:11px;color:var(--mut)}.foot{color:var(--mut);font-size:12px;margin-top:40px}
"""


def _metric_cards(latest: dict) -> str:
    picks = [
        ("RAG faithfulness", latest.get("rag", {}).get("faithfulness")),
        ("RAG context precision", latest.get("rag", {}).get("context_precision")),
        ("Intent accuracy", latest.get("intent", {}).get("intent_accuracy")),
        ("Substitution MRR", latest.get("substitution", {}).get("mrr")),
        ("Triage macro-F1", latest.get("triage", {}).get("type_accuracy")),
        ("Hinglish accuracy", latest.get("hinglish", {}).get("hinglish_accuracy")),
        ("Festival lowers price", latest.get("festival_counterfactual", {}).get("festival_lowers_price_rate")),
        ("Unit-norm accuracy", latest.get("unit_norm", {}).get("unit_price_accuracy")),
        ("Injection invariance", latest.get("adversarial", {}).get("injection_invariance_rate")),
        ("Over-promise guard", latest.get("adversarial", {}).get("over_promise_guard_rate")),
        ("RAG abstention", latest.get("rag_negation", {}).get("abstention_rate")),
        ("Classifier ECE", latest.get("calibration", {}).get("ece")),
        ("Substitution guardrails", latest.get("substitution_guardrails", {}).get("in_scope_rate")),
        ("Schema valid", latest.get("schema_validity", {}).get("schema_valid_rate")),
        ("Price band ordering", latest.get("price_band_sanity", {}).get("band_ordering_rate")),
        ("Tool correctness", latest.get("tool_correctness", {}).get("pass_rate")),
        ("Severity accuracy", latest.get("severity", {}).get("severity_accuracy")),
    ]
    cards = []
    for k, v in picks:
        if v is None:
            continue
        cards.append(f'<div class="card"><div class="k">{html.escape(k)}</div>'
                     f'<div class="v">{v}</div></div>')
    return f'<div class="grid">{"".join(cards)}</div>' if cards else "<p class='pill'>Run evals to populate metrics.</p>"


def _classifier_table(latest: dict) -> str:
    c = latest.get("classifier", {})
    if not c:
        return ""
    rows = []
    for name, m in c.items():
        if not isinstance(m, dict) or "accuracy" not in m:
            continue
        rows.append(f"<tr><td>{html.escape(name)}</td><td>{m.get('accuracy')}</td>"
                    f"<td>{m.get('macro_f1','-')}</td></tr>")
    if not rows:
        return ""
    return ("<h2>Classifier members (golden set)</h2><table><tr><th>member</th>"
            "<th>accuracy</th><th>macro-F1</th></tr>" + "".join(rows) + "</table>")


def _coverage(cov: dict) -> str:
    out = []
    weeks = cov.get("weeks", {})
    total = covered = 0
    for wk, wd in weeks.items():
        rows = []
        for cid, c in wd["concepts"].items():
            total += 1
            status = c.get("status", "covered")
            covered += int(status == "covered")
            badge = "b-ok" if status == "covered" else "b-rb"
            rows.append(
                f'<div class="cpt"><div><b>{html.escape(c["desc"])}</b><br>'
                f'<span class="ev">{html.escape(c.get("evidence",""))}</span><br>'
                f'<span class="ev">eval: {html.escape(c.get("eval_evidence",""))}</span></div>'
                f'<span class="badge {badge}">{status}</span></div>')
        out.append(f'<div class="week"><h3>{html.escape(wd["title"])}</h3>{"".join(rows)}</div>')
    header = f'<p class="sub">Coverage: <b>{covered}/{total}</b> concepts evidenced.</p>'
    return header + "".join(out)


def _runs() -> str:
    files = sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:8]
    if not files:
        return "<p class='pill'>No experiment runs yet.</p>"
    rows = []
    for fp in files:
        try:
            r = json.loads(fp.read_text())
        except Exception:  # noqa: BLE001
            continue
        summ = r.get("config", {}).get("_summary", {})
        s = ", ".join(f"{k}={v}" for k, v in list(summ.items())[:3])
        rows.append(f"<tr><td>{html.escape(r.get('name',''))}</td>"
                    f"<td>{html.escape(r.get('backend',''))}</td>"
                    f"<td>{html.escape(r.get('status',''))}</td>"
                    f"<td class='ev'>{html.escape(s)}</td></tr>")
    return ("<table><tr><th>run</th><th>backend</th><th>status</th><th>summary</th></tr>"
            + "".join(rows) + "</table>")


def build() -> str:
    cov = json.loads(COVERAGE.read_text())
    latest = json.loads(LATEST.read_text()) if LATEST.exists() else {}
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>India Commerce SignalForge — Dashboard</title><style>{CSS}</style></head><body><div class="wrap">
<h1>India Commerce SignalForge</h1>
<p class="sub">One Indian-commerce engine · Deal · Substitute · Triage · prices are INR demo data</p>
<h2>Key metrics (latest eval loop {html.escape(str(latest.get('loop','-')))})</h2>
{_metric_cards(latest)}
{_classifier_table(latest)}
<h2>Experiment tracking</h2>
{_runs()}
<h2>Course-concept coverage (8 weeks)</h2>
{_coverage(cov)}
<p class="foot">Generated by dashboard/build_dashboard.py · Runtime LLM: Groq · Open models/Hub: Hugging Face · Web: Tavily · Tracking: W&B/offline</p>
</div></body></html>"""
    OUT.write_text(page)
    return str(OUT)


if __name__ == "__main__":
    print("wrote", build())
