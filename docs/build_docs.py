"""Generate the India Commerce SignalForge explainer doc (static HTML).

Maps the Indian-commerce product back to all 8 course weeks in the minimalist design.
Run:  python docs/build_docs.py  ->  docs/index.html
"""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "orchestration" / "coverage_map.json"
OUT = Path(__file__).resolve().parent / "index.html"

CSS = """
:root{--bg:#0b1020;--card:#141b2e;--ink:#e6edf6;--mut:#93a1bd;--acc:#5b8cff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:Inter,-apple-system,Segoe UI,Roboto,Arial,sans-serif;line-height:1.6}
.wrap{max-width:900px;margin:0 auto;padding:36px 22px 90px}
h1{font-size:32px;margin:0 0 6px}.sub{color:var(--mut);margin:0 0 26px}
h2{font-size:21px;margin:32px 0 10px}h3{font-size:16px;margin:18px 0 6px;color:var(--acc)}
p,li{color:#d7e0ef}code{background:#1d2740;padding:1px 6px;border-radius:6px;font-size:13px}
.card{background:var(--card);border:1px solid #24304d;border-radius:12px;padding:18px 20px;margin:14px 0}
.flow{font-family:ui-monospace,Menlo,monospace;background:#0e1426;border:1px solid #24304d;
border-radius:10px;padding:14px;color:#aebede;white-space:pre;overflow:auto;font-size:13px}
.tag{display:inline-block;background:rgba(91,140,255,.16);color:#9db8ff;border-radius:999px;
padding:2px 10px;font-size:12px;margin:0 6px 6px 0}
.foot{color:var(--mut);font-size:12px;margin-top:44px}
"""

FLOW = """scout ──▶ classify ──▶ verify ──▶ IntentRouter
                                            │
              ┌─────────────────────────────┼─────────────────────────────┐
              ▼                              ▼                              ▼
       DealPredictor              SubstitutionRanker               ComplaintTriage
   (meta-learner + Groq        (catalog ranking + unit          (type + policy-grounded
    + RAG + festival)            value + availability)            resolution, escalate)
              └─────────────────────────────┼─────────────────────────────┘
                                            ▼
                                   Brief + Memory + UI"""

WEEK_NOTES = {
    "week1": "Groq powers every runtime LLM call. Tavily scrapes Blinkit/Amazon-style "
             "evidence; pages become markdown briefs and strict JSON SKU/price signals.",
    "week2": "Groq fast/strong/OSS router, a six-tab Gradio cockpit, streaming chat, "
             "five callable tools, product-screenshot vision, and deal-card + TTS generation hooks.",
    "week3": "HF pipelines do zero-shot signals, Hinglish aspect-sentiment, and brand NER; "
             "sentence-transformers embed the KB; Groq Whisper transcribes voice complaints.",
    "week4": "Groq synthesizes product-listing parsers (scored vs a hand-written baseline) and a "
             "C++ unit-normalization kernel that is compiled and timed against Python.",
    "week5": "LangChain + Chroma over festival/pricing/policy/substitution docs and the catalog, "
             "with hybrid reranking, query rewriting, and full RAG evals (MRR, nDCG, faithfulness).",
    "week6": "Curated Indian datasets with dedup/leakage checks, unit-normalization features, "
             "random/constant/rule/sklearn/PyTorch baselines, a RandomForest+linear meta-learner, "
             "and W&B/offline experiment tracking. Frontier FT is structurally complete (runbook).",
    "week7": "LoRA/PEFT fine-tunes bert-tiny on Hinglish commerce labels, evaluated against Groq/"
             "sklearn/PyTorch; adapter + dataset publish to the HF Hub; QLoRA 4-bit is a CUDA runbook.",
    "week8": "The agentic spine emits a typed Brief with one of three decision variants, blends a "
             "trained meta-learner ensemble, remembers prior briefs, notifies on high-confidence "
             "deals/escalations, resists prompt injection, and deploys as a Gradio app.",
}


def build() -> str:
    cov = json.loads(COVERAGE.read_text())
    weeks = cov.get("weeks", {})
    sections = []
    for wk, wd in weeks.items():
        tags = "".join(f'<span class="tag">{html.escape(c["desc"])}</span>'
                       for c in wd["concepts"].values())
        note = WEEK_NOTES.get(wk, "")
        sections.append(
            f'<div class="card"><h3>{html.escape(wd["title"])}</h3>'
            f'<p>{html.escape(note)}</p>{tags}</div>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>India Commerce SignalForge — How it works</title><style>{CSS}</style></head><body><div class="wrap">
<h1>India Commerce SignalForge</h1>
<p class="sub">A single Indian-commerce engine with three decision surfaces, mapping to all 8 course weeks.</p>
<div class="card">
<p><b>The product.</b> Indian shoppers ask three recurring questions: <i>is this a good price right
now?</i>, <i>what should I buy instead?</i>, and <i>how do I resolve this order complaint?</i>
SignalForge answers all three from one agent spine, with festival-aware INR pricing, catalog-grounded
substitution, and policy-grounded complaint triage. Review understanding is shared Hinglish NLP
enrichment, not a separate app.</p>
<p><b>Scope &amp; safety.</b> Prices are curated demo data, clearly labelled — we do not claim live
Blinkit/Amazon quotes. Complaint triage cites policy, never issues binding refunds, and marks uncertain
cases as requiring human confirmation.</p>
</div>
<h2>Architecture</h2>
<div class="flow">{html.escape(FLOW)}</div>
<h2>How each course week shows up</h2>
{''.join(sections)}
<p class="foot">Runtime LLM: Groq · Open models/Hub: Hugging Face · Web evidence: Tavily ·
Tracking: W&amp;B/offline · Translate: Gemini/Groq · See RUNBOOK.md for external steps.</p>
</div></body></html>"""
    OUT.write_text(page)
    return str(OUT)


if __name__ == "__main__":
    print("wrote", build())
