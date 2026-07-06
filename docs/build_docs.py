"""Generate the Blinkit Price Intelligence explainer doc (static HTML).

Reads orchestration/coverage_map.json and generates docs/index.html with
architecture flow and course-week mapping. Two decision surfaces only.
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
              ┌─────────────────────────────┤
              ▼                              ▼
        PricePredictor              SubstituteFinder
    (5-arm comparison:        (catalog ranking + unit
     RF + LoRA + Claude       value + availability
     + Groq/RAG + ensemble        + stock check)
     arbitrator)
              └─────────────────────────────┘
                            ▼
                   Brief + Memory + UI"""

WEEK_NOTES = {
    "week1": "Groq powers every runtime LLM call. Tavily scrapes Blinkit/Amazon-style "
             "evidence; pages become markdown briefs and strict JSON SKU/price signals.",
    "week2": "Groq fast/strong/OSS router, a four-tab Gradio cockpit, streaming chat, "
             "callable tools, product-screenshot vision, and deal-card + TTS generation hooks.",
    "week3": "HF pipelines do zero-shot signals, Hinglish aspect-sentiment, and brand NER; "
             "sentence-transformers embed the KB; Groq Whisper transcribes voice queries.",
    "week4": "Deterministic parser candidates benchmarked against gold listings; "
             "unit-normalization kernel timed head-to-head across kg/g/l/ml. "
             "Optional Groq-synthesised codegen if key present.",
    "week5": "Chroma + LangChain over festival/pricing/policy/substitution docs and the catalog, "
             "with hybrid reranking, query rewriting, and retrieval evals (recall, MRR).",
    "week6": "Curated Indian datasets with unit-normalization features, "
             "random/constant/rule/sklearn baselines, a RandomForest+linear meta-learner "
             "with train/test MAE visibility, and W&B/offline experiment tracking.",
    "week7": "LoRA/PEFT fine-tunes bert-tiny on Hinglish commerce labels; "
             "LoRA price regression on distilbert-base via Colab notebook; "
             "adapter + dataset publish to the HF Hub; QLoRA 4-bit CUDA runbook.",
    "week8": "The agentic spine emits a typed Brief with two decision variants, blends a "
             "trained meta-learner ensemble, remembers prior briefs in SQLite, "
             "resists prompt injection, and deploys as a Gradio app.",
}


def build() -> str:
    cov = json.loads(COVERAGE.read_text())
    weeks = cov.get("weeks", {})
    sections = []
    for wk in sorted(weeks.keys()):
        wd = weeks[wk]
        tags = "".join(f'<span class="tag">{html.escape(c["desc"])}</span>'
                       for c in wd["concepts"].values())
        note = WEEK_NOTES.get(wk, "")
        proof = wd.get("proof_commands", [])
        proof_html = ""
        if proof:
            proof_lines = "".join(f"<li><code>{html.escape(c)}</code></li>" for c in proof[:3])
            proof_html = f'<p style="margin-top:8px;font-size:13px;color:var(--mut)">Proof: <ul style="margin:4px 0 0 18px;font-size:12px">{proof_lines}</ul></p>'
        sections.append(
            f'<div class="card"><h3>{html.escape(wd["title"])}</h3>'
            f'<p>{html.escape(note)}</p>{tags}{proof_html}</div>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blinkit Price Intelligence — How it works</title><style>{CSS}</style></head><body><div class="wrap">
<h1>Blinkit Price Intelligence</h1>
<p class="sub">Festival-aware Indian quick-commerce price prediction — two decision surfaces, mapped to all 8 course weeks.</p>
<div class="card">
<p><b>The product.</b> Indian shoppers ask two recurring questions: <i>is this a good price right
now?</i> and <i>what should I buy instead?</i>
Blinkit Price Intelligence answers both from one agent spine, with festival-aware INR pricing
and catalog-grounded substitution. Review understanding is shared Hinglish NLP enrichment.</p>
<p><b>Scope &amp; safety.</b> Catalog prices combine real scraped Blinkit products (BL-* SKUs)
with curated synthetic SKUs (SF-*), clearly labelled — this project does <b>not</b> claim live
quotes. Festival context is auto-detected from the current month, never parsed from the user's query.</p>
</div>
<h2>Architecture</h2>
<div class="flow">{html.escape(FLOW)}</div>
<h2>How each course week shows up</h2>
{''.join(sections)}
<p class="foot">Runtime LLM: Groq · Frontier pricing: Anthropic/Claude · Open models/Hub: Hugging Face ·
Web evidence: Tavily · Tracking: W&amp;B/offline · Translation: Gemini/Groq · See RUNBOOK.md for full guide.</p>
</div></body></html>"""
    OUT.write_text(page)
    return str(OUT)


if __name__ == "__main__":
    print("wrote", build())
