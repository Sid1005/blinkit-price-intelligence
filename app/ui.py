"""Blinkit Price Intelligence — Gradio UI.

Tabs:
  Predictor  — festival-aware INR price with 5-arm model comparison.
  Substitute — ranked alternatives table (+ vision for product images).
  Translate  — Hinglish / regional query translation (Gemini Live + fallback).
  Analytics  — eval metrics and experiment-tracking runs.

Launch:  python -m app.ui
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr

from app import config, i18n
from app.agents import ensemble, memory
from app.finetune import infer_price_lora
from app.llm import groq_client
from app.monitoring import experiment_tracking as tracking
from app.rag import store

SYM = config.CURRENCY_SYMBOL
LATEST_EVAL = Path(__file__).resolve().parents[1] / "evals" / "results" / "latest.json"

ARM_LABELS = {
    "random_forest": "RandomForest (classical ML)",
    "lora_regression": "LoRA regression (fine-tuned)",
    "claude_frontier": "Claude frontier (zero-shot)",
    "groq_rag": "Groq + RAG",
    "ensemble_arbitrator": "Groq ensemble arbitrator",
}

CSS = """
.gradio-container { max-width: 1100px !important }
.provider-row { font-size: 13px; color: #555; margin-bottom: 8px; }
.provider-row span { margin-right: 12px; }
.ok  { color: #16a34a; font-weight: 600; }
.off { color: #94a3b8; }
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _provider_item(name: str, ok: bool) -> str:
    status = "ok" if ok else "offline"
    cls = "ok" if ok else "off"
    return f'<span class="{cls}">{name}: {status}</span>'


def provider_status() -> str:
    items = [
        ("Groq", bool(config.GROQ_API_KEY)),
        ("Claude API", bool(config.ANTHROPIC_API_KEY)),
        ("Tavily", bool(config.TAVILY_API_KEY)),
        ("Gemini", bool(config.GEMINI_API_KEY)),
        ("HuggingFace", bool(config.HF_TOKEN)),
        ("Price LoRA", infer_price_lora.adapter_exists()),
    ]
    row = " ".join(_provider_item(n, ok) for n, ok in items)
    return f'<div class="provider-row">{row}</div>'


def _format_rag_context(rag_ctx: dict | None) -> str:
    if not rag_ctx:
        return "_RAG context not available._"
    lines = ["### RAG Retrieval", ""]
    lines.append(f"**Question:** {rag_ctx.get('question', '—')}")
    lines.append("")
    lines.append(f"**Answer:** {rag_ctx.get('answer', '—')}")
    lines.append("")
    lines.append("**Retrieved documents:**")
    for ctx in rag_ctx.get("contexts", []):
        src = ctx.get("source", "unknown")
        text = ctx.get("text", "")[:280]
        lines.append(f"- **[{src}]** {text}")
    lines.append("")
    lines.append(f"_Sources: {', '.join(rag_ctx.get('sources', []))}_")
    return "\n".join(lines)


def _extract_from_media(image=None, audio=None):
    """Augment query with product details from optional image/audio. Returns (query_augment, notes_md)."""
    parts = []
    notes = []
    if image:
        try:
            result = groq_client.vision(
                "Identify the product, brand, and price in INR if visible. Be concise.",
                image_path=image,
            )
            notes.append(f"**Image extraction:** {result[:200]}")
            parts.append(result[:150])
        except Exception as e:
            notes.append(f"**Image extraction failed:** {e}")
    if audio:
        try:
            result = groq_client.transcribe(audio_path=audio)
            notes.append(f"**Audio transcription:** {result[:200]}")
            parts.append(result[:150])
        except Exception as e:
            notes.append(f"**Audio transcription failed:** {e}")
    aug = " [" + "; ".join(parts) + "]" if parts else ""
    md = "\n\n".join(notes) if notes else ""
    return aug, md


# ---------------------------------------------------------------------------
# Predictor tab
# ---------------------------------------------------------------------------

def _comparison_rows(comparison: dict) -> list[list]:
    rows = []
    for key, label in ARM_LABELS.items():
        arm = comparison.get(key) or {}
        price = arm.get("price")
        if price is None:
            note = "run Colab training" if key == "lora_regression" else "not available"
            rows.append([label, "—", note])
        else:
            rows.append([label, f"{SYM}{price:g}", "ok"])
    return rows


def predictor_fn(query, month, image=None, audio=None):
    aug, notes = _extract_from_media(image, audio)
    full_query = query + aug if aug else query
    month = int(month) if month else None
    brief = ensemble.run(full_query, intent="predictor", current_month=month)
    d = brief["decision"]

    summary = (
        f"**Product:** {d['title'] or full_query}\n\n"
        f"**Price band:** {SYM}{d['low_inr']:g} – {SYM}{d['high_inr']:g} "
        f"| point estimate: **{SYM}{d['point_inr']:g}**"
        + (f" | unit price: {SYM}{d['unit_price_inr']:g}/{d['unit']}" if d.get("unit_price_inr") else "")
        + f"\n\n**Festival:** {d['festival_context']}"
        + f"\n\n**Method:** {d['estimator']}"
        + f"\n\n{d['rationale']}"
    )

    rows = _comparison_rows(d.get("comparison", {}))
    rag_display = _format_rag_context(d.get("rag_context"))
    return summary, rows, rag_display, json.dumps(brief, indent=2, ensure_ascii=False), notes


# ---------------------------------------------------------------------------
# Substitute tab
# ---------------------------------------------------------------------------

def substitute_fn(query, image=None, audio=None):
    aug, notes = _extract_from_media(image, audio)
    full_query = query + aug if aug else query
    brief = ensemble.run(full_query, intent="substitute")
    d = brief["decision"]
    rows = [
        [
            c["title"], c["sku"], c["platform"],
            f"{SYM}{c['price_inr']:g}",
            (f"{SYM}{c['unit_price_inr']:g}" if c.get("unit_price_inr") else "—"),
            "yes" if c["in_stock"] else "no",
            c["rating"],
            f"{c['score']:.2f}",
            c["reason"],
        ]
        for c in d["candidates"]
    ]
    vi = d.get("value_improvement_pct")
    summary = (
        f"**Substitutes for:** {d['original_title'] or full_query}\n\n"
        f"**Reason:** {d['reason_for_substitution']}"
        + (f"\n\n**Value improvement:** {vi}%" if vi is not None else "")
    )
    return summary, rows, json.dumps(brief, indent=2, ensure_ascii=False), notes


def vision_fn(image, question):
    if image is None:
        return "Upload a product screenshot or receipt first."
    prompt = (
        question or
        "Identify the product, brand, price in INR if visible, and any visible defect. Be concise."
    )
    try:
        return groq_client.vision(prompt, image_path=image)
    except Exception as e:  # noqa: BLE001
        return f"Vision call failed: {e}"


# ---------------------------------------------------------------------------
# Translate
# ---------------------------------------------------------------------------

def translate_fn(text, target, use_live):
    if not text or not text.strip():
        yield "Enter text to translate."
        return
    if use_live:
        from app import gemini_live
        res = gemini_live.live_translate(text, target=target)
        yield f"{res['text']}\n\n_provider: {res['provider']}_"
        return
    acc = ""
    for piece in i18n.translate_stream(text, target=target):
        acc += piece
        yield acc


# ---------------------------------------------------------------------------
# Analytics tab
# ---------------------------------------------------------------------------

_HEADLINE = [
    ("Intent accuracy",         ("intent",             "intent_accuracy")),
    ("Substitution MRR",        ("substitution",       "substitution_mrr")),
    ("Ensemble MAE",            ("price_comparison",   "ensemble_mae")),
    ("Unit price accuracy",     ("unit_norm",          "unit_price_accuracy")),
    ("Schema validity",         ("schema_validity",    "schema_valid_rate")),
    ("Injection invariance",    ("guardrails",         "injection_invariance_rate")),
    ("Guardrail in-scope",      ("guardrails",         "in_scope_rate")),
    ("RAG retrieval recall",    ("rag_retrieval",      "retrieval_recall")),
    ("RAG retrieval MRR",       ("rag_retrieval",      "retrieval_mrr")),
]


def _latest_eval() -> dict:
    try:
        return json.loads(LATEST_EVAL.read_text())
    except Exception:  # noqa: BLE001
        return {}


def analytics_metrics_md() -> str:
    latest = _latest_eval()
    if not latest:
        return "Run `python evals/harness.py 1` to populate metrics."
    loop = latest.get("loop", "—")
    lines = [f"### Eval metrics (loop {loop})", ""]
    lines.append("| Metric | Score |")
    lines.append("|--------|-------|")
    for label, (suite, key) in _HEADLINE:
        v = (latest.get(suite, {}) or {}).get(key)
        if v is not None:
            lines.append(f"| {label} | {v} |")
    return "\n".join(lines)


def analytics_plot_df():
    import pandas as pd
    latest = _latest_eval()
    data = []
    for label, (suite, key) in _HEADLINE:
        v = (latest.get(suite, {}) or {}).get(key)
        if isinstance(v, (int, float)):
            data.append({"metric": label, "score": float(v)})
    return pd.DataFrame(data or [{"metric": "n/a", "score": 0.0}])


def analytics_runs_md() -> str:
    mem = memory.stats()
    runs = tracking.load_runs(limit=5)
    lines = [
        f"**Memory:** {mem['total']} briefs — "
        + ", ".join(f"{k}={v}" for k, v in mem["by_intent"].items()),
        "",
        "**Recent experiment runs:**",
    ]
    for r in runs:
        summ = r.get("config", {}).get("_summary", {})
        lines.append(
            f"- `{r['name']}` ({r['backend']}, {r['status']}) "
            + ", ".join(f"{k}={v}" for k, v in list(summ.items())[:3])
        )
    return "\n".join(lines)


def refresh_analytics():
    return analytics_metrics_md(), analytics_plot_df(), analytics_runs_md()


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

def build_demo():
    with gr.Blocks(title="Blinkit Price Intelligence", css=CSS) as demo:
        gr.Markdown("## Blinkit Price Intelligence")
        gr.Markdown(
            "Festival-aware INR price prediction — compare model approaches "
            "side-by-side. Prices are INR demo data, not live quotes."
        )
        gr.HTML(provider_status())

        # ── Predictor ──────────────────────────────────────────────────────
        with gr.Tab("Predictor"):
            with gr.Row():
                with gr.Column(scale=3):
                    q = gr.Textbox(
                        label="Product / query",
                        value="Cadbury Dairy Milk Silk ka price kya hoga?",
                    )
                with gr.Column(scale=1):
                    month = gr.Dropdown(
                        choices=[("auto", "")] + [(str(m), m) for m in range(1, 13)],
                        value="",
                        label="Month override (festival)",
                    )
            gr.Examples(
                [
                    ["Diwali pe Snickers ka price kitna hoga?", 11],
                    ["Cadbury Bournville fair price kya hai?", ""],
                    ["KitKat Big Billion Days pe kitne ka milega?", 10],
                ],
                [q, month],
                label="Examples",
            )
            with gr.Accordion("Add image or audio", open=False):
                with gr.Row():
                    pred_img = gr.Image(label="Product screenshot", type="filepath")
                    pred_audio = gr.Audio(label="Voice query", type="filepath")
            pred_notes = gr.Markdown()
            predict_btn = gr.Button("Predict price", variant="primary")

            with gr.Row():
                pred_summary = gr.Markdown(label="Result")

            pred_table = gr.Dataframe(
                headers=["Arm", "Predicted price", "Status"],
                datatype=["str", "str", "str"],
                interactive=False,
                wrap=True,
                label="5-arm comparison",
            )

            with gr.Accordion("RAG — retrieved context", open=False):
                rag_display = gr.Markdown()

            with gr.Accordion("Full brief (JSON)", open=False):
                pred_json = gr.Code(language="json")

            predict_btn.click(
                predictor_fn,
                [q, month, pred_img, pred_audio],
                [pred_summary, pred_table, rag_display, pred_json, pred_notes],
            )

        # ── Substitute ─────────────────────────────────────────────────────
        with gr.Tab("Substitute"):
            sq = gr.Textbox(
                label="Product to replace",
                value="Snickers is out of stock, alternative?",
            )
            gr.Examples(
                [
                    ["Cadbury Dairy Milk Silk out of stock, alternative?"],
                    ["KitKat cheaper option?"],
                    ["Amul Dark Chocolate better rated alternative"],
                ],
                [sq],
                label="Examples",
            )
            with gr.Accordion("Add image or audio", open=False):
                with gr.Row():
                    sub_img = gr.Image(label="Product screenshot", type="filepath")
                    sub_audio = gr.Audio(label="Voice query", type="filepath")
            sub_notes = gr.Markdown()
            sub_btn = gr.Button("Find substitutes", variant="primary")
            sub_summary = gr.Markdown()
            sub_table = gr.Dataframe(
                headers=["Title", "SKU", "Platform", "Price", "Unit price",
                         "In stock", "Rating", "Score", "Reason"],
                datatype=["str"] * 9,
                interactive=False,
                wrap=True,
                label="Ranked substitutes",
            )
            with gr.Accordion("Full brief (JSON)", open=False):
                sub_json = gr.Code(language="json")
            sub_btn.click(substitute_fn, [sq, sub_img, sub_audio], [sub_summary, sub_table, sub_json, sub_notes])

            with gr.Accordion("Analyze product image (vision)", open=False):
                img = gr.Image(label="Product screenshot / receipt", type="filepath")
                vq = gr.Textbox(
                    label="Question about the image",
                    value="What product and price is shown?",
                )
                v_out = gr.Markdown()
                gr.Button("Analyze").click(vision_fn, [img, vq], v_out)

        # ── Translate ──────────────────────────────────────────────────────
        with gr.Tab("Translate"):
            gr.Markdown(
                "Translate Hinglish or regional product queries to English (or other languages). "
                "Uses Gemini streaming; enable Live API if your project supports it."
            )
            with gr.Row():
                tin = gr.Textbox(
                    label="Input text",
                    lines=3,
                    value="Delivery boy ne 100 rupaye extra liye, product bhi damaged tha",
                )
                tout = gr.Markdown(label="Translation")
            with gr.Row():
                tgt = gr.Dropdown(i18n.LANGUAGES, value="English", label="Target language")
                use_live = gr.Checkbox(label="Use realtime Gemini Live API", value=False)
            gr.Button("Translate", variant="primary").click(
                translate_fn, [tin, tgt, use_live], tout
            )

        # ── Analytics ──────────────────────────────────────────────────────
        with gr.Tab("Analytics"):
            refresh_btn = gr.Button("Refresh", variant="secondary")
            metrics_md = gr.Markdown(analytics_metrics_md())
            try:
                plot = gr.BarPlot(
                    analytics_plot_df(),
                    x="metric",
                    y="score",
                    title="Eval scores",
                    height=280,
                    y_lim=[0, 1],
                )
            except Exception:  # noqa: BLE001
                plot = gr.Markdown("_Bar plot unavailable in this Gradio version._")
            runs_md = gr.Markdown(analytics_runs_md())
            refresh_btn.click(refresh_analytics, None, [metrics_md, plot, runs_md])

    return demo


if __name__ == "__main__":
    store.build_index()
    build_demo().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        theme=gr.themes.Default(),
    )
