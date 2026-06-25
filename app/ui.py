"""India Commerce SignalForge — Gradio cockpit (weeks 2 & 8 deployment surface).

Tabs:
  * Deal       — festival-aware INR price band + buy/wait/avoid (badge, band bar, card).
  * Substitute — ranked alternatives in a table (+ product-screenshot vision).
  * Triage     — complaint classification + policy-grounded resolution (+ voice note).
  * RAG        — ask the grounded knowledge base.
  * Translate  — Gemini live/streaming translation (+ realtime Live API toggle).
  * Chat       — streaming Groq chat with model switching.
  * Analytics  — eval metrics chart + memory stats + experiment-tracking runs.

Launch:  python -m app.ui     (deploy to HF Spaces / Modal per RUNBOOK.md)
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from app import config, i18n
from app.agents import ensemble, memory
from app.finetune import infer_lora
from app.llm import groq_client
from app.media import generate as media
from app.monitoring import experiment_tracking as tracking
from app.rag import rag, store

MODEL_CHOICES = list(config.GROQ_MODELS.values())
SYM = config.CURRENCY_SYMBOL
LATEST_EVAL = Path(__file__).resolve().parents[1] / "evals" / "results" / "latest.json"

REC_COLOR = {"buy_now": "#16a34a", "wait": "#d97706", "avoid": "#dc2626"}
SEV_COLOR = {"high": "#dc2626", "medium": "#d97706", "low": "#16a34a"}

CSS = """
.gradio-container{max-width:1180px !important}
#sf-header{background:linear-gradient(135deg,#0f172a,#1e293b);border-radius:16px;
padding:22px 26px;margin-bottom:10px;border:1px solid #24304d}
#sf-header h1{color:#f8fafc;margin:0 0 4px;font-size:28px}
#sf-header p{color:#93a1bd;margin:0}
.sf-badge{display:inline-block;padding:3px 11px;border-radius:999px;font-size:12px;
font-weight:600;margin:2px 4px 2px 0;color:#fff}
.sf-pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;
margin:2px 6px 2px 0;background:#eef2ff;color:#3730a3;border:1px solid #c7d2fe}
.sf-metric{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:12px 14px;
text-align:center;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.sf-metric .k{color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.sf-metric .v{font-size:22px;font-weight:800;color:#0f172a;margin-top:2px}
.sf-band{position:relative;height:38px;background:linear-gradient(90deg,#bbf7d0,#fde68a,#fecaca);
border-radius:8px;margin:6px 0 2px}
.sf-band .pt{position:absolute;top:-4px;width:3px;height:46px;background:#0f172a}
.sf-band .lbl{position:absolute;top:42px;font-size:11px;color:#475569;transform:translateX(-50%)}
"""


# --- helpers ---------------------------------------------------------------------
def _badge(text: str, color: str) -> str:
    return f'<span class="sf-badge" style="background:{color}">{text}</span>'


def provider_status() -> str:
    items = [
        ("Groq", bool(config.GROQ_API_KEY)),
        ("Tavily", bool(config.TAVILY_API_KEY)),
        ("Gemini", bool(config.GEMINI_API_KEY)),
        ("HuggingFace", bool(config.HF_TOKEN)),
        ("LoRA adapter", infer_lora.adapter_exists()),
    ]
    out = []
    for name, ok in items:
        out.append(_badge(f"{name} {'✓' if ok else '×'}", "#16a34a" if ok else "#94a3b8"))
    return " ".join(out)


def _band_bar(low: float, point: float, high: float) -> str:
    if high <= low:
        return ""
    pct = max(0.0, min(100.0, 100.0 * (point - low) / (high - low)))
    return (f'<div class="sf-band"><div class="pt" style="left:{pct:.1f}%"></div>'
            f'<div class="lbl" style="left:0">{SYM}{low:g}</div>'
            f'<div class="lbl" style="left:{pct:.1f}%">{SYM}{point:g}</div>'
            f'<div class="lbl" style="left:100%">{SYM}{high:g}</div></div>')


# --- Deal ------------------------------------------------------------------------
def deal_fn(query, month, make_card):
    month = int(month) if month else None
    brief = ensemble.run(query, intent="deal", current_month=month)
    d = brief["decision"]
    rec = d["recommendation"]
    badges = (_badge(rec.replace("_", " ").upper(), REC_COLOR.get(rec, "#2563eb"))
              + _badge(d["festival_context"], "#6366f1")
              + _badge(f"trend: {brief['trend_strength']}", "#0ea5e9"))
    band = _band_bar(d["low_inr"], d["point_inr"], d["high_inr"])
    md = (f"### {d['title'] or query}\n"
          f"**Fair price band:** {SYM}{d['low_inr']:g} – {SYM}{d['high_inr']:g} "
          f"· point **{SYM}{d['point_inr']:g}**"
          + (f" · unit {SYM}{d['unit_price_inr']:g}/{d['unit']}" if d.get('unit_price_inr') else "")
          + f"\n\n**Estimator:** `{d['estimator']}`\n\n{d['rationale']}")
    card_path = media.generate_deal_card(brief, use_hf=False)["path"] if make_card else None
    return badges + band, md, card_path, json.dumps(brief, indent=2, ensure_ascii=False)


# --- Substitute ------------------------------------------------------------------
def substitute_fn(query):
    brief = ensemble.run(query, intent="substitute")
    d = brief["decision"]
    rows = [[c["title"], c["sku"], c["platform"], f"{SYM}{c['price_inr']:g}",
             (f"{SYM}{c['unit_price_inr']:g}" if c.get("unit_price_inr") else "—"),
             "yes" if c["in_stock"] else "no", c["rating"], f"{c['score']:.2f}", c["reason"]]
            for c in d["candidates"]]
    vi = d.get("value_improvement_pct")
    summary = (f"### Substitutes for {d['original_title'] or query}\n"
               f"_Reason: {d['reason_for_substitution']}_"
               + (f"\n\n**Top value change vs original:** {vi}%" if vi is not None else ""))
    return summary, rows, json.dumps(brief, indent=2, ensure_ascii=False)


def vision_fn(image, question):
    if image is None:
        return "_Upload a product screenshot / receipt / damaged-item photo first._"
    prompt = (question or "Identify the product, brand, price in INR if visible, and any "
              "visible defect. Be concise.")
    try:
        return groq_client.vision(prompt, image_path=image)
    except Exception as e:  # noqa: BLE001
        return f"Vision call failed: {e}"


# --- Triage ----------------------------------------------------------------------
def triage_fn(text, translate_first):
    if translate_first and text:
        text = i18n.to_english(text)
    brief = ensemble.run(text, intent="triage")
    d = brief["decision"]
    badges = (_badge(d["complaint_type"], "#6366f1")
              + _badge(f"severity: {d['severity']}", SEV_COLOR.get(d["severity"], "#64748b"))
              + _badge("ESCALATE" if d["escalate"] else "no escalation",
                       "#dc2626" if d["escalate"] else "#16a34a")
              + _badge("needs confirmation" if d["requires_confirmation"] else "auto",
                       "#0ea5e9"))
    md = ("**Policy citations:** " + ", ".join(d["policy_citations"]) +
          "\n\n**Steps:**\n" + "\n".join(f"1. {s}" for s in d["steps"]) +
          f"\n\n**Draft reply:**\n> {d['draft_message']}")
    return badges, md, json.dumps(brief, indent=2, ensure_ascii=False)


def voice_fn(audio):
    if audio is None:
        return "Record or upload a voice complaint first.", "", "", "{}"
    res = media.transcribe_voice(audio)
    if res["status"] != "ok":
        return f"Transcription failed: {res.get('error')}", "", "", "{}"
    text = res["text"]
    badges, md, js = triage_fn(text, translate_first=False)
    return text, badges, md, js


# --- RAG / Translate / Chat ------------------------------------------------------
def rag_fn(question):
    r = rag.answer(question)
    src = ", ".join(r["sources"]) if r["sources"] else "—"
    return f"{r['answer']}\n\n_sources: {src}_"


def translate_fn(text, target, use_live):
    """Streaming translation. With use_live, try the realtime Gemini Live API first."""
    if not text or not text.strip():
        yield "_Enter text to translate._"
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


def chat_fn(message, history, model):
    msgs = [{"role": "system", "content": "You are India Commerce SignalForge, an Indian "
             "marketplace pricing, substitution, and complaint analyst. Prices are INR demo data."}]
    for turn in (history or []):
        if isinstance(turn, dict) and turn.get("role") in ("user", "assistant"):
            msgs.append({"role": turn["role"], "content": turn.get("content", "")})
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            msgs += [{"role": "user", "content": turn[0]},
                     {"role": "assistant", "content": turn[1]}]
    msgs.append({"role": "user", "content": message})
    acc = ""
    for delta in groq_client.stream(msgs, model=model, max_tokens=600):
        acc += delta
        yield acc


# --- Analytics -------------------------------------------------------------------
_HEADLINE = [
    ("RAG faithfulness", ("rag", "faithfulness")),
    ("Intent acc", ("intent", "intent_accuracy")),
    ("Substitution MRR", ("substitution", "mrr")),
    ("Triage acc", ("triage", "type_accuracy")),
    ("Hinglish acc", ("hinglish", "hinglish_accuracy")),
    ("Festival↓price", ("festival_counterfactual", "festival_lowers_price_rate")),
    ("Unit-norm", ("unit_norm", "unit_price_accuracy")),
    ("Injection inv.", ("adversarial", "injection_invariance_rate")),
    ("RAG abstain", ("rag_negation", "abstention_rate")),
    ("Tool correctness", ("tool_correctness", "pass_rate")),
    ("Schema valid", ("schema_validity", "schema_valid_rate")),
    ("Guardrails", ("substitution_guardrails", "in_scope_rate")),
]


def _latest_eval() -> dict:
    try:
        return json.loads(LATEST_EVAL.read_text())
    except Exception:  # noqa: BLE001
        return {}


def analytics_metrics_html() -> str:
    latest = _latest_eval()
    cards = []
    for label, (suite, key) in _HEADLINE:
        v = (latest.get(suite, {}) or {}).get(key)
        if v is None:
            continue
        cards.append(f'<div class="sf-metric"><div class="k">{label}</div>'
                     f'<div class="v">{v}</div></div>')
    grid = ('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));'
            f'gap:10px">{"".join(cards)}</div>') if cards else "<p>Run evals to populate.</p>"
    loop = latest.get("loop", "—")
    return f"<h3>Eval metrics (loop {loop})</h3>{grid}"


def analytics_plot_df():
    latest = _latest_eval()
    import pandas as pd
    data = []
    for label, (suite, key) in _HEADLINE:
        v = (latest.get(suite, {}) or {}).get(key)
        if isinstance(v, (int, float)):
            data.append({"metric": label, "score": float(v)})
    return pd.DataFrame(data or [{"metric": "n/a", "score": 0.0}])


def analytics_memory_md() -> str:
    mem = memory.stats()
    runs = tracking.load_runs(limit=5)
    md = [f"**Memory:** {mem['total']} briefs — "
          + ", ".join(f"{k}={v}" for k, v in mem["by_intent"].items()), "", "**Recent runs:**"]
    for r in runs:
        summ = r.get("config", {}).get("_summary", {})
        md.append(f"- `{r['name']}` ({r['backend']}, {r['status']}) "
                  + ", ".join(f"{k}={v}" for k, v in list(summ.items())[:3]))
    return "\n".join(md)


def refresh_analytics():
    return analytics_metrics_html(), analytics_plot_df(), analytics_memory_md()


# --- app -------------------------------------------------------------------------
def build_demo():
    with gr.Blocks(title="India Commerce SignalForge", theme=gr.themes.Soft(
            primary_hue="indigo", secondary_hue="slate"), css=CSS) as demo:
        gr.HTML('<div id="sf-header"><h1>India Commerce SignalForge</h1>'
                '<p>One Indian-commerce engine · Deal · Substitute · Triage · '
                'prices are INR demo data, not live quotes</p></div>')
        gr.HTML(provider_status())

        with gr.Tab("Deal"):
            with gr.Row():
                with gr.Column(scale=3):
                    q = gr.Textbox(label="Product / query", value="Is iPhone 15 a good deal right now?")
                with gr.Column(scale=1):
                    month = gr.Dropdown(choices=[("auto", "")] + [(str(m), m) for m in range(1, 13)],
                                        value="", label="Month (festival)")
                    card = gr.Checkbox(label="Deal-card image", value=True)
            gr.Examples([["Should I wait for Diwali to buy a laptop?", 6, True],
                         ["Tata Salt 1kg ka daam theek hai?", "", False],
                         ["boAt Airdopes 161 worth buying now?", 11, True]],
                        [q, month, card], label="Try")
            deal_badges = gr.HTML()
            deal_md = gr.Markdown()
            with gr.Row():
                deal_img = gr.Image(label="Deal card", type="filepath", height=240)
            with gr.Accordion("Brief (JSON)", open=False):
                deal_json = gr.Code(language="json")
            gr.Button("Analyze deal", variant="primary").click(
                deal_fn, [q, month, card], [deal_badges, deal_md, deal_img, deal_json])

        with gr.Tab("Substitute"):
            sq = gr.Textbox(label="Product to replace",
                            value="Fortune Sunflower Oil is out of stock, alternative?")
            gr.Examples([["Red Label Tea out of stock, alternative?"],
                         ["OnePlus Nord CE4 cheaper option?"],
                         ["boAt Airdopes 161 better rated alternative"]], [sq], label="Try")
            sub_summary = gr.Markdown()
            sub_table = gr.Dataframe(
                headers=["Title", "SKU", "Platform", "Price", "Unit price", "In stock",
                         "Rating", "Score", "Reason"],
                datatype=["str"] * 9, interactive=False, wrap=True, label="Ranked substitutes")
            with gr.Accordion("Brief (JSON)", open=False):
                sub_json = gr.Code(language="json")
            gr.Button("Find substitutes", variant="primary").click(
                substitute_fn, sq, [sub_summary, sub_table, sub_json])
            with gr.Accordion("Read a product screenshot (multimodal vision)", open=False):
                img = gr.Image(label="Product screenshot / receipt", type="filepath")
                vq = gr.Textbox(label="Question about the image",
                                value="What product and price is shown?")
                v_out = gr.Markdown()
                gr.Button("Analyze image").click(vision_fn, [img, vq], v_out)

        with gr.Tab("Triage"):
            tq = gr.Textbox(label="Complaint (English or Hinglish)", lines=2,
                            value="Delivery boy ne 100 rupaye extra liye COD pe")
            tr = gr.Checkbox(label="Translate to English first (Gemini/Groq)", value=False)
            gr.Examples([["Ye iPhone duplicate lag raha hai, seal nahi tha", False],
                         ["Maggi ki expiry nikal chuki hai", False],
                         ["Refund of 21999 initiated last month still not credited", False]],
                        [tq, tr], label="Try")
            tri_badges = gr.HTML()
            tri_md = gr.Markdown()
            with gr.Accordion("Brief (JSON)", open=False):
                tri_json = gr.Code(language="json")
            gr.Button("Triage complaint", variant="primary").click(
                triage_fn, [tq, tr], [tri_badges, tri_md, tri_json])
            with gr.Accordion("Voice complaint (Groq Whisper)", open=False):
                aud = gr.Audio(label="Voice complaint", type="filepath")
                a_text = gr.Textbox(label="Transcript")
                a_badges = gr.HTML()
                a_md = gr.Markdown()
                a_json = gr.Code(language="json", visible=False)
                gr.Button("Transcribe & triage").click(
                    voice_fn, aud, [a_text, a_badges, a_md, a_json])

        with gr.Tab("RAG"):
            rq = gr.Textbox(label="Ask the knowledge base",
                            value="What is the refund timeline for a prepaid order?")
            gr.Examples([["How should a counterfeit product complaint be handled?"],
                         ["Which festival has the biggest discounts on phones?"],
                         ["How do I compare value across different pack sizes?"]], [rq], label="Try")
            ra = gr.Markdown()
            gr.Button("Ask", variant="primary").click(rag_fn, rq, ra)

        with gr.Tab("Translate"):
            gr.Markdown("Live / streaming translation for Hinglish & regional reviews and "
                        "complaints. Default uses Gemini streaming REST (Groq fallback). "
                        "Enable the realtime **Live API** toggle if your Google project has "
                        "Live models (else it falls back automatically).")
            with gr.Row():
                tin = gr.Textbox(label="Text", lines=3,
                                 value="Delivery boy ne 100 rupaye extra liye, product bhi damaged tha")
                tout = gr.Markdown(label="Translation")
            with gr.Row():
                tgt = gr.Dropdown(i18n.LANGUAGES, value="English", label="Translate to")
                use_live = gr.Checkbox(label="Use realtime Gemini Live API", value=False)
            gr.Button("Translate", variant="primary").click(translate_fn, [tin, tgt, use_live], tout)

        with gr.Tab("Chat"):
            model = gr.Dropdown(MODEL_CHOICES, value=config.DEFAULT_MODEL, label="Groq model")
            gr.ChatInterface(fn=chat_fn, additional_inputs=[model], type="messages")

        with gr.Tab("Analytics"):
            metrics_html = gr.HTML(analytics_metrics_html())
            try:
                plot = gr.BarPlot(analytics_plot_df(), x="metric", y="score",
                                  title="Eval scores", height=300, y_lim=[0, 1])
            except Exception:  # noqa: BLE001 — BarPlot unavailable -> skip
                plot = gr.Markdown("_Plot unavailable._")
            mem_md = gr.Markdown(analytics_memory_md())
            gr.Button("Refresh", variant="primary").click(
                refresh_analytics, None, [metrics_html, plot, mem_md])
    return demo


if __name__ == "__main__":
    store.build_index()
    build_demo().launch(server_name="0.0.0.0", server_port=7860)
