# India Commerce SignalForge

One Indian-commerce decision engine with three surfaces over a single agent spine:

- **Deal predictor** — festival-aware INR price band + buy / wait / avoid.
- **Substitution intelligence** — ranked alternatives when a SKU is unavailable,
  overpriced, poor value, or poorly reviewed.
- **Complaint triage** — classify COD / refund / fake / expiry / wrong / damaged
  complaints and draft policy-grounded resolutions.

Review understanding is shared **Hinglish NLP enrichment**, not a separate app.

> **Scope & safety.** All prices are curated demo data, clearly labelled — this project
> does **not** claim live Blinkit/Amazon quotes. Complaint triage cites policy, never
> issues binding refunds, and flags uncertain cases as requiring human confirmation.

## Provider policy

| Capability | Provider |
|---|---|
| Runtime LLM inference | **Groq only** (`GROQ_API_KEY`) |
| Open models / datasets / Hub | **Hugging Face** (`HF_TOKEN`) |
| Web evidence | **Tavily** (`TAVILY_API_KEY`) |
| Experiment tracking | **W&B** (`WANDB_API_KEY`, optional) → offline JSONL fallback |
| Live translation | **Gemini** (`GEMINI_API_KEY`) → Groq fallback |

## Architecture

```
scout → classify → verify → IntentRouter → { DealPredictor | SubstitutionRanker | ComplaintTriage } → Brief → Memory/UI
```

The pricing decision is a **trained stacked ensemble** (RandomForest + linear
meta-learner) blended with a Groq estimate and RAG playbook context, conditioned on the
Indian festival calendar.

## Layout

```
capstone_final/
  app/
    config.py              # keys, Groq registry, festival calendar, taxonomy
    llm/                   # groq_client (chat/stream/json/vision/whisper), router
    ingest/scraper.py      # Tavily search/extract + BeautifulSoup
    nlp/                   # embeddings, hf_pipeline (sentiment/aspect/zeroshot/NER/translate)
    rag/                   # Chroma store (hybrid rerank, headlines, rewrite) + grounded RAG
    agents/                # schemas, intent_router, tools, memory, meta_learner, ensemble
    finetune/              # dataset, baselines (Groq/sklearn/PyTorch), LoRA, frontier runbook
    codegen/               # parser synthesis + C++ compile-and-time benchmark
    monitoring/            # experiment tracking (W&B / offline)
    media/                 # deal-card image, TTS, voice transcription
    i18n.py                # Gemini live translate (Groq fallback)
    notify.py              # deal/escalation notification hook
    ui.py                  # Gradio cockpit (Deal/Substitute/Triage/RAG/Chat/Analytics)
    knowledge_base/*.md    # festival, pricing, policy, substitution docs
  data/                    # generated catalog/prices/reviews/complaints, splits, adapter, runs
  evals/                   # harness + golden files + results + methodology
  dashboard/ docs/         # generated static HTML
  orchestration/coverage_map.json
  fixtures/                # drop demo image/audio here for multimodal smoke (see fixtures/README.md)
```

## Quickstart

See [`RUNBOOK.md`](RUNBOOK.md) for the full step-by-step including every external action.
The short version (from `capstone_final/`, using the repo `.venv`):

```bash
uv pip install -r requirements.txt
python -m app.commerce_data            # curate world data
python -m app.finetune.dataset         # signal dataset splits
python -m app.agents.meta_learner      # train price meta-learner
python -m app.rag.store                # build the Chroma index
python -m app.finetune.train_lora      # local LoRA fine-tune (week 7)
python evals/harness.py 1              # run all eval suites
python dashboard/build_dashboard.py && python docs/build_docs.py
python -m app.ui                       # launch the Gradio app on :7860
```
