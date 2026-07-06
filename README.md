# Blinkit Price Intelligence

**Live pages:** [project explainer](https://sid1005.github.io/blinkit-price-intelligence/) · [metrics dashboard](https://sid1005.github.io/blinkit-price-intelligence/dashboard.html) · demo script: [`DEMO.md`](DEMO.md)

Festival-aware Indian quick-commerce price prediction, with two decision surfaces over a
single agent spine:

- **Price Predictor** — festival-aware INR price prediction shown as a **5-arm model
  comparison**: RandomForest, LoRA regression, Claude frontier (zero-shot), Groq + RAG,
  and a Groq ensemble arbitrator that reasons over the other four.
- **Substitute Finder** — ranked alternatives when a SKU is unavailable, overpriced, poor
  value, or poorly reviewed, grounded in the substitution guide.

Review understanding is shared **Hinglish NLP enrichment**, not a separate app.

> **Scope & safety.** Catalog prices combine real scraped Blinkit products (BL-* SKUs)
> with curated synthetic SKUs (SF-*), clearly labelled — this project does **not** claim
> live quotes. Festival context is auto-detected from the current month, never parsed from
> the user's query.

## The 5 arms

| Arm | What it is | Claim |
|---|---|---|
| RandomForest | Stacked classical ML over structured features | Domain baseline trained on Blinkit data |
| LoRA regression | `distilbert-base-uncased` fine-tuned `{product, festival, platform} -> price` | Domain-adapted small model (trained on Colab) |
| Claude frontier | Anthropic API, pure zero-shot, no RAG, no training data | Large model with no Blinkit exposure |
| Groq + RAG | Groq grounded in the pricing playbook + festival calendar | RAG beats pure zero-shot |
| Groq ensemble arbitrator | Groq reasons over all four estimates, picks the final price | Meta-reasoning |

The comparison is shown both live per query (UI table) and as offline MAE on a held-out
20% test split (`price_comparison` eval suite).

## Provider policy

| Capability | Provider |
|---|---|
| Runtime LLM inference | **Groq** (`GROQ_API_KEY`) |
| Frontier pricing arm | **Anthropic / Claude** (`ANTHROPIC_API_KEY`) |
| Open models / datasets / Hub | **Hugging Face** (`HF_TOKEN`) |
| Web evidence | **Tavily** (`TAVILY_API_KEY`) |
| Experiment tracking | **W&B** (`WANDB_API_KEY`, optional) -> offline JSONL fallback |
| Live translation | **Gemini** (`GEMINI_API_KEY`) -> Groq fallback |

## Architecture

```
scout -> classify -> verify -> IntentRouter -> { PricePredictor (5 arms) | SubstituteFinder } -> Brief -> Memory/UI
```

The RandomForest arm is a **trained stacked ensemble** (RandomForest + linear
meta-learner) fit on an 80/20 split (train + held-out test MAE both reported, so
overfitting is visible), conditioned on the Indian festival calendar.

## Layout

```
capstone/
  app/
    config.py              # keys, Groq registry, festival calendar, taxonomy
    llm/                   # groq_client (chat/stream/json/vision/whisper), router
    ingest/scraper.py      # Tavily search/extract; scrape_blinkit.py (data provenance)
    nlp/                   # embeddings, hf_pipeline (sentiment/aspect/zeroshot/NER/translate)
    rag/                   # Chroma store (hybrid rerank, headlines, rewrite) + grounded RAG
    agents/                # schemas, intent_router, tools, memory, meta_learner, ensemble
    finetune/              # dataset, baselines, signal LoRA, price-LoRA inference, frontier pricer
    monitoring/            # experiment tracking (W&B / offline)
    media/                 # price-card image, TTS
    codegen/               # parser benchmarks, unit-norm kernel (Week 4, offline-safe)
    i18n.py                # Gemini live translate (Groq fallback)
    notify.py              # deal notification hook
    ui.py                  # Gradio cockpit (Predictor/Substitute/Translate/Analytics)
    knowledge_base/*.md    # festival, pricing, policy, substitution docs
  data/
    blinkit_products.json  # real scraped Blinkit products (seed for BL-* catalog)
    catalog/products.json  # combined catalog (BL-* real + SF-* synthetic)
    price_lora_adapter/    # drop the Colab-trained adapter here (auto-detected)
  dashboard/
    build_dashboard.py     # generates dashboard/index.html with summary stats
  evals/
    harness.py             # offline-safe eval harness with 6 suites
    golden/golden.json     # golden test data for all suites
  orchestration/
    coverage_map.json      # Week 1-8 to capstone concept mapping
  docs/
    build_docs.py          # generates docs/index.html explainer
  lora.ipynb               # Colab notebook: trains the LoRA price-regression arm
  DEMO.md                  # 3-5 minute demo script for GitHub
```

## Results (eval loop 7, offline-safe suites)

| Suite | Metric | Score |
|---|---|---|
| Intent routing | accuracy | **1.00** (15/15) |
| Substitution | MRR / coverage | **1.00 / 1.00** |
| Unit normalization | accuracy | **1.00** (10/10) |
| RAG retrieval | recall / MRR | **0.86 / 0.93** |
| Schema validity | valid rate | **1.00** (9/9) |
| Price comparison | ensemble MAE | **₹34.87** (band coverage 3/5) |

Honest overfitting disclosure, on purpose: the RandomForest arm reports **train MAE ₹106 vs held-out test MAE ₹391** — the gap is visible in the dashboard rather than hidden, and it's exactly why the ensemble arbitrator (MAE ₹35) beats any single arm. Guardrail suites run live with `RUN_LIVE_EVALS=1`.

Every eval loop appends to `evals/results/` and W&B (offline JSONL fallback), so regressions show up as a trend, not an anecdote.

## Course concept coverage

All 8 weeks of the LLM engineering course map to running code — the full
week-by-week table (concept → file → how to demo it) is generated from
[`orchestration/coverage_map.json`](orchestration/coverage_map.json) into the
[explainer page](docs/index.html):

| Week | Theme | Concepts wired in |
|---|---|---|
| 1 | Frontier APIs, scraping, prompting, JSON | 7 |
| 2 | Multi-model APIs, Gradio, tools, multimodal | 5 |
| 3 | HuggingFace pipelines, tokenizers, open models | 5 |
| 4 | Code generation & benchmarking | 3 |
| 5 | RAG: LangChain, Chroma, hybrid retrieval | 5 |
| 6 | Data curation, classical ML baselines, tracking | 5 |
| 7 | LoRA/PEFT fine-tuning, HF Hub | 4 |
| 8 | Agentic spine, safety, deployment | 6 |

## Quickstart

See [`RUNBOOK.md`](RUNBOOK.md) for the full step-by-step. The short version (from
`capstone/`, using the repo `.venv`):

```bash
pip install -r requirements.txt
python -m app.commerce_data            # curate world data (real Blinkit + synthetic)
python -m app.agents.meta_learner      # train RandomForest arm (80/20 split, reports MAE)
python -m app.finetune.dataset         # signal + price dataset splits
python -m app.rag.store                # build the Chroma index
python evals/harness.py 1              # run all eval suites (incl. price_comparison MAE)
python dashboard/build_dashboard.py    # generate dashboard/index.html
python docs/build_docs.py              # generate docs/index.html
python -m app.ui                       # launch the Gradio app on :7860
```

### Offline fallback behaviour

Everything is designed to degrade gracefully when API keys are missing:

| If missing... | Then... |
|---|---|
| `GROQ_API_KEY` | Groq-based arms fail; ensemble uses available arms only. Agent classifies purely locally. |
| `ANTHROPIC_API_KEY` | Claude frontier arm excluded; comparison shows 4 arms instead of 5. |
| `TAVILY_API_KEY` | Web evidence disabled; scout uses user snippets only. |
| `GEMINI_API_KEY` | Translation falls back to Groq; Live API toggle auto-degrades. |
| `WANDB_API_KEY` | Experiment tracking writes offline JSONL to `data/runs/`. |
| Price LoRA adapter | Arm 2 shows N/A; advisor suggests running lora.ipynb on Colab. |

**Key design property:** data curation, meta-learner training, evals, and dashboard are
all fully offline-safe — they use only local data and deterministic computations.

The LoRA price arm (Arm 2) is trained separately on Colab via `lora.ipynb`; download the
adapter into `data/price_lora_adapter/` and it is auto-detected on the next app start.
Until then the system runs with the other four arms and shows LoRA as N/A.
