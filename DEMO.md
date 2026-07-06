# DEMO.md — Blinkit Price Intelligence Demo Script (3-5 minutes)

## Pre-flight

```bash
cd /Users/siddharthceri/Documents/capstone
source .venv/bin/activate  # Python 3.12

# One-shot rebuild (offline-safe)
python -m app.commerce_data && \
python -m app.finetune.dataset && \
python -m app.agents.meta_learner && \
python -m app.rag.store && \
python -m app.finetune.train_lora && \
python evals/harness.py 1 && \
python dashboard/build_dashboard.py && \
python docs/build_docs.py
```

## Demo flow

### 0. Show README (30s)

Point out:
- The architecture diagram: scout -> classify -> verify -> IntentRouter -> Predictor | Substitute -> Brief
- The 5-arm price comparison table
- Provider policy: Groq, Claude, HF, Tavily, Gemini, W&B

### 1. Launch the app (30s)

```bash
python -m app.ui
```

Open http://localhost:7860. Note the provider status bar.

### 2. Predictor: three example queries (60s)

| Query | Set month | What to point out |
|---|---|---|
| "Diwali pe Snickers ka price kitna hoga?" | 11 (Nov) | Festival-aware pricing across all 5 arms |
| "Cadbury Bournville fair price kya hai?" | auto | RAG context accordion shows retrieved KB chunks |
| "KitKat Big Billion Days pe kitne ka milega?" | 10 (Oct) | 30% discount bias visible in price drop |

Show: the 5-arm comparison table, RAG context accordion, and the full JSON brief.

### 3. Substitute: rank alternatives (30s)

Query: "Snickers is out of stock, alternative?"

Point out:
- Ranked substitutes with scores, reasons, prices, stock
- Value improvement % (unit-price comparison)

### 4. Analytics tab (45s)

Click **Refresh**. Show:
- Eval metrics table (intent accuracy, substitution MRR, ensemble MAE, etc.)
- Bar chart of scores
- Memory stats (brief history by intent)
- Recent experiment runs

### 5. Open dashboard (30s)

```bash
open dashboard/index.html
```

Point out:
- Catalog split: BL-* (real scraped) vs SF-* (synthetic)
- Meta-learner train/test MAE
- Provider/adapter availability
- 5-arm explanation table
- Eval metrics table

### 6. Course week mapping (30s)

Open `docs/index.html` or show the coverage map:
- Week 1: Groq + Tavily scraping (live at runtime)
- Week 2: Gradio UI, model router, vision, multimodal
- Week 3: HF pipelines, embeddings, Whisper
- Week 4: Codegen parser benchmarks, unit-norm kernel
- Week 5: Chroma RAG with hybrid reranking
- Week 6: Curated data, meta-learner, experiment tracking
- Week 7: LoRA fine-tuning, HF Hub publishing
- Week 8: Agentic spine, safety, deploy

### 7. Run evals (30s)

```bash
python evals/harness.py 1
```

Output shows per-suite scores. Open `evals/results/latest.json`.

## Key takeaways for viewers

- **Real + synthetic data**: 146 BL-* real Blinkit products + 25 SF-* synthetic expansion
- **5 independent arms**: RF, LoRA, Claude, Groq+RAG, ensemble arbitrator — compared per query
- **Offline-safe everywhere**: all evals, data curation, and dashboard work without API keys
- **Train/test visibility**: meta-learner reports both train MAE (105 INR) and test MAE (390 INR)
- **Agentic spine**: typed Pydantic Briefs, signal classification, verifier safety, SQLite memory
