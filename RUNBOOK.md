# RUNBOOK — Blinkit Price Intelligence

Everything needed to run the project, with **external steps you must do yourself**
called out as **[EXTERNAL]**. Run all commands from the repo root
(`/Users/siddharthceri/Documents/capstone`). The repo `.venv` (Python 3.12) is
assumed; substitute your own interpreter if different.

---

## 0. Credentials (`.env` at repo root)

Required / present in this environment:

```
GROQ_API_KEY=...      # runtime LLM (required)
HF_TOKEN=...          # Hugging Face Hub + Inference, present
TAVILY_API_KEY=...    # web evidence, present
GEMINI_API_KEY=...    # live translation, present
```

Optional / external:

- **[EXTERNAL] `ANTHROPIC_API_KEY`** — Claude frontier pricing arm (see L123-127).
- **[EXTERNAL] `WANDB_API_KEY`** — only if you want online W&B tracking (see L95-113).
- **[EXTERNAL] `SIGNALFORGE_WEBHOOK_URL`** — only for real notifications (see L138-143).

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

`wandb` is included but optional — tracking falls back to offline JSONL if it is
absent or `WANDB_API_KEY` is unset.

## 2. Generate curated data

```bash
python -m app.commerce_data      # catalog (real Blinkit + synthetic), prices
python -m app.finetune.dataset   # signal classifier splits + price_train/test splits
```

Data provenance: if `data/blinkit_products.json` is missing, the build script
reconstructs it from existing BL-* catalog entries so the rich real-scraped data
survives across rebuilds.

## 3. Train the price meta-learner (Week 6/8 classical-ML ensemble)

```bash
python -m app.agents.meta_learner
```

Writes `data/rf_price_model.pkl` and `data/price_meta.json` (train **and** held-out test
MAE printed, so overfitting is visible). This is Arm 1 of the 5-arm comparison.

## 4. Build the RAG index (Week 5)

```bash
python -m app.rag.store
```

Indexes the markdown knowledge base + a text view of the catalog into Chroma at
`data/chroma`. Lazy index build on first use means this step is optional at launch.

## 5. Fine-tune the LoRA classifier (Week 7, local)

```bash
python -m app.finetune.train_lora
```

Trains a PEFT LoRA over `prajjwal1/bert-tiny` on Hinglish commerce labels, logs
train/eval loss + macro-F1 per epoch through the tracker, and saves the adapter to
`data/lora_adapter`. Runs on CPU/MPS in seconds.

### [EXTERNAL] QLoRA 4-bit variant (CUDA / Colab)

Mac/MPS cannot run `bitsandbytes` 4-bit. To run the course's QLoRA path:

1. Open a CUDA Colab/GPU runtime.
2. `pip install -U bitsandbytes peft transformers accelerate datasets`.
3. Load a larger base (e.g. a 1-3B model) with
   `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`.
4. Reuse `app/finetune/dataset.py` splits (upload them or pull from the Hub, step 6).
5. Train with the same `LoraConfig` shape, then download the adapter to
   `data/lora_adapter` to use it locally.

## 6. [EXTERNAL-ish] Publish adapter + dataset to the HF Hub (Week 7)

`HF_TOKEN` is present, so this works directly. Choose your own repo ids:

```python
from app.finetune import infer_lora
infer_lora.publish_adapter_to_hub("<your-username>/india-commerce-signal-lora")
infer_lora.publish_dataset_to_hub("<your-username>/india-commerce-signals")
```

**[EXTERNAL]** You must own / create the namespace; the token must have `write` scope
(create one at https://huggingface.co/settings/tokens if needed).

## 7. Experiment tracking (W&B with offline fallback)

Default (no setup) writes W&B-compatible JSON to `data/runs/` and surfaces it in the
dashboard. To verify offline:

```bash
python -m app.monitoring.experiment_tracking   # prints backend=offline + a run path
```

### [EXTERNAL] Enable online W&B

1. Create a free account + project at https://wandb.ai.
2. `export WANDB_API_KEY=...` (or add it to `.env`).
3. `wandb login` (one-time, stores the key locally).
4. Re-run training/evals — the same metric schema now logs to your W&B project.

## 8. LoRA price-regression arm (Arm 2 — Colab) + Claude frontier arm (Arm 3)

**LoRA (Arm 2):** open `lora.ipynb` in Colab (Runtime -> T4 GPU). It fine-tunes
`distilbert-base-uncased` with a PEFT LoRA regression head on `{product, festival,
platform} -> price`, logs `train_mae_inr` / `test_mae_inr` per epoch to the W&B project
`blinkit-price-lora`, and saves the adapter + `price_lora_meta.json` to Drive. Download
that folder into `data/price_lora_adapter/` — `infer_price_lora.adapter_exists()`
auto-detects it on next start. Until then the predictor runs with the other four arms.

**Claude frontier (Arm 3):** zero-shot, no RAG, via the Anthropic API. Set
`ANTHROPIC_API_KEY` in `.env`; the model is `config.ANTHROPIC_PRICE_MODEL`
(`claude-haiku-4-5`). If the key is absent the arm returns `None` and is excluded from
the arbitration.

## 9. Multimodal generation

- **Vision (works now):** Groq vision reads product screenshots/receipts — used in the
  UI Substitute tab and `app/llm/groq_client.py::vision`.
- **Price-card image:** `generate_deal_card` always produces an SVG offline; the raster
  path uses HF Inference text-to-image.
  - **[EXTERNAL]** HF Inference text-to-image/TTS models can be gated or rate-limited.
    Accept the model terms on its HF page, or wire an external image/TTS provider
    (e.g. ElevenLabs/Google TTS) and swap the call in `app/media/generate.py`.

## 10. [EXTERNAL] Notifications

Set a webhook to receive real alerts; otherwise payloads log to
`data/notifications.log`:

```bash
export SIGNALFORGE_WEBHOOK_URL=https://hooks.slack.com/services/...
```

## 11. Run evaluations

```bash
python evals/harness.py 1                       # all suites, loop 1
python evals/harness.py 2 intent,substitution    # a subset
```

Suites: `intent`, `substitution`, `unit_norm`, `price_comparison`,
`schema_validity`, `guardrails`. All suites are offline-safe — no live LLM required.

Results -> `evals/results/results_loop<N>.json` + `latest.json`; metrics also logged to
the tracker.

## 12. Build dashboard + docs

```bash
python dashboard/build_dashboard.py   # dashboard/index.html
python docs/build_docs.py             # docs/index.html
```

Open the HTML files in a browser. Dashboard shows: catalog split (BL vs SF),
meta-learner MAE, eval metrics, experiment runs, provider status, 5-arm explanation.

## 13. Launch the app

```bash
python -m app.ui      # http://localhost:7860
```

### [EXTERNAL] Deploy to Hugging Face Spaces

1. Create a new **Gradio** Space at https://huggingface.co/new-space.
2. Add repo contents (or `git push` to the Space repo).
3. Set Space **secrets**: `GROQ_API_KEY`, `HF_TOKEN`, `TAVILY_API_KEY`,
   `GEMINI_API_KEY` (and `WANDB_API_KEY` if used).
4. Ensure `requirements.txt` is at the Space root and the app entry runs
   `app/ui.py::build_demo().launch()`.

### [EXTERNAL] Deploy to Modal

1. `pip install modal && modal token new`.
2. Wrap `build_demo().launch()` in a Modal ASGI/web function with the same secrets as
   above mounted via `modal.Secret`.

## 14. Live translation (Gemini)

`GEMINI_API_KEY` is present; translation prefers Gemini and falls back to Groq:

```bash
python -m app.i18n     # translates Hinglish samples to English
```

The Translate tab streams token-by-token via Gemini `streamGenerateContent`
(Groq fallback).

### [EXTERNAL] Realtime Gemini Live API (optional)

`app/gemini_live.py` implements the *true* realtime Live API
(`client.aio.live.connect` -> `bidiGenerateContent` over WebSockets) for low-latency
translation, with the streaming REST path as an automatic fallback.

At build time the current `GEMINI_API_KEY` works for REST `generateContent` but the
Live models return *"model not found / not supported for bidiGenerateContent"* — i.e.
the Live API is **not enabled for this credential**. To activate the realtime path:

1. Use an API key from a Google project with the **Gemini Live API** enabled
   (AI Studio key on a Live-eligible project, or Vertex AI with Live access).
2. Set `GEMINI_API_KEY` to that key in `.env`.
3. Verify: `python -m app.gemini_live` should print `live_available: True`.
4. In the UI Translate tab, tick **"Use realtime Gemini Live API"**.

Until then, the toggle transparently falls back to the working streaming translator —
no code change needed.

```bash
python -m app.gemini_live   # prints live_available + a translation (live or fallback)
```

---

### One-shot rebuild

```bash
python -m app.commerce_data && python -m app.finetune.dataset && \
python -m app.agents.meta_learner && python -m app.rag.store && \
python -m app.finetune.train_lora && python evals/harness.py 1 && \
python dashboard/build_dashboard.py && python docs/build_docs.py
```
