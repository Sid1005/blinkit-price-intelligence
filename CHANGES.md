# Capstone — Required Changes

## 1. Remove triage surface
- Delete `triage_agent()` from `app/agents/ensemble.py`
- Delete `Resolution` schema from `app/agents/schemas.py`
- Remove `"triage"` from `config.INTENTS`
- Remove Triage tab from `app/ui.py`
- Remove `ComplaintType`, `COMPLAINT_TYPES`, `COMPLAINT_KB` references from `config.py`
- Remove triage golden files from `evals/golden/` and harness suites

## 2. Switch to real Blinkit data
- `app/ingest/scrape_blinkit.py` already exists — wire it in
- Replace `app/commerce_data.py` synthetic catalog with scraped Blinkit products
- Replace `data/prices/price_observations.jsonl` with real scraped prices
- Rebuild Chroma index, retrain meta-learner on real data
- Update knowledge_base docs to reflect real Blinkit return/COD policies (scraped)

## 3. Fix train/test split in meta-learner
- `app/agents/meta_learner.py::train()` currently fits and evaluates on same data
- Add proper 80/20 split: train on 80%, report MAE on held-out 20%
- Report both train MAE and test MAE so overfitting is visible

## 4. Add frontier LLM price comparison (Claude API, ~$5 budget)
- New file: `app/finetune/frontier_pricer.py`
- Send product description + festival context to Claude
- Ask it to output `{"price_inr": <number>}`
- Compare MAE: RandomForest vs LoRA (once built) vs Claude frontier
- Show comparison table in Analytics tab and in README

## 5. Vision in price predictor + substitute
- Price Predictor tab: add image upload → vision model reads product name/brand/pack size → feeds into price prediction
- Substitute tab: add image upload → identify out-of-stock product from photo → find alternatives
- Uses Groq vision model already wired in `groq_client.vision()`

## 6. Whisper in price predictor
- Add voice input to Price Predictor tab
- User speaks: "Diwali pe boAt Airdopes ka price kya hoga?"
- Transcribe with Groq Whisper → feed transcript into price predictor
- Uses `groq_client.transcribe()` already wired

## 7. Fine-tune LLM for price regression (the main upgrade)
- New file: `app/finetune/train_price_lora.py`
- Base model: small generative model (e.g. GPT-2 or small Llama via Colab QLoRA)
- Training data format: `{"text": "boAt Airdopes 161, Diwali, Flipkart", "price": 1299.0}`
- Output head: regression (num_labels=1), loss: MAE
- At inference: `predict_price("iPhone 15, Big Billion Days, Amazon.in")` → `67999.0`
- This replaces the current LoRA signal classifier as the headline fine-tuning demo
- Run on Colab (GPU needed for anything beyond bert-tiny)

### Colab setup (do this when ready for item 7)
1. Open [colab.research.google.com](https://colab.research.google.com) → New notebook → Runtime → Change runtime type → **T4 GPU**
2. Mount Google Drive: `from google.colab import drive; drive.mount('/content/drive')`
3. Upload `app/finetune/train_price_lora.py` and the scraped Blinkit training data (from item 2) to your Drive
4. In Colab, install deps: `!pip install transformers peft datasets accelerate`
5. Run training — adapter weights save to `models/price_lora/`
6. Download the adapter folder back to your local `models/price_lora/` (or push to HF Hub: `model.push_to_hub("Sid1005/blinkit-price-lora")` using your `HF_TOKEN`)
7. `infer_lora.py` will auto-detect the adapter via `adapter_exists()` on next app start

## 8. Simplify signal pipeline
- Remove or simplify `scout()` + `classifier_agent()` + `verifier_agent()`
- For price predictor: evidence = scraped Blinkit price history, not text snippets
- Keep verifier concept only if web evidence is enabled (injection risk)

## Weights & Biases setup (do this before running evals)
1. Sign up at [wandb.ai](https://wandb.ai) if you haven't — free tier is enough
2. Go to User Settings → API Keys → copy your key
3. It's already in your `.env` as `WANDB_API_KEY=` — paste the key there
4. Run evals: `python evals/harness.py 1` — results auto-log to W&B
5. Go to wandb.ai → your project → you'll see per-suite metrics (faithfulness, intent accuracy, MAE, etc.) tracked across runs
6. Each time you make a change and re-run evals, increment the loop number (`python evals/harness.py 2`, `3`, ...) — W&B plots the trend so you can see if a change improved or regressed metrics
7. The `results/` folder also keeps local JSON copies as a backup

## 9. Rename / reframe for GitHub
- Project name: **Blinkit Price Intelligence** (or keep SignalForge)
- Three surfaces: **Price Predictor · Substitute · (drop triage)**
- Pitch: festival-aware Indian quick-commerce price prediction using the full LLM stack
- README sections: problem, architecture diagram, each technique used + course week mapping
