# Price Intelligence

Predict product prices with a set of models, then compare them fairly on one shared scoreboard.

The same pipeline runs over two datasets:

- **Amazon** (USD): 20,000 train / 1,000 test.
- **Blinkit** (INR): 921 real SKUs scraped live, no synthetic data.

What it does:

- Trains classical ML: LinearRegression, BoW + LinearRegression, RandomForest, XGBoost.
- Adds zero-shot LLM pricing: Groq and Claude.
- Scores every model the same way (MAE, RMSE, R², hit-rate), so the leaderboard is honest.
- Runs a RAG substitution engine that I A/B test against a no-retrieval baseline.
- Pushes the curated datasets to my own Hugging Face account.

## How it works

Both datasets flow through the same steps. Only the loader and the currency change.

```
curate (scrub + filter + Groq rewrite)  ->  push to HF Hub  ->  pull back
   ->  classical baselines (LinReg / BoW / RandomForest / XGBoost)
   ->  zero-shot LLMs (Groq + Claude)
   ->  [Blinkit only] RAG substitution (retrieve real aisle -> grounded pick)
   ->  one shared scorer: MAE / RMSE / R² / hit-rate
```

- Every model sees the same prompt, so no model gets extra information the others don't.
- One scorer grades everything. Add a model, implement one `predict(item) -> float`, and it lands on the same board.
- The RAG substitution arm can only suggest a product that was actually retrieved. If nothing fits, it says "no substitute" instead of making one up.
- Every Blinkit price is a real scraped number. Nothing is synthetic.

## Results

### RAG vs no-RAG substitution (Blinkit, 40 out-of-stock SKUs)

Same task both ways: a shopper can't get product X, suggest one alternative. No-RAG is Claude answering from general knowledge. RAG is Groq grounded in the real retrieved aisle.

| Metric | No-RAG (Claude) | **RAG (Groq)** |
|---|---|---|
| Suggestion exists in the Blinkit catalog | 37.5% | **97.5%** |
| Suggestion is in the right aisle | 80.0% | **89.7%** |
| Stated price is accurate | 28.6% | **100%** |
| Avg cost / query | $0.0020 | **$0.0009** |
| Avg latency | 2.85s | **1.02s** |

Retrieval takes "exists in the real catalog" from 37.5% to 97.5% and stated-price accuracy from 28.6% to 100%, while also being cheaper and faster.

### Price prediction leaderboard

Scored on the held-out test split. MAE/RMSE are in the dataset's currency (smaller is better). R² is variance explained (0 means no better than always guessing the mean, negative means worse than that). Hit-rate is the fraction of predictions in the "green" band.

**Amazon** (20,000 train / 1,000 test, USD)

| Model | MAE | RMSE | R² | Hit rate |
|---|---|---|---|---|
| **XGBoost** | 30.50 | **65.60** | **0.624** | 83.4% |
| **RandomForest** | **28.30** | 68.71 | 0.588 | 84.0% |
| Claude (zero-shot) | 32.94 | 105.82 | 0.022 | **84.3%** |
| Groq (zero-shot) | 34.16 | 100.74 | 0.114 | 81.7% |
| BoW + LinearRegression | 40.24 | 67.94 | 0.597 | 70.7% |
| LinearRegression | 40.85 | 94.12 | 0.227 | 83.0% |

**Blinkit** (736 train / 93 test, INR, real scraped data)

| Model | MAE | RMSE | R² | Hit rate |
|---|---|---|---|---|
| **RandomForest** | **126.33** | **240.35** | **0.421** | **45.2%** |
| XGBoost | 143.54 | 260.50 | 0.320 | 37.6% |
| Groq (zero-shot) | 173.90 | 435.32 | -0.900 | 41.9% |
| Claude (zero-shot) | 176.37 | 404.18 | -0.637 | 43.0% |
| LinearRegression | 190.77 | 345.17 | -0.194 | 22.6% |
| BoW + LinearRegression | 286.78 | 497.78 | -1.484 | 21.5% |

What the numbers say:

- Tree ensembles win on both datasets. RandomForest and XGBoost lead on MAE, and on Blinkit they are the only models with a positive R².
- The zero-shot LLMs beat the linear baselines on typical accuracy, but lose to the trees on MAE and do badly on R².
- Data volume matters a lot. Amazon has about 27x more training data, so every model does better there. On Blinkit's 736 rows, BoW + LinearRegression overfits hard and posts an R² of -1.48.

## What's in here

- **Classical ML baselines** (`app/pricer/baselines.py`): LinearRegression on engineered features (weight, text length, word count, category), BoW `CountVectorizer` + LinearRegression, RandomForest, and XGBoost.
- **Zero-shot LLM arms** (`app/pricer/frontier.py`): Groq and Claude price an item from its description alone, no training and no retrieval.
- **A shared scorer** (`app/pricer/evaluator.py`): one `Tester` grades every model (MAE / RMSE / R² / hit-rate) and renders scatter and cumulative-error charts.
- **Data curation** (`app/pricer/loaders.py`, `parser.py`, `preprocessor.py`): streams Amazon's raw JSONL, scrubs and filters rows, rewrites each description into a clean summary with Groq, and builds train/val/test splits.
- **Blinkit scraper** (`app/ingest/scrape_blinkit.py`): scrapes category pages with Tavily, parses each card into JSON with Groq, dedupes, and keeps only rows with a real price. 921 real SKUs, no synthetic data.
- **RAG substitution engine** (`app/substitution/`): builds a Chroma index over 51 real aisle docs (embedded with `all-MiniLM-L6-v2`), then recommends a substitute grounded only in what was retrieved.
- **Your own Hugging Face datasets** (`app/pricer/hub.py`): curated splits are pushed to your HF account, resolved from `HF_TOKEN`, and read back from there.

## Provider policy

| Capability | Provider |
|---|---|
| Runtime LLM inference (rewrite, parse, price, RAG) | **Groq** (`GROQ_API_KEY`) |
| Second LLM arm (pricing + no-RAG substitution) | **Anthropic / Claude** (`ANTHROPIC_API_KEY`) |
| Embeddings, open models, dataset Hub | **Hugging Face** (`HF_TOKEN`) |
| Live web scraping | **Tavily** (`TAVILY_API_KEY`) |

## Setting it up yourself

```bash
git clone https://github.com/Sid1005/blinkit-price-intelligence.git
cd blinkit-price-intelligence
pip install -r requirements.txt

cp .env.example .env
# fill in GROQ_API_KEY, ANTHROPIC_API_KEY, HF_TOKEN, TAVILY_API_KEY
```

Curate a dataset (streams, scrubs, Groq-rewrites, splits, pushes to your HF account):

```bash
python -m app.pricer.loaders --source amazon      # Amazon, 20k/1k/1k
python -m app.ingest.scrape_blinkit --full        # scrape ~900+ real Blinkit SKUs
python -m app.pricer.loaders --source blinkit     # curate and split the scrape
```

Run the leaderboard (any subset of models, all share the same scorer):

```bash
python -m app.pricer.baselines --source amazon    # LinReg / BoW / RandomForest / XGBoost
python -m app.pricer.baselines --source blinkit
python -m app.pricer.frontier  --source amazon    # Groq + Claude zero-shot
python -m app.pricer.frontier  --source blinkit
```

Run the RAG substitution A/B (Blinkit only):

```bash
python -m app.substitution.rag_index              # build the Chroma index from real aisle docs
python scripts/eval_substitution_rag.py           # RAG vs no-RAG over the eval set
```

Charts for each model land in `data/<source>/eval_charts/`. The substitution results write to `evals/results/substitution_rag_ablation.json`.

## Repo map

```
app/pricer/
  items.py          the Item data-point, shared price prompt, HF push/pull
  loaders.py        AmazonLoader (raw JSONL, splits) + BlinkitLoader (80/10/10)
  parser.py         scrub and filter rules for both sources
  preprocessor.py   Groq description to clean summary rewrite
  baselines.py      LinearRegression / BoW / RandomForest / XGBoost
  frontier.py       Groq + Claude zero-shot pricing
  evaluator.py      the shared Tester: MAE/RMSE/R²/hit-rate + charts
  hub.py            push/pull curated datasets to your Hugging Face account
  BASELINES.md      every baseline, what it predicts from, why it scores as it does
  FRONTIER.md       the zero-shot arms vs the classical leaderboard

app/substitution/
  rag_index.py       Chroma index over the real Blinkit aisle docs
  compare.py         the two arms: no_rag (Claude) vs rag (Groq, grounded)
  blinkit_catalog.py catalog helpers and candidate ranking for grading
  evaluate.py        scores both arms: catalog-hit / same-aisle / price-accuracy

app/ingest/scrape_blinkit.py   Tavily scrape to Groq JSON parse to priced catalog
app/llm/                        Groq client + model router
app/nlp/embeddings.py           all-MiniLM-L6-v2 embedding function for Chroma
scripts/                        dataset build, substitution-doc build, eval scripts

data/blinkit/     921-SKU scrape, curated splits, 51 aisle docs, eval charts
data/amazon/      category map + eval charts (splits live on the HF Hub)
evals/results/    raw JSON for every scored run
```
