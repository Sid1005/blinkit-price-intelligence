# Price Intelligence

A price-prediction lab that runs the **same pipeline over two very different datasets** and
lets the models fight it out on a shared scoreboard.

I curate a dataset, engineer features, and then throw the whole zoo at the price:
**linear regression, bag-of-words + linear regression, random forest, XGBoost**, and
**zero-shot frontier LLMs** (Groq + Claude). Every arm is scored by the *same* evaluator —
MAE, RMSE, R², hit-rate — so the leaderboard is honest and directly comparable.<br>
On top of that there's a **RAG** substitution engine for Blinkit that I A/B against a
no-retrieval baseline, and the fine-tuned **QLoRA** specialist + **ensemble** arms are next
in line (see [Roadmap](#roadmap)).

Two datasets, one pipeline:

- **Amazon** — a faithful reproduction of Ed Donner's *"The Price Is Right"* course pricer on
  `McAuley-Lab/Amazon-Reviews-2023` (USD). 20,000-row lite train split, 1,000 test.
- **Blinkit** — the exact same code on **real scraped Indian quick-commerce data** (INR, no
  synthetic rows). 921 SKUs pulled live via Tavily, plus a RAG substitution engine on top.

Running one pipeline over both sources gives an honest Amazon-vs-Blinkit comparison and shows
where a big pile of data (Amazon) vs a small real one (Blinkit) changes which model wins.

## What's in here

- **Classical ML baselines** (`app/pricer/baselines.py`) — `LinearRegression` on engineered
  features (weight, text length, word count, category), **BoW `CountVectorizer` +
  LinearRegression**, **RandomForest**, and **XGBoost**. Course week-6/day-3-4 parity.
- **Frontier zero-shot arms** (`app/pricer/frontier.py`) — **Groq** (`llama-3.3-70b`) and
  **Anthropic Claude**, each asked to price an item from its description alone, no training,
  no retrieval. The "can a big model just guess it?" arm the classical models have to beat.
- **A shared evaluator** (`app/pricer/evaluator.py`) — one `Tester` scores every arm the same
  way (MAE / RMSE / R² / green-orange-red hit-rate) and renders scatter + cumulative-error
  charts, so nothing gets to grade its own homework.
- **Real data curation** (`app/pricer/loaders.py`, `parser.py`, `preprocessor.py`) — streams
  Amazon's raw per-category JSONL directly (the old `trust_remote_code` loader script is dead
  in `datasets>=3.0`), scrubs/filters with the course rules, rewrites each description into a
  clean summary with Groq, and curates train/val/test splits.
- **A live Blinkit scraper** (`app/ingest/scrape_blinkit.py`) — iterates category listing
  pages via **Tavily** search + extract, parses each card into structured JSON with Groq,
  dedupes by `(name, unit)`, and keeps only rows with a real numeric price. **No synthetic
  SKUs anywhere** — 921 real priced products.
- **A RAG substitution engine** (`app/substitution/`) — when a Blinkit SKU is out of stock,
  retrieve real in-aisle alternatives from a dedicated **Chroma** index (built from 51 real
  aisle docs, embedded with `all-MiniLM-L6-v2`) and have the LLM recommend one **grounded only
  in what was retrieved** — then A/B it against the same model with no retrieval at all.
- **Your own Hugging Face datasets** (`app/pricer/hub.py`) — curated splits are pushed to
  *your* HF namespace (resolved from `HF_TOKEN` via `whoami()`), not someone else's, and read
  back from there.
- **Plain JSONL for everything** — every eval writes its raw records to `evals/results/`, so
  every number in this README is reproducible from the data in the repo.

## Provider policy

| Capability | Provider |
|---|---|
| Runtime LLM inference (rewrite, parse, price, RAG) | **Groq** (`GROQ_API_KEY`) |
| Second frontier arm (pricing + no-RAG substitution) | **Anthropic / Claude** (`ANTHROPIC_API_KEY`) |
| Embeddings, open models, dataset Hub | **Hugging Face** (`HF_TOKEN`) — `all-MiniLM-L6-v2` |
| Live web scraping | **Tavily** (`TAVILY_API_KEY`) |

## How it actually works

Both datasets flow through the identical pipeline — only the loader and the currency change:

```
curate (scrub + filter + Groq rewrite)  ->  push to HF Hub  ->  pull back
   ->  classical baselines (LinReg / BoW / RandomForest / XGBoost)
   ->  frontier zero-shot (Groq + Claude)
   ->  [Blinkit only] RAG substitution (retrieve real aisle -> grounded pick)
   ->  one shared Tester scores every arm: MAE / RMSE / R² / hit-rate
```

1. **Same prompt, every arm.** Each item becomes the course price prompt —
   `"What does this cost to the nearest {dollar|rupee}?\n\n{summary}\n\nPrice is {sym}{price}"`
   — and the frontier arms see exactly the truncated version the baselines train on, so no arm
   gets extra information the others don't.
2. **The evaluator is the referee.** `Tester.test(predictor, label, test_set)` runs any
   `predict(item) -> float` through the same scoring and charting. Add a new model, implement
   one method, and it lands on the same leaderboard.
3. **RAG is grounded, not vibes.** The Blinkit substitution arm can *only* recommend a product
   that showed up in the retrieved context. If the retrieved aisle has nothing suitable, it
   returns "no substitute" instead of hallucinating one — and I measure exactly that.
4. **No synthetic data on the Blinkit side.** Every Blinkit price is a real scraped number.
   The catalog, the splits, and the substitution docs are all built from the 921-SKU scrape.

## Results

### Price prediction leaderboard

Every arm, scored by the same `Tester` on the held-out test split. MAE/RMSE are in the
dataset's native currency (smaller is better); R² is variance explained (0 = no better than
always guessing the mean; negative = *worse* than that); hit-rate is the fraction of
predictions landing in the "green" band.

**Amazon** — 20,000 train / 1,000 test (USD)

| Model | MAE | RMSE | R² | Hit rate |
|---|---|---|---|---|
| **XGBoost** | 30.50 | **65.60** | **0.624** | 83.4% |
| **RandomForest** | **28.30** | 68.71 | 0.588 | 84.0% |
| Claude frontier (zero-shot) | 32.94 | 105.82 | 0.022 | **84.3%** |
| Groq frontier (zero-shot) | 34.16 | 100.74 | 0.114 | 81.7% |
| BoW + LinearRegression | 40.24 | 67.94 | 0.597 | 70.7% |
| LinearRegression | 40.85 | 94.12 | 0.227 | 83.0% |

**Blinkit** — 736 train / 93 test (INR, real scraped data)

| Model | MAE | RMSE | R² | Hit rate |
|---|---|---|---|---|
| **RandomForest** | **126.33** | **240.35** | **0.421** | **45.2%** |
| XGBoost | 143.54 | 260.50 | 0.320 | 37.6% |
| Groq frontier (zero-shot) | 173.90 | 435.32 | -0.900 | 41.9% |
| Claude frontier (zero-shot) | 176.37 | 404.18 | -0.637 | 43.0% |
| LinearRegression | 190.77 | 345.17 | -0.194 | 22.6% |
| BoW + LinearRegression | 286.78 | 497.78 | -1.484 | 21.5% |

What the numbers say:

- **Tree ensembles win on both datasets.** RandomForest and XGBoost lead on MAE *and* are the
  only arms with a comfortably positive R² on Blinkit — everything else there is worse than
  guessing the average price for every item.
- **Frontier LLMs beat the *linear* baselines on typical-case accuracy**, on both datasets,
  purely from world knowledge — but they lose to the trees on MAE and fall apart on R²/RMSE
  (negative R² on Blinkit means a handful of confident, wildly-wrong guesses blow up the
  variance).
- **Data volume decides everything.** Amazon's numbers are categorically stronger across every
  model because it has ~27× the training data. On Blinkit's 736 rows, `BoW+LinearRegression`
  (8,000 vocab params, 736 data points) overfits so hard it posts R² = **-1.48** — the same
  model is perfectly respectable on Amazon.

### RAG vs. no-RAG substitution (Blinkit, 40 out-of-stock SKUs)

Same task both ways: a shopper can't get product X, recommend one alternative. **No-RAG** is
Claude zero-shot from general knowledge; **RAG** is Groq grounded in the retrieved real aisle.

| Metric | No-RAG (Claude) | **RAG (Groq)** |
|---|---|---|
| Suggestion actually exists in the Blinkit catalog | 37.5% | **97.5%** |
| Suggestion is in the right aisle | 80.0% | **89.7%** |
| Stated price is accurate | 28.6% | **100%** |
| Avg cost / query | $0.0020 | **$0.0009** |
| Avg latency | 2.85s | **1.02s** |

Retrieval takes "exists in the real catalog" from **37.5% → 97.5%** and stated-price accuracy
from **28.6% → 100%** — while being *cheaper and faster*, because a grounded 70B model doesn't
need to reason as hard as a frontier model guessing blind.

## Setting it up yourself

```bash
git clone https://github.com/Sid1005/blinkit-price-intelligence.git
cd blinkit-price-intelligence
pip install -r requirements.txt

cp .env.example .env
# fill in GROQ_API_KEY, ANTHROPIC_API_KEY, HF_TOKEN, TAVILY_API_KEY
```

**Curate a dataset** (streams, scrubs, Groq-rewrites, splits, pushes to *your* HF account):

```bash
python -m app.pricer.loaders --source amazon      # Amazon lite, 20k/1k/1k
python -m app.ingest.scrape_blinkit --full        # scrape ~900+ real Blinkit SKUs
python -m app.pricer.loaders --source blinkit     # curate + split the scrape
```

**Run the leaderboard** (any subset of arms; every arm shares the same scorer):

```bash
python -m app.pricer.baselines --source amazon    # LinReg / BoW / RandomForest / XGBoost
python -m app.pricer.baselines --source blinkit
python -m app.pricer.frontier  --source amazon    # Groq + Claude zero-shot
python -m app.pricer.frontier  --source blinkit
```

**Run the RAG substitution A/B** (Blinkit only):

```bash
python -m app.substitution.rag_index              # build the Chroma index from real aisle docs
python scripts/eval_substitution_rag.py           # RAG vs no-RAG over the eval set
```

Scatter/cumulative-error charts for each arm land in `data/<source>/eval_charts/`, and the
substitution ablation writes to `evals/results/substitution_rag_ablation.json`.

## Repo map

```
app/pricer/
  items.py          the Item data-point + shared price-prompt template + HF push/pull
  loaders.py        AmazonLoader (streams raw JSONL, lite splits) + BlinkitLoader (80/10/10)
  parser.py         course scrub/filter rules, generalized so Blinkit rows pass too
  preprocessor.py   Groq description -> clean summary rewrite
  baselines.py      LinearRegression / BoW / RandomForest / XGBoost (+ Groq category cleanup)
  frontier.py       Groq + Claude zero-shot pricing arms
  evaluator.py      the shared Tester: MAE/RMSE/R²/hit-rate + charts
  hub.py            push/pull curated datasets to YOUR Hugging Face namespace
  BASELINES.md      deep-dive: every baseline, what it predicts from, why it scores as it does
  FRONTIER.md       deep-dive: the zero-shot arms vs the classical leaderboard

app/substitution/
  rag_index.py       dedicated Chroma index over the real Blinkit aisle docs
  compare.py         the two substitution arms: no_rag (Claude) vs rag (Groq, grounded)
  blinkit_catalog.py catalog helpers + candidate ranking for grading
  evaluate.py        scores both arms: catalog-hit / same-aisle / price-accuracy

app/ingest/scrape_blinkit.py   Tavily scrape -> Groq JSON parse -> deduped priced catalog
app/llm/                        Groq client + model router
app/nlp/embeddings.py           all-MiniLM-L6-v2 embedding function for Chroma
scripts/                        dataset + substitution-doc build + eval entry points

data/blinkit/     921-SKU scrape, curated splits, 51 aisle substitution docs, eval charts
data/amazon/      category map + eval charts (splits live on the HF Hub)
evals/results/    raw JSONL/JSON for every scored run
```

## Roadmap

Everything above is implemented and scored. Two more arms are landing on the same leaderboard
next — the code lives in a branch and gets pushed once each is fully scored end-to-end:

- **QLoRA price specialist** — 4-bit nf4 QLoRA fine-tune of `meta-llama/Llama-3.2-3B` on the
  curated price prompts (trained on Colab), scored through the *same* `Tester` as every other
  arm so the fine-tuned model sits right next to XGBoost and the frontier arms.
- **Ensemble** — a weighted/arbitrated blend over the available arms
  (`frontier·0.8 + specialist·0.1 + classical·0.1`, course parity), with the Blinkit ensemble
  additionally folding in the RAG arm.

## Credit

The pipeline architecture, the price-prompt format, the scrub/filter rules, and the
classical → frontier → QLoRA → ensemble progression follow Ed Donner's **"LLM Engineering"**
course ("The Price Is Right", weeks 6–8). This repo reproduces that faithfully on Amazon and
then runs the identical pipeline over a real, self-scraped Blinkit dataset with a RAG
substitution engine bolted on.
