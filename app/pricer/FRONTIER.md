# Frontier zero-shot pricing arms

Two LLM-based pricing arms live in [`frontier.py`](frontier.py), evaluated on
both the Amazon (`amazon-pricer-lite`) and Blinkit (`blinkit-pricer`) datasets
via the same `app/pricer/evaluator.py::Tester` used by
[`baselines.py`](baselines.py) / [`BASELINES.md`](BASELINES.md). Both arms see
only `item.test_prompt()` — the product text, no weight/category features, no
retrieval, no fine-tuning — and are asked to guess a price from general world
knowledge alone. This is the arm that RAG (`rag_pricer.py`, Blinkit only) and
the QLoRA specialist are meant to beat by grounding the guess in this specific
catalog's actual price distribution.

Run both arms and get a ranked comparison:

```bash
python3 -m app.pricer.frontier --source amazon
python3 -m app.pricer.frontier --source blinkit
```

Run one specific arm, capped to a sample (each item is a live API call):

```bash
python3 -m app.pricer.frontier --source blinkit --model claude --size 20
```

Predict calls are I/O-bound network requests, so `Tester` fans them out
concurrently (`--workers`, default 8) instead of running one at a time — the
full 1,000-item Amazon test set completes in a few minutes instead of the
20-30+ it would take sequentially.

## Results

Full test sets — 1,000 items (Amazon), 93 items (Blinkit) — both arms, no
`--size` cap:

```
AMAZON (20,000 train / 1,000 test)
Model                              MAE      RMSE        R²    Hit rate
Claude Frontier                  32.94    105.82     0.022       84.3%
Groq Frontier                    34.16    100.74     0.114       81.7%

BLINKIT (736 train / 93 test)
Model                              MAE      RMSE        R²    Hit rate
Groq Frontier                   173.90    435.32    -0.900       41.9%
Claude Frontier                 176.37    404.18    -0.637       43.0%
```

For reference, the classical baselines on the same test sets
([`BASELINES.md`](BASELINES.md#results)):

```
AMAZON            MAE      RMSE        R²    Hit rate
RandomForest     28.30     68.71     0.588       84.0%
XGBoost          30.50     65.60     0.624       83.4%
BoW+LinearReg    40.24     67.94     0.597       70.7%
LinearRegression 40.85     94.12     0.227       83.0%

BLINKIT           MAE      RMSE        R²    Hit rate
RandomForest    126.33    240.35     0.421       45.2%
XGBoost         143.54    260.50     0.320       37.6%
LinearRegression190.77    345.17    -0.194       22.6%
BoW+LinearReg   286.78    497.78    -1.484       21.5%
```

## What the numbers say

- **Frontier LLMs beat both linear baselines on typical-case accuracy, on
  both datasets** — Amazon: Claude/Groq MAE (32.94 / 34.16) beats
  `BoW+LinearRegression` (40.24) and `LinearRegression` (40.85). Blinkit:
  Groq/Claude MAE (173.90 / 176.37) beats `LinearRegression` (190.77) and
  `BoW+LinearRegression` (286.78) by a wide margin. Zero-shot world knowledge
  about "what does a chocolate bar / a spark module cost" generalizes better
  than 4 hand-picked features or an 8,000-word bag on a 736-row training set.
- **But neither frontier arm beats the tree ensembles on MAE, and both lose
  badly on R²/RMSE** — RandomForest and XGBoost still lead on MAE on both
  datasets, and their R² is comfortably positive (0.42-0.62) while both
  frontier arms post **negative R² on Blinkit** (-0.90 / -0.64) and
  **near-zero R² on Amazon** (0.02 / 0.11). Negative R² there means: on the
  errors that matter most to a squared-error metric, a frontier LLM's guesses
  are effectively *worse than always predicting the Blinkit mean price*.
- **Why MAE and R² tell such different stories for the same arm.** MAE
  (mean absolute error) treats every miss equally; R²/RMSE square the error,
  so a handful of huge misses dominates the score. A frontier LLM has no
  visibility into *this catalog's* actual price scale — it estimates from
  general priors, so it's usually close (driving MAE and hit-rate up) but
  occasionally wildly wrong on an item its priors don't cover well (a
  region-specific SKU, an unusual pack size, a category it under/overrates) —
  and those rare big misses get squared and blow up RMSE/R². Random Forest
  and XGBoost, by contrast, are fit directly to this catalog's price
  distribution, so they rarely guess wildly outside the range they were
  trained on — fewer huge misses, better R², even with a slightly higher
  average error on the easy majority of items.
- **Claude edges out Groq on Amazon (lower MAE, higher hit-rate, higher R²);
  Groq edges out Claude on Blinkit's headline MAE but Claude has the better
  R²/RMSE there too.** Neither model has any Blinkit-specific training data —
  both are guessing INR grocery/retail prices from general knowledge, and
  Blinkit's mixed catalog (groceries next to electronics next to baby care,
  ₹14-₹2,000+ range) is a harder zero-shot target than Amazon's narrower
  appliance-parts category, which is reflected in every arm's much worse
  Blinkit numbers.
- **This is the gap RAG is built to close.** `rag_pricer.py` (Blinkit only)
  gives the LLM the 5 most similar *actual* Blinkit SKUs and their real
  prices as context before it guesses — trading pure world-knowledge
  zero-shot for a grounded, catalog-aware estimate. Frontier's negative R²
  here is the baseline that arm needs to beat.

---

## Design

**Shared prompt contract.** Both arms are handed the *identical* input the
classical baselines never see directly and the RAG/QLoRA arms build on:
`item.test_prompt()` — the same product-description text used everywhere
else in the pipeline, truncated right before the price so nothing leaks. A
system prompt asks for a single JSON number in the item's native currency:

```python
_SYSTEM = (
    "You estimate retail prices from a product description. Reply with strict "
    'JSON only, nothing else: {{"price": <number>}}. The number is your single '
    "best point estimate in {currency}, with no currency symbol, no commas, "
    "and no explanation."
)
```

`{currency}` is `"US dollars"` for Amazon items, `"Indian rupees"` for
Blinkit — so the same code path produces correctly-scaled guesses for both
parts without a source-specific branch anywhere in the predict logic.

**Parsing.** The response is expected to be `{"price": 42.5}`, but LLMs
occasionally wrap it in prose or drop the JSON braces under `json_mode`
edge cases, so parsing falls back from a `"price": <number>` regex to a bare
first-number match, and clamps to `>= 0`:

```python
_PRICE_RE = re.compile(r'"?price"?\s*[:=]\s*"?(-?\d+(?:\.\d+)?)', re.I)
_NUMBER_RE = re.compile(r"(-?\d+(?:\.\d+)?)")

def _parse_price(text: str) -> float:
    match = _PRICE_RE.search(text)
    if not match:
        match = _NUMBER_RE.search(text)
    return max(0.0, float(match.group(1))) if match else 0.0
```

**`GroqFrontier`** calls `app.llm.groq_client.chat(..., json_mode=True)` with
`config.DEFAULT_MODEL` (`llama-3.3-70b-versatile`, the "strong" tier),
`temperature=0.0` for deterministic scoring.

**`ClaudeFrontier`** calls the Anthropic SDK directly with
`config.ANTHROPIC_PRICE_MODEL` (`claude-haiku-4-5`) — a second, fully
independent frontier arm with no shared infrastructure with Groq, so any
systematic bias in one provider's price priors doesn't quietly show up as
"the frontier arm" being wrong in both places at once.

Both expose `.predictor() -> Callable[[Item], float]`, the same adapter
signature `Tester` and every other arm in this repo use, so `frontier.py`,
`baselines.py`, `rag_pricer.py`, and `ensemble.py` are all interchangeable
inputs to one evaluation harness.

## Concurrency

`Tester` (`evaluator.py`) gained an optional `max_workers` parameter used
only by I/O-bound predictors like these — a `ThreadPoolExecutor` fans out
`predictor(item)` calls, then results are recorded and printed in original
order so verbose output and chart data stay deterministic. Classical
baselines are unaffected (`max_workers=1` default; local sklearn/xgboost
inference doesn't benefit from threading and isn't thread-contended by it).

## Source

Full implementation: [`frontier.py`](frontier.py). Evaluation harness
(MAE/RMSE/R²/hit-rate, scatter + cumulative-error charts, now with optional
concurrency): [`evaluator.py`](evaluator.py).
