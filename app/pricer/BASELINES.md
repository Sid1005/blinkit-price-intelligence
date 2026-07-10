# Classical ML pricing baselines

Four non-LLM baselines live in [`baselines.py`](baselines.py), evaluated on both
the Amazon (`amazon-pricer-lite`) and Blinkit (`blinkit-pricer`) datasets via
`app/pricer/evaluator.py::Tester`. They exist to set a floor: any LLM-based
pricing arm elsewhere in this repo should beat these, or it isn't earning its
extra cost/latency.

Run all four and get a ranked comparison:

```bash
python3 -m app.pricer.baselines --source amazon
python3 -m app.pricer.baselines --source blinkit
```

Run one specific baseline:

```bash
python3 -m app.pricer.baselines --source blinkit --model rf
```

## Results

```
AMAZON (20,000 train / 1,000 test)
Model                              MAE      RMSE        R²    Hit rate
RandomForest                     28.30     68.71     0.588       84.0%
XGBoost                          30.50     65.60     0.624       83.4%
BoW+LinearRegression             40.24     67.94     0.597       70.7%
LinearRegression                 40.85     94.12     0.227       83.0%

BLINKIT (736 train / 93 test)
Model                              MAE      RMSE        R²    Hit rate
RandomForest                    126.33    240.35     0.421       45.2%
XGBoost                         143.54    260.50     0.320       37.6%
LinearRegression                190.77    345.17    -0.194       22.6%
BoW+LinearRegression            286.78    497.78    -1.484       21.5%
```

`MAE`/`RMSE` are in the dataset's native currency ($ for Amazon, ₹ for
Blinkit) — smaller is better. `R²` is the fraction of price variance the
model explains; **0 means "no better than always predicting the mean," and
negative means worse than that**. `Hit rate` is the fraction of predictions
landing in `Tester`'s "green" band (error < ₹/\$40 or < 20% of truth).

### What the numbers say

- **Tree ensembles (RandomForest, XGBoost) win on both datasets**, and by a
  wide margin on Blinkit. They're the only two baselines with positive R² on
  Blinkit — everything else there is worse than guessing the average price
  for every item.
- **Amazon's results are categorically stronger than Blinkit's** across every
  model, mostly because Amazon has ~27x more training data (20,000 vs 736
  rows) — more data narrows the gap between a weak and strong model.
- **`BoW+LinearRegression` is the worst model on Blinkit** (R² = -1.48) despite
  being competitive on Amazon (R² = 0.597). With only 736 training rows and
  an 8,000-word vocabulary, the linear solver has far more parameters than
  data points to constrain them, so it overfits noise in which words happened
  to co-occur with which prices, and generalizes badly. The same model on
  Amazon's 20,000 rows has enough data to constrain those parameters
  properly.
- **Plain `LinearRegression` (4 engineered features) is more data-efficient
  than `BoW+LinearRegression`** on the small Blinkit set — 4 parameters vs
  8,000 is far less prone to overfitting — but its R² is still negative
  there, because 4 hand-picked features (weight, text length, word count,
  category) just don't carry enough price signal for Blinkit's wide price
  range (₹14–₹2,000+) and heterogeneous catalog (groceries next to
  earbuds next to baby food).
- **Blinkit's raw category field had 112 near-duplicate labels** (HTML-entity
  and casing drift from scraping, e.g. `"Audio &amp; Accessories"` vs
  `"Audio Accessories"`). `baselines.py` collapses these to ≤10 canonical
  buckets via a one-time Groq call, cached to
  `data/blinkit/category_map.json`, so the `LinearRegression` baseline's
  category one-hot columns each get enough training support to carry
  signal instead of being mostly-empty scrap. See `_ensure_category_map()`
  in `baselines.py`.

---

## 1. LinearRegression (engineered features)

**What it predicts from:** four hand-picked numbers per item — `weight`,
`text_length` (characters), `word_count`, and a one-hot-encoded `category`.

**The model.** Linear regression assumes price is a weighted sum of the
features plus a constant:

```
price ≈ w1·weight + w2·text_length + w3·word_count + w4·(category one-hot) + b
```

`fit()` finds the `w`'s and `b` that minimize the sum of squared errors
between predicted and true price across all training items — the smaller the
total squared miss, the better. This loss function is a smooth, convex
"bowl" in weight-space with exactly one minimum, so instead of the iterative
nudge-and-check optimization used by neural nets, sklearn solves for the
minimum directly: it sets the derivative of the loss to zero and solves the
resulting **normal equations**:

```
w = (XᵀX)⁻¹ Xᵀy
```

`X` is the training feature matrix, `y` is the true prices. In practice
sklearn computes this via SVD (singular value decomposition) rather than a
literal matrix inverse, because `XᵀX` can be near-singular when features are
correlated (e.g. `text_length` and `word_count` usually move together) —
SVD stays numerically stable where a direct inverse would blow up. Either
way, this is a **closed-form solve**: one linear-algebra operation, exact,
deterministic, no training loop.

**Where the category cleanup matters.** `DictVectorizer` one-hot-encodes
`category` — each distinct string becomes its own 0/1 column. Blinkit's raw
scrape has 112 near-duplicate category strings; `_ensure_category_map()`
collapses them to ≤10 canonical buckets first, so each one-hot column
actually has enough rows behind it to carry a learnable weight.

**Limitation.** Purely additive and linear — it can learn "each gram adds
$X" but can't learn interactions ("weight matters more for the appliances
bucket than for snacks") or diminishing returns. That's the gap the other
three baselines close.

```python
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import numpy as np
import re

_WORD_RE = re.compile(r"[A-Za-z]+")

def _features(item, category_map):
    text = item.summary or item.full or item.title
    return {
        "weight": item.weight or 0.0,
        "text_length": len(text),
        "word_count": len(_WORD_RE.findall(text)),
        "category": category_map.get(item.category, item.category),
    }

class LinearRegressionBaseline:
    def __init__(self):
        self.vectorizer = DictVectorizer(sparse=False)
        self.model = LinearRegression()
        self.category_map = {}

    def fit(self, train_items):
        self.category_map = _ensure_category_map(train_items)  # 112 -> <=10 buckets
        X = self.vectorizer.fit_transform(
            [_features(item, self.category_map) for item in train_items])
        y = np.array([item.price for item in train_items])
        self.model.fit(X, y)          # closed-form normal-equations solve
        return self

    def predict(self, item):
        X = self.vectorizer.transform([_features(item, self.category_map)])
        return max(0.0, float(self.model.predict(X)[0]))
```

---

## 2. BoW + LinearRegression

**What it predicts from:** raw item text only (no weight, no category) —
represented as **Bag of Words**: build a vocabulary of up to 8,000 words
seen across training text (ignoring common "stop words" like "the"/"and"),
then represent each item as a vector of how many times each vocabulary word
appears in its description. "Amul Salted Butter 200g" becomes a mostly-zero
vector with a `1` in the "amul", "salted", "butter" columns and zeros
everywhere else.

**The model.** Same `LinearRegression` machinery as baseline #1 — same
closed-form least-squares solve — just fed a much wider, sparser feature
matrix (up to 8,000 columns instead of ~13). Each vocabulary word ends up
with its own learned weight: effectively "how many dollars/rupees does
seeing this word add to the price."

**Why it's the worst performer on Blinkit (R² = -1.48) despite being
competitive on Amazon (R² = 0.597).** Blinkit has only 736 training rows but
up to 8,000 feature columns — far more parameters than data points to pin
them down. The solver finds weights that fit the training noise almost
perfectly (which specific words happened to appear on which training-set
items) but those weights don't generalize — classic overfitting. Amazon's
20,000 rows give the same-sized vocabulary enough data to constrain the
weights properly, so the same code performs reasonably there.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
import numpy as np

def _text(item):
    return item.summary or item.full or item.title

class BagOfWordsLinearRegressionBaseline:
    def __init__(self, max_features=8000):
        self.vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
        self.model = LinearRegression()

    def fit(self, train_items):
        X = self.vectorizer.fit_transform([_text(item) for item in train_items])
        y = np.array([item.price for item in train_items])
        self.model.fit(X, y)
        return self

    def predict(self, item):
        X = self.vectorizer.transform([_text(item)])
        return max(0.0, float(self.model.predict(X)[0]))
```

---

## 3. RandomForest

**What it predicts from:** the same Bag-of-Words text vectors as baseline #2.

**The model.** Instead of one global linear formula, a Random Forest trains
many independent decision trees (100 here) and averages their predictions
— this is called **bagging** (bootstrap aggregating). Each tree:

1. Trains on a random bootstrap sample of the training rows (sampled with
   replacement).
2. At each split, only considers a random subset of the feature columns.
3. Recursively asks yes/no questions about word counts — "does 'organic'
   appear? is 'kg' present more than once?" — partitioning items into
   smaller and smaller buckets, each ending in a leaf that predicts the
   average price of the training items that landed there.

The forest's final prediction for a new item is the **average** of what all
100 trees independently predict for it. The randomization in steps 1–2 means
each tree overfits its own slice of the data in a different, uncorrelated
way — averaging many differently-overfit trees cancels most of that noise
out, which is why forests are far more robust to small/noisy datasets than
a single linear model with thousands of parameters.

**Why it wins.** Trees can capture **non-linear, conditional**
relationships that a linear model structurally cannot represent — e.g. "the
word 'earbuds' only matters combined with 'wireless.'" Combined with
bagging's noise-cancellation, this is the best performer on both datasets
here, and by a wide margin on the small Blinkit set (R² = 0.421 vs
BoW+LinearRegression's -1.484 on the identical feature matrix).

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class RandomForestBaseline:
    def __init__(self, max_features=8000, n_estimators=100):
        self.vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)

    def fit(self, train_items):
        X = self.vectorizer.fit_transform([_text(item) for item in train_items])
        y = np.array([item.price for item in train_items])
        self.model.fit(X, y)   # trains 100 independent bootstrapped trees
        return self

    def predict(self, item):
        X = self.vectorizer.transform([_text(item)])
        return max(0.0, float(self.model.predict(X)[0]))   # average of 100 tree predictions
```

---

## 4. XGBoost

**What it predicts from:** the same Bag-of-Words text vectors again.

**The model.** Also an ensemble of decision trees, but built by **gradient
boosting** rather than bagging: trees are trained **sequentially**, and each
new tree is trained specifically to predict the *residual error* (what's
left over) of the trees built so far, rather than the raw price. Roughly:

1. Tree 1 makes a first rough guess for every item.
2. Compute each item's remaining error (`true_price - current_prediction`).
3. Tree 2 is trained to predict *that error*, not the price itself.
4. Add a shrunk version of Tree 2's prediction to the running total; repeat
   for `n_estimators` rounds (200 here).

This is the tree-based analogue of the iterative gradient-descent
optimization mentioned in baseline #1's closed-form contrast — instead of
nudging numeric weights downhill, XGBoost adds a new tree each round that
nudges the ensemble's predictions in the direction that reduces the loss
the most. Each round focuses specifically on the hardest-to-predict items
left over from previous rounds.

**Why it's close to but not always ahead of RandomForest here.** Boosting
usually squeezes out more accuracy than bagging when there's enough data for
the sequential refinement to pay off — it edges out RandomForest on Amazon's
R² (0.624 vs 0.588). But sequential boosting has more room to overfit on
small data, since each new tree is chasing residuals of an already-tiny
736-row training set; RandomForest's independent-and-averaged trees are more
robust when data is scarce, which is likely why RandomForest still leads on
Blinkit.

```python
from xgboost import XGBRegressor
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class XGBoostBaseline:
    def __init__(self, max_features=8000, n_estimators=200):
        self.vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
        self.model = XGBRegressor(n_estimators=n_estimators, random_state=42,
                                   objective="reg:squarederror", n_jobs=-1)

    def fit(self, train_items):
        X = self.vectorizer.fit_transform([_text(item) for item in train_items])
        y = np.array([item.price for item in train_items])
        self.model.fit(X, y)   # 200 rounds of sequential residual-correcting trees
        return self

    def predict(self, item):
        X = self.vectorizer.transform([_text(item)])
        return max(0.0, float(self.model.predict(X)[0]))
```

---

## Source

Full implementation: [`baselines.py`](baselines.py). Evaluation harness
(MAE/RMSE/R²/hit-rate, scatter + cumulative-error charts):
[`evaluator.py`](evaluator.py).
