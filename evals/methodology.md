# Evaluation Methodology

All suites live in `evals/harness.py`; golden files in `evals/golden/`. The Groq
`JUDGE_MODEL` (llama-3.3-70b) acts as the LLM judge. Every `run_all` writes a
W&B-compatible record via the shared tracker plus `results/results_loop<N>.json`.

## Suites

- **rag** — faithfulness & answer-relevance (Groq judge, 0–1), source-level context
  precision, hit-rate@1/4, MRR@4, nDCG@4 against `rag_golden.json` expected sources.
- **retrieval_ablation** — context-precision and MRR with hybrid reranking on vs off;
  reports `precision_gain` to justify the reranker.
- **classifier** — accuracy + macro-F1 on the golden signal set for four members:
  Groq few-shot, sklearn TF-IDF+LogReg, PyTorch hashed-MLP, and the LoRA adapter.
- **intent** — intent-router accuracy (deal/substitute/triage) on `agent_golden.json`.
- **substitution** — hit-rate@1/3, MRR, nDCG against acceptable substitute SKUs, plus
  average value-improvement %.
- **triage** — complaint-type accuracy + macro-F1, escalation correctness, and policy
  groundedness (fraction of resolutions citing a policy doc).
- **hinglish** — clean vs Hinglish accuracy on paraphrase pairs, the perturbation delta
  (clean − Hinglish), and label-flip rate.
- **festival_counterfactual** — for each SKU, predicted price in a festival month vs a
  non-festival month; reports the rate at which the festival lowers price and the
  average implied discount.
- **unit_norm** — per-kg/litre/piece/100ml unit-price correctness and basis correctness.
- **adversarial** — prompt-injection resist rate (decision must not follow injected
  "buy_now") and over-promise guard rate (triage must keep `requires_confirmation`).
- **model_tournament** — pairwise Groq-judged Elo across Groq tiers (order-swapped to
  de-bias). Groq-only: cross-vendor comparisons are structural, runtime uses Groq tiers.
- **e2e** — schema-level smoke across all three surfaces (band valid, candidates present,
  policy cited, confirmation required).

### Added suites (robustness, safety, calibration)

- **rag_negation** — out-of-scope questions (capital of France, biryani recipe, etc.)
  must be refused with "insufficient evidence"; reports abstention rate.
- **calibration** — Expected Calibration Error (ECE) of the LoRA classifier on the
  golden set, with mean confidence, accuracy, and over/under-confidence.
- **substitution_guardrails** — over every grouped SKU: never returns the original,
  candidate scores are in [0,1] and sorted descending, candidates stay in
  category/group, and an in-stock candidate is ranked first when one exists.
- **price_band_sanity** — for a sample of SKUs: strict band ordering
  (0 < low ≤ point ≤ high), point within an MRP bound, and OOS → avoid.
- **schema_validity** — every emitted Brief validates against the pydantic schema and
  per-decision invariants (band ordering, candidate score range/sorting, triage
  requires_confirmation).
- **tool_correctness** — the tool layer validates inputs and never crashes on bad/missing
  args (zero pack size, missing keys, wrong arg type, unknown tool, unknown SKU).
- **severity** — complaint severity mapping matches the policy-defined expectation.
- **determinism** — deterministic surfaces (substitution, unit-norm, the data-driven
  deal recommendation) are reproducible across repeated runs.

## Reading results

`evals/results/latest.json` holds the most recent loop; the dashboard surfaces the
headline metrics and the per-member classifier table. Scalar metrics are flattened to
`suite.metric` keys and logged to the tracker for run-over-run comparison.

## Known honest limitations

- Substitution value-improvement can be negative when the only in-group alternative is
  pricier per unit (e.g. an out-of-stock budget SKU's nearest substitute costs more);
  the metric reports reality rather than forcing a positive number.
- Judge-based metrics (faithfulness/relevance) carry LLM-judge variance; they are
  averaged over the golden set and should be read as directional.
