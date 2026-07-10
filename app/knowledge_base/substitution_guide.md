# Substitution Intelligence Guide

How to recommend alternatives when a SKU is unavailable, overpriced, poor value, or
poorly reviewed.

Grounding for the Blinkit (Part B) substitution arm comes from real scraped data, not
synthetic catalog rows: `data/blinkit/substitutions/index.md` lists 50 real in-aisle
buckets (canonicalized from Blinkit's own — sometimes duplicated — category labels)
built from the 921-SKU scrape in `data/blinkit/blinkit_products.json`, each with a
ranked top-3 substitute list per SKU. `data/blinkit/substitutions/_no_substitute.md`
covers the handful of singleton SKUs with no in-aisle alternative in this scrape.
Every scraped rating came back null, so the ranked buckets substitute brand diversity
and discount depth for the "quality" signal described below.

## When to substitute

- **Out of stock:** the requested SKU has zero stock.
- **Overpriced:** the SKU's current price is materially above its fair band or above a
  comparable item's unit price.
- **Bad value:** a higher unit price than an equivalent pack/SKU.
- **Poorly reviewed:** low rating or strong negative authenticity/quality sentiment.
- **Mismatch:** the SKU does not fit the stated need (e.g., wrong pack size).

## Ranking candidates

Score each candidate substitute (0–1) on a weighted blend:

1. **Same need / category fit** (weight 0.30) — must be in the same substitute group or
   category and serve the same purpose.
2. **Unit-price value** (weight 0.30) — lower normalised unit price is better.
3. **Availability** (weight 0.20) — in stock beats out of stock.
4. **Quality signal** (weight 0.20) — higher rating and positive review aspects.

Rank candidates by total score, descending. Always state the **reason** for each
recommended substitute (cheaper per kg, in stock, better rated, genuine, etc.).

## Value improvement

Report the **value improvement** of the top substitute versus the original: the percent
reduction in unit price, or the availability/quality gain when price is similar. Do not
recommend a substitute that is strictly worse on every axis.

## Guardrails

- Keep substitutes within the same broad need (do not swap a flagship phone for a
  budget feature phone unless explicitly asked).
- Prefer in-stock, well-rated, genuine items.
- If no sensible substitute exists, say so rather than forcing a weak recommendation.
