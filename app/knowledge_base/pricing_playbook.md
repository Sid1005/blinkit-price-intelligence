# India Commerce Pricing & Deal Playbook

Guidance for estimating fair INR price bands and buy/wait/avoid calls. All figures are
demo heuristics for a curated catalog, not live market quotes.

## Unit-normalised pricing

Always compare on a normalised unit so different pack sizes are comparable:

- **Per kg** for atta, salt, sugar, pulses, snacks.
- **Per litre** for edible oil, milk, detergents, shampoo, handwash.
- **Per piece** for electronics and durables.
- **Per 100 ml / per 100 g** for small personal-care packs.

A larger pack with a lower per-unit price is usually better value unless perishability
or storage is a constraint. Flag when a "deal" has a *higher* unit price than a smaller
pack (a common dark pattern).

## Fair price band logic

1. Start from the trailing median observed price for the SKU.
2. Subtract the expected festival discount bias if a sale is active this month.
3. The **band low** is the best festival-adjusted price seen; the **band high** is the
   typical non-sale street price. The **point** estimate is the festival-adjusted median.
4. Never quote below 60% of MRP for electronics or below 50% of MRP for groceries
   unless evidence explicitly supports a clearance.

## Buy / wait / avoid decision

- **BUY NOW:** observed price ≤ band low, item in stock, and no bigger festival within
  the next 30 days.
- **WAIT:** a larger festival (Big Billion Days, Great Indian Festival, Diwali) is this
  month or next and the item is a non-urgent electronic.
- **AVOID:** price is above the band high (demand-spike premium), or stock is zero and a
  cheaper in-stock substitute exists.

## Discount quality checks

- A genuine **festival_discount** shows a real cut versus the trailing median, not just
  an inflated MRP. Treat "X% off MRP" with suspicion if MRP looks inflated.
- A **demand_spike** is a price rise with low stock; do not mistake it for a deal.
- Platform fees, delivery charges, and minimum-cart thresholds change effective value;
  mention them when material.

## Confidence

Report **high** only when there are at least two corroborating price observations and a
clear festival/non-festival context. Otherwise report **medium** or **low** and prefer a
wider band over a falsely precise point estimate.
