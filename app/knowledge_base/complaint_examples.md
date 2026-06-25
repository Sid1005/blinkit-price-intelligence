# Complaint Triage Examples & Categories

Reference examples that ground complaint classification and resolution drafting. Demo
content; triage cites policy and never issues binding decisions.

## Categories

- **cod_dispute:** customer charged more/less than the app total at the door.
- **refund_delay:** refund initiated but not credited within policy timelines.
- **fake_product:** authenticity/counterfeit concern.
- **expiry_issue:** expired or near-expiry perishable delivered.
- **wrong_item:** wrong or missing item delivered.
- **damaged_item:** item physically damaged on delivery.
- **other:** anything that does not fit the above.

## Examples (text -> type)

- "Delivery boy ne 100 rupaye extra liye COD pe." -> cod_dispute
- "Refund initiated last week is still not credited." -> refund_delay
- "Ye iPhone duplicate lag raha hai, seal nahi tha." -> fake_product
- "Maggi ki expiry nikal chuki hai." -> expiry_issue
- "Wrong item delivered, ordered atta got salt." -> wrong_item
- "Phone delivery me screen crack ho gayi." -> damaged_item

## Resolution drafting rules

1. Identify the complaint type and severity.
2. Ground every step in the cited policy document (return_refund_policy.md, cod_policy.md).
3. List concrete next steps (share order ID, upload photo, expected timeline).
4. Set `escalate = true` for fake_product, COD safety violations, and out-of-policy
   refund delays.
5. Set `requires_confirmation = true` whenever a refund or replacement would be issued —
   triage drafts the resolution but a human confirms it.
6. Never promise a specific refund amount for counterfeit claims before verification;
   avoid over-promising timelines beyond the policy windows.

## Severity guide

- **high:** fake_product, COD safety/fraud, expired food, long-overdue refunds.
- **medium:** wrong_item, damaged_item, standard refund_delay, COD overcharge.
- **low:** general queries, cosmetic packaging issues.
