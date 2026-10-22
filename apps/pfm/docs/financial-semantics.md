# Financial semantics

## Ledger calculations

All amounts use integers representing **paise**; there is no floating-point arithmetic in
financial calculations. The period is the Indian calendar month in `Asia/Kolkata`, independent
of the timezone on the incoming event.

- Income = total of trusted `income` events.
- Gross spending = total of `purchase` events.
- Refunds = total of validated `refund` events posted in the selected month.
- Net spending = gross spending − refunds.
- Cash surplus = income − net spending.
- Transfers are reported separately and excluded from both income and spending.

Cash surplus is **not account balance**, disposable cash, a predicted saving, or an investment
recommendation. Income/spending depend on the completeness of the bank event feed. No opening
balance is guessed. If there are no events, the assistant says it cannot infer the finances.
A cross-month refund offsets spending in the refund's month, and can produce a negative net
category/spend value. The original month is not silently restated.

The reference transfer kind is for movements which the bank has determined should be excluded
from consumption (for example verified own-account transfers). Do not label every UPI/NEFT/IMPS
payment a transfer: a merchant UPI payment is still a purchase. The trusted adapter owns this
mapping. Credit-card settlement double-counting, loan principal/interest and investment flows
need bank-specific event semantics before integration.

## Categories and provenance

Priority: trusted kind → MCC → word-boundary merchant rules → AI suggestion → Other.
The merchant rules are illustrative synthetic/demo coverage, not a complete Indian merchant
catalogue. Conflicting MCC wins over merchant string. Unknown MCC does not imply a category.

Supported spending categories: Groceries, Dining, Transport, Shopping, Bills, Health,
Education, Entertainment, Rent, Other. Income/Transfers are reserved for bank-supplied kinds.
An AI proposal requires confidence ≥0.85, but **that confidence is uncalibrated model output**,
not an accuracy guarantee. Every AI category is marked `needs_review`. A valid category must
come from the fixed taxonomy; malformed/low-confidence outputs become Other for review.
The user can correct purchases. Linked refunds follow that correction. AI never overrides
customer corrections when an event is replayed.

## Nudges

Budget nudges are deterministic and opt-out. Incoming current-month purchases trigger a
notification at ≥80% of a customer-defined category budget, and another at ≥100%. Each threshold
is deduplicated by customer/category/month. Historical backfills do not create current alerts.
Refunds, budget changes and customer corrections supersede affected unread category alerts; persisted alerts represent the ledger
at the time of creation and include the triggering event and timestamp.

Nudges are only displayed in-app. No marketing recommendation, credit offer, investment product,
automatic transfer, external push, SMS or email is produced. Channel consent and quiet hours
would be required in a bank's external notification adapter. A simple monthly threshold is
not a predictive cash-flow model or personalised financial advice.

## Goals

The customer enters a target, already-saved amount and future date. Saved cannot exceed target.
Progress is `floor(saved × 100 / target)`. The illustrative daily plan is
`ceil((target − saved) / remaining_days)` in paise. Rounding up avoids underfunding the displayed
plan by a fraction of a paisa. Overdue goals are flagged rather than given a fictitious date.

Goals are tracking records. Saved amounts are customer-entered, not verified bank sub-balances.
There is no automatic funding, payment mandate, interest estimate or promise of a return.
This version creates goals and reports progress; goal contribution/edit/delete workflows remain
bank-specific extensions. A customer can create at most 50 goals.

## Language scope

English and Hindi summary, category, goal, savings, help and unsupported-request templates are
implemented. Intent recognition can handle Hinglish depending on the selected model, but no
broad multilingual quality claim is made. Category names currently remain English. Use a bank's
reviewed terminology, accessibility, speech and language acceptance suites before customer rollout.
