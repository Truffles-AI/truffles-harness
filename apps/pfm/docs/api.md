# HTTP API and bank integration

The local API browser is <http://127.0.0.1:8765/api/docs>, served entirely from this
application without third-party scripts. Expand a route to inspect its OpenAPI definition.
The machine-readable schema is `/api/openapi.json`. Bank mode disables both endpoints.

All customer endpoints require `Authorization: Bearer <token>`. Tokens stay server-side or
in the embedding bank application's authenticated session; never pass them in URLs.

## Identity contract

Bank mode verifies RS256 only, expiry, issued-at, issuer and audience. Required claims:

```json
{
  "sub": "opaque-customer-id",
  "bank_id": "opaque-bank-id",
  "iss": "https://identity.bank.example",
  "aud": "truffles-pfm",
  "iat": 1792530000,
  "exp": 1792530300,
  "permissions": ["pfm:read", "pfm:write"]
}
```

Values above are illustrative, not valid credentials. `sub` and `bank_id` accept 1–64 ASCII
letters, digits, underscores and hyphens. Service-to-service ingestion needs `pfm:ingest` and
an explicitly customer-scoped identity issued by the trusted bank gateway. The ordinary customer
frontend must not receive this permission. Diagnostics require `pfm:ops`.

The demo uses random, process-local bearer tokens for synthetic Ananya and Kabir profiles.
Its public session endpoint intentionally grants Ananya demo read/write/ingestion/ops access.
**Never expose demo mode to an untrusted network.** There is no demo token endpoint in bank mode.
JWT key rotation/JWKS refresh, session revocation, mTLS and identity federation are gateway
integration responsibilities; this reference verifies a pinned PEM public key.

## Routes

| Method / path | Permission | Behaviour |
|---|---|---|
| GET `/healthz` | none | Process liveness; no model request |
| GET `/readyz` | none | Reference database readiness |
| GET `/api/dashboard` | read | Current-month summary, transactions, budgets, goals, nudges and preferences |
| POST `/api/transactions` | ingest | Idempotent trusted posted event ingestion |
| PATCH `/api/transactions/{event_id}/category` | write | Customer correction of a purchase category; updates linked refunds |
| PUT `/api/budget` | write | Set a category's monthly budget |
| POST `/api/goals` | write | Create a tracking goal; no transfer instruction |
| PUT `/api/preferences` | write | Set AI consent, voice consent and nudge preference |
| POST `/api/nudges/{id}/dismiss` | write | Dismiss a scoped in-app notification |
| POST `/api/chat` | read | Interpret and answer a financial question |
| GET `/api/audit` | read | Latest 100 audit entries within customer scope |
| GET `/api/diagnostics` | ops | Process model counts, fallback counts, latency and safe error type |
| POST `/api/voice` | read + AI/voice consent | Multipart audio → transcript and chat answer |
| GET `/api/speech/{answer_id}` | read + AI/voice consent | Speak a scoped answer no more than 15 minutes old |

## Incoming event

```json
{
  "event_id": "bank-posting-000123",
  "amount_paise": 145000,
  "currency": "INR",
  "occurred_at": "2026-10-21T10:30:00+05:30",
  "merchant": "Swiggy",
  "mcc": "5812",
  "kind": "purchase",
  "original_event_id": null
}
```

Amounts are positive integer paise, never floating rupees. Booleans, strings and floats are
rejected as amounts. Only INR is accepted. The bank adapter supplies timezone-aware posting time,
kind and authoritative MCC. `event_id` must remain stable across retries. Future postings beyond
five minutes are rejected. Supported kinds: purchase, income, transfer, refund. Refunds require
the original purchase ID and cannot precede it or cumulatively exceed it. A late/out-of-order
refund is rejected until its original exists; the bank adapter must retry or dead-letter it.

A successful response includes `transaction`, `duplicate`, and `nudges`.
The visible demo UI refreshes its ledger every 10 seconds, so events posted by a bank adapter
appear with their nudges without a customer button click. Background tabs pause this polling. Exact retries return
HTTP 200 with `duplicate: true` and no new nudge. Mismatched retries return 409. Ingestion errors
do not partially commit a ledger event or audit entry. Pending transactions, reversals,
chargebacks and multi-currency conversion require explicit additional bank event mappings;
do not relabel them as supported kinds to bypass validation.

## Chat request/response

```json
{"message":"How much did I spend on dining?", "conversation_id":null}
```

Response includes controlled `answer`, `intent`, `mode` (`truffles` or `fallback`), `facts`,
ledger `citations`, `conversation_id`, `answer_id` and timestamp. Return the conversation ID on
follow-up turns. An ID issued to another bank/customer returns 404. Source citations are current
ledger event IDs, not external web references. AI-produced text is never executed or treated
as a ledger write. Goal/budget creation is intentionally a validated form/API action.

## Error and resource contract

- 401 invalid/missing identity; 403 insufficient permission or missing consent.
- 404 unknown or out-of-scope resource; 409 idempotency conflict.
- 413 oversized request; 415 unsupported audio MIME type; 422 validation failure.
- 429 per-process throttle or saturated voice service; 503 unavailable voice/capacity.
- Generic 500 with request ID for unexpected failures, never a raw provider exception.
- Ordinary bodies ≤16 KiB, voice multipart ≤2.2 MB, audio bytes ≤2 MB, question ≤1,000 characters.
- Chat/reading requests are throttled at 40/minute per bank/customer/permission in this process.
  A bank gateway must implement shared limits and request deadlines across replicas.

This reference returns all scoped transactions in the dashboard; production must add pagination,
range queries, retention, and indexes appropriate to the bank's expected ledger sizes.
