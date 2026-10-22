# Production readiness and security boundaries

This is an RFP reference application with a working synthetic demonstration. It is **not a
bank-certified production system**, a complete regulatory compliance solution or a replacement
for the bank's ledger. The following is an engineering handover, not legal advice.

## Implemented controls and evidence

| Concern | Implemented | Evidence |
|---|---|---|
| Cross-customer access | Verified scope on every repository read/write; no request-body customer selection | API/service isolation tests, including conversations and speech |
| Identity | RS256 only, required claims, issuer/audience/expiry checks, separated permissions | Forged algorithm, wrong issuer/audience, expiry and role tests |
| Financial integrity | Integer paise, validated refunds, posted-event semantics, no account balance inference | Arithmetic, concurrency, refunds and month-boundary tests |
| Duplicate delivery | Canonical fingerprint, scoped event key, in-transaction recheck | 20 simultaneous replays commit one event |
| Model safety | No ledger data in prompt, no write tools, whitelist JSON, deterministic answer templates | Invalid output, injection-shaped input, provider outage and import-boundary tests |
| Degraded operation | Bounded requests, process-local breaker, visible fallback, safe diagnostics | Invalid JSON and exhausted-model tests; live regression documented |
| Consent | AI/voice off by default in bank mode; voice requires both; opt-out nudges | Consent-gated model calls and speech tests |
| API protection | Body limits, strict fields, same-origin policy, host allowlist, no-store, CSP, no raw error echo | API adversarial tests |
| Audit | Hash-linked change records within transaction; no raw prompts or audio | Chain verification and rollback tests |
| Credentials | Environment/vault inputs; no keys in source/browser/artifacts | Source and artifact scan before handover |
| Synthetic evaluation | Two synthetic profiles, offline tests, real SDK stub contracts, live OpenAI flows | Test logs, live-results JSON and recorded browser demo |

## Required bank-owned work before live customer data

| Area | Remaining integration / qualification |
|---|---|
| Identity / gateway | Connect real bank SSO, session revocation, rotating keys/JWKS, mTLS for bank ingestion, WAF, CSRF posture appropriate to bank cookies, shared rate limits |
| Event correctness | Approve CBS/card/UPI event mappings, posted/pending/reversal/refund semantics, account linkage, transfers, backfills, reconciliation, ordering, replay and dead-letter policies |
| Data platform | Replace local SQLite with managed HA storage, enforce DB-level RLS, schema migrations, indexes and pagination, encrypted disks/backups, keys in KMS/HSM, PITR and tested recovery |
| Audit | Export to independent append-only/WORM audit with external integrity anchors; the local hash chain can be rewritten by a privileged database operator |
| Privacy / consent | Approved notices and purpose mapping, consent evidence/versioning, minimisation, rights requests, erasure/retention/legal hold, vendor processing agreements and approved data residency |
| Model hosting | Bank-approved model endpoint/egress policy, server isolation, inference load tests, held-out evaluations, prompt-injection red team, confidence calibration, drift monitoring and rollback |
| Voice | Bank-approved ASR/TTS hosting, audio duration/content validation, provider retention controls, accessibility, Hindi/Hinglish speech evaluation, device permission flows |
| Observability | SIEM integration and alerting, vendor failures, latency/cost dashboards, reconciliation alarms, SLOs; do not export raw customer prompts by default |
| Resilience | Queue consumers/outbox, shared circuit breakers/limits, bounded background workers, failover, disaster recovery, operational runbooks, penetration and load tests |
| Notification policy | Opt-in channel adapters, quiet hours, frequency limits, reviewed templates, opt-out propagation and delivery receipts |
| Release assurance | Dependency/image scanning, SBOM/signatures, reproducible release pipeline, staging/UAT, approvals, independent security review and bank-specific acceptance criteria |

## Important limits

The local dashboard loads scoped records into memory and is not a scale qualification. The
reference rate limiter and circuit breaker are process-local. No guaranteed throughput, latency,
classification accuracy, financial outcome or bank regulatory compliance is claimed. The small
self-hosted model path is contract-tested, not evaluated against an actual self-hosted model.

Redaction is a defence in depth regex, not comprehensive DLP: names or unusual identifiers may
remain in merchant/question text. Synthetic OpenAI testing must not be taken as permission to
send real bank data to that endpoint. Audio necessarily contains the speaker's raw words; the
application does not save it, but provider handling remains a separate vendor/privacy decision.

Recent generated answer text is stored for playback (up to 20 answer entries per customer,
playback expires after 15 minutes). Expiry blocks access but is not automatic physical deletion.
Conversation context, ledger, goals and audit need an approved retention scheduler. Revoking
AI/voice consent deletes stored answers and stops new external model use; it does not cancel
an already-sent provider request or delete the provider's prior data.

Local demo tokens intentionally have ingestion permission to make a button-driven demonstration
possible. Bank customers must only get read/write permissions; a trusted bank adapter gets
customer-scoped ingestion permissions. Never expose the local demo session endpoint publicly.

## Official reference points for the bank's review

Applicability, amendments, commencement dates and institution-specific obligations must be
confirmed by the bank's legal and security teams. This project does not claim to implement them.

- [RBI Master Directions index](https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx)
  — locate the current applicable Digital Payment Security Controls and IT governance material.
- [MeitY Digital Personal Data Protection Rules 2025](https://www.meity.gov.in/documents/act-and-policies/digital-personal-data-protection-rules-2025-gDOxUjMtQWa?pageTitle=Digital-Personal-Data-Protection-Rules-2025)
  — review the final official rules and phased commencement with counsel.
- [OpenAI transcription contract](https://developers.openai.com/api/reference/python/resources/audio/subresources/transcriptions/methods/create)
  and [speech contract](https://developers.openai.com/api/reference/python/resources/audio/subresources/speech/methods/create)
  — the implemented speech wire formats, verified with native-SDK stub tests and live synthetic calls.

The exact RFP text beyond the IB3 description was not provided. This reference maps that
capability, not undisclosed contractual security, performance, integration or certification terms.
