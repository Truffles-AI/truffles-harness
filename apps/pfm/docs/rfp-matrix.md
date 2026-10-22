# IB3 RFP capability matrix

**Use case:** conversational personal finance management inside mobile and internet banking.

| Requirement | Delivered behaviour | Boundary / remaining bank work |
|---|---|---|
| Chat assistant | English/Hindi monthly summaries, category spending, cash surplus, goals, follow-up context | Reviewed language set and bank-specific intents; no open-ended investment advice |
| Voice assistant | Opt-in upload/microphone, transcription, same chat flow, AI spoken response | Production ASR/TTS hosting, device testing and privacy/vendor approval |
| Automatic categorisation | Posted-event rules, MCC, merchant rules, Truffles AI suggestions, review/correction | Bank taxonomy/merchant master, held-out accuracy tests, trusted event mapping |
| Spend insights | Income, gross/net spend, refunds, excluded transfers, category distribution, ledger citations | Complete bank event feed; multi-currency, pending, reversal and credit-product semantics |
| Savings insights | Exact cash-surplus computation and user-entered goal progress | No account-balance inference or automated funds movement |
| Goal setting | Customer form/API, target date, saved amount, progress, illustrative daily plan | Verified balances, contributions/edit workflows and bank product integration |
| Proactive nudges | In-app 80%/100% category-budget alerts, deduplication, opt-out and dismissal | Outbox-driven external push/SMS/email, quiet hours and channel policies |
| Multi-customer safety | Composite bank/customer keys, verified identity-derived scope, scoped conversations | Production DB RLS, bank SSO/gateway and session lifecycle |
| Model portability | Configurable OpenAI-compatible chat endpoint, no tools required | Actual small-model evaluation and load qualification |
| Minimal model cost | Rules first, one bounded model call, ledger never embedded or sent for summarisation | Validate against real transaction mix and cost/latency targets |
| Explainability | Category method/review flag, source event IDs, visible fallback, audit activity | Reviewed explanation language and independent audit storage |
| Banking deployment | Clean domain/application/adapter boundaries, RS256 bank mode, container recipe, locked deps | Full security/reliability/operational readiness work in the readiness document |

## Why development becomes shorter

The AI implementation uses the same Truffles entry points throughout:

```python
from truffles import agent, speech_model

router = agent("openai:gpt-4.1-mini", kind="chat", instructions="Return a bounded intent JSON.")
speech = speech_model("openai")
```

Production PFM then adds the necessary bank domain boundaries around those calls: strict event
schemas, exact arithmetic, scoped persistence, consent, safe degradation and validated UI actions.
Provider SDK construction, model invocation and speech wire formats stay in the harness. A new
compatible model changes configuration rather than the ledger service or frontend.

This deliberately does not add multiple autonomous money-moving agents. Two bounded specialists
(intent and categorisation) cover the model tasks; deterministic services handle financial state.
Additional domain use cases can reuse the ports and composition root without embedding banking
business rules into the general-purpose Truffles package.

## Demonstration evidence

See [testing and evidence](testing.md) for offline coverage, live synthetic checks, performance
scope and the recorded video. Successful synthetic demonstrations establish working flows,
not production regulatory compliance or real-customer accuracy.
