# IB3 — Conversational Personal Finance Management

A runnable reference implementation for an Indian banking RFP, built on **Truffles**.

- Incoming posted transactions → automatic category → scoped ledger → budget nudge.
- English/Hindi chat → Truffles intent agent → exact deterministic financial answer.
- Audio question → Truffles speech transcription → same PFM flow → optional AI voice reply.
- Customer-controlled budgets, savings goals, category corrections and consent.
- OpenAI today; an OpenAI-compatible self-hosted chat model by configuration.

**Status:** working synthetic-data reference, not a certified or approved banking deployment.
The controls implemented here are useful integration foundations. The bank must complete
its own architecture review, security assessment, privacy/legal review, performance qualification
and model evaluation. See [production readiness](production-readiness.md).

## Run from the repository root

Requires Python 3.11+ and `uv`. No external database is required for the demo.

```sh
uv sync --project apps/pfm --locked
uv run --project apps/pfm truffles-pfm
```

Open <http://127.0.0.1:8765>. Without a key, the UI explicitly identifies local fallback mode.
For live OpenAI inference, export a key from your secret manager or terminal environment:

```sh
export OPENAI_API_KEY='YOUR_KEY'
export PFM_AUDIO_ENABLED=1
uv run --project apps/pfm truffles-pfm
```

Never put a real key in source, a browser bundle, the video, or a ticket. The provided key was
used only in the local server process for this demonstration; it is not shipped with the project.
The app is loopback-only by default. It is not published on the internet.

If you use the existing root virtual environment:

```sh
uv pip install --python .venv/bin/python -e '.[openai,audio]' -e apps/pfm
.venv/bin/python -m truffles_pfm
```

## Try the complete flow

1. The synthetic Ananya profile starts with ₹95,000 income and ₹33,539 net spending for
   the current Indian calendar month. An own-account ₹10,000 transfer is excluded.
2. Ask **“Summarise my spending this month.”** The response cites the ledger event IDs.
3. Add a **Swiggy ₹1,450** incoming transaction. It is categorised as Dining without an
   LLM call, moves Dining from ₹2,880 to ₹4,330, and triggers the 80% budget nudge.
4. Add **Saffron Leaf Vegetarian Restaurant ₹350** to exercise the model classifier.
   Its category is explicitly a suggestion requiring review. Choose Edit to correct it.
5. Create a **Goa holiday** goal: target ₹60,000, saved ₹15,000, future date.
   Ask **“How are my goals doing?”** No funds are moved.
6. Ask **“इस महीने मेरे खर्च का सारांश बताओ”** for a Hindi summary.
7. Open Preferences and opt into voice. Record up to 20 seconds, or upload the supplied
   [synthetic question](../artifacts/voice-question.wav). Click Listen to hear the answer.
8. Open Activity trail. It records state changes without storing raw prompts or audio.
9. Turn off AI interpretation: questions use an explicitly labelled conservative local fallback.

The live-test run may have added transactions and goals to the running demo. For a clean
independent dataset, set `PFM_DATABASE=.pfm/fresh-demo.sqlite3` before launch. Do not delete
an existing bank database to reset a demo.

## Documentation map

- [Architecture and boundaries](architecture.md)
- [API and bank integration](api.md)
- [Configuration and self-hosted model migration](configuration.md)
- [Financial semantics and nudge rules](financial-semantics.md)
- [Security and production readiness](production-readiness.md)
- [Testing, evidence and demo recording](testing.md)
- [RFP capability matrix](rfp-matrix.md)

The implementation lives in `apps/pfm/src/truffles_pfm`. Its AI adapter imports only the
public Truffles API. The core harness gained a small speech adapter in `src/truffles/speech.py`.
