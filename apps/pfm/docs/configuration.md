# Configuration and model portability

| Environment variable | Default / meaning |
|---|---|
| `PFM_MODE` | `demo`; use `bank` to disable demo identity issuance and synthetic seeding |
| `PFM_DATABASE` | `.pfm/pfm.sqlite3`; local reference store |
| `PFM_PORT` | `8765`; launcher binds to 127.0.0.1 |
| `PFM_MODEL` | `openai:gpt-4.1-mini` |
| `PFM_MODEL_API_KEY` | Explicit model secret; otherwise `OPENAI_API_KEY` |
| `PFM_MODEL_BASE_URL` | Native provider default; explicit compatible URL for self-hosting |
| `PFM_MODEL_TIMEOUT` | `12` seconds, allowed range 1–60; SDK retries disabled |
| `PFM_AUDIO_ENABLED` | `1` to enable speech; disabled by default |
| `PFM_AUDIO_BASE_URL` | Independent compatible audio endpoint |
| `PFM_AUDIO_API_KEY` | Audio-specific key; otherwise `OPENAI_API_KEY` |
| `PFM_TRANSCRIPTION_MODEL` | `whisper-1` |
| `PFM_SYNTHESIS_MODEL` | `tts-1` |
| `PFM_ALLOWED_HOSTS` | `127.0.0.1,localhost`; comma-separated trusted hosts |
| `PFM_JWT_PUBLIC_KEY_FILE` | PEM RS256 public verification key, required in bank mode |
| `PFM_JWT_ISSUER` | Exact expected issuer, required in bank mode |
| `PFM_JWT_AUDIENCE` | `truffles-pfm` |

The application disables inherited LangSmith tracing to avoid accidental third-party prompt
export. No credentials are written to the audit log or returned to the frontend. Secrets belong
in the bank's vault/workload environment. Use separate provider projects and budget limits.

## Self-hosted small model

An OpenAI-compatible endpoint needs plain chat completions; tool calling and constrained-output
extensions are **not required**. Use the actual served model name, not a product name assumed
by the application:

```sh
export PFM_MODEL='openai:YOUR_SERVED_MODEL_ID'
export PFM_MODEL_BASE_URL='http://127.0.0.1:8000/v1'
export PFM_MODEL_API_KEY='LOCAL_SERVER_KEY'
export PFM_AUDIO_ENABLED=0
uv run --project apps/pfm truffles-pfm
```

For a remote endpoint, HTTPS is mandatory. Credentials in URLs, URL query strings and fragments
are rejected. A server operator controls the endpoint, never an API request body. Configure
network-level egress allowlists and prevent redirects to unapproved services in production.

The only AI work is short intent JSON and occasional unknown-merchant classification. Rules
and MCC categories avoid inference for common transactions. Two reusable Truffles agents reuse
native client pools. Classification does not send the transaction amount. Summaries and nudges
use exact arithmetic without inference. Changing the model does not change the ledger rules.

A different model must pass the same held-out intent/category/language/adversarial evaluation.
The compatible HTTP contract is tested with a stub server; **a real self-hosted model has not
been benchmarked or qualified by this project**. Never infer quality parity from API compatibility.

## Independent speech hosting

```sh
export PFM_AUDIO_ENABLED=1
export PFM_AUDIO_BASE_URL='https://speech.bank.internal/v1'
export PFM_AUDIO_API_KEY='SPEECH_SERVICE_KEY'
export PFM_TRANSCRIPTION_MODEL='YOUR_ASR_MODEL_ID'
export PFM_SYNTHESIS_MODEL='YOUR_TTS_MODEL_ID'
```

This requires the `/audio/transcriptions` multipart and `/audio/speech` contracts. For other
ASR/TTS protocols, implement `transcribe(bytes, filename=...) -> str` and `speak(str) -> bytes`,
and supply it to `create_app(speech=adapter)` through a custom composition root. The native
Truffles `speech_model(adapter)` accepts custom adapters without introducing SDK dependencies
into PFM business code.

## Container reference

Build from the repository root:

```sh
docker build -f apps/pfm/Dockerfile -t truffles-pfm:local .
```

The image installs from `apps/pfm/uv.lock`, runs as UID 10001, defaults to `bank` mode and
requires explicit identity configuration. It has been smoke-tested with a read-only root
filesystem, a temporary writable data directory and loopback-only port publication. Missing
bank identity configuration is rejected at startup. See [container smoke evidence](../artifacts/container-smoke.json).

A bank deployment must inject `PFM_JWT_PUBLIC_KEY_FILE`, `PFM_JWT_ISSUER`, approved model
configuration and `PFM_ALLOWED_HOSTS`, mount data storage with appropriate ownership, terminate
TLS at the trusted gateway, and pin/scan the base image according to its release policy.
The container recipe is not a managed HA bank data deployment.
