# Truffles PFM / IB3

Working synthetic Indian banking PFM reference: categorisation, budget nudges, goals,
English/Hindi chat and opt-in voice, with all AI routed through Truffles.

Start with [the guide](docs/README.md), [architecture](docs/architecture.md),
[production readiness](docs/production-readiness.md), and [testing evidence](docs/testing.md).

From the repository root:

```sh
uv sync --project apps/pfm --locked
uv run --project apps/pfm truffles-pfm
```

Open http://127.0.0.1:8765. Set `OPENAI_API_KEY` to enable live inference and
`PFM_AUDIO_ENABLED=1` for voice. Without a key, the UI labels its local fallback.

[Watch the demo](artifacts/truffles-pfm-demo.mp4).
