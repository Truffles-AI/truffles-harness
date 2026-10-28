# Truffles

A small Python harness for AI agents, multi-agent workflows and retrieval-augmented
generation (RAG). Create an agent in two lines, add tools or knowledge, and compose
it into a workflow. LangChain and LangGraph execute the work; Truffles shortens the
setup while keeping their native APIs accessible.

```python
from truffles import agent
answer = agent("openai:gpt-4.1-mini").run("Explain RAG in one sentence.")
print(answer.text)
```

Python **3.11+** · MIT license · Optional provider and vector-store dependencies

[Quickstart](docs/quickstart.md) · [Documentation](docs/README.md) ·
[API reference](docs/api.md) · [Banking demo](apps/pfm/README.md)

## Install

This project is **not published on PyPI**. Install from this repository; do not
assume an unrelated package named `truffles` on a package registry is this library.
Install the released wheel directly, without cloning the repository:

```bash
python -m pip install 'truffles[openai] @ https://github.com/Truffles-AI/truffles-harness/releases/download/v0.3.0/truffles-0.3.0-py3-none-any.whl'
```

Replace `[openai]` with the extras you need, or omit it for the core library.
Alternatively, clone the public repository for an editable installation:

```bash
git clone https://github.com/Truffles-AI/truffles-harness.git
cd truffles-harness
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[openai]'
export OPENAI_API_KEY='your-key'
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell and set the
key using `$env:OPENAI_API_KEY = 'your-key'`. Credentials belong in your environment
or secret manager. Truffles does not load `.env` files automatically.

Choose only the integrations you need:

```bash
python -m pip install -e .                       # Core, local BM25 and native workflows
python -m pip install -e '.[anthropic,qdrant]'    # One provider and one vector backend
python -m pip install -e '.[providers]'          # All six named chat providers
python -m pip install -e '.[openai,audio]'       # OpenAI chat, transcription and speech
```

For reproducible development, use [uv](https://docs.astral.sh/uv/) and the committed
lockfiles; see [Development and tests](#development-and-tests).

## Create an agent with tools

```python
from truffles import agent, tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

bot = agent("openai:gpt-4.1-mini", tools=[multiply], max_model_calls=5)
print(bot.run("What is 17 times 23?").text)
```

The native ReAct loop chooses tools, validates arguments and feeds results back to
the model. Agents have a model-call budget (20 by default), support `arun`,
streaming, structured output and explicit conversation memory. Reuse agent instances
across calls to reuse their provider clients. See [agents](docs/agents.md) and
[runtime controls](docs/runtime.md).

## Add RAG

```python
from truffles import rag
knowledge = rag(["Support is available Monday to Friday, 09:00–18:00 IST."])
answer = knowledge.ask("When is support available?", model="openai:gpt-4.1-mini")
print(answer.text)
print(answer.sources)
```

Without embeddings, retrieval uses local BM25. `knowledge.retrieve(query)` requires
no model or credentials. For hybrid lexical and vector retrieval:

```python
from truffles import rag
knowledge = rag(
    ["Truffles wraps LangChain and LangGraph.", "Qdrant stores vectors."],
    embedding="openai:text-embedding-3-small",
    backend="qdrant",
    algorithm="hybrid",
    store_options={"location": ":memory:"},
)
```

Install `.[openai,qdrant]` for that example. Other algorithms include similarity,
MMR, score thresholds, multi-query retrieval and pluggable reranking. Pass
`knowledge=knowledge` to an agent to let it choose when to retrieve, or use
`knowledge.ask(...)` to retrieve before every answer. See [RAG](docs/rag.md) and
[vector stores](docs/vector-stores.md) for persistence and backend-specific settings.

## Compose multiple agents

```python
from truffles import agent, sequence
writer = agent("openai:gpt-4.1-mini", instructions="Write a concise draft.")
editor = agent("openai:gpt-4.1-mini", instructions="Improve the supplied draft.")
pipeline = sequence(writer, editor)
print(pipeline.invoke("Explain embeddings to a beginner."))
```

Use `parallel` for independent specialists, `route` for conditional execution and
`supervisor` for agent delegation. Existing LangChain Runnables compose directly;
native `StateGraph` and an agent's `.native` graph remain available for cycles,
handoffs, subgraphs and durable execution. [Workflow examples](docs/workflows.md).

## Integrations

| Capability | Included integrations |
| --- | --- |
| Chat models | OpenAI, Anthropic, Gemini, Groq, xAI/Grok, Perplexity |
| Web search | Perplexity Sonar/Agent and xAI Responses web search |
| Embeddings | OpenAI, Gemini, Cohere, Hugging Face, FastEmbed |
| Vector stores | Memory, Qdrant, Chroma, Pinecone, Weaviate, Milvus, PGVector, Redis, FAISS, Elasticsearch |
| Speech | OpenAI transcription and text-to-speech; compatible endpoints or custom adapters |
| Extensions | Native models, tools, retrievers, stores and Runnables; custom factory registration |

Provider prefixes, install extras and credential variables are documented in
[providers](docs/providers.md). Groq inference and xAI's Grok are separate
integrations. Backend capabilities differ: accepting a native integration does not
mean every provider or vector database implements every feature.

An OpenAI-compatible self-hosted model can use the same agent API:

```python
import os
from truffles import agent

bot = agent(
    "openai:your-served-model-id",
    kind="chat",
    model_options={
        "base_url": "http://localhost:8000/v1",
        "api_key": os.environ["MODEL_API_KEY"],
        "temperature": 0,
        "max_tokens": 256,
    },
)
print(bot.run("Say hello.").text)
```

Use your server's model identifier and supported parameters. Tool calling and
structured-output support depend on that server and model. See
[compatibility](docs/compatibility.md), [web search](docs/search.md),
[speech](docs/speech.md) and [extensions](docs/extensions.md).

## Try it without credentials

After installing the core library:

```bash
python examples/offline.py
python examples/async_team.py
python examples/graph.py
```

These examples use deterministic test doubles, not live inference. They demonstrate
tool execution, RAG, pipelines, parallel teams, delegation and native graphs.
[More examples](examples/README.md).

## Process images, documents and audio

Install the `processing` extra and the Tesseract system executable for local OCR (add `tables` for table layout extraction):

```python
from truffles import ocr, read_documents, transcribe

image_text = ocr("invoice.png")
pages = read_documents("statement.pdf")  # Text extraction + OCR for textless pages
transcript = transcribe("meeting.wav")  # Configured OpenAI speech endpoint by default
```

Each processor returns a native Document (or a list for `read_documents`), ready
for RAG. Since 0.3.0, OCR also supports automatic quarter-turn correction.
Install `tables` for `extract_tables("statement.pdf")`, which returns reviewable
rows/cells with CSV/JSON exports. Use `processing_tools("./uploads")` to give an agent directory-scoped file
reading and OCR; supply a speech adapter to enable its transcription tool.
[Processing guide](docs/processing.md) covers setup, custom speech endpoints, input
limits and the distinction between local processing and external transcription.

## Evaluate and observe

Evaluate agents, workflows or retrievers against a dataset and fail CI on regressions:

```python
from truffles import EvalCase, evaluate, exact_match

report = evaluate(str.upper, [EvalCase("hello", "HELLO", id="greeting")],
                  evaluators={"accuracy": exact_match})
report.assert_passed()
report.save("evaluation.json")
```

Built-in checks cover exact answers, substrings, JSON/schema validity and retrieval
recall. Add business-specific scorers or an opt-in LLM judge through Truffles.
`aevaluate` supports bounded concurrency. Reports include pass rates, target latency,
observed token usage and errors; saved reports omit outputs by default.

```python
from truffles import Observer, agent

observer = Observer()
bot = agent("openai:gpt-4.1-mini")
bot.run("Explain RAG.", config={"callbacks": [observer]})
print(observer.summary())
```

The observer captures parent-child spans for graphs, models, tools and retrieval
without recording prompts or answers. Export to local JSONL or a custom sink;
existing native monitoring callbacks can run alongside it.
[Evaluation guide](docs/evaluation.md) · [Observability guide](docs/observability.md)

## Banking reference application: IB3 PFM

The included [PFM application](apps/pfm/README.md) demonstrates automatic transaction
categorisation, spend insights, budget nudges, savings goals, English/Hindi chat and
opt-in voice on synthetic Indian banking data. All AI calls go through Truffles.

From the repository root, with uv installed:

```bash
uv sync --project apps/pfm --locked
uv run --project apps/pfm truffles-pfm
```

Open **http://127.0.0.1:8765**. Without a key, the UI labels its local fallback.
Set `OPENAI_API_KEY` for live inference and `PFM_AUDIO_ENABLED=1` for voice.
See [configuration](apps/pfm/docs/configuration.md) for self-hosted models.

[Watch/download the two-minute demo](apps/pfm/artifacts/truffles-pfm-demo.mp4) ·
[Architecture](apps/pfm/docs/architecture.md) ·
[API](apps/pfm/docs/api.md) · [Test evidence](apps/pfm/docs/testing.md)

This is a synthetic RFP reference implementation. A bank deployment still requires
bank identity/core-system integration, production persistence and independent
security and privacy qualification. The [production readiness document](apps/pfm/docs/production-readiness.md)
records these boundaries explicitly.

## Development and tests

```bash
uv sync --locked --extra providers --extra memory --extra qdrant --extra chroma --extra faiss --extra processing --extra tables
uv run ruff check src tests examples scripts
uv run ruff format --check src tests examples scripts
uv run pytest --cov=truffles --cov-fail-under=90
uv run python scripts/check_docs.py
uv run python -m build
```

The application has its own dependency environment and tests:

```bash
uv sync --project apps/pfm --locked
uv run --project apps/pfm ruff check apps/pfm
uv run --project apps/pfm ruff format --check apps/pfm
uv run --project apps/pfm pytest apps/pfm/tests --cov=truffles_pfm --cov-fail-under=85
uv build --project apps/pfm
```

Install the Tesseract executable before running the real OCR checks
(`brew install tesseract` on macOS or `apt-get install tesseract-ocr` on Ubuntu).

GitHub Actions runs both suites on Python 3.11, 3.12 and 3.13. Provider contract
tests use real adapters with synthetic keys and intercepted HTTP; pytest disables
outbound network sockets. Tests need no paid credentials. Live checks are separate,
explicit opt-in scripts. See [testing](docs/testing.md) and
[validation](docs/validation.md) for scope and recorded results.

## Repository layout

```text
src/truffles/       Library, provider factories, agents, workflows and RAG
tests/              Library unit, provider-contract and local integration tests
examples/           Runnable offline and local-vector-store examples
docs/               Detailed library guides and API reference
apps/pfm/           Banking reference app, tests, docs, Dockerfile and demo artifacts
scripts/            Documentation checks and local benchmarks
.github/workflows/  Library and PFM CI workflows
```

Start with the [documentation index](docs/README.md) or
[project guide](docs/PROJECT_GUIDE.md). Licensed under [MIT](LICENSE).
