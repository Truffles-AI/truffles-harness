# Testing and reproducibility

## Run the same checks as CI

```bash
uv sync --locked --extra providers --extra memory --extra qdrant --extra chroma --extra faiss --extra processing
uv run ruff check src tests examples scripts
uv run ruff format --check src tests examples scripts
uv run pytest --cov=truffles --cov-fail-under=90
uv run python scripts/check_docs.py
uv run python -m build
```

CI is configured for Python 3.11, 3.12 and 3.13. The validation report distinguishes
configured jobs from environments actually exercised during development.

`uv.lock` resolves optional integrations as well as the selected test environment.
Only selected extras are installed by `uv sync`. For an application, pin a tested
lock and update integrations deliberately; upper major-version bounds are not a
promise that every future minor release is compatible.

## Test layers

1. Agent behavior uses a real compiled LangGraph with `ScriptedChatModel` responses.
   It covers tools, limits, structured output, history isolation, callbacks,
   interruptions, resume, streaming and async execution.
2. Workflow tests execute sequential, parallel, routed and hierarchical systems.
   A synchronization barrier verifies that async branches actually overlap.
3. Retrieval tests check ranking, empty results, chunk metadata, deduplication,
   weighting, query bounds, rerankers, source artifacts and validation errors.
4. Local integration tests execute actual InMemoryVectorStore, Qdrant, Chroma and FAISS local
   stores with deterministic test embeddings. They do not mock vector search.
5. Provider contract tests instantiate the actual integration and provider SDK,
   intercept HTTP, inspect outgoing JSON/headers and return protocol-shaped JSON
   or SSE. Tests cover all six named chat providers, search, tools, embeddings,
   errors and native streaming where explicitly listed in compatibility notes.

The HTTP stubs use `respx`. Some current SDKs use the `httpx2` fork; a test-only
transport bridge routes its serialized requests through the same strict fixtures.
No production model or parser is replaced by that bridge. Perplexity search tests
exercise the public SDK transport and Truffles' native search response conversion. Test fixtures set only
synthetic credentials and disable tracing. `pytest-socket` blocks outbound network
sockets; Unix sockets are permitted for local async event-loop infrastructure.

## What a contract test proves

It proves that the installed adapter and SDK serialize the exercised request in
the expected shape and parse the supplied response correctly. It cannot prove
that every field combination is accepted by a remote service, that a model is
currently offered to your account, or that an LLM will choose the right tool.
Fixtures are based on official documentation and installed SDK contracts. A
provider upgrade requires rechecking both. See [sources](sources.md).

There are no hidden live tests, credential reads or paid API calls in this suite.
Optional service-backed vector integrations are not labeled live-verified based
on a generic factory mock.

## Write an offline test

```python
from langchain_core.messages import AIMessage
from truffles import agent
from truffles.testing import ScriptedChatModel

model = ScriptedChatModel(responses=[AIMessage(content="Expected answer")])
assert agent(model).run("Question").text == "Expected answer"
assert model.requests[0][-1].content == "Question"
```

For tools, place an `AIMessage` with `tool_calls` before the final answer. The
runtime will execute the actual local tool. Scripts never cycle automatically;
exhaustion raises so accidental extra model calls fail tests.

For meaningful RAG evaluation, add domain-specific expected retrieval sets and
answer-grounding checks. Hash embeddings validate plumbing, not semantic quality.
Use your chosen production embeddings in an explicitly authorized integration
suite when that evaluation is needed.

## Evaluation and observability tests

`tests/test_evaluation.py` checks deterministic and semantic-judge contracts,
retrieval recall, invalid scores, target/scorer failures, report privacy, CI gates,
async concurrency, stable ordering and cancellation. `tests/test_observability.py`
executes real agent/tool graphs, parallel workflows, retrieval and streaming to
verify span ancestry, per-call usage, bounded buffers, export failures and thread
safety. Provider contract tests additionally assert observed model/tool counts and
usage for the OpenAI, Groq, xAI and Anthropic adapters over stubbed HTTP.

`examples/evaluation.py` is a credential-free CI example. Its scripted judge only
tests the judging interface; it does not assess a live model's answer quality.

## Processing tests

Install the Tesseract system executable for the real OCR integration check; CI
installs it explicitly. `tests/test_processing.py` covers real image/scanned-PDF
OCR, PDF text extraction, local concurrent readers, resource bounds, tool path
restrictions, source artifacts, async tools and native speech SDK wire contracts
against stubbed HTTP. No speech credentials or paid calls are needed.
