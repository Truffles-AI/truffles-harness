# Public API reference

## Agent construction

`agent(model, **options) -> Agent` is the convenience factory for:

```python
from truffles import Agent
# Signature overview; model is required.
# Agent(model, tools=(), instructions=None, name=None, kind="react",
#       knowledge=None, memory=False, checkpointer=None, max_model_calls=20,
#       model_options=None, middleware=(), response_format=None, **native_options)
```

| Parameter | Meaning |
| --- | --- |
| `model` | `provider:model` identifier or native chat model |
| `tools` | Native tools, annotated callables, or supported native tool dictionaries |
| `instructions` | System prompt |
| `name` | Native graph/runnable name |
| `kind` | `react` or `chat`; chat rejects tools/knowledge |
| `knowledge` | Object exposing `as_tool()`, normally a `RAG` |
| `memory` | Use an in-process checkpointer |
| `checkpointer` | Caller-managed native saver; exclusive with `memory=True` |
| `max_model_calls` | Positive per-run model request bound; default 20 |
| `model_options` | Native provider constructor kwargs, only for string models |
| `middleware` | Native middleware appended after the model-call limit |
| `response_format` | Schema or native structured-output strategy |
| `**native_options` | Other `create_agent` options |

Methods:

- `run(prompt, *, config=None, thread_id=None, **kwargs) -> Result`
- `await arun(...) -> Result`
- `stream(prompt, *, config=None, thread_id=None, stream_mode="updates", **kwargs)`
- `astream(...)` returns an async iterator of native events.
- `as_runnable(text_only=True)` returns a native RunnableLambda; `False` returns Results.
- `as_tool(name=None, description=None)` returns a stateless specialist tool.
- `.native` is the compiled graph; `.model` is the configured chat model.

Run kwargs go to graph invocation, for example `context=...`. Stream kwargs go to
graph streaming. Native return-shape-changing options should be used on `.native`
when they no longer produce an agent-state dictionary.

## Result

`Result` fields are `text`, `messages`, `structured`, `sources`, `usage` and `state`.
`str(result)` returns `text`. `interrupted` checks the native interruption output.
`Result.from_state(state)` converts a native agent-state dictionary. Usage covers
all returned messages, which includes earlier history for a checkpointed thread.

## Provider factories

- `chat_model(model, **options)`: native chat model factory or instance passthrough.
- `embeddings(model, **options)`: native embeddings factory or instance passthrough.
- `register_provider(name, factory, replace=False)`
- `register_embeddings(name, factory, replace=False)`

Factory options retain their native spelling and semantics. New model identifiers
are not restricted by an internal model catalog.

## RAG and ingestion

`rag(documents=None, *, embedding=None, backend="memory", store=None,
retriever=None, algorithm=None, k=4, chunk_size=1000, chunk_overlap=150,
search_options=None, store_options=None, max_context_chars=16000) -> RAG`

`documents` is a list of strings or native Documents. `algorithm=None` selects
similarity if embeddings/a store is supplied, otherwise BM25. Explicit algorithms
are `bm25`, `similarity`, `mmr`, `similarity_score_threshold` and `hybrid`.

`retriever=` is exclusive with documents/store/embedding/algorithm. Other ingestion
settings are irrelevant when directly supplying a retriever; configure that
retriever's `k` and search policy before passing it in. `store=` attaches an existing
native store and is exclusive with documents/embedding/store_options.

`RAG(retriever, *, store=None, max_context_chars=16000)` provides:

- `retrieve(query, *, config=None) -> list[Document]`
- `await aretrieve(...) -> list[Document]`
- `ask(query, *, model, config=None, **run_options) -> Result`
- `await aask(...) -> Result`
- `as_tool(name="search_knowledge", description="Search the knowledge base.")`
- `.retriever` and `.store` native access.

`split(documents, *, chunk_size=1000, chunk_overlap=150, **options)` uses a native
recursive character splitter with start indices. Additional options reach the
splitter. Chunk IDs are content/metadata hashes.

## Retrieval components

- `BM25Retriever.from_texts(texts, k=4, k1=1.5, b=0.75)`
- `BM25Retriever.from_documents(documents, ...)`
- `HybridRetriever(retrievers=[...], weights=None, k=4, rank_constant=60)`
- `MultiQueryRetriever(retriever=..., generator=..., max_queries=3, k=8)`
- `RerankRetriever(retriever=..., compressor=..., k=4)`

These are native BaseRetriever subclasses with `invoke`, `ainvoke`, batch methods
and callbacks. They return native Documents. Multi-query allows 1–20 queries.
Hybrid weights must be finite, nonnegative, match the retriever count and have a
positive sum. BM25 returns positive lexical matches only.

## Vector stores

`vectorstore(backend, *, texts, embedding, metadatas=None, **options)` creates and
indexes a native store through `from_texts`. Texts must be nonempty; supplied
metadata must match their count. Embeddings may be a string or an existing native
object. `register_vectorstore(name, factory, replace=False)` adds a backend.

## Workflows

- `wrap(value)`: native Runnable passthrough, callable adaptation, or Agent adapter.
- `sequence(*steps)`: nonempty linear composition.
- `parallel(**branches)`: nonempty named fan-out.
- `route(selector, branches, default=None)`: choose one branch with a native/callable selector.
- `supervisor(model, specialists, **agent_options)`: manager Agent with named specialist tools.

## Search

`search_tool(provider="perplexity", *, model=None, model_options=None,
search_options=None, name="web_search") -> StructuredTool`

The tool's input schema has one `query: str` field. Its output is a JSON string.
Perplexity defaults to Sonar; xAI/Grok requires an explicit model. See
[search](search.md) for route-specific options and source metadata.

## Native exports and offline utilities

`Document`, `tool`, `StateGraph`, `START`, and `END` are native re-exports.
`truffles.testing.ScriptedChatModel(responses=[AIMessage(...)])` and
`HashEmbeddings(dimensions=64)` are explicit test utilities, never automatic
production fallbacks.

## Evaluation

- `EvalCase(input, expected=None, id=<generated UUID>, reference_ids=(), config={})`
- `Score(value, threshold=1.0, reason=None)`; `.passed` compares value to threshold.
- `evaluate(target, cases, *, evaluators, config=None) -> EvalReport`
- `aevaluate(target, cases, *, evaluators, max_concurrency=4, config=None) -> EvalReport`
- `exact_match(case, output)`, `contains_expected(case, output)`
- `json_valid(schema=None)`, `retrieval_recall(k=5, id_key=None, threshold=1.0)`
- `llm_judge(model, *, rubric, threshold=0.8, max_input_chars=20000, model_options=None)`
- `EvalReport.rows`, `.summary`, `.assert_passed(min_pass_rate=1.0, max_p95_ms=None)`,
  `.to_dict(include_content=False)`, `.save(path, include_content=False)`.

An `EvalRow` holds case ID, output, named scores, latency, trace ID, usage, target
error type and scorer error types. See [evaluation](evaluation.md) for execution,
privacy, scoring and concurrency contracts.

## Observability

- `Observer(max_spans=1000, max_active=1000, sink=None)`: native callback handler.
- `.spans`: completed `Span` snapshots; `.summary()`: lifetime counters and usage.
- `Span.to_dict()`: serializable completed operation, without input/output content.
- `JSONLTraceSink(path)`: callable sink for local JSONL export.

Pass `config={"callbacks": [observer]}` to agent or native workflow execution.
See [observability](observability.md) for bounded storage and export semantics.

## Basic processing (0.2.0)

- `ProcessingLimits(max_bytes=20971520, max_pages=50, max_pixels=25000000, max_chars=200000)`
- `ocr(source, *, language="eng", psm=3, timeout=30, preprocess=True, limits=None) -> Document`
- `read_documents(path, *, pdf_mode="auto", language="eng", psm=3, timeout=30, dpi=200,
  password=None, encoding="utf-8", limits=None) -> list[Document]`
- `transcribe(source, *, speech=None, filename=None, language=None, limits=None) -> Document`
- `processing_tools(root, *, speech=None, language="eng", transcription_language=None,
  limits=None, max_tool_chars=12000, pdf_mode="auto") -> list[StructuredTool]`

OCR/audio sources may be paths or bytes; `read_documents` takes a file path.
PDF page metadata is one-based. Tools return bounded model-visible text and full
bounded Document artifacts. See [processing](processing.md) for installation,
resource limits, local/remote boundaries and native-engine requirements.
