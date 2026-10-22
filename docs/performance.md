# Performance and operational costs

Truffles keeps its own work small and delegates execution to native libraries.
It does not claim to make a remote model generate faster. Latency is usually
controlled by inference, network calls, retrieval and the architecture you choose.

## Implementation choices

- Public imports are lazy. A plain `import truffles` does not load LangChain,
  LangGraph, provider SDKs, numpy or vector databases.
- A constructed agent retains its model and compiled graph. Provider clients keep
  their native connection pools; no client is reconstructed per tool-loop step.
- There is one native tool loop and one provider retry policy. Truffles does not
  add an outer retry layer that multiplies requests.
- Async calls delegate to native async APIs. Parallel workflows use native
  scheduling. Async hybrid and query-expansion retrieval fan out independent work.
- BM25 builds term postings and document lengths once. Queries score matching
  postings rather than tokenize every document again.
- Hybrid fusion uses rank positions instead of attempting cross-store score
  normalization. Candidate and output sizes are bounded by retrieval settings.
- RAG's source excerpt section has an explicit character bound. Agent model calls
  have an explicit per-run limit.

## Measure locally

```bash
uv run python scripts/benchmark.py
```

The script compares the same compiled agent graph invoked natively and through
`Agent.run`, using a scripted model. It also measures import time in subprocesses
and BM25 retrieval over 10,000 short synthetic documents. These are local overhead
measurements, not semantic quality or live throughput measurements. Cold imports,
CPU scheduling, tracing and Python versions affect results. The validation report
records a sample run without presenting it as a service-level guarantee.

## Application tuning

Reuse agents, tools, embeddings and stores. Choose `arun`/`ainvoke` for concurrent
I/O. Use parallel branches only for independent work. Keep `k`, `fetch_k`, query
expansion count and reranker candidates aligned with an evaluated quality budget.
Use native embedding batching and caches when your corpus contains repeated text.
Truffles does not enable a global cache because cache scope must respect your
model settings, credentials, tenant boundaries and freshness requirements.

Groq is available as a provider for inference. Selecting it does not automatically
change your architecture or route unsupported models to another provider. Provider
quotas and model choice still control achievable throughput.

## Bounds are different controls

`max_model_calls` counts model invocations, not tokens or dollars. A specialist
has its own counter. A single invocation can use several tools, retries or a
provider-hosted search loop. Use native tool limits, token caps and SDK timeouts
as appropriate. Provider retries happen inside a model invocation and may produce
more than one HTTP request.

Memory grows with conversation history unless you configure native summarization
or trimming. BM25 and in-memory vectors are single-process data structures, not
large-scale search services. For large corpora use an external indexed store;
for durable conversations use a persistent checkpointer. Those choices belong
in application configuration, not hidden library defaults.
