# Observability

`Observer` is a native LangChain callback handler for agents, workflows, models,
tools and retrievers. It requires no additional dependency or hosted service.
Attach it using the existing native callback configuration:

```python
from truffles import Observer, agent

observer = Observer()
bot = agent("openai:gpt-4.1-mini")
result = bot.run("Explain RAG.", config={"callbacks": [observer]})
print(observer.summary())
for span in observer.spans:
    print(span.kind, span.name, span.duration_ms, span.status)
```

The same config works with `arun`, `stream`, `astream` and native
`invoke`/`ainvoke`. Native child callbacks carry the parent ID, so parallel
branches, nested graphs, tools and model calls retain their relationship.
A root run ID identifies a trace. Construct observers per request for isolation,
or share one within a process for aggregate metrics.

## What is recorded

A completed `Span` contains:

- Run ID, parent run ID and trace ID.
- Operation kind (`chain`, `model`, `tool` or `retriever`) and a bounded name.
- Start/end Unix timestamps in nanoseconds and monotonic duration in milliseconds.
- `ok` or `error` status and exception **class name** on failure.
- Model-reported input/output/total token counts when available.

The summary counts completed/error spans, model/tool/retrieval calls, retained and
active spans, evictions, dropped starts, sink failures and observed token totals.
Errors count spans, not requests: a failing model can also fail parent graph spans.
Nested span durations must not be added together as request latency.

Usage comes from each model completion callback. It does not recount the whole
conversation when memory is enabled. Alternative generations use their first
candidate's usage; providers that omit usage produce no token data. Provider-internal
HTTP retries are not separate model spans. No cost calculation, retry count or
token completeness is inferred. Streaming usage is recorded at completion only
if the native integration reports it. Interrupted graphs follow native callback
semantics and are not automatically labelled as business failures.

## Privacy and storage

Inputs, outputs, prompts, documents, metadata, tags, credentials and exception
messages are never retained by this observer. It does not have a content-capture
switch. **Names are retained:** use static operation names and do not put customer
identifiers or secrets in `run_name`, tool names or custom model names.

This policy applies only to Truffles' observer. Other callbacks, SDK logs and
LangSmith tracing enabled by your environment can have different collection rules.
The harness does not change global tracing settings behind your back.

Completed spans use a bounded buffer (`max_spans=1000`). Old spans are evicted,
while summary counters remain cumulative. Pending starts are also bounded
(`max_active=1000`): excess starts are dropped and counted. Missing completion
callbacks can leave spans active, but cannot grow this storage without bound.
Use an observer scoped to a request/job rather than keeping incomplete observers
forever. If an ancestor start was dropped or was never observed, a descendant's
trace ID falls back to its own run ID; incomplete traces are not reconstructed.

Summary/spans are snapshots protected by a lock and safe under threaded callback
execution. They provide local diagnostics, not a durable audit ledger or a
cross-process monitoring service.

## JSONL and custom sinks

```python
from truffles import JSONLTraceSink, Observer

observer = Observer(
    max_spans=500,
    max_active=1000,
    sink=JSONLTraceSink("traces.jsonl"),
)
```

The sink appends one JSON object per completed span. Parent directories must
already exist. A file is opened and closed on each write; no flush/shutdown method
is needed. Writes are serialized within one sink instance only. Use separate
files per process, a logging collector or your own sink for multi-process writes.
File size is not bounded: configure external rotation, access control and retention.

Any callable accepting a `Span` can serve as a sink. It executes outside the
observer's state lock and may be called concurrently; make your sink thread-safe.
Export failures increment `sink_errors` without failing business execution. They
are not retried or queued by Truffles. Monitor this counter to detect lost telemetry.
Sinks are synchronous, so slow writes add callback latency. Production exporters
should enqueue to their own bounded queue with an explicit overflow policy.

## Existing monitoring backends

You can attach an existing native LangChain callback alongside the observer:

```python
# external_handler is your configured backend's native LangChain callback.
config = {"callbacks": [observer, external_handler]}
```

Native LangSmith tracing remains available through its own configuration. For
Langfuse, OpenTelemetry or another backend, use its maintained LangChain adapter
or adapt `Span` records in a custom sink. This release does not bundle a dashboard,
OTLP exporter, remote ingestion service, distributed metrics store or alert manager.
No hosted backend is enabled implicitly, and none is live-verified by the offline
suite. The callback surface follows the official
[LangChain callback interfaces](https://reference.langchain.com/python/langchain-core/callbacks/base/AsyncCallbackHandler).

## Evaluation correlation

```python
from truffles import EvalCase, Observer, evaluate, exact_match

observer = Observer()
report = evaluate(
    str.upper,
    [EvalCase("hello", "HELLO", id="greeting")],
    evaluators={"accuracy": exact_match},
    config={"callbacks": [observer]},
)
row = report.rows[0]
trace = [span for span in observer.spans if span.trace_id == row.trace_id]
assert trace
```

See [evaluation](evaluation.md) for datasets, retrieval metrics, semantic judging,
regression gates and report privacy. Instrumenting execution is separate from
checking answer quality; use both when assessing an agent.
