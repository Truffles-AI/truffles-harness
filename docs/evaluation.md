# Evaluation

Truffles evaluates agents, native Runnables (including compiled graphs and
retrievers), and callables against a dataset. Deterministic evaluators require no
provider, API key or additional dependency. Semantic judging is explicitly opt-in.

## First evaluation

```python
from truffles import EvalCase, evaluate, exact_match

report = evaluate(
    str.upper,
    [EvalCase("hello", "HELLO", id="greeting")],
    evaluators={"accuracy": exact_match},
)
print(report.summary)
report.assert_passed()
report.save("evaluation.json")
```

Replace `str.upper` with a Truffles agent or native Runnable. A callable is wrapped
as a native `RunnableLambda`, so normal callback propagation remains available.
A bare `RAG` is not an executable target: pass `knowledge.retriever` for retrieval
or wrap `knowledge.ask` in a callable for answer evaluation.

Each `EvalCase` has `input`, optional `expected`, a stable nonempty unique `id`,
optional `reference_ids` and optional native runtime `config`. Inputs may be text,
mappings, documents or other objects accepted by your target. Generated IDs are
available for ad-hoc runs; use explicit IDs for reproducible dataset comparisons.

Cases execute real work. Use a sandbox or read-only tools for evaluation: this
runner does not suppress tool side effects or provider charges. Persistent agents
require explicit thread IDs in case config; choose distinct IDs across independent
cases and repeated experiments to avoid contaminating results with prior history.

## Built-in metrics

| Evaluator | Meaning |
| --- | --- |
| `exact_match` | Exact answer text, including case and whitespace |
| `contains_expected` | Case-sensitive reference substring in the answer |
| `json_valid()` | Valid JSON; rejects NaN and Infinity |
| `json_valid(MySchema)` | JSON plus validation against a Pydantic model |
| `retrieval_recall(k=5)` | Fraction of unique reference document IDs in the first k retrieved documents |
| `llm_judge(model, rubric=...)` | Model-scored semantic criterion with validated JSON output |

Exact match and substring checks require a reference. Missing references are
scorer errors, not passing scores. `json_valid` validates syntax/schema, not the
truth of field values. Schema coercion follows your Pydantic model configuration.

`retrieval_recall` accepts a list of Documents or a `Result` with `.sources`.
It uses `Document.id` by default; use `id_key="source"` to read IDs from metadata.
Reference IDs must be nonempty. Duplicate retrieved IDs do not inflate recall.
For chunked documents, use stable source metadata if the labels refer to whole
documents instead of individual chunks. This metric measures retrieval coverage,
not answer correctness or faithfulness.

```python
from truffles import BM25Retriever, Document, EvalCase, evaluate, retrieval_recall

retriever = BM25Retriever.from_documents([
    Document(page_content="Savings accounts earn interest.", id="savings"),
    Document(page_content="Cards have credit limits.", id="cards"),
])
report = evaluate(
    retriever,
    [EvalCase("savings interest", id="r1", reference_ids=("savings",))],
    evaluators={"recall": retrieval_recall(k=2)},
)
report.assert_passed()
```

## Custom metrics and semantic judges

A custom evaluator receives `(case, output)` and returns a number between 0 and 1
or `Score(value, threshold=1.0, reason=None)`. It receives the original output,
including `Result.structured`, `.sources` and `.messages` when evaluating agents.
Use business-specific checks for category correctness, tool selection and policy
compliance. Non-finite/out-of-range scores are scorer errors.

```python
from truffles import EvalCase, Score, evaluate

report = evaluate(
    str,
    [EvalCase("A concise answer.", id="brevity")],
    evaluators={"brevity": lambda case, output: Score(float(len(output) < 80))},
)
```

For a semantic check, install the selected model integration and configure its
credentials as usual:

```python
from truffles import agent, evaluate, EvalCase, llm_judge

judge = llm_judge(
    "openai:gpt-4.1-mini",
    rubric="Score whether the answer is correct and supported by the provided sources.",
    threshold=0.8,
    model_options={"temperature": 0, "timeout": 20, "max_retries": 0},
)
report = evaluate(
    agent("openai:gpt-4.1-mini"),
    [EvalCase("What is 6 times 7?", "42", id="arithmetic")],
    evaluators={"correctness": judge},
)
```

The judge is a tool-free Truffles agent with a one-model-call budget. It receives
input, reference, answer text and source text as JSON; it must return exactly
`{"score": <0..1>, "reason": "..."}`. Malformed or extra fields are errors.
Inputs above `max_input_chars` (default 20,000) are rejected rather than silently
truncated. The prompt treats evidence as untrusted, but an LLM judge can still be
fooled or biased. Calibrate rubrics with human-labelled examples. A judge score is
not proof of factual accuracy, safety or regulatory compliance.

Judge requests are additional model calls. Their usage and latency are **not**
included in target usage/latency; instrument a supplied native judge model with
callbacks if you need separate telemetry. No monetary cost is guessed from tokens.

## Async and concurrency

```python
from truffles import EvalCase, aevaluate, exact_match

async def run_suite(bot):
    return await aevaluate(
        bot,
        [EvalCase("Say hello", "Hello", id="hello")],
        evaluators={"exact": exact_match},
        max_concurrency=4,
    )
```

`evaluate` is sequential. `aevaluate` uses at most `max_concurrency` worker tasks
and preserves dataset order; it does not spawn one task per case. Sync targets and
scorers run in threads, async targets/scorers are awaited. Targets and scorers must
be safe to share across workers. Cancellation propagates; Python cannot forcibly
stop synchronous work already running in a thread. Configure provider timeouts
and application deadlines separately. Concurrency is per invocation/process,
not a distributed rate limit. Cases and result outputs remain in memory, so split
very large datasets into batches and save each report.

## Reports and CI gates

Rows contain the case ID, original output, scores, target latency in milliseconds,
root trace ID, observed target token usage, target error type and evaluator error
types. Exceptions do not abort other cases. KeyboardInterrupt and cancellation are not
swallowed. A graph pause is returned as native state or `Result.interrupted`; use
a custom metric to decide whether that pause is expected in your evaluation. A case passes only if the target succeeds and every evaluator succeeds
and meets its threshold.

Summary fields include pass rate, mean scores, scored-case counts, target/scorer
error counts, mean and nearest-rank p95 target latency, and observed token totals.
Mean scores exclude unscored errors; read the counts with them. Token usage is
only what model callbacks report and excludes judge calls. Latency covers the
target and its callback overhead, excluding scoring and dataset queue time.

`report.assert_passed(min_pass_rate=0.95, max_p95_ms=2000)` raises `AssertionError`
when the pass rate/latency gate fails. Empty datasets and any target/scorer errors
always fail the gate, even with `min_pass_rate=0`. Choose latency thresholds on
representative hardware and workloads; unit-test latency is not a service SLA.

`report.save(path)` writes schema-versioned JSON with scores, IDs and operational
metrics. It excludes inputs, outputs, references and explanations by default.
`include_content=True` adds output text and explanations, which may contain
sensitive information. Reports in memory still hold original output objects.
Case IDs and metric names must also be non-sensitive. Callers control retention
and file permissions.

Use `config={"callbacks": [observer]}` to connect report trace IDs to the
[observability layer](observability.md). Existing callbacks are preserved. Per-case
config merges with suite config; a fresh root run ID is assigned to every case.

Run the full offline example with `python examples/evaluation.py`. It writes
reports and traces under `.truffles/evaluation/` and is exercised in CI. Scripted
judge answers in that example validate the integration contract, not model quality.
