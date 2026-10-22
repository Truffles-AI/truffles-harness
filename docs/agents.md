# Agents and tools

## Construction

```python
from truffles import agent
bot = agent(
    "openai:gpt-4.1-mini",
    instructions="You are a careful research assistant.",
    name="researcher",
    max_model_calls=10,
    model_options={"temperature": 0, "max_retries": 2, "timeout": 30},
)
```

`agent` constructs `Agent`; both have the same options. `model_options` reach the
provider's native constructor. Other extra keyword arguments reach LangChain's
`create_agent` (for example `state_schema`, `context_schema`, `store`,
`interrupt_before`, `interrupt_after`, `debug` and `cache`). Invalid options fail
at their native boundary.

`kind="react"` is the default. The graph implements model → tool execution → model
until completion. This is native tool-calling ReAct, not parsing a model's private
reasoning or requiring text labeled “Thought”. `kind="chat"` disallows tools and
knowledge; structured output remains possible through native response strategies.

## Input and output

`run` and `arun` accept a string, a list of native messages/message dictionaries,
a state dictionary, or a native `Command` for resumption. Strings become a user
message. Multimodal blocks are forwarded unchanged when using native messages.

```python
result = bot.run({"messages": [{"role": "user", "content": "Hello"}]})
print(result.text)
print(result.usage)
print(result.messages)
```

`Result.text` contains visible text from the final assistant message. Reasoning
blocks are not flattened into the answer. `structured` holds a validated structured
response when requested. `sources` holds retrieved documents from knowledge-tool
artifacts. `state` retains the full native output, including interruptions.

`usage` aggregates the standard integer token counters present in returned AI
messages. With checkpointed memory, the returned state contains previous messages,
so these counters cover that returned thread state, **not just the current call**.
SDK-specific cache/reasoning/cost fields remain on the original messages. Result
objects are frozen containers, but nested native messages and dictionaries are
not deep-frozen snapshots.

## Tool schemas and failures

```python
from truffles import tool

@tool
async def lookup_order(order_id: str) -> str:
    """Look up an order by its public order ID."""
    # Replace with an authorized application integration.
    return f"Order {order_id}: shipped"
```

Async-only tools require `arun`. Sync functions also work in async runs through
native runtime execution. Multiple tools may run concurrently when the runtime
and model request that; tools must handle their own shared resources correctly.

Schema validation errors and unknown tool names become tool observations under
the native default behavior. Arbitrary application exceptions can propagate.
Truffles does not swallow provider or tool errors. Add native middleware to define
application-specific retry or error-handling policies.

## Structured responses

```python
from pydantic import BaseModel
from langchain.agents.structured_output import ToolStrategy
from truffles import agent

class Classification(BaseModel):
    category: str
    confidence: float

classifier = agent("openai:gpt-4.1-mini", response_format=ToolStrategy(Classification))
result = classifier.run("Classify: the shipment arrived damaged.")
print(result.structured.category)
```

Passing the schema itself allows native strategy selection. An explicit
`ToolStrategy` requests function-based structured output; `ProviderStrategy`
requests a native provider schema. Availability depends on the model and endpoint.
Validation retries also consume the model-call budget. Perplexity's Responses
adapter rejects forced `tool_choice`; do not assume every strategy works there.

## Execution bounds

`max_model_calls` defaults to 20 and is scoped to each run. Exceeding it raises
LangChain's model-call limit error. A separate default recursion limit accommodates
the graph and middleware nodes. Override that safeguard in `config` if custom
middleware legitimately adds more graph steps.

A model call can emit several tool calls, and nested specialists have their own
budgets. For a tool-count budget use `ToolCallLimitMiddleware`. Token caps, timeouts
and retries are provider settings. There is no hidden extra retry loop around the
provider SDK.

## Native escape hatch

```python
native_graph = bot.native
native_state = native_graph.invoke({"messages": [{"role": "user", "content": "Hello"}]})
runnable = bot.as_runnable(text_only=False)
```

Calling `.native` bypasses Truffles' input/config normalization and result wrapper;
you must supply native state and checkpoint configuration yourself. The model-call
middleware is part of the compiled graph and still applies.
