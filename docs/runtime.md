# Runtime, memory and observability

## Async and concurrency

```python
import asyncio
from truffles import agent

async def main():
    bot = agent("openai:gpt-4.1-mini")
    result = await bot.arun("Hello")
    print(result.text)

# asyncio.run(main())  # call from a synchronous program's entry point
```

Do not call `asyncio.run` inside an already running event loop; await `main()` there.
Native async models and tools remain async. Integrations without native async may
use thread fallback. Cancelling an async task does not forcibly stop a synchronous
function already running in a worker thread. Set native client/tool timeouts.

## Streaming

```python
from truffles import agent
bot = agent("openai:gpt-4.1-mini")
for event in bot.stream("Explain RAG", stream_mode="updates"):
    print(event)
```

Updates are native graph state updates. For token chunks use `stream_mode="messages"`:

```python
for message_chunk, metadata in bot.stream("Hello", stream_mode="messages"):
    print(message_chunk.content, end="")
```

Models can emit non-text blocks, tool-call deltas and reasoning metadata. Preserve
native chunks if your application needs them. `astream` yields the same native
modes asynchronously. Streaming graph updates does not promise token streaming
from a model that lacks it. Provider errors can arrive after earlier chunks have
already been consumed.

## Batch execution

```python
runnable = bot.as_runnable(text_only=False)
results = runnable.batch(["Question one", "Question two"], config={"max_concurrency": 2})
```

`abatch` is the async counterpart. Use independent inputs for stateless agents.
Persistent conversations need distinct per-item configurations. Avoid concurrent
writes to the same checkpointed thread unless your chosen runtime/storage contract
explicitly supports the ordering you require. Native `return_exceptions=True`
can retain batch failures as results when that matches application policy.

## Memory and threads

```python
from truffles import agent
bot = agent("openai:gpt-4.1-mini", memory=True)
bot.run("My name is Ada", thread_id="conversation-1")
result = bot.run("What is my name?", thread_id="conversation-1")
```

`memory=True` uses native `InMemorySaver`, which lasts only as long as the process.
The wrapper requires an explicit `thread_id` whenever a checkpointer is supplied.
This prevents unrelated users from sharing a default conversation accidentally.

A `config={"configurable": {"thread_id": "..."}}` also works. Conflicting thread
identifiers raise an error. Runtime config is copied before defaults are added;
the caller's configuration is not mutated.

Pass an already-open durable native saver via `checkpointer=` for persistence.
Its lifecycle, credentials, schema setup and cleanup remain your responsibility.
Do not combine `memory=True` with an explicit checkpointer. Native stores for
long-term memory are separate from thread checkpointing and can be supplied using
`store=` at agent construction.

## Human interruption and resume

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Command
from truffles import agent, tool

@tool
def update_record(record_id: str) -> str:
    """Update a record in an example application."""
    return f"Updated {record_id}"

bot = agent("openai:gpt-4.1-mini", tools=[update_record], memory=True,
            middleware=[HumanInTheLoopMiddleware(interrupt_on={"update_record": True})])
result = bot.run("Update record 123", thread_id="review-1")
if result.interrupted:
    requests = result.state["__interrupt__"]
    # Display requests to the application's reviewer before actually approving.
    # approved = bot.run(Command(resume={"decisions": [{"type": "approve"}]}),
    #                    thread_id="review-1")
```

The graph must be resumed with its native command and the same thread. Approval,
edit and rejection formats belong to the installed middleware. Merely receiving
an interruption is not a successful final answer. The test suite exercises an
actual interrupt followed by an explicit synthetic approval.

## Callbacks, tracing and middleware

Pass native callbacks, tags, metadata, configurable values and run names through
`config`. The adapters propagate these into native calls and sub-workflows.
Truffles does not enable external tracing automatically. LangSmith and other
native callback backends remain usable if you configure them.

Middleware can implement retries, summarization, dynamic model selection, tool
limits or application authorization. Add native middleware objects to
`middleware=[...]`; the model-call budget is added first. The wrapper does not
rewrite middleware state or callback payloads. Provider-specific usage and raw
metadata remain on native messages.
