# Chains and multi-agent workflows

All workflow factories return native LangChain Runnables. They therefore compose
with `|`, support `invoke`/`ainvoke`, and inherit batching and callback behavior.
Truffles agents are adapted to return answer text by default.

## Sequential / linear

```python
from truffles import sequence
pipeline = sequence(lambda s: s.strip(), str.upper, lambda s: f"Result: {s}")
assert pipeline.invoke(" hello ") == "Result: HELLO"
```

Each step receives only the previous output. To preserve the original request,
use a dictionary and native `RunnablePassthrough.assign`, or carry typed state in
a graph. An existing LCEL prompt/model/parser chain can be a step unchanged.
A single-step sequence is allowed; an empty one is rejected.

## Parallel teams and fan-in

```python
from truffles import parallel, sequence
team = parallel(optimist=lambda topic: f"Benefits of {topic}",
                critic=lambda topic: f"Risks of {topic}")
pipeline = sequence(team, lambda opinions: "\n".join(opinions.values()))
print(pipeline.invoke("automation", config={"max_concurrency": 2}))
```

Each branch receives the same input. Sync execution uses native thread execution;
async execution uses native async concurrency. Treat shared inputs as immutable.
Branches return a mapping in declaration order. Exceptions propagate by default.
`max_concurrency` controls native runnable scheduling where supported; it is not a
provider rate limit and does not automatically constrain tools inside a graph.

## Routing

```python
from truffles import route
router = route(lambda text: "billing" if "invoice" in text else "support", {
    "billing": lambda text: f"Billing: {text}",
    "support": lambda text: f"Support: {text}",
})
print(router.invoke("Find my invoice"))
```

A selector is any callable or Runnable returning a branch key. An async selector
works with `ainvoke`. The chosen branch receives the original input, not the key.
Unknown keys raise `ValueError` unless `default=` is supplied. An LLM classifier
can be the selector if its output is parsed into a valid key.

## Supervisor and specialists

```python
from truffles import agent, supervisor
researcher = agent("openai:gpt-4.1-mini", instructions="Research the supplied topic.")
editor = agent("anthropic:claude-sonnet-4-6", instructions="Edit the supplied draft.")
team = supervisor("openai:gpt-4.1-mini", {"research": researcher, "edit": editor},
                  max_model_calls=8)
print(team.run("Draft a short guide to vector search.").text)
```

The supervisor decides whether and how to delegate. Specialist names become tool
names, and specialists receive a `query` string. Each specialist is an ordinary
agent with its own tools, prompt and model-call budget. Add specific descriptions
with `specialist.as_tool(name=..., description=...)` and pass those tools to a
manager agent when better routing hints are needed.

Specialists exposed by `as_tool` must be stateless. Reusing a manager's thread ID
inside multiple persistent specialists could mix conversation histories. Use
native subgraphs/checkpoint namespaces for persistent multi-agent memory.

## Native graphs, cycles and handoffs

```python
from typing import TypedDict
from truffles import StateGraph, START, END, wrap

class State(TypedDict):
    count: int

graph = StateGraph(State)
graph.add_node("advance", lambda state: {"count": state["count"] + 1})
graph.add_edge(START, "advance")
graph.add_conditional_edges("advance", lambda s: END if s["count"] >= 3 else "advance")
workflow = wrap(graph.compile())
assert workflow.invoke({"count": 0}) == {"count": 3}
```

`StateGraph`, `START` and `END` are native re-exports. Use `Send` for dynamic fan-out,
reducers for parallel state writes, `Command` for handoffs, subgraphs for nested
architectures, and native checkpointers for durable execution. No graph structure
is copied or flattened by `wrap`.

A Truffles agent can be used as a text-producing Runnable node, or its `.native`
graph can be mounted as a subgraph when state schemas are compatible. Native graph
nodes must obey their state's update contract; wrapping an agent does not invent
that state mapping.

## Plan/execute, reflection and debate

A fixed plan/execute pipeline can be `sequence(planner, executor, reviewer)`.
For adaptive planning, keep the plan and cursor in graph state and conditionally
loop through execution nodes. A reflection loop alternates draft and review nodes
with a termination condition. A debate runs independent agents in `parallel`, then
passes their named answers to a judge. Set graph recursion and model/tool budgets
for every loop; a prompt alone is not an execution bound.
