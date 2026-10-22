import asyncio
from typing import TypedDict

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from truffles import END, START, StateGraph, agent, parallel, route, sequence, supervisor, wrap
from truffles.testing import ScriptedChatModel


def test_sequence_with_agents_and_native_runnable():
    model = ScriptedChatModel(responses=[AIMessage(content="draft")])
    pipe = sequence(agent(model), str.upper, RunnableLambda(lambda s: s + "!"))
    assert pipe.invoke("write") == "DRAFT!"


async def test_parallel_is_actually_concurrent():
    entered = 0
    both_started = asyncio.Event()

    async def worker(value):
        nonlocal entered
        entered += 1
        if entered == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=2)
        return value + 1

    assert await parallel(a=worker, b=worker).ainvoke(1) == {"a": 2, "b": 2}


def test_parallel_sync_batch():
    team = parallel(double=lambda n: 2 * n, square=lambda n: n * n)
    assert team.batch([2, 3], config={"max_concurrency": 2}) == [
        {"double": 4, "square": 4},
        {"double": 6, "square": 9},
    ]


def test_router_and_fallback():
    router = route(lambda x: x[0], {"a": str.upper}, default=lambda x: "fallback")
    assert router.invoke("abc") == "ABC"
    assert router.invoke("xyz") == "fallback"
    with pytest.raises(ValueError, match="No route"):
        route(lambda x: "absent", {"a": str.upper}).invoke("x")


async def test_async_router():
    async def select(value):
        return "selected"

    assert await route(select, {"selected": lambda n: n + 1}).ainvoke(3) == 4


@pytest.mark.parametrize(
    "factory",
    [
        lambda: sequence(),
        lambda: parallel(),
        lambda: route(str, {}),
        lambda: supervisor("unused", {}),
    ],
)
def test_empty_architectures_rejected(factory):
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize("async_mode", [False, True])
async def test_supervisor_delegates(async_mode):
    specialist = agent(ScriptedChatModel(responses=[AIMessage(content="specialist answer")]))
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "delegate", "name": "research", "args": {"query": "research this"}}
                ],
            ),
            AIMessage(content="combined answer"),
        ]
    )
    boss = supervisor(model, {"research": specialist})
    result = await boss.arun("go") if async_mode else boss.run("go")
    assert result.text == "combined answer"
    assert "specialist answer" in str(model.requests[1])


def test_native_graph_conditional_cycle_and_chain():
    class State(TypedDict):
        count: int

    graph = StateGraph(State)
    graph.add_node("increment", lambda s: {"count": s["count"] + 1})
    graph.add_edge(START, "increment")
    graph.add_conditional_edges("increment", lambda s: "increment" if s["count"] < 3 else END)
    native = graph.compile()
    assert wrap(native) is native
    assert sequence(native, lambda s: s["count"]).invoke({"count": 0}) == 3


def test_callbacks_propagate():
    from langchain_core.callbacks import BaseCallbackHandler

    class Handler(BaseCallbackHandler):
        seen = []

        def on_chat_model_start(self, serialized, messages, *, tags=None, **kwargs):
            self.seen.append(tags)

    handler = Handler()
    pipe = sequence(agent(ScriptedChatModel(responses=[AIMessage(content="ok")])), str.upper)
    assert pipe.invoke("hi", config={"callbacks": [handler], "tags": ["trace-me"]}) == "OK"
    assert "trace-me" in handler.seen[0]
