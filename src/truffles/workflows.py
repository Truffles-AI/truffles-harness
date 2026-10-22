"""Composition uses native Runnables: callbacks, batching and async stay intact."""

from collections.abc import Mapping
from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from langchain_core.runnables.base import coerce_to_runnable

from .agents import Agent, agent


def wrap(value: Any) -> Runnable:
    """Accept a Truffles agent, LCEL chain, compiled graph, callable or runnable."""
    return value.as_runnable() if isinstance(value, Agent) else coerce_to_runnable(value)


def sequence(*steps: Any) -> Runnable:
    """Linear pipeline; every step receives the previous step's output."""
    if not steps:
        raise ValueError("A sequence needs at least one step.")
    result = wrap(steps[0])
    for step in steps[1:]:
        result = result | wrap(step)
    return result


def parallel(**branches: Any) -> RunnableParallel:
    """Fan out the same input; return a name->output mapping in declaration order."""
    if not branches:
        raise ValueError("Provide at least one named branch.")
    return RunnableParallel({name: wrap(step) for name, step in branches.items()})


def route(selector: Any, branches: Mapping[str, Any], *, default: Any = None) -> Runnable:
    """Select a branch with a sync/async callable or runnable returning its name."""
    if not branches:
        raise ValueError("Provide at least one branch.")
    choose = wrap(selector)
    targets = {name: wrap(value) for name, value in branches.items()}
    fallback = wrap(default) if default is not None else None

    def target(key):
        if key in targets:
            return targets[key]
        if fallback is not None:
            return fallback
        raise ValueError(f"No route for {key!r}; available routes: {sorted(targets)}")

    def invoke(value, config):
        return target(choose.invoke(value, config)).invoke(value, config)

    async def ainvoke(value, config):
        key = await choose.ainvoke(value, config)
        return await target(key).ainvoke(value, config)

    return RunnableLambda(invoke, afunc=ainvoke)


def supervisor(model: Any, specialists: Mapping[str, Agent], **options: Any) -> Agent:
    """Hierarchical multi-agent system: a bounded manager delegates through tools."""
    if not specialists:
        raise ValueError("Provide at least one specialist.")
    tools = list(options.pop("tools", ()))
    tools.extend(member.as_tool(name=name) for name, member in specialists.items())
    options.setdefault("instructions", "Delegate to the appropriate specialists, then synthesize.")
    return agent(model, tools=tools, **options)
