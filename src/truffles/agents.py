"""A bounded ReAct harness backed by LangChain's native compiled agent."""

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from .providers import chat_model
from .results import Result


class Agent:
    """Reusable agent; conversations are isolated unless memory is explicitly enabled.

    `run` returns Result. `native` is the full compiled LangGraph. Constructor
    `**options` go to create_agent; provider options belong in `model_options`.
    """

    def __init__(
        self,
        model: Any,
        *,
        tools: Sequence[Any] = (),
        instructions: str | None = None,
        name: str | None = None,
        kind: str = "react",
        knowledge: Any = None,
        memory: bool = False,
        checkpointer: Any = None,
        max_model_calls: int = 20,
        model_options: Mapping[str, Any] | None = None,
        middleware: Sequence[Any] = (),
        response_format: Any = None,
        **options: Any,
    ):
        if kind not in {"chat", "react"}:
            raise ValueError(
                "kind must be 'chat' or 'react'; compose other architectures as workflows."
            )
        if max_model_calls < 1:
            raise ValueError("max_model_calls must be positive.")
        if kind == "chat" and (tools or knowledge is not None):
            raise ValueError("A chat agent cannot have tools or knowledge; use kind='react'.")
        if memory and checkpointer is not None:
            raise ValueError("Use memory=True or an explicit checkpointer, not both.")
        self.name = name
        self.max_model_calls = max_model_calls
        self.model = chat_model(model, **dict(model_options or {}))
        self.checkpointer = InMemorySaver() if memory else checkpointer
        all_tools = list(tools)
        if knowledge is not None:
            all_tools.append(knowledge.as_tool())
        self.native = create_agent(
            self.model,
            tools=all_tools,
            system_prompt=instructions,
            name=name,
            checkpointer=self.checkpointer,
            response_format=response_format,
            middleware=[
                ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior="error"),
                *middleware,
            ],
            **options,
        )

    def _config(self, config: RunnableConfig | None, thread_id: str | None) -> RunnableConfig:
        result = dict(config or {})
        configurable = dict(result.get("configurable", {}))
        if thread_id is not None:
            if "thread_id" in configurable and configurable["thread_id"] != thread_id:
                raise ValueError("Conflicting thread_id values.")
            configurable["thread_id"] = thread_id
        if self.checkpointer is not None and not configurable.get("thread_id"):
            raise ValueError("A checkpointed agent requires an explicit thread_id.")
        if configurable:
            result["configurable"] = configurable
        # Middleware adds graph nodes. This graph safeguard is distinct from the model budget.
        result.setdefault("recursion_limit", self.max_model_calls * 8 + 10)
        return result

    @staticmethod
    def _input(value: Any) -> Any:
        if isinstance(value, str):
            return {"messages": [{"role": "user", "content": value}]}
        if isinstance(value, (list, tuple)):
            return {"messages": list(value)}
        return value  # native state dictionaries and LangGraph Commands

    def run(self, prompt: Any, *, config=None, thread_id=None, **kwargs: Any) -> Result:
        return Result.from_state(
            self.native.invoke(self._input(prompt), self._config(config, thread_id), **kwargs)
        )

    async def arun(self, prompt: Any, *, config=None, thread_id=None, **kwargs: Any) -> Result:
        return Result.from_state(
            await self.native.ainvoke(
                self._input(prompt), self._config(config, thread_id), **kwargs
            )
        )

    def stream(
        self, prompt: Any, *, config=None, thread_id=None, stream_mode="updates", **kwargs: Any
    ) -> Iterator[Any]:
        """Yield native graph events; use stream_mode='messages' for token chunks."""
        yield from self.native.stream(
            self._input(prompt), self._config(config, thread_id), stream_mode=stream_mode, **kwargs
        )

    async def astream(
        self, prompt: Any, *, config=None, thread_id=None, stream_mode="updates", **kwargs: Any
    ) -> AsyncIterator[Any]:
        async for event in self.native.astream(
            self._input(prompt), self._config(config, thread_id), stream_mode=stream_mode, **kwargs
        ):
            yield event

    def as_runnable(self, *, text_only: bool = True) -> RunnableLambda:
        """Adapt to any LCEL chain, preserving callbacks and runtime configuration."""

        def invoke(value, config: RunnableConfig):
            result = self.run(value, config=config)
            return result.text if text_only else result

        async def ainvoke(value, config: RunnableConfig):
            result = await self.arun(value, config=config)
            return result.text if text_only else result

        return RunnableLambda(invoke, afunc=ainvoke, name=self.name)

    def as_tool(self, *, name: str | None = None, description: str | None = None):
        """Expose a stateless specialist to a supervisor, with sync and async execution."""
        if self.checkpointer is not None:
            raise ValueError("Specialist tools must be stateless; use native subgraphs for memory.")

        def invoke(query: str, config: RunnableConfig) -> str:
            return self.run(query, config=config).text

        async def ainvoke(query: str, config: RunnableConfig) -> str:
            return (await self.arun(query, config=config)).text

        return StructuredTool.from_function(
            invoke,
            coroutine=ainvoke,
            name=name or self.name or "specialist",
            description=description or f"Delegate a task to {self.name or 'a specialist'}.",
        )


def agent(model: Any, **options: Any) -> Agent:
    """Create a Truffles agent in one line."""
    return Agent(model, **options)
