import pytest
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel

from truffles import Result, agent, rag, tool
from truffles.testing import ScriptedChatModel


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def call(name="add", args=None, id="call-1"):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args or {"a": 2, "b": 3},
                "id": id,
                "type": "tool_call",
            }
        ],
    )


def test_react_loop_tool_result_and_usage():
    model = ScriptedChatModel(
        responses=[
            call(),
            AIMessage(
                content="5",
                usage_metadata={"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
            ),
        ]
    )
    result = agent(model, tools=[add], instructions="Be accurate.").run("2 + 3?")
    assert result.text == str(result) == "5"
    assert result.usage["total_tokens"] == 4
    observation = next(m for m in model.requests[1] if isinstance(m, ToolMessage))
    assert observation.content == "5"
    assert observation.tool_call_id == "call-1"
    assert model.requests[0][0].content == "Be accurate."


async def test_async_agent_and_async_tool():
    @tool
    async def double(value: int) -> int:
        """Double a value."""
        return value * 2

    model = ScriptedChatModel(responses=[call("double", {"value": 4}), AIMessage(content="8")])
    assert (await agent(model, tools=[double]).arun("double 4")).text == "8"


def test_model_call_limit_stops_before_extra_request():
    model = ScriptedChatModel(responses=[call(), call(id="call-2"), AIMessage(content="never")])
    with pytest.raises(Exception, match="(?i)model call limit"):
        agent(model, tools=[add], max_model_calls=2).run("Keep going")
    assert len(model.requests) == 2


def test_memory_isolation_and_explicit_thread():
    model = ScriptedChatModel(responses=[AIMessage(content="ok") for _ in range(3)])
    bot = agent(model, memory=True)
    with pytest.raises(ValueError, match="thread_id"):
        bot.run("hi")
    bot.run("I am Ada", thread_id="a")
    bot.run("Who am I?", thread_id="a")
    bot.run("Who am I?", thread_id="b")
    assert len(model.requests[1]) == 3
    assert len(model.requests[2]) == 1


def test_stateless_calls_and_config_not_mutated():
    model = ScriptedChatModel(responses=[AIMessage(content="ok") for _ in range(2)])
    bot = agent(model)
    config = {"configurable": {"custom": 3}, "tags": ["test"]}
    bot.run("one", config=config, thread_id="a")
    bot.run([HumanMessage(content="two")], config=config)
    assert all(len(r) == 1 for r in model.requests)
    assert config == {"configurable": {"custom": 3}, "tags": ["test"]}
    with pytest.raises(ValueError, match="Conflicting"):
        bot.run("x", config={"configurable": {"thread_id": "a"}}, thread_id="b")


@pytest.mark.parametrize(
    "options",
    [
        {"kind": "unknown"},
        {"max_model_calls": 0},
        {"kind": "chat", "tools": [add]},
        {"memory": True, "checkpointer": object()},
    ],
)
def test_bad_agent_config(options):
    with pytest.raises(ValueError):
        agent(ScriptedChatModel(responses=[]), **options)


def test_structured_output():
    class Answer(BaseModel):
        value: int

    model = ScriptedChatModel(responses=[call("Answer", {"value": 5})])
    result = agent(model, response_format=ToolStrategy(Answer)).run("Give five")
    assert result.structured == Answer(value=5)


def test_tool_validation_error_is_observed():
    model = ScriptedChatModel(
        responses=[
            call(args={"a": "not-an-integer", "b": 3}),
            AIMessage(content="Please provide a number"),
        ]
    )
    result = agent(model, tools=[add]).run("bad input")
    assert any(isinstance(m, ToolMessage) and m.status == "error" for m in result.messages)


def test_unknown_tool_is_observed():
    model = ScriptedChatModel(responses=[call("missing"), AIMessage(content="Recovered")])
    result = agent(model, tools=[add]).run("try")
    assert any(isinstance(m, ToolMessage) and m.status == "error" for m in result.messages)


def test_knowledge_tool_retains_documents():
    knowledge = rag(["Truffles is an agent harness."])
    model = ScriptedChatModel(
        responses=[
            call("search_knowledge", {"query": "Truffles"}),
            AIMessage(content="An agent harness [1]."),
        ]
    )
    result = agent(model, knowledge=knowledge).run("What is Truffles?")
    assert result.sources[0].page_content == "Truffles is an agent harness."


def test_human_interrupt_and_resume():
    model = ScriptedChatModel(responses=[call(), AIMessage(content="5")])
    bot = agent(
        model,
        tools=[add],
        memory=True,
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"add": True})],
    )
    result = bot.run("add", thread_id="approval")
    assert result.interrupted
    result = bot.run(Command(resume={"decisions": [{"type": "approve"}]}), thread_id="approval")
    assert result.text == "5" and not result.interrupted


def test_sync_stream():
    bot = agent(ScriptedChatModel(responses=[AIMessage(content="hello")]))
    assert any("model" in event for event in bot.stream("hi"))


async def test_async_stream_and_runnable():
    bot = agent(ScriptedChatModel(responses=[AIMessage(content="hello") for _ in range(2)]))
    events = [event async for event in bot.astream("hi")]
    assert any("model" in event for event in events)
    result = await bot.as_runnable(text_only=False).ainvoke("again")
    assert isinstance(result, Result)


def test_result_extracts_text_without_reasoning():
    result = Result.from_state(
        {
            "messages": [
                AIMessage(
                    content=[
                        {"type": "reasoning", "reasoning": "private"},
                        {"type": "text", "text": "public"},
                    ]
                )
            ]
        }
    )
    assert result.text == "public"
    assert Result.from_state({}).text == ""


def test_script_exhaustion_and_memory_specialist_rejected():
    with pytest.raises(RuntimeError, match="exhausted"):
        agent(ScriptedChatModel(responses=[])).run("hello")
    with pytest.raises(ValueError, match="stateless"):
        agent(ScriptedChatModel(responses=[]), memory=True).as_tool()


def test_multiple_tool_calls_keep_ids_and_outputs():
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "add", "args": {"a": 1, "b": 2}, "id": "first"},
                    {"name": "add", "args": {"a": 5, "b": 6}, "id": "second"},
                ],
            ),
            AIMessage(content="3 and 11"),
        ]
    )
    result = agent(model, tools=[add]).run("Calculate both")
    assert {m.tool_call_id: m.content for m in result.messages if isinstance(m, ToolMessage)} == {
        "first": "3",
        "second": "11",
    }
