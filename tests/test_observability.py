import json
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.runnables import RunnableLambda

from truffles import JSONLTraceSink, Observer, agent, parallel, rag, tool
from truffles.testing import ScriptedChatModel


def model(*answers):
    return ScriptedChatModel(responses=list(answers))


def answer(text="secret-answer"):
    return AIMessage(
        content=text,
        usage_metadata={
            "input_tokens": 8,
            "output_tokens": 2,
            "total_tokens": 10,
        },
    )


def test_agent_tool_tree_and_no_content(tmp_path):
    @tool
    def echo(value: str) -> str:
        """Echo a value."""
        return value

    path = tmp_path / "traces.jsonl"
    observer = Observer(sink=JSONLTraceSink(path))
    bot = agent(
        model(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "echo",
                        "args": {"value": "secret-tool"},
                        "id": "call-1",
                    }
                ],
            ),
            answer(),
        ),
        tools=[echo],
    )
    result = bot.run(
        "secret-input", config={"callbacks": [observer], "metadata": {"secret-key": "secret-value"}}
    )
    assert result.text == "secret-answer"
    spans = observer.spans
    assert {s.kind for s in spans} == {"chain", "model", "tool"}
    roots = [s for s in spans if s.parent_run_id is None]
    assert len(roots) == 1
    assert {s.trace_id for s in spans} == {roots[0].run_id}
    assert all(s.duration_ms >= 0 and s.ended_ns >= s.started_ns for s in spans)
    assert all(s.parent_run_id in {p.run_id for p in spans} for s in spans if s.parent_run_id)
    stats = observer.summary()
    assert stats["model_calls"] == 2 and stats["tool_calls"] == 1
    assert stats["usage"] == {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
    assert stats["active"] == stats["errors"] == 0
    saved = path.read_text()
    assert "secret-" not in saved
    assert len(saved.splitlines()) == len(spans)
    assert all(json.loads(line)["status"] == "ok" for line in saved.splitlines())
    # Returned spans and sinks cannot mutate aggregate or retained usage.
    model_span = next(s for s in spans if s.usage)
    model_span.usage["total_tokens"] = 900
    assert next(s for s in observer.spans if s.usage).usage["total_tokens"] == 10


def test_memory_usage_counts_only_new_calls():
    observer = Observer()
    bot = agent(model(answer("first"), answer("second")), memory=True)
    bot.run("one", thread_id="a", config={"callbacks": [observer]})
    second = bot.run("two", thread_id="a", config={"callbacks": [observer]})
    assert second.usage["total_tokens"] == 20
    assert observer.summary()["usage"]["total_tokens"] == 20  # Not 10 + 20.


@pytest.mark.asyncio
async def test_parallel_async_retrieval_and_stream():
    observer = Observer()
    team = parallel(left=agent(model(answer("a"))), right=agent(model(answer("b"))))
    assert await team.ainvoke("question", config={"callbacks": [observer]}) == {
        "left": "a",
        "right": "b",
    }
    knowledge = rag(["truffles agents"])
    await knowledge.aretrieve("agents", config={"callbacks": [observer]})
    bot = agent(model(answer("stream")))
    chunks = [chunk async for chunk in bot.astream("hi", config={"callbacks": [observer]})]
    assert chunks
    assert observer.summary()["retrievals"] == 1
    assert observer.summary()["model_calls"] == 3
    assert observer.summary()["active"] == 0


def test_errors_never_capture_exception_message():
    observer = Observer()

    def fail(value):
        raise RuntimeError("secret-database-credentials")

    with pytest.raises(RuntimeError):
        RunnableLambda(fail).invoke("private", config={"callbacks": [observer]})
    (span,) = observer.spans
    assert span.status == "error" and span.error_type == "RuntimeError"
    assert "secret" not in json.dumps(span.to_dict())
    assert observer.summary()["errors"] == 1


@pytest.mark.parametrize("kind", ["llm", "tool", "retriever"])
def test_callback_error_paths(kind):
    observer = Observer()
    run_id = uuid4()
    getattr(observer, f"on_{kind}_start")({"name": "operation"}, [], run_id=run_id)
    getattr(observer, f"on_{kind}_error")(ValueError("private"), run_id=run_id)
    assert observer.spans[0].error_type == "ValueError"
    assert observer.summary()["active"] == 0


def test_bounded_storage_duplicate_end_and_sink_failure():
    def broken_sink(span):
        raise OSError("private path")

    observer = Observer(max_spans=1, max_active=1, sink=broken_sink)
    first, dropped = uuid4(), uuid4()
    observer.on_chain_start(None, {}, run_id=first)
    observer.on_chain_start(None, {}, run_id=first)  # Duplicate start is ignored.
    observer.on_chain_start(None, {}, run_id=dropped)
    observer.on_chain_end({}, run_id=dropped)
    observer.on_chain_end({}, run_id=first)
    observer.on_chain_end({}, run_id=first)
    second = uuid4()
    observer.on_chain_start(None, {}, run_id=second)
    observer.on_chain_end({}, run_id=second)
    assert observer.spans[0].run_id == str(second)
    assert observer.summary() | {} == {
        "completed": 2,
        "errors": 0,
        "model_calls": 0,
        "tool_calls": 0,
        "retrievals": 0,
        "dropped_starts": 1,
        "evicted_spans": 1,
        "sink_errors": 2,
        "active": 0,
        "retained": 1,
        "usage": {},
    }
    for options in ({"max_spans": 0}, {"max_active": 0}):
        with pytest.raises(ValueError):
            Observer(**options)


def test_native_llm_usage_fallback_and_alternative_candidates():
    observer = Observer()
    run_id = uuid4()
    observer.on_llm_start(None, [], run_id=run_id)
    observer.on_llm_end(
        LLMResult(
            generations=[],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "total_tokens": 4,
                }
            },
        ),
        run_id=run_id,
    )
    assert observer.spans[0].usage["total_tokens"] == 4
    other = uuid4()
    observer.on_chat_model_start(None, [], run_id=other)
    observer.on_llm_end(
        LLMResult(
            generations=[[], [ChatGeneration(message=answer()), ChatGeneration(message=answer())]]
        ),
        run_id=other,
    )
    assert observer.summary()["usage"]["total_tokens"] == 14


def test_concurrent_sink_writes(tmp_path):
    path = tmp_path / "events.jsonl"
    observer = Observer(sink=JSONLTraceSink(path))

    def work(index):
        RunnableLambda(str).invoke(index, config={"callbacks": [observer]})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(work, range(50)))
    assert len({json.loads(line)["run_id"] for line in path.read_text().splitlines()}) == 50
    assert observer.summary()["completed"] == 50
