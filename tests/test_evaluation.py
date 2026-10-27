import asyncio
import json

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from truffles import (
    Document,
    EvalCase,
    Observer,
    Result,
    Score,
    aevaluate,
    agent,
    contains_expected,
    evaluate,
    exact_match,
    json_valid,
    llm_judge,
    retrieval_recall,
    sequence,
)
from truffles.testing import ScriptedChatModel


def test_agent_dataset_scores_and_export_privacy(tmp_path):
    bot = agent(
        ScriptedChatModel(
            responses=[
                AIMessage(
                    content="private answer",
                    usage_metadata={
                        "input_tokens": 3,
                        "output_tokens": 2,
                        "total_tokens": 5,
                    },
                )
            ]
        )
    )
    observer = Observer()
    report = evaluate(
        bot,
        [EvalCase("private input", "private answer", id="case-1")],
        evaluators={"exact": exact_match},
        config={"callbacks": [observer]},
    )
    assert report.assert_passed() is report
    assert report.summary["usage"]["total_tokens"] == 5
    assert report.summary["mean_scores"] == {"exact": 1.0}
    assert report.summary["scored_cases"] == {"exact": 1}
    assert report.rows[0].trace_id in {span.trace_id for span in observer.spans}
    path = tmp_path / "report.json"
    report.save(path)
    assert "private" not in path.read_text()
    assert json.loads(path.read_text())["schema_version"] == 1
    assert report.to_dict(include_content=True)["rows"][0]["output"] == "private answer"


def test_workflow_callable_and_metadata_config():
    workflow = sequence(str.strip, str.upper)
    assert (
        evaluate(workflow, [EvalCase(" hi ", "HI")], evaluators={"exact": exact_match})
        .rows[0]
        .passed
    )

    def configured(value, config):
        assert config["metadata"] == {"suite": "offline", "case": "specific"}
        return value

    report = evaluate(
        configured,
        [EvalCase("x", "x", config={"metadata": {"case": "specific"}})],
        evaluators={"exact": exact_match},
        config={"metadata": {"suite": "offline"}},
    )
    report.assert_passed()


def test_target_and_evaluator_failures_do_not_abort_dataset():
    def target(value):
        if value == "bad":
            raise RuntimeError("secret-key")
        return value

    def scorer(case, output):
        if output == "scorer-bad":
            raise ValueError("private-data")
        return 1

    report = evaluate(
        target,
        [EvalCase(x, id=x) for x in ["bad", "scorer-bad", "ok"]],
        evaluators={"custom": scorer, "other": lambda c, o: 1},
    )
    assert report.summary["target_errors"] == 1
    assert report.summary["evaluator_errors"] == 1
    assert report.summary["pass_rate"] == 1 / 3
    assert report.rows[1].scores["other"].passed
    assert "secret" not in json.dumps(report.to_dict())
    assert "private" not in json.dumps(report.to_dict())
    with pytest.raises(AssertionError):
        report.assert_passed(min_pass_rate=0)


def test_thresholds_latency_gate_and_empty_dataset():
    report = evaluate(
        str,
        [EvalCase("a"), EvalCase("b")],
        evaluators={"custom": lambda c, o: Score(0.8, threshold=0.7, reason="secret")},
    )
    report.assert_passed()
    assert "secret" not in json.dumps(report.to_dict())
    assert "secret" in json.dumps(report.to_dict(include_content=True))
    with pytest.raises(AssertionError):
        report.assert_passed(max_p95_ms=0)
    for kw in ({"min_pass_rate": -1}, {"min_pass_rate": float("nan")}, {"max_p95_ms": -1}):
        with pytest.raises(ValueError):
            report.assert_passed(**kw)
    empty = evaluate(str, [], evaluators={"exact": exact_match})
    assert empty.summary["latency_ms"]["p95"] == 0
    with pytest.raises(AssertionError):
        empty.assert_passed(min_pass_rate=0)
    failed = evaluate(str, [EvalCase("a", "b")], evaluators={"exact": exact_match})
    with pytest.raises(AssertionError):
        failed.assert_passed()
    failed.assert_passed(min_pass_rate=0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, "yes"])
def test_invalid_scores(value):
    with pytest.raises(ValueError):
        Score(value)
    report = evaluate(str, [EvalCase("a")], evaluators={"bad": lambda c, o: value})
    assert report.rows[0].evaluator_errors == {"bad": "ValueError"}


def test_invalid_dataset_and_scorers():
    for cases, scorers in [
        ([EvalCase("a")], {}),
        ([EvalCase("a")], {"bad": 1}),
        ([EvalCase("a", id="x"), EvalCase("b", id="x")], {"e": exact_match}),
        (["bad"], {"e": exact_match}),
        ([EvalCase("a", id="")], {"e": exact_match}),
    ]:
        with pytest.raises(ValueError):
            evaluate(str, cases, evaluators=scorers)


def test_deterministic_metrics():
    assert exact_match(EvalCase("", "hello"), AIMessage(content="hello")).passed
    assert exact_match(EvalCase("", {"x": 1}), {"x": 1}).passed
    assert not exact_match(EvalCase("", "Hello"), "hello").passed
    assert contains_expected(EvalCase("", "hello"), "hello there").passed
    for expected in [None, ""]:
        with pytest.raises(ValueError):
            contains_expected(EvalCase("", expected), "answer")
    with pytest.raises(ValueError):
        exact_match(EvalCase(""), "answer")

    class Schema(BaseModel):
        category: str

    metric = json_valid(Schema)
    assert metric(EvalCase(""), '{"category":"food"}').passed
    assert not metric(EvalCase(""), '{"wrong": 2}').passed
    assert not json_valid()(EvalCase(""), "NaN").passed
    assert not json_valid()(EvalCase(""), "not-json").passed


def test_retrieval_recall_deduplicates_and_obeys_k():
    docs = [
        Document(page_content="a", id="a"),
        Document(page_content="a", id="a"),
        Document(page_content="b", id="b"),
        Document(page_content="missing"),
    ]
    case = EvalCase("query", reference_ids=("a", "b", "b"))
    assert retrieval_recall(k=2)(case, docs).value == 0.5
    assert retrieval_recall(k=4)(case, Result("", sources=tuple(docs))).passed
    assert (
        retrieval_recall(id_key="source")(
            case, [Document(page_content="a", metadata={"source": "a"})]
        ).value
        == 0.5
    )
    with pytest.raises(ValueError):
        retrieval_recall()(EvalCase(""), docs)
    with pytest.raises(ValueError):
        retrieval_recall(k=0)


def test_llm_judge_validated_and_size_bounded():
    model = ScriptedChatModel(responses=[AIMessage(content='{"score":0.9,"reason":"grounded"}')])
    judge = llm_judge(model, rubric="Answer must be supported by the sources.")
    output = Result("answer", sources=(Document(page_content="evidence"),))
    score = judge(EvalCase("question", "reference"), output)
    assert score.passed and score.reason == "grounded"
    payload = json.loads(model.requests[0][-1].content)
    assert payload["sources"] == ["evidence"]
    assert payload["expected"] == "reference"
    for kwargs in ({"rubric": ""}, {"rubric": "x", "max_input_chars": 0}):
        with pytest.raises(ValueError):
            llm_judge(model, **kwargs)
    short = llm_judge(model, rubric="x", max_input_chars=1)
    with pytest.raises(ValueError):
        short(EvalCase("question"), "answer")


@pytest.mark.parametrize(
    "response", ["not-json", '{"score":2,"reason":"bad"}', '{"score":0.8,"reason":"ok","extra":1}']
)
def test_judge_errors_are_not_passing_scores(response):
    judge = llm_judge(ScriptedChatModel(responses=[AIMessage(content=response)]), rubric="Accuracy")
    report = evaluate(str, [EvalCase("x")], evaluators={"judge": judge})
    assert report.rows[0].evaluator_errors
    assert not report.rows[0].passed


@pytest.mark.asyncio
async def test_async_concurrency_order_errors_and_custom_scorer():
    active = peak = 0

    async def target(value):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.002 * (5 - value))
            if value == 3:
                raise ValueError("sensitive")
            return str(value)
        finally:
            active -= 1

    async def metric(case, output):
        await asyncio.sleep(0)
        if output == "2":
            raise RuntimeError("sensitive")
        return exact_match(case, output)

    report = await aevaluate(
        target,
        [EvalCase(i, str(i), id=str(i)) for i in range(5)],
        evaluators={"exact": metric},
        max_concurrency=2,
    )
    assert peak == 2 and active == 0
    assert [row.case_id for row in report.rows] == list(map(str, range(5)))
    assert report.rows[2].evaluator_errors == {"exact": "RuntimeError"}
    assert report.rows[3].error == "ValueError"
    assert report.summary["passed"] == 3
    with pytest.raises(ValueError):
        await aevaluate(str, [], evaluators={"exact": exact_match}, max_concurrency=0)


@pytest.mark.asyncio
async def test_async_agent_workflow_judge_and_sync_callable():
    bot = agent(ScriptedChatModel(responses=[AIMessage(content="x")]))
    judge = llm_judge(
        ScriptedChatModel(responses=[AIMessage(content='{"score":1.0,"reason":"ok"}')]),
        rubric="Accuracy",
    )
    (await aevaluate(bot, [EvalCase("x")], evaluators={"judge": judge})).assert_passed()
    (
        await aevaluate(
            sequence(str.upper), [EvalCase("a", "A")], evaluators={"exact": exact_match}
        )
    ).assert_passed()
    (await aevaluate(str, [EvalCase("a", "a")], evaluators={"exact": exact_match})).assert_passed()


@pytest.mark.asyncio
async def test_cancellation_propagates_and_stops_workers():
    active = 0
    started = asyncio.Event()

    async def target(value):
        nonlocal active
        active += 1
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            active -= 1

    task = asyncio.create_task(
        aevaluate(
            target,
            [EvalCase(i) for i in range(8)],
            evaluators={"exact": exact_match},
            max_concurrency=2,
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert active == 0


def test_sync_runner_rejects_async_scorers_and_targets():
    async def scorer(case, output):
        return 1

    async def target(value):
        return value

    report = evaluate(str, [EvalCase("a")], evaluators={"async": scorer})
    assert report.rows[0].evaluator_errors == {"async": "TypeError"}
    report = evaluate(target, [EvalCase("a")], evaluators={"exact": exact_match})
    assert report.rows[0].error == "TypeError"
