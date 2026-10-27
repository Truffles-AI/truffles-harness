"""Run offline evaluation and write content-free reports: python examples/evaluation.py."""

import argparse
import json
from pathlib import Path

from langchain_core.messages import AIMessage

from truffles import (
    BM25Retriever,
    Document,
    EvalCase,
    JSONLTraceSink,
    Observer,
    agent,
    evaluate,
    exact_match,
    llm_judge,
    retrieval_recall,
)
from truffles.testing import ScriptedChatModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".truffles/evaluation"))
    directory = parser.parse_args().output_dir
    directory.mkdir(parents=True, exist_ok=True)
    observer = Observer(sink=JSONLTraceSink(directory / "traces.jsonl"))
    bot = agent(ScriptedChatModel(responses=[AIMessage(content="42"), AIMessage(content="Paris")]))
    cases = [
        EvalCase("Six times seven?", "42", id="arithmetic"),
        EvalCase("Capital of France?", "Paris", id="geography"),
    ]
    report = evaluate(
        bot, cases, evaluators={"exact": exact_match}, config={"callbacks": [observer]}
    )
    report.assert_passed().save(directory / "answers.json")

    retriever = BM25Retriever.from_documents(
        [
            Document(page_content="Savings accounts earn interest.", id="savings"),
            Document(page_content="Cards have credit limits.", id="cards"),
        ],
        k=1,
    )
    retrieval = evaluate(
        retriever,
        [EvalCase("savings interest", id="retrieval", reference_ids=("savings",))],
        evaluators={"recall_at_1": retrieval_recall(k=1)},
        config={"callbacks": [observer]},
    )
    retrieval.assert_passed().save(directory / "retrieval.json")

    # Exercise the judge contract without calling a live model. This is NOT a quality claim.
    judge = llm_judge(
        ScriptedChatModel(
            responses=[
                AIMessage(content='{"score": 1.0, "reason": "Scripted test judgment"}'),
            ]
        ),
        rubric="The answer should match the reference.",
    )
    judged = evaluate(str, [EvalCase("42", "42", id="judge-contract")], evaluators={"judge": judge})
    judged.assert_passed().save(directory / "judge.json")
    print(
        json.dumps(
            {
                "answers": report.summary,
                "retrieval": retrieval.summary,
                "telemetry": observer.summary(),
            },
            indent=2,
        )
    )
    print(f"Reports and traces: {directory.resolve()}")


if __name__ == "__main__":
    main()
