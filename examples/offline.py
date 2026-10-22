"""Run: python examples/offline.py. No API keys, services or model downloads."""

from langchain_core.messages import AIMessage

from truffles import agent, parallel, rag, sequence, supervisor, tool
from truffles.testing import ScriptedChatModel


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def scripted(*responses):
    return ScriptedChatModel(
        responses=[AIMessage(content=r) if isinstance(r, str) else r for r in responses]
    )


bot = agent(
    scripted(
        AIMessage(
            content="",
            tool_calls=[{"name": "multiply", "args": {"a": 7, "b": 6}, "id": "multiply-1"}],
        ),
        "7 × 6 = 42.",
    ),
    tools=[multiply],
)
print("ReAct:", bot.run("Multiply 7 by 6."))

knowledge = rag(["Truffles wraps LangChain and LangGraph.", "Qdrant stores vectors."])
answer = knowledge.ask("What does Truffles wrap?", model=scripted("LangChain and LangGraph [1]."))
print("RAG:", answer)
print("Sources:", [d.page_content for d in answer.sources])

pipeline = sequence(agent(scripted("A draft.")), agent(scripted("An edited draft.")))
print("Pipeline:", pipeline.invoke("Write a draft."))

team = parallel(research=agent(scripted("Evidence.")), critique=agent(scripted("Limitations.")))
print("Parallel:", team.invoke("Review this topic."))

manager_model = scripted(
    AIMessage(
        content="",
        tool_calls=[{"name": "research", "args": {"query": "Find evidence"}, "id": "delegate-1"}],
    ),
    "Summary of the evidence.",
)
manager = supervisor(manager_model, {"research": agent(scripted("Evidence found."))})
print("Supervisor:", manager.run("Research this topic."))
