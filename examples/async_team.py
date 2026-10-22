"""Async team and RAG, fully offline."""

import asyncio

from langchain_core.messages import AIMessage

from truffles import agent, parallel, rag
from truffles.testing import ScriptedChatModel


async def main():
    team = parallel(
        **{
            name: agent(ScriptedChatModel(responses=[AIMessage(content=reply)]))
            for name, reply in {"writer": "Draft", "reviewer": "Review"}.items()
        }
    )
    print(await team.ainvoke("Write and review", config={"max_concurrency": 2}))
    knowledge = rag(["Python supports asyncio."])
    result = await knowledge.aask(
        "What does Python support?",
        model=ScriptedChatModel(responses=[AIMessage(content="asyncio [1].")]),
    )
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
