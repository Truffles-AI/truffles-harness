"""Local overhead measurements, not a provider-throughput benchmark."""

import json
import statistics
import subprocess
import sys
import time

from langchain_core.messages import AIMessage

from truffles import BM25Retriever, agent
from truffles.testing import ScriptedChatModel


def per_call(fn, count=100):
    start = time.perf_counter()
    for _ in range(count):
        fn()
    return (time.perf_counter() - start) * 1_000 / count


def compare():
    wrapped, native = [], []
    for _ in range(3):
        bot = agent(ScriptedChatModel(responses=[AIMessage(content="ok") for _ in range(202)]))
        bot.run("warm up")
        wrapped.append(per_call(lambda bot=bot: bot.run("hello")))
        native.append(
            per_call(
                lambda bot=bot: bot.native.invoke(
                    {"messages": [{"role": "user", "content": "hello"}]}
                )
            )
        )
    retriever = BM25Retriever.from_texts(
        [f"Document {i} about topic_{i % 100} and retrieval" for i in range(10_000)]
    )
    import_times = [
        float(
            subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    "import time; t=time.perf_counter(); import truffles; "
                    "print((time.perf_counter()-t)*1000)",
                ],
                text=True,
            )
        )
        for _ in range(5)
    ]
    print(
        json.dumps(
            {
                "python": sys.version.split()[0],
                "truffles_import_ms_median": round(statistics.median(import_times), 3),
                "wrapped_agent_ms_per_call_median": round(statistics.median(wrapped), 3),
                "native_same_graph_ms_per_call_median": round(statistics.median(native), 3),
                "bm25_10000_docs_ms_per_query": round(
                    per_call(lambda: retriever.invoke("topic_42")), 3
                ),
                "note": "Local scripted model; hardware-dependent. Not a live inference benchmark.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    compare()
