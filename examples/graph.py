"""A native conditional graph, wrapped without losing graph behavior."""

from typing import TypedDict

from truffles import END, START, StateGraph, wrap


class State(TypedDict):
    count: int


builder = StateGraph(State)
builder.add_node("increment", lambda state: {"count": state["count"] + 1})
builder.add_edge(START, "increment")
builder.add_conditional_edges(
    "increment", lambda state: END if state["count"] >= 3 else "increment"
)
print(wrap(builder.compile()).invoke({"count": 0}))
