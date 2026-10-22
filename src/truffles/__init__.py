"""Truffles: small conveniences, native LangChain and LangGraph underneath.

Public objects load on first access. Importing Truffles alone imports no SDKs.
"""

from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "0.1.0"

_EXPORTS = {
    "Agent": ".agents",
    "agent": ".agents",
    "speech_model": ".speech",
    "Result": ".results",
    "chat_model": ".providers",
    "embeddings": ".providers",
    "register_provider": ".providers",
    "register_embeddings": ".providers",
    "vectorstore": ".stores",
    "register_vectorstore": ".stores",
    "RAG": ".knowledge",
    "rag": ".knowledge",
    "split": ".knowledge",
    "BM25Retriever": ".retrieval",
    "HybridRetriever": ".retrieval",
    "MultiQueryRetriever": ".retrieval",
    "RerankRetriever": ".retrieval",
    "sequence": ".workflows",
    "parallel": ".workflows",
    "route": ".workflows",
    "supervisor": ".workflows",
    "wrap": ".workflows",
    "search_tool": ".search",
    "tool": "langchain_core.tools",
    "Document": "langchain_core.documents",
    "StateGraph": "langgraph.graph",
    "START": "langgraph.graph",
    "END": "langgraph.graph",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'truffles' has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from langchain_core.documents import Document as Document
    from langchain_core.tools import tool as tool
    from langgraph.graph import END as END
    from langgraph.graph import START as START
    from langgraph.graph import StateGraph as StateGraph

    from .agents import Agent as Agent
    from .agents import agent as agent
    from .knowledge import RAG as RAG
    from .knowledge import rag as rag
    from .knowledge import split as split
    from .providers import (
        chat_model as chat_model,
    )
    from .providers import (
        embeddings as embeddings,
    )
    from .providers import (
        register_embeddings as register_embeddings,
    )
    from .providers import (
        register_provider as register_provider,
    )
    from .results import Result as Result
    from .retrieval import (
        BM25Retriever as BM25Retriever,
    )
    from .retrieval import (
        HybridRetriever as HybridRetriever,
    )
    from .retrieval import (
        MultiQueryRetriever as MultiQueryRetriever,
    )
    from .retrieval import (
        RerankRetriever as RerankRetriever,
    )
    from .search import search_tool as search_tool
    from .speech import speech_model as speech_model
    from .stores import register_vectorstore as register_vectorstore
    from .stores import vectorstore as vectorstore
    from .workflows import (
        parallel as parallel,
    )
    from .workflows import (
        route as route,
    )
    from .workflows import (
        sequence as sequence,
    )
    from .workflows import (
        supervisor as supervisor,
    )
    from .workflows import (
        wrap as wrap,
    )
