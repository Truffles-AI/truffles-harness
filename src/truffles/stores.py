"""Vector-store plugins return native objects; no invented universal filter schema."""

from collections.abc import Callable
from typing import Any

from .providers import _load, _register
from .providers import embeddings as make_embeddings

_STORES = {
    "memory": ("langchain_core.vectorstores", "InMemoryVectorStore", "memory"),
    "qdrant": ("langchain_qdrant", "QdrantVectorStore", "qdrant"),
    "chroma": ("langchain_chroma", "Chroma", "chroma"),
    "pinecone": ("langchain_pinecone", "PineconeVectorStore", "pinecone"),
    "weaviate": ("langchain_weaviate", "WeaviateVectorStore", "weaviate"),
    "milvus": ("langchain_milvus", "Milvus", "milvus"),
    "postgres": ("langchain_postgres", "PGVector", "postgres"),
    "redis": ("langchain_redis", "RedisVectorStore", "redis"),
    "faiss": ("langchain_community.vectorstores", "FAISS", "faiss"),
    "elasticsearch": ("langchain_elasticsearch", "ElasticsearchStore", "elasticsearch"),
}
_custom_stores: dict[str, Callable[..., Any]] = {}


def register_vectorstore(name: str, factory: Callable[..., Any], *, replace=False) -> None:
    """Register a factory accepting texts, embedding, metadatas and native kwargs."""
    _register(_custom_stores, _STORES, name, factory, replace)


def vectorstore(
    backend: str,
    *,
    texts: list[str],
    embedding: Any,
    metadatas: list[dict] | None = None,
    **options: Any,
):
    """Create/index a store using its native from_texts contract.

    To attach an existing store, pass that native object straight to rag(store=...).
    Construction writes documents to the selected store and can call embedding APIs.
    """
    backend = backend.lower().strip()
    if not texts:
        raise ValueError(
            "Provide at least one text; attach an existing native store for empty indexes."
        )
    if metadatas is not None and len(metadatas) != len(texts):
        raise ValueError("metadatas must have one entry per text.")
    embedding = make_embeddings(embedding)
    if backend in _custom_stores:
        return _custom_stores[backend](
            texts=texts, embedding=embedding, metadatas=metadatas, **options
        )
    if backend not in _STORES:
        raise ValueError(
            f"Unknown vector store {backend!r}; register it or pass a native store to rag."
        )
    cls = _load(*_STORES[backend])
    return cls.from_texts(texts=texts, embedding=embedding, metadatas=metadatas, **options)
