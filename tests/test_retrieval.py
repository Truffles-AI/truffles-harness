import pytest
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from truffles import (
    BM25Retriever,
    HybridRetriever,
    MultiQueryRetriever,
    RerankRetriever,
    agent,
    rag,
    split,
    vectorstore,
)
from truffles.knowledge import _context
from truffles.retrieval import document_key
from truffles.testing import HashEmbeddings, ScriptedChatModel

TEXTS = [
    "Python is a programming language.",
    "Qdrant is a vector database.",
    "Python supports asynchronous programming.",
    "Oranges are fruit.",
]


def test_bm25_ranking_empty_and_unicode():
    retriever = BM25Retriever.from_texts(TEXTS, k=2)
    docs = retriever.invoke("QDRANT database")
    assert len(docs) == 1 and docs[0].page_content == TEXTS[1]
    assert retriever.invoke("unmatched") == []
    assert BM25Retriever.from_texts([]).invoke("hello") == []
    assert BM25Retriever.from_texts([""]).invoke("hello") == []
    assert BM25Retriever.from_texts(["café crème"]).invoke("CAFÉ")[0].page_content == "café crème"


async def test_bm25_async_and_no_mutation():
    original = Document(page_content="Python", metadata={"source": "guide"})
    retriever = BM25Retriever.from_documents([original])
    original.page_content = "changed"
    result = await retriever.ainvoke("Python")
    result[0].page_content = "changed again"
    assert retriever.invoke("Python")[0].page_content == "Python"


def test_bm25_known_formula_ranking():
    retriever = BM25Retriever.from_texts(["cat cat", "cat dog dog dog", "dog"])
    assert [d.page_content for d in retriever.invoke("cat")] == ["cat cat", "cat dog dog dog"]


def test_split_preserves_sources_and_assigns_unique_chunk_ids():
    doc = Document(
        id="parent", page_content="alpha beta gamma delta epsilon", metadata={"source": "a"}
    )
    chunks = split([doc], chunk_size=12, chunk_overlap=3)
    assert len(chunks) > 1
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(c.metadata["source"] == "a" and c.metadata["start_index"] >= 0 for c in chunks)
    assert doc.id == "parent"


@pytest.mark.parametrize(
    "options", [{"chunk_size": 0}, {"chunk_size": 2, "chunk_overlap": 2}, {"chunk_overlap": -1}]
)
def test_invalid_split(options):
    with pytest.raises(ValueError):
        split(["x"], **options)


def test_rrf_duplicate_suppression_and_weights():
    a, b, c = [Document(page_content=x) for x in "abc"]
    retriever = HybridRetriever(
        retrievers=[RunnableLambda(lambda q: [a, a, b]), RunnableLambda(lambda q: [b, c])], k=3
    )
    assert retriever.invoke("query") == [b, a, c]
    weighted = HybridRetriever(
        retrievers=[RunnableLambda(lambda q: [a]), RunnableLambda(lambda q: [b])], weights=[0, 1]
    )
    assert weighted.invoke("query") == [b]


@pytest.mark.parametrize(
    "options",
    [
        {"retrievers": []},
        {"retrievers": [1], "weights": []},
        {"retrievers": [1], "weights": [-1]},
        {"retrievers": [1], "weights": [float("nan")]},
        {"retrievers": [1], "weights": [0]},
    ],
)
def test_invalid_rrf(options):
    with pytest.raises(ValueError):
        HybridRetriever(**options)


async def test_hybrid_async():
    lexical = BM25Retriever.from_texts(TEXTS)
    hybrid = HybridRetriever(retrievers=[lexical, lexical])
    assert await hybrid.ainvoke("Python") == hybrid.invoke("Python")


async def test_multiquery_expansion_dedup_and_bound():
    queries = []

    def retrieve(query):
        queries.append(query)
        return [Document(page_content=query)]

    def generate(prompt):
        return AIMessage(content="1. alternate\n2. alternate\n- second\nthird")

    retriever = MultiQueryRetriever(
        retriever=RunnableLambda(retrieve), generator=RunnableLambda(generate), max_queries=3, k=2
    )
    assert [d.page_content for d in retriever.invoke("original")] == ["original", "alternate"]
    assert queries == ["original", "alternate", "second"]
    assert [d.page_content for d in await retriever.ainvoke("original")] == [
        "original",
        "alternate",
    ]
    assert len(queries) == 6


def test_multiquery_one_query_does_not_call_model():
    def fail(_):
        raise AssertionError("generator should not run")

    retriever = MultiQueryRetriever(
        retriever=BM25Retriever.from_texts(TEXTS), generator=RunnableLambda(fail), max_queries=1
    )
    assert retriever.invoke("Qdrant")


class ReverseCompressor(BaseDocumentCompressor):
    def compress_documents(self, documents, query, callbacks=None):
        return list(reversed(documents))


async def test_reranker_and_empty_results():
    base = BM25Retriever.from_texts(TEXTS)
    reranker = RerankRetriever(retriever=base, compressor=ReverseCompressor(), k=1)
    assert reranker.invoke("Python") == list(reversed(base.invoke("Python")))[:1]
    assert await reranker.ainvoke("Python") == reranker.invoke("Python")
    assert reranker.invoke("missing") == await reranker.ainvoke("missing") == []


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ["similarity", "mmr", "hybrid"])
async def test_memory_rag_algorithms(algorithm):
    knowledge = rag(TEXTS, embedding=HashEmbeddings(), algorithm=algorithm, k=2)
    assert knowledge.retrieve("Qdrant database")[0].page_content == TEXTS[1]
    assert await knowledge.aretrieve("Qdrant database") == knowledge.retrieve("Qdrant database")
    assert len({d.page_content for d in knowledge.retrieve("Python")}) == 2


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ["similarity", "mmr", "similarity_score_threshold", "hybrid"])
def test_real_qdrant_local(algorithm):
    knowledge = rag(
        TEXTS,
        embedding=HashEmbeddings(),
        backend="qdrant",
        algorithm=algorithm,
        store_options={"location": ":memory:", "collection_name": "contract"},
        search_options={"score_threshold": 0.2} if "threshold" in algorithm else {},
    )
    assert knowledge.retrieve("Qdrant database")[0].page_content == TEXTS[1]
    # Confirm the native store remains fully usable for mutations.
    if algorithm == "hybrid":
        results = knowledge.retrieve("Qdrant database")
        assert len({d.page_content for d in results}) == len(results)
    ids = knowledge.store.add_texts(["Bananas are yellow fruit."])
    assert ids
    assert knowledge.retrieve("Bananas yellow")[0].page_content == "Bananas are yellow fruit."
    assert knowledge.store.delete(ids)
    knowledge.store.client.close()


async def test_rag_generation_sync_async_and_tool():
    knowledge = rag(TEXTS, k=1, max_context_chars=100)
    model = ScriptedChatModel(responses=[AIMessage(content="A database [1].") for _ in range(2)])
    bot = agent(model)
    result = knowledge.ask("Qdrant", model=bot)
    assert result.sources[0].page_content == TEXTS[1]
    assert "source_excerpts" in model.requests[0][-1].content
    assert (await knowledge.aask("Qdrant", model=bot)).text == result.text
    assert "Qdrant" in await knowledge.as_tool().ainvoke({"query": "Qdrant"})


def test_rag_existing_native_store_and_retriever():
    store = vectorstore("memory", texts=TEXTS, embedding=HashEmbeddings())
    assert rag(store=store).retrieve("Qdrant")
    assert rag(retriever=BM25Retriever.from_texts(TEXTS)).retrieve("Qdrant")
    assert rag([], chunk_size=None).retrieve("anything") == []


def test_context_limit_and_document_identity():
    doc = Document(page_content="a" * 100, metadata={"source": "s" * 100})
    assert len(_context([doc, doc], 25)) <= 25
    assert document_key(doc) == document_key(doc.model_copy())
    assert document_key(doc) != document_key(Document(page_content=doc.page_content))
    assert document_key(Document(id="abc", page_content="anything")) == "abc"


@pytest.mark.parametrize(
    "options",
    [
        {"k": 0},
        {"algorithm": "wrong"},
        {"algorithm": "mmr"},
        {"max_context_chars": 0},
        {"embedding": HashEmbeddings(), "algorithm": "bm25"},
        {"store": object(), "embedding": HashEmbeddings()},
        {"retriever": object(), "embedding": HashEmbeddings()},
        {"algorithm": "hybrid", "store": object()},
    ],
)
def test_rag_bad_options(options):
    with pytest.raises(ValueError):
        rag(**options)


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["chroma", "faiss"])
@pytest.mark.parametrize("algorithm", ["similarity", "mmr", "hybrid"])
def test_additional_local_stores(backend, algorithm):
    options = {}
    if backend == "chroma":
        from uuid import uuid4

        from chromadb.config import Settings

        options = {
            "collection_name": "truffles_" + uuid4().hex,
            "client_settings": Settings(anonymized_telemetry=False),
        }
    knowledge = rag(
        TEXTS,
        backend=backend,
        embedding=HashEmbeddings(),
        algorithm=algorithm,
        store_options=options,
        k=2,
    )
    try:
        assert knowledge.retrieve("Qdrant database")[0].page_content == TEXTS[1]
        results = knowledge.retrieve("Qdrant database")
        assert len({d.page_content for d in results}) == len(results)
    finally:
        if backend == "chroma":
            knowledge.store.delete_collection()


def test_unsupported_native_retrieval_does_not_silently_fallback():
    knowledge = rag(
        TEXTS,
        embedding=HashEmbeddings(),
        algorithm="similarity_score_threshold",
        search_options={"score_threshold": 0.8},
    )
    with pytest.raises(NotImplementedError):
        knowledge.retrieve("Python")
