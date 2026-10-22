"""Composable retrievers with native LangChain sync/async and callback support."""

import asyncio
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, model_validator

from .results import text_content


def document_key(document: Document) -> str:
    if document.id is not None:
        return document.id
    payload = json.dumps(
        [document.page_content, document.metadata], sort_keys=True, default=str, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


class BM25Retriever(BaseRetriever):
    """Dependency-free BM25 with precomputed postings; Unicode word tokenization.

    Build once with from_documents. Treat the corpus as immutable; rebuild to update.
    Only documents with positive lexical evidence are returned.
    """

    documents: list[Document]
    k: int = Field(default=4, gt=0)
    k1: float = Field(default=1.5, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)
    postings: dict[str, list[tuple[int, int]]] = Field(default_factory=dict, exclude=True)
    lengths: list[int] = Field(default_factory=list, exclude=True)
    average_length: float = Field(default=0, exclude=True)

    @model_validator(mode="after")
    def _index(self):
        postings = defaultdict(list)
        self.lengths = []
        for index, document in enumerate(self.documents):
            counts = Counter(_tokens(document.page_content))
            self.lengths.append(sum(counts.values()))
            for word, count in counts.items():
                postings[word].append((index, count))
        self.postings = dict(postings)
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0
        return self

    @classmethod
    def from_documents(cls, documents: Sequence[Document], **options):
        return cls(documents=[d.model_copy(deep=True) for d in documents], **options)

    @classmethod
    def from_texts(cls, texts: Sequence[str], **options):
        return cls.from_documents([Document(page_content=t) for t in texts], **options)

    def _get_relevant_documents(self, query: str, *, run_manager):
        if not self.average_length:
            return []
        scores: dict[int, float] = defaultdict(float)
        n = len(self.documents)
        for term in set(_tokens(query)):
            postings = self.postings.get(term, [])
            idf = math.log(1 + (n - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, frequency in postings:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average_length)
                scores[index] += idf * frequency * (self.k1 + 1) / (frequency + norm)
        ranked = sorted(scores, key=lambda i: (-scores[i], i))[: self.k]
        return [self.documents[i].model_copy(deep=True) for i in ranked]


class HybridRetriever(BaseRetriever):
    """Weighted reciprocal rank fusion (RRF); independent of store score scales."""

    retrievers: list[Any]
    weights: list[float] | None = None
    k: int = Field(default=4, gt=0)
    rank_constant: int = Field(default=60, gt=0)

    @model_validator(mode="after")
    def _validate(self):
        if not self.retrievers:
            raise ValueError("At least one retriever is required.")
        if self.weights is None:
            self.weights = [1.0] * len(self.retrievers)
        if len(self.weights) != len(self.retrievers):
            raise ValueError("Provide one weight per retriever.")
        if any(not math.isfinite(w) or w < 0 for w in self.weights) or not sum(self.weights):
            raise ValueError("Weights must be finite, nonnegative, and have a positive sum.")
        return self

    def _fuse(self, batches):
        scores: dict[str, float] = defaultdict(float)
        documents: dict[str, Document] = {}
        for weight, batch in zip(self.weights, batches, strict=True):
            if not weight:
                continue
            seen = set()
            for rank, doc in enumerate(batch, 1):
                key = document_key(doc)
                if key in seen:
                    continue
                seen.add(key)
                documents.setdefault(key, doc)
                scores[key] += weight / (self.rank_constant + rank)
        return [documents[key] for key in sorted(scores, key=lambda key: -scores[key])[: self.k]]

    def _get_relevant_documents(self, query, *, run_manager):
        return self._fuse(
            [
                r.invoke(query, config={"callbacks": run_manager.get_child()})
                for r in self.retrievers
            ]
        )

    async def _aget_relevant_documents(self, query, *, run_manager):
        batches = await asyncio.gather(
            *[
                r.ainvoke(query, config={"callbacks": run_manager.get_child()})
                for r in self.retrievers
            ]
        )
        return self._fuse(batches)


class MultiQueryRetriever(BaseRetriever):
    """Generate newline-separated query variants, retrieve, deduplicate, and cap results."""

    retriever: Any
    generator: Any
    max_queries: int = Field(default=3, gt=0, le=20)
    k: int = Field(default=8, gt=0)

    def _prompt(self, query):
        return (
            f"Write at most {self.max_queries - 1} alternative search queries for the question. "
            f"Return one query per line, no numbering or explanation.\nQuestion: {query}"
        )

    def _queries(self, query, answer):
        text = text_content(answer) if isinstance(answer, BaseMessage) else str(answer)
        variants = [
            re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line).strip() for line in text.splitlines()
        ]
        return list(dict.fromkeys([query, *[v for v in variants if v]]))[: self.max_queries]

    def _merge(self, batches):
        docs = {}
        for batch in batches:
            for doc in batch:
                docs.setdefault(document_key(doc), doc)
        return list(docs.values())[: self.k]

    def _get_relevant_documents(self, query, *, run_manager):
        config = {"callbacks": run_manager.get_child()}
        answer = self.generator.invoke(self._prompt(query), config) if self.max_queries > 1 else ""
        return self._merge([self.retriever.invoke(q, config) for q in self._queries(query, answer)])

    async def _aget_relevant_documents(self, query, *, run_manager):
        config = {"callbacks": run_manager.get_child()}
        answer = (
            await self.generator.ainvoke(self._prompt(query), config)
            if self.max_queries > 1
            else ""
        )
        batches = await asyncio.gather(
            *[self.retriever.ainvoke(q, config) for q in self._queries(query, answer)]
        )
        return self._merge(batches)


class RerankRetriever(BaseRetriever):
    """Adapt any native document compressor/reranker, such as CohereRerank."""

    retriever: Any
    compressor: Any
    k: int = Field(default=4, gt=0)

    def _get_relevant_documents(self, query, *, run_manager):
        callbacks = run_manager.get_child()
        docs = self.retriever.invoke(query, {"callbacks": callbacks})
        if not docs:
            return []
        return list(self.compressor.compress_documents(docs, query, callbacks=callbacks))[: self.k]

    async def _aget_relevant_documents(self, query, *, run_manager):
        callbacks = run_manager.get_child()
        docs = await self.retriever.ainvoke(query, {"callbacks": callbacks})
        if not docs:
            return []
        return list(await self.compressor.acompress_documents(docs, query, callbacks=callbacks))[
            : self.k
        ]
