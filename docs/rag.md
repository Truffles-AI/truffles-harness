# Retrieval-augmented generation

RAG has three independently usable stages: ingest documents, retrieve candidates,
and generate a grounded answer. The `rag` factory makes common configurations short
while returning a `RAG` object with its native retriever and store available.

## Text, documents and chunking

```python
from truffles import Document, rag
knowledge = rag([
    Document(page_content="The support desk opens at 09:00.", metadata={"source": "handbook"}),
    Document(page_content="Invoices are payable within 30 days.", metadata={"source": "billing"}),
], chunk_size=500, chunk_overlap=75, k=3)
```

Without an embedding or store, the algorithm defaults to BM25. With either, it
defaults to similarity search. Texts become native `Document` objects. Metadata is
preserved. Recursive splitting uses character counts, not tokens, and attaches
`start_index` metadata. Chunk identities are hashes of content and metadata, not
the original parent ID. Use source metadata to preserve a parent identifier.

`chunk_size=None` preserves already-chunked documents. `0 <= chunk_overlap <
chunk_size` is required. A custom splitter can be run first; pass its output with
splitting disabled. Native document loaders can load PDFs, web pages, databases
and other formats before passing their documents to Truffles.

## Dense retrieval and vector stores

```python
from truffles import rag
knowledge = rag(
    ["Qdrant is a vector database.", "Truffles is an agent harness."],
    embedding="openai:text-embedding-3-small",
    backend="qdrant",
    store_options={"location": ":memory:", "collection_name": "demo"},
)
documents = knowledge.retrieve("What stores vectors?")
```

This example embeds through OpenAI and stores vectors in local Qdrant. For an
entirely local test, replace the embedder with `HashEmbeddings()`. For production
local semantic embeddings, choose an appropriate installed local model.

Construction indexes the supplied texts and may make remote calls. Reuse the
result. Attaching an existing store performs no ingestion:

```python
attached = rag(store=knowledge.store, algorithm="mmr", k=4,
               search_options={"fetch_k": 20, "lambda_mult": 0.5})
```

Existing stores cannot be combined with input documents or a new embedding model;
perform mutations through the native store. For lexical+dense retrieval over an
existing corpus, construct each retriever explicitly and combine with RRF.

## Algorithms

| Algorithm | Construction | Behavior |
| --- | --- | --- |
| BM25 | `rag(texts, algorithm="bm25")` | Lexical ranking; no embedding cost |
| Similarity | `algorithm="similarity"` | Native store nearest-neighbor search |
| MMR | `algorithm="mmr"` | Native diversity/relevance tradeoff |
| Threshold | `algorithm="similarity_score_threshold"` | Native normalized relevance threshold |
| Hybrid | `algorithm="hybrid"` | Dense + BM25 combined by reciprocal rank fusion |
| Multi-query | `MultiQueryRetriever(...)` | Expand the query and merge candidates |
| Rerank | `RerankRetriever(...)` | Retrieve first, then use a native compressor |

Backend support for MMR, thresholds and filters varies. Unsupported native
operations raise their native error; Truffles does not silently fall back to
another algorithm. Thresholds and score normalization belong to the backend.
For example, a raw cosine distance is not interchangeable with a relevance score.

BM25 uses Unicode word tokens with case folding, `k1=1.5` and `b=0.75`. It returns
only documents with lexical overlap, so fewer than `k` documents may be returned.
The index is built once, and query scoring visits term postings. It does not apply
stemming, synonyms or language-specific segmentation. Rebuild after corpus changes.

## Hybrid fusion

```python
from truffles import BM25Retriever, HybridRetriever, rag
lexical = BM25Retriever.from_texts(["first document", "second document"], k=10)
# dense is any native retriever over the corresponding corpus.
dense = lexical  # replace with your_store.as_retriever(search_kwargs={"k": 10})
hybrid = HybridRetriever(retrievers=[dense, lexical], weights=[0.7, 0.3], k=4,
                         rank_constant=60)
knowledge = rag(retriever=hybrid)
```

Each list contributes `weight / (rank_constant + rank)`, with ranks starting at 1.
Documents are deduplicated per list. A native document ID defines identity when
present; otherwise content and metadata define identity. If different retrievers
assign different IDs to the same chunks, normalize them before manual fusion. The
`algorithm="hybrid"` ingestion convenience stores a reserved `_truffles_chunk_id`
metadata field and uses it as the shared identity. This handles backend-assigned
IDs and injected metadata without returning the same chunk twice. Do not use that
reserved metadata name for application data. It is a
snapshot of the ingested corpus; mutating only the vector store leaves BM25 stale.

## Query expansion and reranking

```python
from truffles import MultiQueryRetriever, chat_model, rag
expanded = MultiQueryRetriever(
    retriever=knowledge.retriever,
    generator=chat_model("openai:gpt-4.1-mini"),
    max_queries=3,
    k=8,
)
knowledge = rag(retriever=expanded)
```

The original query is always first. At most `max_queries - 1` generated variants
are included, duplicates are removed, and results are merged in query order. This
is a union, not a learned rerank. `max_queries=1` skips generation entirely. Async
retrieval runs the bounded query list concurrently. Query expansion incurs an
additional model call and may increase backend query volume.

`RerankRetriever(retriever=..., compressor=..., k=4)` accepts any native document
compressor, including a configured `CohereRerank` or contextual compression
pipeline. Native compressors implement `compress_documents` and
`acompress_documents`; the base compressor provides async fallback. Retrieval
filters should enforce access policy before documents reach a remote reranker.

## Generation and citations

```python
answer = knowledge.ask("What is the policy?", model="openai:gpt-4.1-mini")
print(answer.text)
for document in answer.sources:
    print(document.metadata)
```

`ask` always retrieves before calling the model. `aask` uses async retrieval and
generation. Passing an existing `Agent` reuses its client and configuration.
`max_context_chars` defaults to 16,000 and bounds the serialized source excerpt
section, including metadata. It does not cap question or instruction length.

Candidate sources remain in `Result.sources`, even if context truncation prevents
all of them from fitting. The prompt asks for numeric citations, but Truffles
does not prove the generated answer is supported. Inspect the retrieved sources
and evaluate grounding for your application.

`agent(model, knowledge=knowledge)` exposes a `search_knowledge` tool. The model
chooses whether to retrieve. Tool artifacts preserve documents in the final
result. Retrieved excerpts are labeled untrusted in `ask`; applications should
also include appropriate instructions for agentic retrieval.
