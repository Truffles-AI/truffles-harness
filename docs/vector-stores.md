# Vector-store plugins

The factory calls each integration's native `from_texts` method and returns its
real vector-store object. This supports native filters, delete/update operations,
retrievers, async APIs and persistence without recreating their interfaces.

```python
from truffles import vectorstore
from truffles.testing import HashEmbeddings
store = vectorstore("qdrant", texts=["A sample document"], embedding=HashEmbeddings(),
                    location=":memory:", collection_name="example")
print(store.similarity_search("sample", k=1))
store.client.close()
```

## Named integrations

| Backend | Class | Extra | Typical native creation settings |
| --- | --- | --- | --- |
| `memory` | `InMemoryVectorStore` | `memory` | No service; process-local vectors |
| `qdrant` | `QdrantVectorStore` | `qdrant` | `location`, or `url`, `api_key`, `collection_name` |
| `chroma` | `Chroma` | `chroma` | `collection_name`, `persist_directory` or `client` |
| `pinecone` | `PineconeVectorStore` | `pinecone` | Existing `index_name`, optional `namespace` |
| `weaviate` | `WeaviateVectorStore` | `weaviate` | Configured `client`, `index_name`, `text_key` |
| `milvus` | `Milvus` | `milvus` | `connection_args`, `collection_name` |
| `postgres` | `PGVector` | `postgres` | `connection`, `collection_name` |
| `redis` | `RedisVectorStore` | `redis` | Configured native `RedisConfig` via `config` |
| `faiss` | `FAISS` | `faiss` | Native index options; local index |
| `elasticsearch` | `ElasticsearchStore` | `elasticsearch` | `es_url`/`es_connection`, `index_name` |

This table is an integration registry, not a claim that all services were exercised
live. Local Qdrant, Chroma, FAISS and in-memory execution are covered by integration tests. See
[compatibility](compatibility.md) for the exact verification scope.

Backend libraries may require pre-created indexes, service-side collection
configuration, extensions, TLS settings or credentials. Truffles forwards native
kwargs and errors. It does not provision infrastructure automatically beyond what
the native `from_texts` implementation does.

## Attach an existing store

```python
from truffles import rag
# store may be any configured LangChain vector-store instance.
knowledge = rag(store=store, k=5, search_options={})
```

This is the widest integration path: any store implementing `as_retriever` can
participate, even when it is not named in Truffles' registry. Any native retriever
can instead be passed as `rag(retriever=...)`. These interfaces cover integrations
such as OpenSearch, Azure AI Search, LanceDB and other community stores without
adding compulsory dependencies or claiming service-specific behavior.

## Filtering and score semantics

Pass filters in `search_options` using your backend's native schema. Qdrant accepts
its `Filter` models; other stores may use dictionaries or differently named
arguments. Truffles does not translate these formats. MMR needs backend support;
`similarity_score_threshold` requires a valid native relevance-score implementation.

For access control, apply trusted filters in your application and native retriever.
Do not let a model freely choose tenant or permission filters. Use separate indexes
or namespaces where required by your application's isolation model.

## Persistence and updates

Keep the returned store and use its native API. For Qdrant local persistence use
`path=...`; for remote Qdrant use its service URL. Chroma can use a persistent
directory. FAISS exposes save/load methods. Production services retain data
according to their own settings. Only load serialized indexes from trusted
sources; native pickle-based formats may execute code when deserialized.

Adding or deleting dense documents does not update a previously built BM25 index.
Rebuild both sides of a hybrid index together. Reusing the same embedding model
and dimensions across writes and reads is your responsibility.

## Custom backend

`register_vectorstore(name, factory)` accepts a trusted callable receiving
`texts`, `embedding`, `metadatas` and native options. The factory can call
`YourVectorStore.from_texts` directly. Existing names cannot be overwritten unless
`replace=True` is explicit. Registration is in-process; import your plugin during
application startup.
