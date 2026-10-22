# Extending Truffles

Prefer passing an existing native object when possible. Registration is useful
when your application wants reusable string identifiers.

## Model and embedding factories

```python
from truffles import register_provider, register_embeddings
from truffles.testing import ScriptedChatModel, HashEmbeddings
from langchain_core.messages import AIMessage

register_provider("demo", lambda model, **options: ScriptedChatModel(
    responses=[AIMessage(content=f"Selected {model}")]
))
register_embeddings("demo", lambda model, **options: HashEmbeddings())
```

Factories receive `model=` and all supplied options. They should return a native
chat model or embeddings object. Names are normalized to lowercase; provider
aliases resolve before registration. Existing names require `replace=True` to
overwrite. Registration is process-local and should happen during startup, before
concurrent requests begin. Plugin code is trusted Python, not sandboxed code.

To create production models, return a real `BaseChatModel` integration. A custom
chat model must support `bind_tools` for tool agents; a text-only model does not
gain function calling by being wrapped. Unknown string chat providers can also
flow through LangChain's universal factory without registration.

## Vector stores

```python
from langchain_core.vectorstores import InMemoryVectorStore
from truffles import register_vectorstore
register_vectorstore("my-memory", InMemoryVectorStore.from_texts)
```

A factory receives `texts`, `embedding`, `metadatas` and backend options. It must
return a native store. Install that store's dependencies separately. For a
preconfigured store use `rag(store=store)`; no registration is needed.

## Retrievers and rerankers

Any native retriever can be passed to `RAG(retriever)` or `rag(retriever=retriever)`.
A compatible Runnable returning a list of `Document` objects also works. Implement
both `invoke` and `ainvoke`, or derive from `BaseRetriever`, whose base provides
async fallback. Keep metadata and identifiers intact for deduplication/citations.

`RerankRetriever` accepts native document compressors. You can use embedding
filters, a local cross-encoder adapter, a remote reranker, or a compressor pipeline.
Truffles delegates the scoring/compression policy and does not require a specific
reranking service.

## Tools, MCP and other ecosystems

Use native LangChain tools directly. If an MCP adapter returns LangChain tools,
pass the loaded tools to `agent(..., tools=tools)` while keeping the adapter's
connection/session lifecycle alive. Install and configure the chosen MCP adapter
separately; Truffles does not launch arbitrary servers automatically.

Similarly, existing SQL agents, toolkits, retrievers, legacy chains and compiled
LangGraphs can be used through their native Runnable interface. `wrap` accepts
Runnables and callables; it does not assume every arbitrary Python object implements
a chain. For a legacy API, adapt its invocation with `RunnableLambda` explicitly.

## Middleware and graphs

All additional constructor keyword arguments on `Agent` go to `create_agent`.
Use its state/context schemas and middleware instead of extending private wrapper
methods. For architecture changes, compose native graphs or Runnables. Keep agent
`.native`, RAG `.retriever`/`.store`, and returned model/store objects as the stable
extension boundaries.

A good plugin test checks the actual package import/class, native constructor
arguments, request serialization (if remote), result parsing and meaningful
failure behavior. A fake returning a hard-coded answer proves only the wrapper
interface, not external compatibility.
