# Compatibility and support boundary

Truffles 0.1.0 is a local package with native extension points. “Supported” means
the wrapper exposes the integration; the verification level is stated separately.
It does not mean every provider/model/backend combination offers every feature.

## Verified offline

| Area | Verification |
| --- | --- |
| Agent runtime | Real LangChain/LangGraph execution with scripted model outputs |
| Sync/async tools, memory, schemas, interruptions | Behavior tests through the actual runtime |
| Linear, parallel, router and supervisor | End-to-end local execution |
| OpenAI, Groq, xAI chat | Actual SDK HTTP tool-call round trips, sync and async |
| Anthropic | Actual SDK Messages tool-use/tool-result round trips, sync and async |
| Gemini | Actual SDK generateContent tool-call round trips, sync and async |
| xAI search | Actual SDK Responses request with `web_search`, sync and async |
| Perplexity Sonar | Actual SDK request/response with source metadata, sync and async |
| Perplexity Responses | SDK transport to documented Agent route with built-in search, sync and async |
| OpenAI/Gemini embeddings | Actual SDK embedding payloads and vector response parsing |
| OpenAI streaming | SSE parsing through the actual SDK and chat adapter |
| In-memory/Qdrant/Chroma/FAISS retrieval | Real local indexing/search, including multiple algorithms |
| BM25, RRF, query expansion, reranking | Ranking/composition and sync/async tests |

## Configured integration factories

Pinecone, Weaviate, Milvus, PostgreSQL/PGVector, Redis and Elasticsearch
have lazy named factories with optional dependency declarations. Their native
`from_texts`/`as_retriever` interfaces are used directly. Remote deployments were
not created or queried. Native objects provide the extension route for additional
LangChain stores. MMR, filters, thresholds, async execution and persistence remain
backend-specific; unsupported operations do not silently fall back.

Embedding factories are provided for OpenAI, Google/Gemini, Cohere, Hugging Face
and FastEmbed. OpenAI and Gemini document-embedding payloads are contract-tested; other providers
remain native factories.
Local neural embedding weights were not downloaded during validation.

## API-version drift

Provider SDKs are the live transport boundary. The project pins a reproducible
resolved environment in `uv.lock`, while package metadata allows a bounded range
for consumers. The installed Perplexity generated Responses resource uses an older
URL, so `search_tool` uses the SDK's public transport with the documented
`/v1/agent` and `/v1/sonar` paths explicitly. Tests assert those exact URLs and
payloads. The generic Perplexity chat factory returns the native LangChain object
unchanged and follows its native endpoint configuration. Provider schema or SDK
changes still require contract-test updates and deployment validation.

No live calls were made for any provider. API-key validity, endpoint availability,
model retirement, quotas, current web content and semantic output quality remain
outside offline verification. These limitations are material and are not hidden
behind coverage percentages.

## Native ecosystem support

LCEL Runnables and compiled LangGraphs pass through without flattening. Arbitrary
state schemas, conditional edges, reducers, cycles, subgraphs and checkpoints are
configured through native APIs. A non-Runnable historical chain may need a
`RunnableLambda` adapter. “All architectures” is supported through composability,
not through an exhaustive catalog of named agent classes.

The package is synchronous/asynchronous Python, not a distributed worker system.
A model supporting text completion alone cannot be made tool-capable by a wrapper.
Structured-output modes, multimodal inputs and hosted tools require matching
provider/model capabilities. Errors retain their native types.

## Packaging

Python 3.11+ is declared. The full suite passed on Python 3.11 and 3.12 on macOS; a clean base-wheel smoke
test passed on Python 3.14. CI specifies Linux 3.11/3.12/3.13. Extras install only their chosen
integrations. No package registry publication or name reservation has occurred.

The optional FAISS and FastEmbed factories currently use `langchain-community`,
which emits an upstream sunset/deprecation warning. FAISS local operations pass
the tests; the warning is preserved rather than hidden. A future maintained
adapter can be supplied as a native object or registered factory.
