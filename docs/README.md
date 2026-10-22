# Truffles documentation

Truffles is an installable Python package with a concise API over LangChain and
LangGraph. Start with the offline example before configuring any provider.

| Document | What it covers |
| --- | --- |
| [Project guide](PROJECT_GUIDE.md) | Design, scope, architecture, tradeoffs and project structure |
| [Quickstart](quickstart.md) | Installation and first agent, tool, team and RAG pipeline |
| [Agents](agents.md) | ReAct, chat, structured output, tool schemas, bounds and results |
| [Providers](providers.md) | OpenAI, Anthropic, Gemini, Groq, xAI, Perplexity and embeddings |
| [Workflows](workflows.md) | Linear, parallel, routed, hierarchical and custom graph architectures |
| [RAG](rag.md) | Ingestion, chunking, context, citations and retrieval algorithms |
| [Vector stores](vector-stores.md) | Backend plugins, configuration, filters and persistence |
| [Search](search.md) | Grok and Perplexity grounding and source metadata |
| [Runtime](runtime.md) | Async, streaming, batching, memory, checkpoints, interruptions and callbacks |
| [Extensions](extensions.md) | Custom models, embeddings, stores, tools, middleware and retrievers |
| [API reference](api.md) | Public functions, constructors, methods and result fields |
| [Performance](performance.md) | What is optimized, benchmarking, budgets and concurrency |
| [Testing](testing.md) | Offline doubles, HTTP contracts, CI and reproducible commands |
| [Compatibility](compatibility.md) | Tested capabilities, optional integrations and limitations |
| [Sources](sources.md) | Official API references used for contract checks |
| [Validation report](validation.md) | Recorded build, tests and example results |

Examples live in [examples](../examples/README.md). All code uses the local
checkout; nothing has been uploaded to a package registry.

- [Speech adapter](speech.md)
- [IB3 banking PFM reference](../apps/pfm/docs/README.md)
