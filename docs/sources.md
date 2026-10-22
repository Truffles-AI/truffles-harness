# Official contract references

Reviewed on 2026-10-19. These sources informed integration decisions and stub
fixtures. Executable serialization checks use the versions recorded in `uv.lock`.
A documentation example is not a substitute for a live service compatibility test.

| Topic | Primary reference | Contract checked |
| --- | --- | --- |
| LangChain agents | [Agents](https://docs.langchain.com/oss/python/langchain/agents) | Native `create_agent`, tools, structured state and middleware |
| LangGraph | [Graph API source](https://github.com/langchain-ai/docs/blob/main/src/oss/langgraph/graph-api.mdx) | StateGraph nodes, edges and compilation |
| OpenAI tools | [Function calling](https://developers.openai.com/api/docs/guides/function-calling) | Tool schemas, arguments and matching tool-call IDs |
| OpenAI chat | [Create chat completion](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) | Chat request/response shape |
| Anthropic tools | [Handle tool calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls) | `tool_use`, `input_schema`, `tool_result` and `is_error` |
| Gemini | [Function calling](https://ai.google.dev/gemini-api/docs/function-calling) | `functionCall` and JSON `functionResponse` |
| Gemini integration | [ChatGoogleGenerativeAI](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai) | Native integration setup and capabilities |
| Groq | [API reference](https://console.groq.com/docs/api-reference) | OpenAI-shaped chat endpoint under `/openai/v1` |
| xAI search | [Web Search](https://docs.x.ai/developers/tools/web-search) | Responses built-in `web_search` and filter shape |
| xAI integration | [ChatXAI](https://docs.langchain.com/oss/python/integrations/chat/xai) | Deprecation of old search configuration |
| Perplexity integration | [ChatPerplexity](https://docs.langchain.com/oss/python/integrations/chat/perplexity) | Sonar vs Responses, metadata and tool limitations |
| Perplexity migration | [Sonar to Agent API](https://docs.perplexity.ai/docs/agent-api/migrate-from-sonar/overview) | Distinct messages/choices and input/output contracts |
| Perplexity Agent route | [Create Agent Response](https://docs.perplexity.ai/api-reference/agent-post) | Explicit `/v1/agent` route used by search tools |
| Qdrant | [Similarity search reference](https://reference.langchain.com/python/langchain-qdrant/qdrant/QdrantVectorStore/similarity_search_with_score) | Native search/filter semantics |
| Embeddings | [Embedding integrations](https://docs.langchain.com/oss/python/integrations/embeddings) | Native document/query embedding interface |

Source examples and SDKs can evolve at different times. The Perplexity search
wrapper deliberately uses the SDK's public transport with the currently documented
`/v1/agent` and `/v1/sonar` routes; tests verify those exact paths. Generic native
model objects keep their own SDK routing. See [compatibility](compatibility.md).
