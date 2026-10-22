# Models and embeddings

## Chat providers

| Prefix | Aliases | Local install extra | Credential variable | Native class |
| --- | --- | --- | --- | --- |
| `openai` | — | `openai` | `OPENAI_API_KEY` | `ChatOpenAI` |
| `anthropic` | — | `anthropic` | `ANTHROPIC_API_KEY` | `ChatAnthropic` |
| `google_genai` | `gemini`, `google` | `gemini` | `GOOGLE_API_KEY` | `ChatGoogleGenerativeAI` |
| `groq` | — | `groq` | `GROQ_API_KEY` | `ChatGroq` |
| `xai` | `grok` | `xai` | `XAI_API_KEY` | `ChatXAI` |
| `perplexity` | — | `perplexity` | `PPLX_API_KEY` | `ChatPerplexity` |

The underlying Perplexity SDK also uses `PERPLEXITY_API_KEY`; `PPLX_API_KEY` is the
LangChain adapter's conventional setting. Explicit native API-key parameters can
be passed in `model_options`. Gemini's native SDK also supports `GEMINI_API_KEY`.

Groq provides inference; Grok is xAI's model family. They use different prefixes,
credentials and packages. Provider names are case-normalized. The model string
after the first colon is preserved, including slashes and further colons.

```python
from truffles import agent, chat_model
model = chat_model("groq:llama-3.3-70b-versatile", temperature=0, max_retries=1)
bot = agent(model)
```

Instances are passed through unchanged. Configure an existing model before
passing it in; combining an instance with `model_options` is rejected to prevent
ignored settings. To use another native provider:

```python
from truffles import chat_model
model = chat_model("ollama:llama3.2")
```

Install its native integration package yourself. Unknown provider names are
forwarded to `init_chat_model`; unsupported names fail there. Alternatively
[register a factory](extensions.md).

## Parameters and endpoints

Truffles does not translate temperature, token limits, retry settings or timeouts
into a fabricated universal schema. Use the integration's native parameter names.
For example, Anthropic uses `max_tokens`, while Gemini exposes
`max_output_tokens`. Passing unsupported settings can raise native validation or
API errors. Constructor defaults are inherited from the installed integration.

OpenAI-compatible servers can use a configured `ChatOpenAI` or
`chat_model("openai:your-model", base_url="https://your-server/v1", api_key="...")`.
This does not certify that a server implements tools, Responses, multimodal content
or structured output correctly. Those are endpoint-specific capabilities.

Provider SDKs own authentication, connection pooling, payload serialization,
response decoding and transport retries. Keep an agent/model instance for repeated
calls. There is no Truffles global model cache, which could mix credentials or
configuration between users.

## Embeddings

```python
from truffles import embeddings
embedder = embeddings("openai:text-embedding-3-small", dimensions=512)
vectors = embedder.embed_documents(["first document", "second document"])
query = embedder.embed_query("search question")
```

`aembed_documents` and `aembed_query` are available through the native object.
Support and concurrency behavior come from that integration.

| Prefix | Native class | Extra |
| --- | --- | --- |
| `openai` | `OpenAIEmbeddings` | `openai` |
| `gemini` / `google_genai` | `GoogleGenerativeAIEmbeddings` | `gemini` |
| `cohere` | `CohereEmbeddings` | `cohere` |
| `huggingface` | `HuggingFaceEmbeddings` | `huggingface` |
| `fastembed` | `FastEmbedEmbeddings` | `fastembed` |

Hugging Face and FastEmbed receive the model identifier as `model_name`; the others
receive `model`. These factories do not imply every provider has an embeddings
endpoint. Anthropic, Groq and xAI chat support does not create an embedding model
for those providers.

Embedding dimensions, normalization and model choice must remain consistent
between ingestion and queries. Switching models generally requires reindexing.
Local model runtimes may download model weights on first use. No local weights
are downloaded by Truffles itself.

For fully offline tests, use `truffles.testing.HashEmbeddings`. It hashes lexical
tokens into vectors; it is intentionally **not** a semantic embedding model and
must not be used to infer real retrieval quality.
