# Web search

`search_tool` creates a native LangChain tool that can be attached to any
Truffles agent. The search provider can differ from the agent's reasoning model.
The same tool also supports direct `invoke` and `ainvoke` calls.

## Perplexity Sonar

```python
from truffles import agent, search_tool
search = search_tool("perplexity", search_options={"search_recency_filter": "week"})
researcher = agent("anthropic:claude-sonnet-4-6", tools=[search])
print(researcher.run("Summarize recent developments in vector databases.").text)
```

Install the `perplexity` extra plus the reasoning provider's extra. Set
`PPLX_API_KEY` for the LangChain adapter. The default search model is `sonar`.
Sonar performs provider-hosted grounding without a local search function schema.
`search_options` are invocation parameters on this path. Native domain/recency
filters can be passed through when supported by that endpoint.

## Grok / xAI

```python
from truffles import search_tool
search = search_tool("xai", model="grok-4.7", search_options={
    "filters": {"allowed_domains": ["docs.python.org"]},
})
print(search.invoke({"query": "What changed in asyncio?"}))
```

`grok` is an alias for `xai`. An explicit xAI model is required. The wrapper enables
Responses and supplies a built-in `web_search` tool. It does not use deprecated
`search_parameters`. On this path, `search_options` are fields of that built-in
tool, so they differ from Sonar request parameters. Use the native provider schema
for filters; do not combine allowed/excluded domains when the provider forbids it.

## Perplexity Responses

```python
from truffles import search_tool
search = search_tool("perplexity", model="openai/gpt-5.5")
```

A Perplexity model containing `/`, or explicit
`model_options={"use_responses_api": True}`, selects the current Agent endpoint
`/v1/agent` and supplies a `web_search` tool. Sonar uses `/v1/sonar`. These are the
current documented routes. The wrapper uses the Perplexity SDK's public `post`
transport because the installed generated `responses.create` still targets an
older route. Authentication, pooling, timeouts, HTTP errors and retries remain
owned by the SDK; Truffles handles the small native response-to-tool conversion.

For Perplexity search, `model_options` accepts SDK transport settings (`api_key`,
`base_url`, `timeout`, `max_retries`, `default_headers`, `default_query`) and native
request fields. Supplied `client` and `async_client` are reused. Other options go
into the request body. On the Agent route, `max_tokens` maps to
`max_output_tokens`; passing both is rejected. A `preset` selects the Agent route
and replaces explicit model routing. Use provider-supported request fields; no
sampling parameters are invented or silently dropped. `search_options` is the
built-in tool configuration for Agent requests and top-level parameters for Sonar.

The generic `chat_model("perplexity:...")` remains the native LangChain class and
therefore follows that adapter's own SDK routing. Use `search_tool` for the
explicit current search endpoints described here.

## Output and citations

Direct invocation returns a JSON string containing:

- `answer`: visible answer text;
- `content`: original provider content blocks, including annotations;
- `sources`: provider-specific additional metadata such as citations or search results;
- `metadata`: native response metadata.

Truffles does not invent citations or normalize heterogeneous provider metadata
into an unreliable common schema. Source URLs may be in `sources`, annotations in
`content`, or response metadata depending on the provider. An outer agent receives
this JSON as its tool observation and may summarize it; inspect raw tool messages
when retaining source provenance is important.

The search client is constructed once per tool. Reuse the tool for repeated runs.
Search calls are separate provider requests and can incur costs in addition to
reasoning-model calls. Tool-provider token usage is not automatically included in
the outer agent's `Result.usage`; native callbacks can track both.
