"""Provider-hosted web search, reusable as a tool in any agent."""

import json
import os
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool

from .providers import _load, chat_model
from .results import text_content


def search_tool(
    provider: str = "perplexity",
    *,
    model: str | None = None,
    model_options: dict[str, Any] | None = None,
    search_options: dict[str, Any] | None = None,
    name: str = "web_search",
) -> StructuredTool:
    """Search with Perplexity Sonar or xAI/Perplexity Responses built-in tools.

    Perplexity defaults to Sonar's native grounded chat. Supply a provider/model
    name (e.g. openai/gpt-5.5) for Perplexity Agent API. xAI needs an explicit
    model to avoid silently selecting a retired model. Options remain native.
    """
    provider = "xai" if provider == "grok" else provider
    if provider not in {"xai", "perplexity"}:
        raise ValueError("search_tool supports 'xai' (Grok) and 'perplexity'.")
    if provider == "xai" and not model:
        raise ValueError("Choose an xAI model explicitly, e.g. model='grok-4.7'.")
    model = model or "sonar"
    options = dict(model_options or {})
    if provider == "perplexity":
        return _perplexity_search(model, options, dict(search_options or {}), name)
    if search_options and search_options.get("type", "web_search") != "web_search":
        raise ValueError("search_options cannot change the web_search tool type.")
    options["use_responses_api"] = True
    llm = chat_model(f"{provider}:{model}", **options)
    invocation = {"tools": [{"type": "web_search", **(search_options or {})}]}

    def serialize(message):
        # Preserve annotations and provider-specific source metadata without inventing citations.
        return json.dumps(
            {
                "answer": text_content(message),
                "content": message.content,
                "sources": message.additional_kwargs,
                "metadata": message.response_metadata,
            },
            default=str,
            ensure_ascii=False,
        )

    def search(query: str, config: RunnableConfig) -> str:
        return serialize(llm.invoke(query, config=config, **invocation))

    async def asearch(query: str, config: RunnableConfig) -> str:
        return serialize(await llm.ainvoke(query, config=config, **invocation))

    return StructuredTool.from_function(
        search,
        coroutine=asearch,
        name=name,
        description="Search the web. Returns a current answer and source metadata.",
    )


def _perplexity_search(model, options, search_options, name):
    """Use SDK transport with current documented paths, avoiding outdated route aliases.

    The SDK still owns authentication, pooling, timeouts and retries. The payload
    is the provider's native schema; no private LangChain/SDK internals are used.
    """
    native_search = options.pop("use_responses_api", False) or "/" in model or "preset" in options
    sync_client = options.pop("client", None)
    async_client = options.pop("async_client", None)
    client_options = {
        key: options.pop(key)
        for key in (
            "api_key",
            "base_url",
            "max_retries",
            "timeout",
            "default_headers",
            "default_query",
        )
        if key in options
    }
    if "api_key" not in client_options and os.environ.get("PPLX_API_KEY"):
        client_options["api_key"] = os.environ["PPLX_API_KEY"]
    sync_client = sync_client or _load("perplexity", "Perplexity", "perplexity")(**client_options)
    async_client = async_client or _load("perplexity", "AsyncPerplexity", "perplexity")(
        **client_options
    )
    path = "/v1/agent" if native_search else "/v1/sonar"
    if native_search:
        if search_options.get("type", "web_search") != "web_search":
            raise ValueError("search_options cannot change the web_search tool type.")
        if "max_tokens" in options:
            if "max_output_tokens" in options:
                raise ValueError("Specify max_tokens or max_output_tokens, not both.")
            options["max_output_tokens"] = options.pop("max_tokens")

    def payload(query):
        params = dict(options)
        if "preset" not in params:
            params["model"] = model
        if native_search:
            params.update(input=query, tools=[{"type": "web_search", **search_options}])
        else:
            params.update(search_options)
            params["messages"] = [{"role": "user", "content": query}]
        return params

    def serialize(response):
        if native_search:
            content = response["output"]
            answer = "".join(
                block["text"]
                for item in content
                if item.get("type") == "message"
                for block in item.get("content", [])
                if block.get("type") == "output_text"
            )
        else:
            content = response["choices"][0]["message"]["content"]
            answer = content
        return json.dumps(
            {
                "answer": answer,
                "content": content,
                "sources": {
                    key: response[key]
                    for key in ("citations", "search_results", "images", "related_questions")
                    if key in response
                },
                "metadata": {
                    key: response[key]
                    for key in ("id", "model", "status", "usage")
                    if key in response
                },
            },
            ensure_ascii=False,
        )

    def search(query: str) -> str:
        return serialize(sync_client.post(path, body=payload(query), cast_to=dict[str, Any]))

    async def asearch(query: str) -> str:
        response = await async_client.post(path, body=payload(query), cast_to=dict[str, Any])
        return serialize(response)

    return StructuredTool.from_function(
        search,
        coroutine=asearch,
        name=name,
        description="Search the web. Returns a current answer and source metadata.",
    )
