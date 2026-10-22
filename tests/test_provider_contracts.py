"""Actual adapters + actual SDK serialization, with HTTP responses stubbed by respx.

These fixtures model provider wire schemas. They do not mock the wrapper's model
interface. The default pytest socket ban catches any un-intercepted network access.
"""

import json
import re

import httpx
import pytest
from langchain_core.messages import AIMessage

from truffles import agent, chat_model, embeddings, search_tool, tool

pytestmark = pytest.mark.contract


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def completion(model, *, tool_call=False):
    message = {"role": "assistant", "content": None if tool_call else "5"}
    if tool_call:
        message["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "add",
                    "arguments": '{"a":2,"b":3}',
                },
            }
        ]
    return {
        "id": "chatcmpl_test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {"index": 0, "message": message, "finish_reason": "tool_calls" if tool_call else "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.mark.parametrize(
    "provider,model,url",
    [
        ("openai", "gpt-4.1-mini", "https://api.openai.com/v1/chat/completions"),
        ("groq", "llama-3.3-70b-versatile", "https://api.groq.com/openai/v1/chat/completions"),
        ("xai", "grok-4", "https://api.x.ai/v1/chat/completions"),
    ],
)
@pytest.mark.parametrize("async_mode", [False, True])
async def test_openai_compatible_tool_roundtrip(respx_mock, provider, model, url, async_mode):
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.headers["authorization"] == "Bearer test-key-not-a-credential"
        assert body["model"] == model
        schema = body["tools"][0]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "add"
        assert schema["function"]["parameters"]["properties"]["a"]["type"] == "integer"
        assert set(schema["function"]["parameters"]["required"]) == {"a", "b"}
        return httpx.Response(200, json=completion(model, tool_call=len(requests) == 1))

    route = respx_mock.post(url).mock(side_effect=respond)
    options = {"max_retries": 0}
    if provider != "groq":
        options["use_responses_api"] = False
    bot = agent(f"{provider}:{model}", tools=[add], model_options=options)
    result = await bot.arun("2 + 3?") if async_mode else bot.run("2 + 3?")
    assert result.text == "5" and route.call_count == 2
    assert requests[1]["messages"][-1] == {
        "role": "tool",
        "content": "5",
        "tool_call_id": "call_1",
    }
    assert result.usage["total_tokens"] == 30


@pytest.mark.parametrize("async_mode", [False, True])
async def test_anthropic_tool_roundtrip(respx_mock, async_mode):
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.headers["x-api-key"] == "test-key-not-a-credential"
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert body["system"] == "Calculate carefully."
        assert body["max_tokens"] == 128
        assert body["tools"][0]["input_schema"]["properties"]["a"]["type"] == "integer"
        tool_call = len(requests) == 1
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"a": 2, "b": 3}}
                ]
                if tool_call
                else [{"type": "text", "text": "5"}],
                "stop_reason": "tool_use" if tool_call else "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    respx_mock.post("https://api.anthropic.com/v1/messages").mock(side_effect=respond)
    bot = agent(
        "anthropic:claude-sonnet-4-6",
        tools=[add],
        instructions="Calculate carefully.",
        model_options={"max_tokens": 128, "max_retries": 0},
    )
    result = await bot.arun("2 + 3?") if async_mode else bot.run("2 + 3?")
    assert result.text == "5"
    returned = requests[1]["messages"][-1]
    assert returned["role"] == "user"
    assert returned["content"][0] == {
        "type": "tool_result",
        "content": "5",
        "tool_use_id": "toolu_1",
        "is_error": False,
    }


@pytest.mark.parametrize("async_mode", [False, True])
async def test_gemini_tool_roundtrip(respx_mock, async_mode):
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.headers["x-goog-api-key"] == "test-key-not-a-credential"
        assert body["systemInstruction"]["parts"][0]["text"] == "Calculate carefully."
        declaration = body["tools"][0]["functionDeclarations"][0]
        assert declaration["name"] == "add"
        schema = declaration.get("parametersJsonSchema", declaration.get("parameters"))
        assert schema["properties"]["a"]["type"].lower() == "integer"
        part = (
            {"functionCall": {"name": "add", "args": {"a": 2, "b": 3}}}
            if len(requests) == 1
            else {"text": "5"}
        )
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [part]},
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15,
                },
                "modelVersion": "gemini-2.5-flash",
            },
        )

    respx_mock.post(
        re.compile(r"https://generativelanguage.googleapis.com/.*:generateContent")
    ).mock(side_effect=respond)
    bot = agent(
        "gemini:gemini-2.5-flash",
        tools=[add],
        instructions="Calculate carefully.",
        model_options={
            "max_retries": 0,
            "client_args": {"transport": httpx.MockTransport(respx_mock.handler)},
        },
    )
    result = await bot.arun("2 + 3?") if async_mode else bot.run("2 + 3?")
    assert result.text == "5"
    returned = requests[1]["contents"][-1]["parts"][0]["functionResponse"]
    assert returned["name"] == "add"
    assert returned["response"] == {"output": 5}


def responses_body(model):
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 1700000000,
        "model": model,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "parallel_tool_calls": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "tool_choice": "auto",
        "tools": [{"type": "web_search"}],
        "output": [
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Grounded answer.",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://example.org",
                                "title": "Source",
                                "start_index": 0,
                                "end_index": 8,
                            }
                        ],
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


@pytest.mark.parametrize("async_mode", [False, True])
async def test_xai_search_uses_responses_and_native_web_search(respx_mock, async_mode):
    def respond(request):
        body = json.loads(request.content)
        assert body["model"] == "grok-4.7"
        assert body["tools"] == [
            {"type": "web_search", "filters": {"allowed_domains": ["example.org"]}}
        ]
        assert body["input"][-1]["role"] == "user"
        assert "search_parameters" not in body
        return httpx.Response(200, json=responses_body("grok-4.7"))

    route = respx_mock.post("https://api.x.ai/v1/responses").mock(side_effect=respond)
    search = search_tool(
        "grok",
        model="grok-4.7",
        model_options={"max_retries": 0},
        search_options={"filters": {"allowed_domains": ["example.org"]}},
    )
    output = (
        await search.ainvoke({"query": "news"}) if async_mode else search.invoke({"query": "news"})
    )
    assert json.loads(output)["answer"] == "Grounded answer."
    assert "https://example.org" in output and route.call_count == 1


@pytest.mark.parametrize("async_mode", [False, True])
async def test_perplexity_sonar_preserves_sources(respx_mock, async_mode):
    def respond(request):
        assert request.url.path == "/v1/sonar"
        body = json.loads(request.content)
        assert body["model"] == "sonar"
        assert body["messages"] == [{"role": "user", "content": "news"}]
        assert body["search_recency_filter"] == "week"
        data = completion("sonar")
        data["citations"] = ["https://example.org"]
        data["search_results"] = [
            {"title": "Source", "url": "https://example.org", "date": "2026-10-01"}
        ]
        return httpx.Response(200, json=data)

    respx_mock.post(re.compile(r"https://api.perplexity.ai/.*")).mock(side_effect=respond)
    search = search_tool(
        model_options={"max_retries": 0}, search_options={"search_recency_filter": "week"}
    )
    output = (
        await search.ainvoke({"query": "news"}) if async_mode else search.invoke({"query": "news"})
    )
    assert json.loads(output)["sources"]["citations"] == ["https://example.org"]


@pytest.mark.parametrize("async_mode", [False, True])
async def test_perplexity_agent_api_search(respx_mock, async_mode):
    def respond(request):
        assert request.url.path == "/v1/agent"
        body = json.loads(request.content)
        assert body["model"] == "openai/gpt-5.5"
        assert body["tools"] == [{"type": "web_search"}]
        assert body["input"] == "news" and "messages" not in body
        return httpx.Response(200, json=responses_body("openai/gpt-5.5"))

    respx_mock.post(re.compile(r"https://api.perplexity.ai/.*")).mock(side_effect=respond)
    search = search_tool("perplexity", model="openai/gpt-5.5", model_options={"max_retries": 0})
    output = (
        await search.ainvoke({"query": "news"}) if async_mode else search.invoke({"query": "news"})
    )
    assert json.loads(output)["answer"] == "Grounded answer."


def test_perplexity_preset_token_cap_and_annotations(respx_mock):
    def respond(request):
        assert request.url.path == "/v1/agent"
        body = json.loads(request.content)
        assert body == {
            "preset": "fast",
            "max_output_tokens": 64,
            "input": "news",
            "tools": [{"type": "web_search"}],
        }
        assert request.headers["authorization"] == "Bearer test-key-not-a-credential"
        return httpx.Response(200, json=responses_body("provider/model"))

    respx_mock.post("https://api.perplexity.ai/v1/agent").mock(side_effect=respond)
    search = search_tool(model_options={"preset": "fast", "max_tokens": 64, "max_retries": 0})
    result = json.loads(search.invoke({"query": "news"}))
    assert result["content"][0]["content"][0]["annotations"][0]["url"] == "https://example.org"
    assert result["metadata"]["usage"]["total_tokens"] == 15


@pytest.mark.parametrize(
    "options",
    [
        {"provider": "xai", "model": "grok-4.7", "search_options": {"type": "other"}},
        {"model": "provider/model", "search_options": {"type": "other"}},
        {"model": "provider/model", "model_options": {"max_tokens": 1, "max_output_tokens": 2}},
    ],
)
def test_search_rejects_ambiguous_native_options(options):
    with pytest.raises(ValueError):
        search_tool(**options)


def test_native_perplexity_chat_passthrough(respx_mock):
    def respond(request):
        # Generic models retain the native SDK route; the search helper uses current explicit paths.
        assert request.url.path == "/chat/completions"
        assert json.loads(request.content)["model"] == "sonar"
        return httpx.Response(200, json=completion("sonar"))

    respx_mock.post("https://api.perplexity.ai/chat/completions").mock(side_effect=respond)
    assert chat_model("perplexity:sonar", max_retries=0).invoke("hello").content == "5"


def test_gemini_embeddings_contract(respx_mock):
    def respond(request):
        body = json.loads(request.content)
        assert request.url.path.endswith(":batchEmbedContents")
        assert [r["content"]["parts"][0]["text"] for r in body["requests"]] == ["one", "two"]
        assert all(r["taskType"] == "RETRIEVAL_DOCUMENT" for r in body["requests"])
        return httpx.Response(
            200, json={"embeddings": [{"values": [1.0, 0.0, 0.0]}, {"values": [0.0, 1.0, 0.0]}]}
        )

    respx_mock.post(re.compile(r"https://generativelanguage.googleapis.com/.*")).mock(
        side_effect=respond
    )
    embedder = embeddings("gemini:gemini-embedding-001")
    assert embedder.embed_documents(["one", "two"]) == [[1, 0, 0], [0, 1, 0]]


def test_provider_retry_is_owned_by_sdk(respx_mock):
    route = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                429,
                json={"error": {"message": "retry", "type": "rate_limit_error"}},
                headers={"retry-after-ms": "1"},
            ),
            httpx.Response(200, json=completion("gpt-4.1-mini")),
        ]
    )
    result = agent("openai:gpt-4.1-mini", model_options={"max_retries": 1}).run("hello")
    assert result.text == "5" and route.call_count == 2


def test_openai_embeddings_wire_format(respx_mock):
    def respond(request):
        body = json.loads(request.content)
        assert body["model"] == "text-embedding-3-small"
        assert body["input"] == ["one", "two"]
        assert body["dimensions"] == 3
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": body["model"],
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [1.0, 0.0, 0.0]},
                    {"object": "embedding", "index": 1, "embedding": [0.0, 1.0, 0.0]},
                ],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    respx_mock.post("https://api.openai.com/v1/embeddings").mock(side_effect=respond)
    embedder = embeddings(
        "openai:text-embedding-3-small",
        dimensions=3,
        max_retries=0,
        check_embedding_ctx_length=False,
    )
    assert embedder.embed_documents(["one", "two"]) == [[1, 0, 0], [0, 1, 0]]


def test_provider_error_is_not_swallowed_or_double_retried(respx_mock):
    import openai

    route = respx_mock.post("https://api.openai.com/v1/chat/completions").respond(
        429,
        json={
            "error": {"message": "rate limited", "type": "rate_limit_error", "code": "rate_limit"}
        },
    )
    with pytest.raises(openai.RateLimitError):
        agent("openai:gpt-4.1-mini", model_options={"max_retries": 0}).run("hello")
    assert route.call_count == 1


def test_openai_sse_streaming_contract(respx_mock):
    events = [
        {
            "id": "chatcmpl_test",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-4.1-mini",
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
        for text in ["hello", " world"]
    ]
    events.append(
        {
            "id": "chatcmpl_test",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-4.1-mini",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    payload = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    respx_mock.post("https://api.openai.com/v1/chat/completions").respond(
        200, content=payload, headers={"content-type": "text/event-stream"}
    )
    chunks = list(chat_model("openai:gpt-4.1-mini", max_retries=0).stream("hi"))
    assert "".join(c.content for c in chunks) == "hello world"


@pytest.mark.parametrize("options", [{"provider": "unknown"}, {"provider": "xai"}])
def test_search_bad_options(options):
    with pytest.raises(ValueError):
        search_tool(**options)


def test_native_model_preserves_multimodal_input(respx_mock):
    def respond(request):
        body = json.loads(request.content)
        assert body["messages"][0]["content"][1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,dGVzdA=="},
        }
        return httpx.Response(200, json=completion("gpt-4.1-mini"))

    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(side_effect=respond)
    output = chat_model("openai:gpt-4.1-mini").invoke(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,dGVzdA=="}},
                ],
            }
        ]
    )
    assert isinstance(output, AIMessage)


def test_small_self_hosted_model_plain_chat_contract(respx_mock):
    response = completion("bank-small-model")
    response["choices"][0]["message"]["content"] = '{"intent":"summary"}'
    route = respx_mock.post("https://models.bank.internal/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=response)
    )
    bot = agent(
        "openai:bank-small-model",
        kind="chat",
        max_model_calls=1,
        model_options={
            "base_url": "https://models.bank.internal/v1",
            "api_key": "bank-test-key",
            "max_tokens": 180,
            "max_retries": 0,
            "temperature": 0,
        },
    )
    assert bot.run("summary").text == '{"intent":"summary"}'
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "bank-small-model"
    assert "tools" not in body and "response_format" not in body
    assert body.get("max_completion_tokens", body.get("max_tokens")) == 180
    assert route.calls.last.request.headers["authorization"] == "Bearer bank-test-key"
