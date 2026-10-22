import pytest


@pytest.fixture(autouse=True)
def bridge_httpx2_transport(respx_mock, monkeypatch):
    """Route newer SDKs' httpx2 transports through the same strict HTTP fixtures.

    respx intercepts httpx, while recent SDK releases can use its httpx2 fork.
    Only the transport is replaced; SDK request/response processing stays real.
    """
    import importlib.util

    if importlib.util.find_spec("httpx2") is None:
        return
    import httpx
    import httpx2

    def handle_request(self, request):
        converted = httpx.Request(
            request.method, str(request.url), headers=request.headers.raw, content=request.read()
        )
        response = respx_mock.handler(converted)
        return httpx2.Response(
            response.status_code, headers=response.headers.raw, content=response.read()
        )

    async def handle_async_request(self, request):
        converted = httpx.Request(
            request.method,
            str(request.url),
            headers=request.headers.raw,
            content=await request.aread(),
        )
        response = await respx_mock.async_handler(converted)
        return httpx2.Response(
            response.status_code, headers=response.headers.raw, content=await response.aread()
        )

    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", handle_request)
    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", handle_async_request)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "XAI_API_KEY",
        "PPLX_API_KEY",
        "PERPLEXITY_API_KEY",
    ):
        monkeypatch.setenv(key, "test-key-not-a-credential")
    for key in (
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "ANTHROPIC_BASE_URL",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_CLOUD_PROJECT",
        "PERPLEXITY_BASE_URL",
        "GROQ_BASE_URL",
        "XAI_API_BASE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LANGCHAIN_OPENAI_TCP_KEEPALIVE", "0")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
