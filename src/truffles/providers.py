"""Lazy, extensible factories. Provider SDKs own wire formats and retries."""

from collections.abc import Callable
from importlib import import_module
from typing import Any

_CHAT = {
    "openai": ("langchain_openai", "ChatOpenAI", "openai"),
    "anthropic": ("langchain_anthropic", "ChatAnthropic", "anthropic"),
    "google_genai": ("langchain_google_genai", "ChatGoogleGenerativeAI", "gemini"),
    "groq": ("langchain_groq", "ChatGroq", "groq"),
    "xai": ("langchain_xai", "ChatXAI", "xai"),
    "perplexity": ("langchain_perplexity", "ChatPerplexity", "perplexity"),
}
_EMBEDDINGS = {
    "openai": ("langchain_openai", "OpenAIEmbeddings", "openai"),
    "google_genai": ("langchain_google_genai", "GoogleGenerativeAIEmbeddings", "gemini"),
    "cohere": ("langchain_cohere", "CohereEmbeddings", "cohere"),
    "huggingface": ("langchain_huggingface", "HuggingFaceEmbeddings", "huggingface"),
    "fastembed": ("langchain_community.embeddings", "FastEmbedEmbeddings", "fastembed"),
}
_ALIASES = {"gemini": "google_genai", "google": "google_genai", "grok": "xai"}
_custom_chat: dict[str, Callable[..., Any]] = {}
_custom_embeddings: dict[str, Callable[..., Any]] = {}


def _load(module: str, name: str, extra: str):
    try:
        return getattr(import_module(module), name)
    except ModuleNotFoundError as exc:
        # Do not hide a broken transitive dependency as a missing integration.
        if exc.name and (module == exc.name or module.startswith(exc.name + ".")):
            raise ImportError(
                f"Install this integration with: pip install 'truffles[{extra}]'"
            ) from exc
        raise


def _parse(spec: str) -> tuple[str, str]:
    provider, sep, model = spec.partition(":")
    if not sep or not provider.strip() or not model.strip():
        raise ValueError("Use 'provider:model', for example 'openai:gpt-4.1-mini'.")
    provider = provider.lower().strip()
    return _ALIASES.get(provider, provider), model


def _register(registry, builtins, name, factory, replace):
    name = name.lower().strip()
    if not name or ":" in name or not callable(factory):
        raise ValueError("Provide a nonempty provider name without ':' and a callable factory.")
    if not replace and (name in registry or name in builtins):
        raise ValueError(f"{name!r} is already registered; use replace=True explicitly.")
    registry[name] = factory


def register_provider(name: str, factory: Callable[..., Any], *, replace: bool = False) -> None:
    """Register a trusted factory accepting model= and provider-specific options."""
    name = _ALIASES.get(name.lower().strip(), name.lower().strip())
    _register(_custom_chat, _CHAT, name, factory, replace)


def register_embeddings(name: str, factory: Callable[..., Any], *, replace: bool = False) -> None:
    """Register a trusted embeddings factory accepting model= and options."""
    name = _ALIASES.get(name.lower().strip(), name.lower().strip())
    _register(_custom_embeddings, _EMBEDDINGS, name, factory, replace)


def chat_model(model: Any, **options: Any):
    """Return an existing chat model, or construct one from 'provider:model'.

    No network call is made by Truffles. All options reach the native constructor.
    """
    if not isinstance(model, str):
        if options:
            raise ValueError("Configure an existing model before passing it to Truffles.")
        return model
    provider, model_name = _parse(model)
    if provider in _custom_chat:
        return _custom_chat[provider](model=model_name, **options)
    if provider in _CHAT:
        return _load(*_CHAT[provider])(model=model_name, **options)
    # Keep the full LangChain provider ecosystem accessible without another registry.
    from langchain.chat_models import init_chat_model

    return init_chat_model(model_name, model_provider=provider, **options)


def embeddings(model: Any, **options: Any):
    """Construct embeddings, or pass through any LangChain Embeddings instance."""
    if not isinstance(model, str):
        if options:
            raise ValueError("Configure an existing embeddings object before passing it in.")
        return model
    provider, model_name = _parse(model)
    if provider in _custom_embeddings:
        return _custom_embeddings[provider](model=model_name, **options)
    if provider not in _EMBEDDINGS:
        raise ValueError(
            f"Unknown embeddings provider {provider!r}; register it or pass an instance."
        )
    key = "model_name" if provider in {"huggingface", "fastembed"} else "model"
    return _load(*_EMBEDDINGS[provider])(**{key: model_name}, **options)
