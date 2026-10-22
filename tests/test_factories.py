import subprocess
import sys

import pytest

import truffles
from truffles import (
    chat_model,
    embeddings,
    register_embeddings,
    register_provider,
    register_vectorstore,
    vectorstore,
)
from truffles.providers import _load
from truffles.testing import HashEmbeddings


def test_lightweight_import():
    code = (
        "import sys, truffles; assert 'langchain' not in sys.modules; "
        "assert 'openai' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    assert "agent" in dir(truffles)
    with pytest.raises(AttributeError):
        _ = truffles.nonexistent


def test_existing_objects_and_bad_options():
    obj = object()
    assert chat_model(obj) is embeddings(obj) is obj
    for factory in (chat_model, embeddings):
        with pytest.raises(ValueError):
            factory(obj, temperature=0)
        for spec in ["missing-colon", ":model", "openai:"]:
            with pytest.raises(ValueError):
                factory(spec)


def test_missing_dependency_is_actionable():
    with pytest.raises(ImportError, match=r"pip install 'truffles\[fake\]'"):
        _load("missing_truffles_integration", "Thing", "fake")


def test_custom_provider_and_embeddings_registration():
    register_provider("test-provider", lambda **kwargs: kwargs, replace=True)
    assert chat_model("test-provider:model:revision", temperature=0) == {
        "model": "model:revision",
        "temperature": 0,
    }
    register_embeddings("test-embeddings", lambda **kwargs: kwargs, replace=True)
    assert embeddings("test-embeddings:small") == {"model": "small"}
    with pytest.raises(ValueError, match="already registered"):
        register_provider("openai", lambda **kw: kw)
    with pytest.raises(ValueError):
        register_provider("invalid:name", lambda **kw: kw)
    with pytest.raises(ValueError):
        embeddings("missing-provider:model")


def test_custom_store_registration_and_validation():
    register_vectorstore("test-store", lambda **kwargs: kwargs, replace=True)
    embedding = HashEmbeddings()
    result = vectorstore("test-store", texts=["hello"], embedding=embedding, collection="x")
    assert result["embedding"] is embedding and result["collection"] == "x"
    with pytest.raises(ValueError):
        vectorstore("missing", texts=["hello"], embedding=embedding)
    with pytest.raises(ValueError):
        vectorstore("memory", texts=[], embedding=embedding)
    with pytest.raises(ValueError):
        vectorstore("memory", texts=["hello"], metadatas=[], embedding=embedding)


def test_hash_embeddings_determinism():
    with pytest.raises(ValueError):
        HashEmbeddings(0)
    assert HashEmbeddings().embed_query("Hello world") == HashEmbeddings().embed_query(
        "hello world"
    )


def test_submodule_import_does_not_shadow_rag_factory():
    code = (
        "import truffles.knowledge; from truffles import rag; "
        "assert callable(rag); assert rag(['hello']).retrieve('hello')"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_custom_store_names_are_not_provider_aliases():
    register_vectorstore("google", lambda **kwargs: kwargs, replace=True)
    assert vectorstore(" google ", texts=["x"], embedding=HashEmbeddings())["texts"] == ["x"]


def test_unknown_chat_provider_delegates_to_native_factory(monkeypatch):
    import langchain.chat_models

    monkeypatch.setattr(langchain.chat_models, "init_chat_model", lambda *a, **kw: (a, kw))
    assert chat_model("custom_native:local-model", temperature=0) == (
        ("local-model",),
        {"model_provider": "custom_native", "temperature": 0},
    )
