import ast
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from truffles_pfm.config import Settings
from truffles_pfm.intelligence import Intelligence

from truffles.testing import ScriptedChatModel


def model(*responses):
    return ScriptedChatModel(responses=[AIMessage(content=r) for r in responses])


def test_real_harness_invokes_json_intent_without_ledger():
    native = model('{"intent":"category","category":"Dining","month":"2026-10","language":"en"}')
    ai = Intelligence(Settings(), model=native)
    result, mode = ai.intent("My dining spend? email me alice@example.com", "2026-10")
    assert result.intent == "category" and mode == "truffles"
    sent = native.requests[0][-1].content
    assert "alice" not in sent and "ledger" not in json.loads(sent)
    assert len(native.requests) == 1


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        "{}",
        '{"intent":"transfer"}',
        '{"intent":"summary","sql":"SELECT *"}',
        '{"intent":"summary","month":"2026-13"}',
        '{"intent":"summary","category":"SECRET"}',
    ],
)
def test_invalid_model_output_uses_safe_fallback(response):
    ai = Intelligence(Settings(), model=model(response))
    result, mode = ai.intent("summarise spending", "2026-10")
    assert result.intent == "summary" and mode == "fallback"


def test_circuit_breaker_and_consent():
    native = model("bad", "bad", "bad")
    ai = Intelligence(Settings(), model=native)
    for _ in range(4):
        ai.intent("summary", "2026-10")
    assert len(native.requests) == 3
    assert ai.fallbacks == 4
    ai.intent("summary", "2026-10", consent=False)
    assert len(native.requests) == 3


def test_timeout_exhaustion_and_unsupported_fallback():
    ai = Intelligence(Settings(), model=model())
    result, mode = ai.intent("Transfer money to another customer", "2026-10")
    assert result.intent == "unsupported" and mode == "fallback"
    result, _ = ai.intent("spend last month", "2026-01", consent=False)
    assert result.month == "2025-12"


def test_app_never_imports_provider_or_orchestration_sdk_directly():
    root = Path(__file__).parents[1] / "src"
    forbidden = {
        "openai",
        "anthropic",
        "langchain",
        "langchain_core",
        "langgraph",
        "langchain_openai",
    }
    for file in root.rglob("*.py"):
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden, file
            if isinstance(node, ast.Import):
                assert not any(alias.name.split(".")[0] in forbidden for alias in node.names), file


def test_small_model_string_null_regression():
    ai = Intelligence(
        Settings(),
        model=model('{"intent":"savings","category":"null","month":"null","language":"en"}'),
    )
    result, mode = ai.intent("cash surplus", "2026-10")
    assert mode == "truffles" and result.category is None and result.month is None
