"""Explicit offline doubles for examples/tests; never selected by production factories."""

import hashlib
import math
import re
from threading import Lock
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


class ScriptedChatModel(BaseChatModel):
    """Return supplied messages in order; fail loudly when the script runs out."""

    responses: list[AIMessage]
    _index: int = PrivateAttr(default=0)
    _lock: Any = PrivateAttr(default_factory=Lock)
    _requests: list = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "truffles-scripted-test-model"

    @property
    def requests(self) -> list:
        return list(self._requests)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self.bind(tools=tools, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        with self._lock:
            if self._index >= len(self.responses):
                raise RuntimeError("ScriptedChatModel exhausted; add a response for this call.")
            self._requests.append(list(messages))
            message = self.responses[self._index].model_copy(deep=True)
            self._index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


class HashEmbeddings(Embeddings):
    """Deterministic lexical hashing for tests. NOT a semantic embedding model."""

    def __init__(self, dimensions: int = 64):
        if dimensions < 1:
            raise ValueError("dimensions must be positive.")
        self.dimensions = dimensions

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"\w+", text.casefold()):
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]
