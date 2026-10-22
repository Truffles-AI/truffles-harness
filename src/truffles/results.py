"""Convenient outputs that retain the complete native state."""

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


def text_content(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block if isinstance(block, str) else block.get("text", "")
        for block in content
        if isinstance(block, str) or block.get("type") in {"text", "output_text"}
    )


@dataclass(frozen=True)
class Result:
    """Agent result. Usage covers returned state (the full thread with memory)."""

    text: str
    messages: tuple[BaseMessage, ...] = ()
    structured: Any = None
    sources: tuple[Document, ...] = ()
    usage: dict[str, int] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict, repr=False)

    def __str__(self) -> str:
        return self.text

    @property
    def interrupted(self) -> bool:
        return bool(self.state.get("__interrupt__"))

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "Result":
        messages = tuple(state.get("messages", []))
        answers = [m for m in messages if isinstance(m, AIMessage)]
        usage: dict[str, int] = {}
        sources: list[Document] = []
        for message in messages:
            if isinstance(message, AIMessage) and message.usage_metadata:
                for key in ("input_tokens", "output_tokens", "total_tokens"):
                    usage[key] = usage.get(key, 0) + message.usage_metadata.get(key, 0)
            if isinstance(message, ToolMessage) and isinstance(message.artifact, list):
                sources.extend(d for d in message.artifact if isinstance(d, Document))
        return cls(
            text=text_content(answers[-1]) if answers else "",
            messages=messages,
            structured=state.get("structured_response"),
            sources=tuple(sources),
            usage=usage,
            state=state,
        )
