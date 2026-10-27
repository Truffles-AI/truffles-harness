"""Bounded, content-free telemetry using native LangChain callbacks."""

import json
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


@dataclass(frozen=True)
class Span:
    """One completed native execution; timestamps are Unix nanoseconds."""

    run_id: str
    parent_run_id: str | None
    trace_id: str
    kind: str
    name: str
    started_ns: int
    ended_ns: int
    duration_ms: float
    status: str
    error_type: str | None = None
    usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class JSONLTraceSink:
    """Append spans locally; serialized within this instance, not across processes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = Lock()

    def __call__(self, span: Span) -> None:
        line = json.dumps(span.to_dict(), ensure_ascii=False, allow_nan=False)
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def _usage(response: Any) -> dict[str, int]:
    """Count only usage reported by this model callback, never conversation history."""
    total: Counter = Counter()
    found = False
    for group in getattr(response, "generations", ()):
        # Alternative candidates belong to one request. Use its first generation's usage.
        if not group:
            continue
        usage = getattr(getattr(group[0], "message", None), "usage_metadata", None)
        if usage:
            found = True
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                value = usage.get(key)
                if type(value) is int and value >= 0:
                    total[key] += value
    if not found:
        output = getattr(response, "llm_output", None) or {}
        usage = output.get("token_usage") or output.get("usage") or {}
        for key, alias in (
            ("input_tokens", "prompt_tokens"),
            ("output_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            value = usage.get(key, usage.get(alias))
            if type(value) is int and value >= 0:
                total[key] += value
    return dict(total)


class Observer(BaseCallbackHandler):
    """Thread-safe native callback with bounded storage and explicit sink failures.

    No inputs, outputs, metadata, tags, credentials or exception messages are stored.
    Names are retained: use static operation names, never customer data in run_name.
    Sinks run synchronously outside the state lock; slow sinks add execution latency.
    """

    def __init__(
        self,
        *,
        max_spans: int = 1000,
        max_active: int = 1000,
        sink: Callable[[Span], None] | None = None,
    ):
        if max_spans < 1 or max_active < 1:
            raise ValueError("max_spans and max_active must be positive.")
        self._lock = Lock()
        self._spans: deque[Span] = deque(maxlen=max_spans)
        self._active: dict[str, dict] = {}
        self._max_active = max_active
        self._sink = sink
        self._counts: Counter = Counter()
        self._tokens: Counter = Counter()

    @property
    def spans(self) -> tuple[Span, ...]:
        with self._lock:
            # Do not expose mutable token dictionaries from the internal buffer.
            return tuple(Span(**span.to_dict()) for span in self._spans)

    def summary(self) -> dict:
        """Lifetime counts; nested durations are not added into request latency."""
        with self._lock:
            return {
                "completed": self._counts["completed"],
                "errors": self._counts["errors"],
                "model_calls": self._counts["model_calls"],
                "tool_calls": self._counts["tool_calls"],
                "retrievals": self._counts["retrievals"],
                "dropped_starts": self._counts["dropped_starts"],
                "evicted_spans": self._counts["evicted_spans"],
                "sink_errors": self._counts["sink_errors"],
                "active": len(self._active),
                "retained": len(self._spans),
                "usage": dict(self._tokens),
            }

    def _start(self, kind, serialized, *, run_id, parent_run_id=None, name=None, **kwargs):
        run_id = str(run_id)
        parent = str(parent_run_id) if parent_run_id is not None else None
        with self._lock:
            if run_id in self._active:
                return
            if len(self._active) >= self._max_active:
                self._counts["dropped_starts"] += 1
                return
            trace_id = self._active.get(parent, {}).get("trace_id", run_id)
            self._active[run_id] = {
                "run_id": run_id,
                "parent_run_id": parent,
                "trace_id": trace_id,
                "kind": kind,
                "name": str(name or (serialized or {}).get("name") or kind)[:160],
                "started_ns": time.time_ns(),
                "clock": time.perf_counter(),
            }

    def _finish(self, *, run_id, error=None, response=None, **kwargs):
        with self._lock:
            start = self._active.pop(str(run_id), None)
            if start is None:
                return
            duration = max(0.0, (time.perf_counter() - start.pop("clock")) * 1000)
            usage = _usage(response) if start["kind"] == "model" and response is not None else {}
            span = Span(
                **start,
                ended_ns=time.time_ns(),
                duration_ms=duration,
                status="error" if error is not None else "ok",
                error_type=type(error).__name__ if error is not None else None,
                usage=usage,
            )
            if len(self._spans) == self._spans.maxlen:
                self._counts["evicted_spans"] += 1
            self._spans.append(span)
            self._counts["completed"] += 1
            self._counts["errors"] += error is not None
            count_key = {"model": "model_calls", "tool": "tool_calls", "retriever": "retrievals"}
            if span.kind in count_key:
                self._counts[count_key[span.kind]] += 1
            self._tokens.update(usage)
        if self._sink is not None:
            try:
                self._sink(Span(**span.to_dict()))
            except Exception:
                # Instrumentation must not turn successful business work into a retry.
                with self._lock:
                    self._counts["sink_errors"] += 1

    def on_chain_start(self, serialized, inputs, **kwargs):
        self._start("chain", serialized, **kwargs)

    def on_chain_end(self, outputs, **kwargs):
        self._finish(**kwargs)

    def on_chain_error(self, error, **kwargs):
        self._finish(error=error, **kwargs)

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self._start("model", serialized, **kwargs)

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._start("model", serialized, **kwargs)

    def on_llm_end(self, response, **kwargs):
        self._finish(response=response, **kwargs)

    def on_llm_error(self, error, **kwargs):
        self._finish(error=error, **kwargs)

    def on_tool_start(self, serialized, input_str, **kwargs):
        self._start("tool", serialized, **kwargs)

    def on_tool_end(self, output, **kwargs):
        self._finish(**kwargs)

    def on_tool_error(self, error, **kwargs):
        self._finish(error=error, **kwargs)

    def on_retriever_start(self, serialized, query, **kwargs):
        self._start("retriever", serialized, **kwargs)

    def on_retriever_end(self, documents, **kwargs):
        self._finish(**kwargs)

    def on_retriever_error(self, error, **kwargs):
        self._finish(error=error, **kwargs)
