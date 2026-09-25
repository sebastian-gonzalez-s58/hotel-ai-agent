"""Opt-in structured timing; no prompts, arguments, answers or credentials are logged."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from functools import wraps
import json
import logging
import re
import time
from uuid import uuid4

from app.core.config import settings

logger = logging.getLogger("chat.latency")


@dataclass(frozen=True)
class Context:
    trace_id: str
    span_id: str | None = None


_current: ContextVar[Context | None] = ContextVar("chat_latency", default=None)


def identifier(value):
    return value if isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", value) else None


def current():
    return _current.get() if settings.chat_latency_enabled else None


@contextmanager
def trace(trace_id=None, parent_span_id=None, *, enabled=True):
    token = _current.set(Context(identifier(trace_id) or str(uuid4()), identifier(parent_span_id))
                         if enabled and settings.chat_latency_enabled else None)
    try:
        yield
    finally:
        _current.reset(token)


def headers():
    context = current()
    if context is None:
        return {}
    return {"X-Chat-Latency-Trace": context.trace_id,
            **({"X-Chat-Latency-Parent": context.span_id} if context.span_id else {})}


class span:
    def __init__(self, stage, **attributes):
        self.stage = stage
        self.attributes = attributes
        self.outcome = "ok"
        self.started_ns = None
        self.active = False

    def __enter__(self):
        self.parent = current()
        self.active = self.parent is not None
        if self.active:
            self.started_at = datetime.now(timezone.utc).isoformat()
            self.started_ns = time.perf_counter_ns()
            self.context = Context(self.parent.trace_id, uuid4().hex[:16])
            self.token = _current.set(self.context)
        return self

    def __exit__(self, error_type, error, traceback):
        if not self.active:
            return False
        elapsed = (time.perf_counter_ns() - self.started_ns) / 1_000_000
        _current.reset(self.token)
        if error_type is not None:
            self.outcome = "cancelled" if error_type.__name__ == "CancelledError" else "error"
            self.attributes["error_type"] = error_type.__name__
        try:
            logger.info("CHAT_LATENCY %s", json.dumps({
                "version": 1, "component": "agent", "trace_id": self.context.trace_id,
                "span_id": self.context.span_id, "parent_span_id": self.parent.span_id,
                "stage": self.stage, "kind": "span", "started_at": self.started_at,
                "duration_ms": round(elapsed, 2), "outcome": self.outcome,
                "slow": elapsed >= settings.chat_latency_slow_ms, **self.attributes,
            }, separators=(",", ":")))
        except Exception:
            # Diagnostic output must not affect a guest request.
            pass
        return False


def timed(stage):
    def decorate(function):
        @wraps(function)
        def measured(*args, **kwargs):
            if current() is None:
                return function(*args, **kwargs)
            with span(stage):
                return function(*args, **kwargs)
        return measured
    return decorate


def waited(stage, started_ns):
    with span(stage) as measurement:
        if measurement.active:
            measurement.started_ns = started_ns
            measurement.started_at = (datetime.now(timezone.utc) - timedelta(
                microseconds=(time.perf_counter_ns() - started_ns) / 1000)).isoformat()
            measurement.attributes["kind"] = "interval"

