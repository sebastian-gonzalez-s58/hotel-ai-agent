from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from fastapi import Request


@dataclass(frozen=True)
class AgentTrackingContext:
    purpose: str
    request_id: str | None = None
    conversation_id: str | None = None
    operation_id: str | None = None
    message_id: str | None = None
    unmatched_message_id: str | None = None
    process_instance_id: str | None = None
    process_activity_id: str | None = None


_tracking_context: ContextVar[AgentTrackingContext | None] = ContextVar(
    "agent_tracking_context",
    default=None,
)


def get_agent_tracking_context() -> AgentTrackingContext | None:
    return _tracking_context.get()


def set_agent_tracking_context(context: AgentTrackingContext):
    return _tracking_context.set(context)


def reset_agent_tracking_context(token) -> None:
    _tracking_context.reset(token)


def tracking_context_from_payload(
    payload: dict[str, Any],
    *,
    purpose: str,
    request_id: str | None,
) -> AgentTrackingContext:
    task_context = payload.get("context") or {}
    known_context = payload.get("knownContext") or {}

    def value(name: str) -> str | None:
        raw = task_context.get(name)
        if raw is None:
            raw = known_context.get(name)
        if raw is None:
            raw = payload.get(name)
        return str(raw) if raw is not None else None

    return AgentTrackingContext(
        purpose=purpose,
        request_id=request_id or value("requestId"),
        conversation_id=value("conversationId"),
        operation_id=value("operationId"),
        message_id=value("messageId"),
        unmatched_message_id=value("unmatchedMessageId"),
        process_instance_id=value("processInstanceId"),
        process_activity_id=value("processActivityId"),
    )


async def bind_agent_tracking_context(request: Request):
    if request.method in {"POST", "PUT", "PATCH"}:
        try:
            payload = await request.json()
        except Exception:
            payload = {}
    else:
        payload = {}

    purpose = payload.get("taskType") or _purpose_from_path(request.url.path)
    context = tracking_context_from_payload(
        payload,
        purpose=str(purpose)[:80],
        request_id=(
            request.headers.get("X-Request-ID")
            or getattr(request.state, "request_id", None)
        ),
    )
    token = set_agent_tracking_context(context)
    try:
        yield
    finally:
        reset_agent_tracking_context(token)


def _purpose_from_path(path: str) -> str:
    value = path.removeprefix("/hotel/").replace("/", "_").replace("-", "_")
    return value.upper() or "HOTEL_AGENT_TASK"
