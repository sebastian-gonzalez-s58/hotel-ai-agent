import json
from typing import Any

from fastapi import HTTPException, status

from app.core.config import settings


def validate_guest_message(message: str | None) -> None:
    if message is None:
        return
    if len(message) > settings.max_guest_message_chars:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"guestMessage exceeds {settings.max_guest_message_chars} characters",
        )


def validate_history(messages: list[Any]) -> None:
    if len(messages) > settings.max_history_messages:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"conversationHistory exceeds {settings.max_history_messages} messages",
        )

    for message in messages:
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if content and len(content) > settings.max_guest_message_chars:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"conversationHistory message exceeds {settings.max_guest_message_chars} characters",
            )


def validate_context(context: dict[str, Any]) -> None:
    serialized_context = json.dumps(context, ensure_ascii=False)
    if len(serialized_context) > settings.max_context_chars:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"knownContext exceeds {settings.max_context_chars} characters",
        )


def validate_agent_task(payload: dict[str, Any]) -> None:
    serialized_payload = json.dumps(payload, ensure_ascii=False)
    if len(serialized_payload) > settings.max_agent_task_chars:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Agent task exceeds {settings.max_agent_task_chars} characters",
        )
