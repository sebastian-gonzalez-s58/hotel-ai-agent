from typing import Any

from app.agents.state import HotelConversationState


def build_history_text(messages: list[dict[str, Any]]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def conversation_message_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump()
    return message.dict()


def initial_state(
    guest_message: str = "",
    conversation_history: list[dict[str, Any]] | None = None,
    known_context: dict[str, Any] | None = None,
    extraction_json: dict[str, Any] | None = None,
    pending_order: dict[str, Any] | None = None,
    history_text: str = "",
) -> HotelConversationState:
    return {
        "guest_message": guest_message,
        "conversation_history": conversation_history or [],
        "known_context": known_context or {},
        "history_text": history_text,
        "extraction_json": extraction_json or {},
        "clarification_message": "",
        "faq_message": "",
        "room_service_confirmation_message": "",
        "room_service_pending_order": pending_order or {},
        "room_service_confirmation_action": "",
        "room_service_missing_fields": [],
        "room_service_request_complete": False,
    }
