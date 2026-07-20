from typing import Any, TypedDict


class HotelConversationState(TypedDict):
    guest_message: str
    conversation_history: list[dict[str, Any]]
    known_context: dict[str, Any]
    history_text: str
    extraction_json: dict[str, Any]
    clarification_message: str
    faq_message: str
    room_service_confirmation_message: str
    room_service_pending_order: dict[str, Any]
    room_service_confirmation_action: str
