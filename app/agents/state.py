from typing import Any, TypedDict


class HotelConversationState(TypedDict):
    guest_message: str
    conversation_history: list[dict[str, Any]]
    known_context: dict[str, Any]
    history_text: str
    extraction_json: dict[str, Any]
    clarification_message: str
    clarification_interaction: dict[str, Any] | None
    faq_message: str
    faq_interaction: dict[str, Any] | None
    room_service_confirmation_message: str
    room_service_confirmation_interaction: dict[str, Any] | None
    room_service_pending_order: dict[str, Any]
    room_service_confirmation_action: str
