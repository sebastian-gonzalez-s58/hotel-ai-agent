from typing import Any

from langgraph.graph import END, StateGraph

from app.agents.helpers import build_history_text
from app.agents.state import HotelConversationState
from app.prompts.hotel import (
    clarification_prompt,
    extraction_prompt,
    faq_prompt,
    generic_message_prompt,
    maintenance_guest_resolution_evaluation_prompt,
    maintenance_initial_response_prompt,
    maintenance_staff_update_prompt,
    room_service_confirmation_evaluation_prompt,
    room_service_confirmation_prompt,
    spa_confirmation_evaluation_prompt,
    spa_menu_prompt,
    spa_reservation_confirmation_prompt,
    unmatched_guest_response_prompt,
)
from app.services.knowledge_client import get_faq_knowledge, get_menu_knowledge
from app.services.openai_client import call_openai_json


def collect_conversation_context(state: HotelConversationState) -> HotelConversationState:
    conversation_history = state["conversation_history"]
    guest_message = state["guest_message"]

    full_history = list(conversation_history)
    if guest_message and not (
        full_history
        and full_history[-1].get("role") == "guest"
        and full_history[-1].get("content") == guest_message
    ):
        full_history.append(
            {
                "role": "guest",
                "content": guest_message,
            }
        )

    return {
        **state,
        "history_text": build_history_text(full_history),
    }


def extract_intent_and_entities(state: HotelConversationState) -> HotelConversationState:
    extraction_json = call_openai_json(
        extraction_prompt(
            history_text=state["history_text"],
            known_context=state["known_context"],
        )
    )

    return {
        **state,
        "extraction_json": extraction_json,
    }


def generate_clarification_question(state: HotelConversationState) -> HotelConversationState:
    clarification_json = call_openai_json(
        clarification_prompt(
            extraction=state["extraction_json"],
            history_text=state["history_text"],
        )
    )

    return {
        **state,
        "clarification_message": clarification_json["message"],
    }


def normalize_room_service_order(state: HotelConversationState) -> HotelConversationState:
    menu_knowledge = get_menu_knowledge()
    confirmation_json = call_openai_json(
        room_service_confirmation_prompt(
            extraction=state["extraction_json"],
            history_text=state["history_text"],
            menu_knowledge=menu_knowledge,
        )
    )

    return {
        **state,
        "room_service_confirmation_message": confirmation_json["message"],
        "room_service_pending_order": confirmation_json["pendingOrder"],
    }


def evaluate_room_service_confirmation_reply(state: HotelConversationState) -> HotelConversationState:
    evaluation_json = call_openai_json(
        room_service_confirmation_evaluation_prompt(
            guest_message=state["guest_message"],
            pending_order=state["room_service_pending_order"],
            history_text=state["history_text"],
        )
    )

    return {
        **state,
        "room_service_confirmation_action": evaluation_json["confirmationAction"],
        "room_service_pending_order": evaluation_json["updatedOrder"],
        "room_service_confirmation_message": evaluation_json["message"],
    }


def generate_faq_response(state: HotelConversationState) -> HotelConversationState:
    known_context = {
        **state["known_context"],
        "faqKnowledge": get_faq_knowledge(),
    }
    faq_json = call_openai_json(
        faq_prompt(
            guest_message=state["guest_message"],
            history_text=state["history_text"],
            known_context=known_context,
        )
    )

    return {
        **state,
        "faq_message": faq_json["message"],
        "faq_answered": faq_json.get("answered", True),
        "faq_needs_human_answer": faq_json.get("needsHumanAnswer", False),
        "faq_category": faq_json.get("category"),
    }


def generate_spa_menu_message(history: list[dict[str, Any]], known_context: dict[str, Any]) -> dict[str, Any]:
    history_text = build_history_text(history)
    return call_openai_json(spa_menu_prompt(history_text=history_text, known_context=known_context))


def generate_spa_reservation_confirmation(
    guest_message: str | None,
    extraction: dict[str, Any],
    history: list[dict[str, Any]],
    known_context: dict[str, Any],
) -> dict[str, Any]:
    history_text = build_history_text(history)
    return call_openai_json(
        spa_reservation_confirmation_prompt(
            guest_message=guest_message,
            extraction=extraction,
            history_text=history_text,
            known_context=known_context,
        )
    )


def evaluate_spa_reservation_confirmation(
    guest_message: str,
    pending_reservation: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    history_text = build_history_text(history)
    return call_openai_json(
        spa_confirmation_evaluation_prompt(
            guest_message=guest_message,
            pending_reservation=pending_reservation,
            history_text=history_text,
        )
    )


def generate_maintenance_initial_response(
    extraction: dict[str, Any],
    history: list[dict[str, Any]],
    known_context: dict[str, Any],
) -> dict[str, Any]:
    history_text = build_history_text(history)
    return call_openai_json(
        maintenance_initial_response_prompt(
            extraction=extraction,
            history_text=history_text,
            known_context=known_context,
        )
    )


def generate_maintenance_staff_update_response(
    staff_status: str,
    staff_message: str | None,
    history: list[dict[str, Any]],
    known_context: dict[str, Any],
) -> dict[str, Any]:
    history_text = build_history_text(history)
    return call_openai_json(
        maintenance_staff_update_prompt(
            staff_status=staff_status,
            staff_message=staff_message,
            history_text=history_text,
            known_context=known_context,
        )
    )


def evaluate_maintenance_guest_resolution(
    guest_message: str,
    history: list[dict[str, Any]],
    known_context: dict[str, Any],
) -> dict[str, Any]:
    history_text = build_history_text(history)
    return call_openai_json(
        maintenance_guest_resolution_evaluation_prompt(
            guest_message=guest_message,
            history_text=history_text,
            known_context=known_context,
        )
    )


def generate_unmatched_guest_response(
    guest_message: str,
    from_phone_number: str | None,
    known_context: dict[str, Any],
) -> dict[str, Any]:
    return call_openai_json(
        unmatched_guest_response_prompt(
            guest_message=guest_message,
            from_phone_number=from_phone_number,
            known_context=known_context,
        )
    )


def generate_inactivity_message(
    kind: str,
    history: list[dict[str, Any]],
    known_context: dict[str, Any],
) -> dict[str, Any]:
    purpose = (
        "Send a friendly reminder that the hotel assistant is waiting for the guest's reply. "
        "Mention that the conversation may be closed if there is no response soon."
        if kind == "reminder"
        else "Tell the guest that due to inactivity the conversation will be closed, and that they can write again if they still need help."
    )
    return call_openai_json(
        generic_message_prompt(
            purpose=purpose,
            history_text=build_history_text(history),
            known_context=known_context,
        )
    )


def generate_front_desk_handoff_response(
    guest_message: str | None,
    history: list[dict[str, Any]],
    known_context: dict[str, Any],
) -> dict[str, Any]:
    return call_openai_json(
        generic_message_prompt(
            purpose=(
                "Tell the guest that front desk has been notified and will help with the request. "
                "If the request appears urgent, use a calm reassuring tone."
            ),
            history_text=build_history_text(history),
            known_context=known_context,
            guest_message=guest_message,
        )
    )


def generate_request_processed_response(
    guest_message: str | None,
    history: list[dict[str, Any]],
    known_context: dict[str, Any],
) -> dict[str, Any]:
    return call_openai_json(
        generic_message_prompt(
            purpose=(
                "Tell the guest their request has been processed and that this conversation will be closed. "
                "If the request was routed to a hotel team, mention that the appropriate team has been notified. "
                "Invite them to write again if they need anything else."
            ),
            history_text=build_history_text(history),
            known_context=known_context,
            guest_message=guest_message,
        )
    )


def generate_request_cancelled_response(
    guest_message: str | None,
    history: list[dict[str, Any]],
    known_context: dict[str, Any],
) -> dict[str, Any]:
    return call_openai_json(
        generic_message_prompt(
            purpose=(
                "Tell the guest their request was not processed because they cancelled it or did not confirm it. "
                "Close the conversation politely and invite them to write again if they need anything else."
            ),
            history_text=build_history_text(history),
            known_context=known_context,
            guest_message=guest_message,
        )
    )


extract_graph_builder = StateGraph(HotelConversationState)
extract_graph_builder.add_node("collect_context", collect_conversation_context)
extract_graph_builder.add_node("extract_intent", extract_intent_and_entities)
extract_graph_builder.set_entry_point("collect_context")
extract_graph_builder.add_edge("collect_context", "extract_intent")
extract_graph_builder.add_edge("extract_intent", END)
extract_graph = extract_graph_builder.compile()


clarification_graph_builder = StateGraph(HotelConversationState)
clarification_graph_builder.add_node("collect_context", collect_conversation_context)
clarification_graph_builder.add_node("clarify", generate_clarification_question)
clarification_graph_builder.set_entry_point("collect_context")
clarification_graph_builder.add_edge("collect_context", "clarify")
clarification_graph_builder.add_edge("clarify", END)
clarification_graph = clarification_graph_builder.compile()


faq_graph_builder = StateGraph(HotelConversationState)
faq_graph_builder.add_node("collect_context", collect_conversation_context)
faq_graph_builder.add_node("faq", generate_faq_response)
faq_graph_builder.set_entry_point("collect_context")
faq_graph_builder.add_edge("collect_context", "faq")
faq_graph_builder.add_edge("faq", END)
faq_graph = faq_graph_builder.compile()
