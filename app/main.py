import asyncio
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, status
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.core.concurrency import agent_request_semaphore
from app.core.config import settings
from app.core.errors import AgentTimeoutError, register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import request_context_middleware
from app.core.security import verify_internal_token
from app.core.agent_tracking import bind_agent_tracking_context
from app.agents.capability_executor import execute_agent_task
from app.agents.helpers import (
    build_history_text,
    conversation_message_dict,
    initial_state,
)
from app.agents.hotel_graphs import (
    collect_spa_reservation_details,
    evaluate_room_service_confirmation_reply,
    evaluate_maintenance_guest_resolution,
    evaluate_spa_reservation_confirmation,
    extract_graph,
    faq_graph,
    generate_front_desk_handoff_response,
    generate_inactivity_message,
    generate_clarification_question,
    generate_maintenance_initial_response,
    generate_maintenance_staff_update_response,
    generate_request_cancelled_response,
    generate_request_processed_response,
    generate_room_service_delivery_location_response,
    generate_room_service_menu_response,
    generate_room_service_order_accepted_response,
    generate_room_service_kitchen_rejected_response,
    generate_spa_menu_message,
    generate_spa_request_received_response,
    generate_spa_reservation_confirmation,
    generate_spa_staff_response,
    generate_unmatched_guest_response,
    normalize_room_service_order,
)
from app.schemas.hotel import (
    ClarificationRequest,
    ClarificationResponse,
    FaqResponse,
    FaqResponseRequest,
    GenericMessageRequest,
    GenericMessageResponse,
    HotelConversationRequest,
    HotelExtractionResponse,
    MaintenanceGuestResolutionEvaluationRequest,
    MaintenanceGuestResolutionEvaluationResponse,
    MaintenanceInitialResponseRequest,
    MaintenanceStaffUpdateResponseRequest,
    RoomServiceConfirmationEvaluationRequest,
    RoomServiceConfirmationEvaluationResponse,
    RoomServiceConfirmationRequest,
    RoomServiceConfirmationResponse,
    SpaMenuResponseRequest,
    SpaReservationDetailsRequest,
    SpaReservationDetailsResponse,
    SpaReservationConfirmationEvaluationRequest,
    SpaReservationConfirmationEvaluationResponse,
    SpaReservationConfirmationRequest,
    SpaReservationConfirmationResponse,
    SpaStaffResponseRequest,
    UnmatchedGuestResponseRequest,
)
from app.schemas.tasks import (
    AgentCapabilitiesResponse,
    AgentTaskRequest,
    AgentTaskResponse,
    AgentTaskType,
    AgentToolName,
)
from app.services.request_limits import (
    validate_agent_task,
    validate_context,
    validate_guest_message,
    validate_history,
)


configure_logging()

app = FastAPI(
    title="Chatbot Inn Hotel Agent",
    version=settings.app_version,
)
app.middleware("http")(request_context_middleware)
register_exception_handlers(app)

hotel_router = APIRouter(
    prefix="/hotel",
    dependencies=[
        Depends(verify_internal_token),
        Depends(bind_agent_tracking_context),
    ],
)


def dump_history(messages):
    return [conversation_message_dict(item) for item in messages]


def validate_conversation_payload(
    guest_message: str | None = None,
    conversation_history: list[Any] | None = None,
    known_context: dict[str, Any] | None = None,
) -> None:
    validate_guest_message(guest_message)
    validate_history(conversation_history or [])
    validate_context(known_context or {})


async def run_agent_step(call: Callable[[], Any]) -> Any:
    async with agent_request_semaphore:
        try:
            return await asyncio.wait_for(
                run_in_threadpool(call),
                timeout=settings.request_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise AgentTimeoutError("Agent request timed out") from exc


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
    }


@app.get("/ready")
async def ready():
    if not settings.is_openai_configured or not settings.is_internal_token_configured:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "openaiConfigured": settings.is_openai_configured,
                "internalTokenConfigured": settings.is_internal_token_configured,
            },
        )

    return {
        "status": "ready",
        "openaiConfigured": True,
        "internalTokenConfigured": True,
    }


@hotel_router.post("/extract-intent", response_model=HotelExtractionResponse)
async def hotel_extract_intent(request: HotelConversationRequest):
    validate_conversation_payload(
        guest_message=request.guestMessage,
        conversation_history=request.conversationHistory,
        known_context=request.knownContext,
    )

    result = await run_agent_step(
        lambda: extract_graph.invoke(
            initial_state(
                guest_message=request.guestMessage,
                conversation_history=dump_history(request.conversationHistory),
                known_context=request.knownContext,
            )
        )
    )
    return result["extraction_json"]


@hotel_router.post("/generate-clarification", response_model=ClarificationResponse)
async def hotel_generate_clarification(request: ClarificationRequest):
    validate_conversation_payload(conversation_history=request.conversationHistory)
    history = dump_history(request.conversationHistory)
    result = await run_agent_step(
        lambda: generate_clarification_question(
            initial_state(
                conversation_history=history,
                history_text=build_history_text(history),
                extraction_json=request.extraction,
            )
        )
    )

    return {
        "message": result["clarification_message"],
        "interaction": result.get("clarification_interaction"),
    }


@hotel_router.post("/faq-response", response_model=FaqResponse)
async def hotel_faq_response(request: FaqResponseRequest):
    validate_conversation_payload(
        guest_message=request.guestMessage,
        conversation_history=request.conversationHistory,
        known_context=request.knownContext,
    )

    result = await run_agent_step(
        lambda: faq_graph.invoke(
            initial_state(
                guest_message=request.guestMessage,
                conversation_history=dump_history(request.conversationHistory),
                known_context=request.knownContext,
            )
        )
    )

    return {
        "message": result["faq_message"],
        "answered": result.get("faq_answered", True),
        "needsHumanAnswer": result.get("faq_needs_human_answer", False),
        "category": result.get("faq_category"),
        "interaction": result.get("faq_interaction"),
    }


@hotel_router.post("/room-service-confirmation", response_model=RoomServiceConfirmationResponse)
async def hotel_room_service_confirmation(request: RoomServiceConfirmationRequest):
    validate_conversation_payload(conversation_history=request.conversationHistory)
    history = dump_history(request.conversationHistory)
    result = await run_agent_step(
        lambda: normalize_room_service_order(
            initial_state(
                conversation_history=history,
                history_text=build_history_text(history),
                extraction_json=request.extraction,
            )
        )
    )

    return {
        "message": result["room_service_confirmation_message"],
        "pendingOrder": result["room_service_pending_order"],
        "missingFields": result.get("room_service_missing_fields", []),
        "requestComplete": result.get("room_service_request_complete", False),
        "interaction": result.get("room_service_confirmation_interaction"),
    }


@hotel_router.post("/room-service/delivery-location-response", response_model=GenericMessageResponse)
async def hotel_room_service_delivery_location_response(request: GenericMessageRequest):
    validate_conversation_payload(
        guest_message=request.guestMessage,
        conversation_history=request.conversationHistory,
        known_context=request.knownContext,
    )
    history = dump_history(request.conversationHistory)
    result = await run_agent_step(
        lambda: generate_room_service_delivery_location_response(
            history,
            request.knownContext,
        )
    )
    return {
        "message": result["message"],
        "interaction": result.get("interaction"),
    }


@hotel_router.post("/room-service/menu-response", response_model=GenericMessageResponse)
async def hotel_room_service_menu_response(request: GenericMessageRequest):
    validate_conversation_payload(
        guest_message=request.guestMessage,
        conversation_history=request.conversationHistory,
        known_context=request.knownContext,
    )
    history = dump_history(request.conversationHistory)
    result = await run_agent_step(
        lambda: generate_room_service_menu_response(
            history,
            request.knownContext,
        )
    )
    return {
        "message": result["message"],
        "interaction": result.get("interaction"),
    }


@hotel_router.post("/room-service/order-accepted-response", response_model=GenericMessageResponse)
async def hotel_room_service_order_accepted_response(request: GenericMessageRequest):
    validate_conversation_payload(
        guest_message=request.guestMessage,
        conversation_history=request.conversationHistory,
        known_context=request.knownContext,
    )
    history = dump_history(request.conversationHistory)
    result = await run_agent_step(
        lambda: generate_room_service_order_accepted_response(
            request.guestMessage,
            history,
            request.knownContext,
        )
    )
    return {
        "message": result["message"],
        "interaction": result.get("interaction"),
    }


@hotel_router.post("/room-service/kitchen-rejected-response", response_model=GenericMessageResponse)
async def hotel_room_service_kitchen_rejected_response(request: GenericMessageRequest):
    validate_conversation_payload(
        guest_message=request.guestMessage,
        conversation_history=request.conversationHistory,
        known_context=request.knownContext,
    )
    history = dump_history(request.conversationHistory)
    result = await run_agent_step(
        lambda: generate_room_service_kitchen_rejected_response(
            request.guestMessage,
            history,
            request.knownContext,
        )
    )
    return {
        "message": result["message"],
        "interaction": result.get("interaction"),
    }


@hotel_router.post("/evaluate-room-service-confirmation", response_model=RoomServiceConfirmationEvaluationResponse)
async def hotel_evaluate_room_service_confirmation(request: RoomServiceConfirmationEvaluationRequest):
    validate_conversation_payload(
        guest_message=request.guestMessage,
        conversation_history=request.conversationHistory,
    )
    history = dump_history(request.conversationHistory)
    result = await run_agent_step(
        lambda: evaluate_room_service_confirmation_reply(
            initial_state(
                guest_message=request.guestMessage,
                conversation_history=history,
                history_text=build_history_text(history),
                pending_order=request.pendingOrder,
            )
        )
    )

    return {
        "confirmationAction": result["room_service_confirmation_action"],
        "updatedOrder": result["room_service_pending_order"],
        "message": result["room_service_confirmation_message"],
        "interaction": result.get("room_service_confirmation_interaction"),
    }


@hotel_router.post("/spa/menu-response", response_model=GenericMessageResponse)
async def hotel_spa_menu_response(request: SpaMenuResponseRequest):
    validate_conversation_payload(
        conversation_history=request.conversationHistory,
        known_context=request.knownContext,
    )
    history = dump_history(request.conversationHistory)
    result = await run_agent_step(
        lambda: generate_spa_menu_message(history, request.knownContext)
    )
    return {
        "message": result["message"],
        "interaction": result.get("interaction"),
    }


@hotel_router.post("/spa/reservation-details", response_model=SpaReservationDetailsResponse)
async def hotel_spa_reservation_details(request: SpaReservationDetailsRequest):
    validate_conversation_payload(
        guest_message=request.guestMessage,
        conversation_history=request.conversationHistory,
        known_context=request.knownContext,
    )
    history = dump_history(request.conversationHistory)
    result = await run_agent_step(
        lambda: collect_spa_reservation_details(
            request.guestMessage,
            request.pendingReservation,
            history,
            request.knownContext,
        )
    )
    return result


@hotel_router.post("/spa/request-received-response", response_model=GenericMessageResponse)
async def hotel_spa_request_received_response(request: GenericMessageRequest):
    validate_conversation_payload(
        guest_message=request.guestMessage,
        conversation_history=request.conversationHistory,
        known_context=request.knownContext,
    )
    history = dump_history(request.conversationHistory)
    result = await run_agent_step(
        lambda: generate_spa_request_received_response(
            request.guestMessage,
            history,
            request.knownContext,
        )
    )
    return {
        "message": result["message"],
        "interaction": result.get("interaction"),
    }


@hotel_router.post("/spa/reservation-confirmation", response_model=SpaReservationConfirmationResponse)
async def hotel_spa_reservation_confirmation(request: SpaReservationConfirmationRequest):
    validate_conversation_payload(
        guest_message=request.guestMessage,
        conversation_history=request.conversationHistory,
        known_context=request.knownContext,
    )
    history = dump_history(request.conversationHistory)
    result = await run_agent_step(
        lambda: generate_spa_reservation_confirmation(
            request.guestMessage,
            request.extraction,
            history,
            request.knownContext,
        )
    )
    return {
        "message": result["message"],
        "pendingReservation": result["pendingReservation"],
        "interaction": result.get("interaction"),
    }


@hotel_router.post("/spa/evaluate-confirmation", response_model=SpaReservationConfirmationEvaluationResponse)
async def hotel_spa_evaluate_confirmation(request: SpaReservationConfirmationEvaluationRequest):
    validate_conversation_payload(
        guest_message=request.guestMessage,
        conversation_history=request.conversationHistory,
    )
    history = dump_history(request.conversationHistory)
    result = await run_agent_step(
        lambda: evaluate_spa_reservation_confirmation(
            request.guestMessage,
            request.pendingReservation,
            history,
        )
    )
    return {
        "confirmationAction": result["confirmationAction"],
        "updatedReservation": result["updatedReservation"],
        "message": result["message"],
        "interaction": result.get("interaction"),
    }


@hotel_router.post("/spa/staff-response", response_model=GenericMessageResponse)
async def hotel_spa_staff_response(request: SpaStaffResponseRequest):
    validate_conversation_payload(
        guest_message=request.staffMessage,
        conversation_history=request.conversationHistory,
        known_context=request.knownContext,
    )
    history = dump_history(request.conversationHistory)
    result = await run_agent_step(
        lambda: generate_spa_staff_response(
            request.staffDecision,
            request.staffMessage,
            history,
            request.knownContext,
        )
    )
    return {
        "message": result["message"],
        "interaction": result.get("interaction"),
    }


@hotel_router.get("/capabilities", response_model=AgentCapabilitiesResponse)
async def hotel_capabilities():
    return {
        "schemaVersion": 1,
        "taskTypes": list(AgentTaskType),
        "tools": list(AgentToolName),
    }


@hotel_router.post("/tasks", response_model=AgentTaskResponse)
async def hotel_execute_task(request: AgentTaskRequest):
    validate_guest_message(request.latestMessage)
    validate_guest_message(request.staffMessage)
    validate_history(request.conversationHistory)
    validate_agent_task(request.model_dump(mode="json"))
    return await run_agent_step(lambda: execute_agent_task(request))


@hotel_router.post("/maintenance/initial-response", response_model=GenericMessageResponse)
async def hotel_maintenance_initial_response(request: MaintenanceInitialResponseRequest):
    validate_conversation_payload(
        conversation_history=request.conversationHistory,
        known_context=request.knownContext,
    )
    history = dump_history(request.conversationHistory)
    result = await run_agent_step(
        lambda: generate_maintenance_initial_response(
            request.extraction,
            history,
            request.knownContext,
        )
    )
    return {
        "message": result["message"],
        "interaction": result.get("interaction"),
    }


@hotel_router.post("/maintenance/staff-update-response", response_model=GenericMessageResponse)
async def hotel_maintenance_staff_update_response(request: MaintenanceStaffUpdateResponseRequest):
    validate_conversation_payload(
        guest_message=request.staffMessage,
        conversation_history=request.conversationHistory,
        known_context=request.knownContext,
    )
    history = dump_history(request.conversationHistory)
    result = await run_agent_step(
        lambda: generate_maintenance_staff_update_response(
            request.staffStatus,
            request.staffMessage,
            history,
            request.knownContext,
        )
    )
    return {
        "message": result["message"],
        "interaction": result.get("interaction"),
    }


@hotel_router.post("/maintenance/evaluate-guest-resolution", response_model=MaintenanceGuestResolutionEvaluationResponse)
async def hotel_maintenance_evaluate_guest_resolution(request: MaintenanceGuestResolutionEvaluationRequest):
    validate_conversation_payload(
        guest_message=request.guestMessage,
        conversation_history=request.conversationHistory,
        known_context=request.knownContext,
    )
    history = dump_history(request.conversationHistory)
    result = await run_agent_step(
        lambda: evaluate_maintenance_guest_resolution(
            request.guestMessage,
            history,
            request.knownContext,
        )
    )
    return {
        "guestConfirmedResolved": result["guestConfirmedResolved"],
        "message": result["message"],
        "interaction": result.get("interaction"),
    }


@hotel_router.post("/unmatched-guest-response", response_model=GenericMessageResponse)
async def hotel_unmatched_guest_response(request: UnmatchedGuestResponseRequest):
    validate_conversation_payload(
        guest_message=request.guestMessage,
        known_context=request.knownContext,
    )
    result = await run_agent_step(
        lambda: generate_unmatched_guest_response(
            request.guestMessage,
            request.fromPhoneNumber,
            request.knownContext,
        )
    )
    return {
        "message": result["message"],
        "interaction": result.get("interaction"),
    }


@hotel_router.post("/inactivity-reminder", response_model=GenericMessageResponse)
async def hotel_inactivity_reminder(request: GenericMessageRequest):
    validate_conversation_payload(
        guest_message=request.guestMessage,
        conversation_history=request.conversationHistory,
        known_context=request.knownContext,
    )
    history = dump_history(request.conversationHistory)
    result = await run_agent_step(
        lambda: generate_inactivity_message("reminder", history, request.knownContext)
    )
    return {
        "message": result["message"],
        "interaction": result.get("interaction"),
    }


@hotel_router.post("/inactivity-closure", response_model=GenericMessageResponse)
async def hotel_inactivity_closure(request: GenericMessageRequest):
    validate_conversation_payload(
        guest_message=request.guestMessage,
        conversation_history=request.conversationHistory,
        known_context=request.knownContext,
    )
    history = dump_history(request.conversationHistory)
    result = await run_agent_step(
        lambda: generate_inactivity_message("closure", history, request.knownContext)
    )
    return {
        "message": result["message"],
        "interaction": result.get("interaction"),
    }


@hotel_router.post("/front-desk-handoff", response_model=GenericMessageResponse)
async def hotel_front_desk_handoff(request: GenericMessageRequest):
    validate_conversation_payload(
        guest_message=request.guestMessage,
        conversation_history=request.conversationHistory,
        known_context=request.knownContext,
    )
    history = dump_history(request.conversationHistory)
    result = await run_agent_step(
        lambda: generate_front_desk_handoff_response(
            request.guestMessage,
            history,
            request.knownContext,
        )
    )
    return {
        "message": result["message"],
        "interaction": result.get("interaction"),
    }


@hotel_router.post("/request-processed-response", response_model=GenericMessageResponse)
async def hotel_request_processed_response(request: GenericMessageRequest):
    validate_conversation_payload(
        guest_message=request.guestMessage,
        conversation_history=request.conversationHistory,
        known_context=request.knownContext,
    )
    history = dump_history(request.conversationHistory)
    result = await run_agent_step(
        lambda: generate_request_processed_response(
            request.guestMessage,
            history,
            request.knownContext,
        )
    )
    return {
        "message": result["message"],
        "interaction": result.get("interaction"),
    }


@hotel_router.post("/request-cancelled-response", response_model=GenericMessageResponse)
async def hotel_request_cancelled_response(request: GenericMessageRequest):
    validate_conversation_payload(
        guest_message=request.guestMessage,
        conversation_history=request.conversationHistory,
        known_context=request.knownContext,
    )
    history = dump_history(request.conversationHistory)
    result = await run_agent_step(
        lambda: generate_request_cancelled_response(
            request.guestMessage,
            history,
            request.knownContext,
        )
    )
    return {
        "message": result["message"],
        "interaction": result.get("interaction"),
    }


@hotel_router.post("/integration-test-response", response_model=GenericMessageResponse)
async def hotel_integration_test_response(request: GenericMessageRequest):
    validate_conversation_payload(
        guest_message=request.guestMessage,
        conversation_history=request.conversationHistory,
        known_context=request.knownContext,
    )
    return {
        "message": "Hola mundo desde el agente de Chatbot Inn. Integracion completa OK.",
        "interaction": None,
    }



if settings.enable_debug_endpoints:
    @hotel_router.post("/debug")
    async def hotel_debug(request: HotelConversationRequest):
        validate_conversation_payload(
            guest_message=request.guestMessage,
            conversation_history=request.conversationHistory,
            known_context=request.knownContext,
        )
        result = await run_agent_step(
            lambda: extract_graph.invoke(
                initial_state(
                    guest_message=request.guestMessage,
                    conversation_history=dump_history(request.conversationHistory),
                    known_context=request.knownContext,
                )
            )
        )

        return {
            "history_text": result["history_text"],
            "extraction": result["extraction_json"],
        }


app.include_router(hotel_router)
