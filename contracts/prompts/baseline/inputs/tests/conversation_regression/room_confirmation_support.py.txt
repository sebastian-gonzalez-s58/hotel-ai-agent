"""Initial unit-test menus use the runtime renderer; never fabricate valid IDs."""
import json
import time

from app.agents.v2_turn_planner import (
    _bind_room_confirmation, _deterministic_turn_response, _room_service_confirmation_message,
)


def current_room_button(response, action="CONFIRM"):
    return next(option.id for message in response.messages if message.interaction
                for option in message.interaction.options
                if option.id.startswith("confirmation:ROOM_SERVICE:") and option.id.endswith(":" + action))


def present_room_confirmation(request):
    state = json.loads(request.conversation.summary)
    assert state["awaitingExplicitConfirmation"]
    offering = next(o for o in request.availableOfferings if o.offeringCode == "ROOM_SERVICE")
    message = _room_service_confirmation_message(request, offering, state["capturedFields"])
    response = _deterministic_turn_response(request, time.perf_counter(), disposition="RESPONSE_READY",
        messages=[message], updated_summary=request.conversation.summary)
    return _bind_room_confirmation(request, response)


def use_presented_room_button(request, action="CONFIRM"):
    response = present_room_confirmation(request)
    request.conversation.summary = response.updatedConversationSummary
    request.conversation.recentMessages[-1].interactionReplyId = current_room_button(response, action)
    return request
