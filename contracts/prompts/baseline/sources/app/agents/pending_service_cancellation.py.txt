"""Cancel an unsubmitted capture, never an existing operation or staff task."""
import json
from uuid import uuid4

from app.services.localized_content import template

TOOL = "CANCEL_GUEST_WEB_SESSION"


def cleared_state(state, code):
    state = dict(state)
    for key in ("pendingOffering", "phase", "capturedFields", "catalogPending",
                "awaitingExplicitConfirmation", "readyToStart"):
        state.pop(key, None)
    return json.dumps(state, ensure_ascii=False)


def reply(request, key, summary):
    return {"disposition": "RESPONSE_READY", "messages": [{"purpose": "ANSWER",
        "text": template(key, request.guest.preferredLanguage),
        "language": request.guest.preferredLanguage, "operationIds": [], "conversationTaskIds": []}],
        "updated_summary": summary}


def plan_pending_cancellation(request, latest, state, scope=None):
    results = [r for r in request.previousToolResults if r.toolName == TOOL]
    if results:
        result = results[-1]
        status = (result.result or {}).get("status") if result.status == "SUCCEEDED" else None
        if status == "CANCELLED":
            return reply(request, "draft.cancelled", cleared_state(state, state.get("pendingOffering")))
        if status == "ALREADY_SUBMITTED":
            return reply(request, "draft.already_submitted", cleared_state(state, state.get("pendingOffering")))
        return reply(request, "draft.cancel_failed", request.conversation.summary)
    if request.previousToolResults or latest is None or latest.interactionReplyId or scope is None:
        return None
    code = state.get("pendingOffering")
    if (not code or request.conversation.focusedConversationTaskId
            or scope.kind != "CONTEXT_REPLY" or scope.separateRequest
            or scope.offeringCode not in (None, code) or scope.replyAction != "CANCEL"
            or scope.replyActionConfidence < 0.85 or scope.replyActionEvidence != latest.text
            or scope.containsUnrelatedTopic):
        return None
    if any(o.offeringCode == code and o.pendingConversationTasks for o in request.activeOperations):
        return None
    offering = next((o for o in request.availableOfferings if o.offeringCode == code), None)
    if offering is None:
        return None
    if offering.guestExperience in ("WEB_ORDER", "WEB_BOOKING"):
        if TOOL not in request.toolPolicy.allowedTools:
            return reply(request, "draft.cancel_failed", request.conversation.summary)
        return {"disposition": "TOOL_CALLS_REQUIRED", "messages": [], "tool_calls": [{
            "toolCallId": str(uuid4()), "toolName": TOOL,
            "targetOperationId": None, "targetConversationTaskId": None,
            "arguments": {"offeringCode": code}, "confidence": scope.replyActionConfidence,
            "evidenceMessageIds": [str(latest.messageId)]}],
            "updated_summary": request.conversation.summary}
    # Existing structured order/SPA drafts own their confirmation-token history.
    if code in ("ROOM_SERVICE", "SPA"):
        return None
    return reply(request, "draft.cancelled", cleared_state(state, code))
