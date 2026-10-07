"""Open the server calendar; selecting a service in chat never books a slot."""
import json
from urllib.parse import urlsplit
from uuid import uuid4

from app.schemas.v2_turns import DomainToolName
from app.services.localized_content import template


def plan_guest_booking_session(request, latest, state, *, selected_code=None, scope=None):
    offerings = {o.offeringCode: o for o in request.availableOfferings if o.guestExperience == "WEB_BOOKING"}
    results = [r for r in request.previousToolResults if r.toolName == "OPEN_GUEST_BOOKING_SESSION"]
    if results:
        result = results[-1]
        data = result.result or {}
        code, url = data.get("offeringCode"), data.get("url")
        try:
            parsed = urlsplit(url) if isinstance(url, str) and not any(c.isspace() for c in url) else None
        except ValueError:
            parsed = None
        valid = (result.status == "SUCCEEDED" and code in offerings and parsed is not None
                 and parsed.scheme in {"https", "http"} and parsed.hostname
                 and not parsed.username and not parsed.password)
        text = template("guest_booking.open", request.guest.preferredLanguage, url=url) if valid else template(
            "guest_booking.unavailable", request.guest.preferredLanguage)
        if valid and data.get("amendment"):
            text = (f"Choose another time or a different service here: {url}. Your selection will return to the hotel for review under the same reference."
                    if request.guest.preferredLanguage.lower().startswith("en") else
                    f"Elige otro horario o un servicio diferente aquí: {url}. Enviaremos tu selección al hotel para revisión con el mismo folio.")
        return {"disposition": "RESPONSE_READY", "messages": [{"purpose": "ANSWER", "text": text,
                "language": request.guest.preferredLanguage, "operationIds": [], "conversationTaskIds": []}],
                "updated_summary": _summary(state, code) if valid and not data.get("amendment") else request.conversation.summary}
    if request.previousToolResults or latest is None:
        return None
    code = selected_code
    if scope and scope.kind == "SERVICE_REQUEST":
        code = scope.offeringCode
    elif scope and scope.kind == "CONTEXT_REPLY" and not request.conversation.focusedConversationTaskId:
        code = state.get("pendingOffering") if state.get("phase") == "WEB_BOOKING" else None
        if any(o.offeringCode == code and o.pendingConversationTasks and not any(t.context.get("guestExperience") == "WEB_BOOKING" for t in o.pendingConversationTasks) for o in request.activeOperations):
            return None
    if scope and scope.kind in {"SERVICE_REQUEST", "CONTEXT_REPLY"}:
        candidates = [(operation, task) for operation in request.activeOperations
                      for task in operation.pendingConversationTasks
                      if task.taskType == "SPA_RESERVATION_CHANGE_DETAILS"
                      and task.context.get("guestExperience") == "WEB_BOOKING"
                      and (operation.offeringCode == code or scope.kind == "CONTEXT_REPLY"
                           and task.conversationTaskId == request.conversation.focusedConversationTaskId)]
        if len(candidates) == 1:
            operation, task = candidates[0]
            if scope.replyAction == "CANCEL":
                return {"disposition": "TOOL_CALLS_REQUIRED", "messages": [], "tool_calls": [{
                    "toolCallId": str(uuid4()), "toolName": "COMPLETE_CONVERSATION_TASK",
                    "targetOperationId": str(operation.operationId), "targetConversationTaskId": str(task.conversationTaskId),
                    "arguments": {"conversationTaskId": str(task.conversationTaskId), "expectedVersion": task.version,
                                  "result": {"decision": "CANCEL"}},
                    "confidence": 1, "evidenceMessageIds": [str(latest.messageId)]}],
                    "updated_summary": request.conversation.summary}
            return {"disposition": "TOOL_CALLS_REQUIRED", "messages": [], "tool_calls": [{
                "toolCallId": str(uuid4()), "toolName": "OPEN_GUEST_BOOKING_SESSION",
                "targetOperationId": str(operation.operationId), "targetConversationTaskId": str(task.conversationTaskId),
                "arguments": {"offeringCode": operation.offeringCode, "language": request.guest.preferredLanguage,
                              "conversationTaskId": str(task.conversationTaskId)},
                "confidence": 1, "evidenceMessageIds": [str(latest.messageId)]}],
                "updated_summary": request.conversation.summary}
    if code not in offerings:
        return None
    if DomainToolName.OPEN_GUEST_BOOKING_SESSION not in request.toolPolicy.allowedTools:
        return {"disposition": "RESPONSE_READY", "messages": [{"purpose": "ANSWER",
                "text": template("guest_booking.unavailable", request.guest.preferredLanguage),
                "language": request.guest.preferredLanguage, "operationIds": [], "conversationTaskIds": []}],
                "updated_summary": request.conversation.summary}
    return {"disposition": "TOOL_CALLS_REQUIRED", "messages": [], "tool_calls": [{
            "toolCallId": str(uuid4()), "toolName": "OPEN_GUEST_BOOKING_SESSION",
            "targetOperationId": None, "targetConversationTaskId": None,
            "arguments": {"offeringCode": code, "language": request.guest.preferredLanguage},
            "confidence": 1, "evidenceMessageIds": [str(latest.messageId)]}],
            "updated_summary": _summary(state, code)}


def _summary(state, code):
    state = {key: value for key, value in state.items() if key != "catalogPending"}
    return json.dumps({**state, "pendingOffering": code, "phase": "WEB_BOOKING", "capturedFields": {},
                       "awaitingExplicitConfirmation": False, "readyToStart": False}, ensure_ascii=False)
