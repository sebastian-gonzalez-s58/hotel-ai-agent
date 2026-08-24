import logging
import time
from uuid import UUID, uuid4

from pydantic import ValidationError

from app.core.config import settings
from app.core.errors import AgentModelError
from app.prompts.v2_turn import build_v2_turn_prompt
from app.schemas.v2_turns import AgentTurnRequest, AgentTurnResponse
from app.schemas.v2_turns import DomainToolName
from app.services.openai_client import call_openai_json_result


logger = logging.getLogger("chatbotinn-agent.v2-turn-planner")
AGENT_TURN_RESPONSE_SCHEMA = AgentTurnResponse.model_json_schema()
MAX_PLAN_ATTEMPTS = 2


def plan_v2_turn(request: AgentTurnRequest) -> AgentTurnResponse:
    started_at = time.perf_counter()
    prompt = build_v2_turn_prompt(request)
    for attempt in range(1, MAX_PLAN_ATTEMPTS + 1):
        result = call_openai_json_result(
            prompt,
            purpose="V2_AGENT_TURN",
            response_schema=AGENT_TURN_RESPONSE_SCHEMA,
            response_schema_name="agent_turn_response_v2",
        )
        payload = _normalize_response_envelope(request, result.payload)
        payload = _normalize_guest_experience(request, payload)
        payload["usage"] = {
            "model": settings.openai_model,
            **result.usage.as_api_dict(),
            "latencyMs": round((time.perf_counter() - started_at) * 1000),
        }
        try:
            response = AgentTurnResponse.model_validate(payload)
            _validate_plan(request, response)
            return response
        except ValidationError as exc:
            error = AgentModelError(
                f"OpenAI returned an invalid V2 agent turn schema response_id={result.response_id}"
            )
            details = exc.errors(include_url=False, include_input=False)
        except AgentModelError as exc:
            error = exc
            details = str(exc)

        logger.warning(
            "Invalid V2 agent plan; retrying when possible agent_turn_id=%s response_id=%s "
            "attempt=%s/%s error=%s",
            request.agentTurnId,
            result.response_id,
            attempt,
            MAX_PLAN_ATTEMPTS,
            details,
        )
        if attempt == MAX_PLAN_ATTEMPTS:
            raise error
        prompt += (
            "\n\nThe previous plan was invalid and must not be repeated. "
            f"Validation error: {details}. Return a corrected plan."
        )

    raise AgentModelError("OpenAI did not return a valid V2 agent plan")


def _normalize_response_envelope(
    request: AgentTurnRequest,
    model_payload: dict,
) -> dict:
    payload = dict(model_payload)
    # These fields are protocol metadata, not model decisions.
    payload["schemaVersion"] = "2.0"
    payload["agentTurnId"] = str(request.agentTurnId)
    payload.setdefault("detectedLanguage", None)
    payload.setdefault("messages", [])
    payload.setdefault("toolCalls", [])
    payload.setdefault("updatedConversationSummary", None)
    payload.setdefault("warnings", [])
    for message in payload["messages"]:
        if isinstance(message, dict):
            message["messageDraftId"] = str(uuid4())
    for tool_call in payload["toolCalls"]:
        if isinstance(tool_call, dict):
            tool_call["toolCallId"] = str(uuid4())
    return payload


def _validate_plan(request: AgentTurnRequest, response: AgentTurnResponse) -> None:
    if response.agentTurnId != request.agentTurnId:
        raise AgentModelError("Agent turn ID does not match the request")

    allowed_tools = set(request.toolPolicy.allowedTools)
    message_ids = {message.messageId for message in request.conversation.recentMessages}
    all_operations = _all_operations(request)
    operation_ids = {operation.operationId for operation in all_operations}
    tasks_by_id = {
        task.conversationTaskId: task
        for operation in request.activeOperations
        for task in operation.pendingConversationTasks
    }
    task_ids = set(tasks_by_id)
    offerings = {offering.offeringCode: offering for offering in request.availableOfferings}
    operations_by_id = {operation.operationId: operation for operation in all_operations}

    if len(response.toolCalls) > request.toolPolicy.maxToolCalls:
        raise AgentModelError("Agent exceeded the tool-call limit")
    if response.disposition == "TOOL_CALLS_REQUIRED" and not response.toolCalls:
        raise AgentModelError("TOOL_CALLS_REQUIRED requires tool calls")
    if response.disposition != "TOOL_CALLS_REQUIRED" and response.toolCalls:
        raise AgentModelError("Only TOOL_CALLS_REQUIRED may contain tool calls")
    if response.disposition in {"RESPONSE_READY", "HANDOFF_REQUIRED"} and not response.messages:
        raise AgentModelError("The disposition requires a guest-facing message")
    if response.disposition == "NO_ACTION" and response.messages:
        raise AgentModelError("NO_ACTION cannot contain messages")

    completed_tasks: set[UUID] = set()
    for call in response.toolCalls:
        if call.toolName not in allowed_tools:
            raise AgentModelError(f"Tool {call.toolName.value} is not allowed")
        if call.targetOperationId is not None and call.targetOperationId not in operation_ids:
            raise AgentModelError("Tool call targets an operation outside the turn context")
        if call.targetConversationTaskId is not None and call.targetConversationTaskId not in task_ids:
            raise AgentModelError("Tool call targets a conversation task outside the turn context")
        if not set(call.evidenceMessageIds).issubset(message_ids):
            raise AgentModelError("Tool call contains evidence outside the turn context")
        _validate_lifecycle_call(call, offerings, operations_by_id)
        _validate_status_call(call, offerings)
        _validate_conversation_task_call(call, tasks_by_id)
        if call.toolName.value == "COMPLETE_CONVERSATION_TASK" and call.targetConversationTaskId:
            completed_tasks.add(call.targetConversationTaskId)

    if not request.toolPolicy.allowMultipleConversationTaskCompletions and len(completed_tasks) > 1:
        raise AgentModelError("Multiple conversation-task completions are not allowed")


def _validate_lifecycle_call(call, offerings, operations_by_id) -> None:
    if call.toolName == DomainToolName.START_SERVICE:
        if call.targetOperationId is not None:
            raise AgentModelError("START_SERVICE cannot target an existing operation")
        offering_code = call.arguments.get("offeringCode")
        offering = offerings.get(offering_code)
        if offering is None:
            raise AgentModelError("START_SERVICE references an unavailable offering")
        if not isinstance(call.arguments.get("input"), dict):
            raise AgentModelError("START_SERVICE input must be an object")
        _validate_offering_input(
            call.arguments["input"],
            offering.inputSchema,
            offering.offeringCode,
        )
        if not call.evidenceMessageIds:
            raise AgentModelError("START_SERVICE requires guest evidence")
        if offering.requiresExplicitGuestConfirmation:
            confirmation = _argument_uuid(
                call.arguments.get("guestConfirmationEvidenceMessageId"),
                "START_SERVICE confirmation evidence",
            )
            if confirmation not in set(call.evidenceMessageIds):
                raise AgentModelError("START_SERVICE confirmation evidence is not declared")

    if call.toolName == DomainToolName.EXECUTE_SERVICE_ACTION:
        if call.targetOperationId is None:
            raise AgentModelError("EXECUTE_SERVICE_ACTION requires targetOperationId")
        operation = operations_by_id.get(call.targetOperationId)
        if operation is None:
            raise AgentModelError("EXECUTE_SERVICE_ACTION targets an unavailable operation")
        argument_operation_id = _argument_uuid(
            call.arguments.get("operationId"),
            "EXECUTE_SERVICE_ACTION operationId",
        )
        if argument_operation_id != call.targetOperationId:
            raise AgentModelError("EXECUTE_SERVICE_ACTION operation IDs do not match")
        if call.arguments.get("expectedVersion") != operation.version:
            raise AgentModelError("EXECUTE_SERVICE_ACTION uses a stale operation version")
        if not isinstance(call.arguments.get("input", {}), dict):
            raise AgentModelError("EXECUTE_SERVICE_ACTION input must be an object")
        action_code = call.arguments.get("actionCode")
        action = next(
            (candidate for candidate in operation.availableActions
             if candidate.actionCode == action_code),
            None,
        )
        if action is None:
            raise AgentModelError("EXECUTE_SERVICE_ACTION references an unavailable action")
        argument_evidence = {
            _argument_uuid(value, "EXECUTE_SERVICE_ACTION evidence")
            for value in call.arguments.get("evidenceMessageIds", [])
        }
        if argument_evidence != set(call.evidenceMessageIds):
            raise AgentModelError("EXECUTE_SERVICE_ACTION evidence declarations do not match")
        if action.requiresExplicitGuestConfirmation and not call.evidenceMessageIds:
            raise AgentModelError("EXECUTE_SERVICE_ACTION requires explicit guest evidence")


def _validate_offering_input(value: dict, schema: dict, offering_code: str) -> None:
    if schema.get("type") not in {None, "object"}:
        raise AgentModelError(f"Offering {offering_code} input schema must describe an object")

    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    for field in required:
        field_value = value.get(field)
        if _is_missing_required_value(field_value):
            raise AgentModelError(
                f"START_SERVICE requires a non-empty value for offering field {field}"
            )
        field_schema = properties.get(field)
        if isinstance(field_schema, dict):
            _validate_schema_value(field_value, field_schema, field)


def _validate_schema_value(value, schema: dict, field: str) -> None:
    expected = schema.get("type")
    valid = {
        "string": isinstance(value, str) and bool(value.strip()),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "array": isinstance(value, list) and bool(value),
        "object": isinstance(value, dict) and bool(value),
    }.get(expected, True)
    if not valid:
        raise AgentModelError(f"START_SERVICE offering field {field} must be {expected}")


def _is_missing_required_value(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or (
        isinstance(value, (list, dict)) and not value
    )


def _validate_status_call(call, offerings) -> None:
    if call.toolName != DomainToolName.GET_OPERATION_STATUS:
        return
    reference_code = call.arguments.get("referenceCode")
    offering_code = call.arguments.get("offeringCode")
    if not isinstance(reference_code, str) and not isinstance(offering_code, str):
        raise AgentModelError("GET_OPERATION_STATUS requires referenceCode or offeringCode")
    if isinstance(offering_code, str) and offering_code not in offerings:
        raise AgentModelError("GET_OPERATION_STATUS references an unavailable offering")


def _all_operations(request: AgentTurnRequest):
    operations = {operation.operationId: operation for operation in request.recentOperations}
    operations.update({operation.operationId: operation for operation in request.activeOperations})
    return list(operations.values())


def _normalize_guest_experience(request: AgentTurnRequest, payload: dict) -> dict:
    normalized = dict(payload)
    messages = [dict(message) for message in normalized.get("messages", []) if isinstance(message, dict)]
    normalized["messages"] = messages
    if normalized.get("disposition") == "RESPONSE_READY":
        _ensure_personalized_service_menu(request, messages)
        _ensure_service_start_acknowledgements(request, messages)
        if messages:
            normalized["disposition"] = "RESPONSE_READY"
    return normalized


def _ensure_personalized_service_menu(request: AgentTurnRequest, messages: list[dict]) -> None:
    if not messages or not request.availableOfferings or not _is_greeting_turn(request):
        return
    message = messages[0]
    display_name = request.guest.displayName.strip()
    first_name = display_name.split()[0] if display_name else ""
    spanish = request.guest.preferredLanguage.lower().startswith("es")
    if spanish:
        salutation = f"Hola, {first_name}. " if first_name else "Hola. "
        text = (
            salutation
            + "\u00bfC\u00f3mo podemos ayudarte hoy? Por favor, elige una opci\u00f3n del men\u00fa."
        )
    else:
        salutation = f"Hello, {first_name}. " if first_name else "Hello. "
        text = salutation + "How can we help you today? Please choose an option from the menu."

    options = [
        {"id": f"offering:{offering.offeringCode}", "label": offering.name[:24]}
        for offering in request.availableOfferings[:10]
    ]
    message["text"] = text
    message["interaction"] = {
        "type": "BUTTONS" if len(options) <= 3 else "LIST",
        "title": "Servicios del hotel" if spanish else "Hotel services",
        "body": text,
        "buttonText": "Ver servicios" if spanish else "View services",
        "options": options,
    }


def _ensure_service_start_acknowledgements(request: AgentTurnRequest, messages: list[dict]) -> None:
    started = [
        result.result
        for result in request.previousToolResults
        if result.status == "SUCCEEDED" and result.toolName == DomainToolName.START_SERVICE.value
        and isinstance(result.result, dict) and result.result.get("referenceCode")
    ]
    if not started:
        return
    spanish = request.guest.preferredLanguage.lower().startswith("es")
    offering_names = {
        offering.offeringCode: offering.name for offering in request.availableOfferings
    }
    acknowledgements: list[dict] = []
    for operation in started:
        reference = str(operation["referenceCode"])
        operation_id = str(operation.get("operationId") or "")
        existing = next((
            message for message in messages
            if reference.casefold() in str(message.get("text") or "").casefold()
        ), None)
        if existing:
            acknowledgements.append(existing)
            continue
        offering_code = str(operation.get("offeringCode") or "")
        offering_name = offering_names.get(offering_code, offering_code.replace("_", " ").title())
        text = (
            f"Se inició tu solicitud de {offering_name} con el folio {reference}. "
            "Te enviaremos las actualizaciones por este medio; por favor, mantente atento."
            if spanish
            else f"Your {offering_name} request was started with reference {reference}. "
            "We will send updates through this channel; please keep an eye on your messages."
        )
        linked_operations = [operation_id] if operation_id else []
        acknowledgements.append({
            "messageDraftId": str(uuid4()),
            "purpose": "STATUS_UPDATE",
            "text": text,
            "language": request.guest.preferredLanguage,
            "operationIds": linked_operations,
            "conversationTaskIds": [],
            "interaction": None,
        })
    messages[:] = acknowledgements


def _is_greeting_turn(request: AgentTurnRequest) -> bool:
    inbound = [
        message for message in request.conversation.recentMessages
        if message.direction == "INBOUND"
    ]
    if not inbound:
        return False
    text = " ".join(inbound[-1].text.casefold().strip().split()).strip("!?., ")
    return text in {
        "hola", "hello", "hi", "hey", "buen dia", "buen día", "buenos dias",
        "buenos días", "buenas tardes", "buenas noches", "que tal", "qué tal",
    }


def _validate_conversation_task_call(call, tasks_by_id) -> None:
    if call.toolName not in {
        DomainToolName.SAVE_CONVERSATION_TASK_PROGRESS,
        DomainToolName.COMPLETE_CONVERSATION_TASK,
    }:
        return
    if call.targetConversationTaskId is None:
        raise AgentModelError(f"{call.toolName.value} requires targetConversationTaskId")

    task = tasks_by_id.get(call.targetConversationTaskId)
    if task is None:
        raise AgentModelError("Conversation-task tool targets an unavailable task")
    argument_task_id = _argument_uuid(
        call.arguments.get("conversationTaskId"),
        f"{call.toolName.value} conversationTaskId",
    )
    if argument_task_id != call.targetConversationTaskId:
        raise AgentModelError("Conversation-task IDs do not match")
    if call.targetOperationId is not None and call.targetOperationId != task.operationId:
        raise AgentModelError("Conversation task belongs to another operation")
    if call.arguments.get("expectedVersion") != task.version:
        raise AgentModelError("Conversation-task tool uses a stale task version")
    if not call.evidenceMessageIds:
        raise AgentModelError("Conversation-task tool requires guest evidence")

    payload_name = (
        "partialResult"
        if call.toolName == DomainToolName.SAVE_CONVERSATION_TASK_PROGRESS
        else "result"
    )
    if not isinstance(call.arguments.get(payload_name), dict):
        raise AgentModelError(f"{call.toolName.value} {payload_name} must be an object")


def _argument_uuid(value, field: str) -> UUID:
    try:
        return UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise AgentModelError(f"{field} must be a UUID") from exc
