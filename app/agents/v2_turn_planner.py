import json
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
MAX_PLAN_ATTEMPTS = 3


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
            + _build_focused_task_repair_instruction(request)
        )

    raise AgentModelError("OpenAI did not return a valid V2 agent plan")


def _build_focused_task_repair_instruction(request: AgentTurnRequest) -> str:
    focused_task_id = request.conversation.focusedConversationTaskId
    if focused_task_id is None:
        return ""

    focused_task = next(
        (
            task
            for operation in request.activeOperations
            for task in operation.pendingConversationTasks
            if task.conversationTaskId == focused_task_id
        ),
        None,
    )
    if focused_task is None:
        return ""

    latest_inbound = next(
        (
            message
            for message in reversed(request.conversation.recentMessages)
            if message.direction == "INBOUND"
        ),
        None,
    )
    if latest_inbound is None:
        return ""

    required_schema = json.dumps(
        focused_task.requiredOutputSchema,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "\nThe turn is answering the focused conversation task. Do not send a guest-facing "
        "acknowledgement yet. Return disposition TOOL_CALLS_REQUIRED, messages [], and exactly "
        "one COMPLETE_CONVERSATION_TASK tool call. Use "
        f"targetOperationId={focused_task.operationId}, "
        f"targetConversationTaskId={focused_task.conversationTaskId}, "
        f"arguments.conversationTaskId={focused_task.conversationTaskId}, "
        f"arguments.expectedVersion={focused_task.version}, and "
        f"evidenceMessageIds=[{latest_inbound.messageId}]. "
        "Infer arguments.result from the guest's latest message and make it satisfy this exact "
        f"requiredOutputSchema: {required_schema}."
    )


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

    successfully_completed_tasks = {
        UUID(str(result.result["conversationTaskId"]))
        for result in request.previousToolResults
        if result.status == "SUCCEEDED"
        and result.toolName == DomainToolName.COMPLETE_CONVERSATION_TASK.value
        and isinstance(result.result, dict)
        and result.result.get("conversationTaskId")
    }
    for message in response.messages:
        referenced_tasks = set(message.conversationTaskIds)
        if not referenced_tasks.issubset(task_ids):
            raise AgentModelError(
                "Agent message references a conversation task outside the turn context"
            )
        if not referenced_tasks.issubset(successfully_completed_tasks):
            raise AgentModelError(
                "Complete the referenced conversation task before acknowledging it"
            )


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

    capture = schema.get("x-chatbotinn-capture")
    if not isinstance(capture, dict):
        return
    input_mode = str(capture.get("inputMode") or "AUTO").upper()
    allowed_codes = _capture_option_codes(capture)
    if input_mode == "SINGLE_SELECT" and allowed_codes and value not in allowed_codes:
        raise AgentModelError(
            f"START_SERVICE offering field {field} must use a configured option code"
        )
    if input_mode == "MULTI_SELECT" and allowed_codes:
        if not isinstance(value, list) or any(item not in allowed_codes for item in value):
            raise AgentModelError(
                f"START_SERVICE offering field {field} must use configured option codes"
            )


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
    _remove_maintenance_issue_interactions(request, messages)
    if _ensure_configured_field_capture(request, normalized, messages):
        return normalized
    if _is_greeting_turn(request) and not request.previousToolResults:
        # A greeting opens a navigation turn. Historical operations remain available
        # for later status questions, but they can never trigger side effects here.
        normalized["disposition"] = "RESPONSE_READY"
        normalized["toolCalls"] = []
        _ensure_personalized_service_menu(request, messages)
        return normalized
    if _successful_started_operations(request):
        # A successful lifecycle result is authoritative. The same guest turn must
        # acknowledge it instead of proposing START_SERVICE again.
        normalized["disposition"] = "RESPONSE_READY"
        normalized["toolCalls"] = []
        _ensure_service_start_acknowledgements(request, messages)
        return normalized
    if normalized.get("disposition") == "RESPONSE_READY":
        _ensure_personalized_service_menu(request, messages)
        _ensure_service_start_acknowledgements(request, messages)
        if messages:
            normalized["disposition"] = "RESPONSE_READY"
    return normalized


def _ensure_configured_field_capture(
    request: AgentTurnRequest,
    normalized: dict,
    messages: list[dict],
) -> bool:
    latest_inbound = _latest_inbound_message(request)
    if latest_inbound is None or request.previousToolResults:
        return False

    selection = _capture_selection(request, latest_inbound)
    if selection is None:
        return False
    offering, completed_field = selection
    fields = _ordered_guest_capture_fields(offering)
    if not fields:
        return False

    field_index = 0
    if completed_field is not None:
        matching_index = next(
            (index for index, (code, _) in enumerate(fields) if code == completed_field),
            None,
        )
        if matching_index is None or matching_index + 1 >= len(fields):
            return False
        field_index = matching_index + 1

    field_code, field_schema = fields[field_index]
    capture = field_schema.get("x-chatbotinn-capture")
    if not isinstance(capture, dict):
        return False
    message = _capture_message(request, offering.offeringCode, field_code, field_schema, capture)
    if message is None:
        return False

    messages[:] = [message]
    normalized["disposition"] = "RESPONSE_READY"
    normalized["toolCalls"] = []
    return True


def _capture_selection(request: AgentTurnRequest, latest_inbound):
    reply_id = (latest_inbound.interactionReplyId or "").strip()
    offerings = {offering.offeringCode: offering for offering in request.availableOfferings}
    if reply_id.startswith("offering:"):
        offering_code = reply_id.split(":", 1)[1]
        offering = offerings.get(offering_code)
        return (offering, None) if offering is not None else None

    if reply_id.startswith("field:"):
        parts = reply_id.split(":", 3)
        if len(parts) != 4:
            return None
        _, offering_code, field_code, option_code = parts
        offering = offerings.get(offering_code)
        if offering is None:
            return None
        properties = offering.inputSchema.get("properties")
        field_schema = properties.get(field_code) if isinstance(properties, dict) else None
        if not isinstance(field_schema, dict):
            return None
        capture = field_schema.get("x-chatbotinn-capture")
        if not isinstance(capture, dict) or option_code not in _capture_option_codes(capture):
            return None
        return offering, field_code

    normalized_text = _normalized_phrase(latest_inbound.text)
    for offering in request.availableOfferings:
        if normalized_text in {
            _normalized_phrase(offering.offeringCode.replace("_", " ")),
            _normalized_phrase(offering.name),
        }:
            return offering, None
    return None


def _ordered_guest_capture_fields(offering) -> list[tuple[str, dict]]:
    schema = offering.inputSchema
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    fields = []
    for position, field_code in enumerate(required):
        field_schema = properties.get(field_code)
        if not isinstance(field_schema, dict):
            continue
        source = str(field_schema.get("x-source") or "GUEST").upper()
        capture = field_schema.get("x-chatbotinn-capture")
        if source == "STAY" or not isinstance(capture, dict):
            continue
        order = capture.get("displayOrder")
        fields.append(
            (order if isinstance(order, int) else position, position, field_code, field_schema)
        )
    fields.sort(key=lambda item: (item[0], item[1]))
    return [(field_code, field_schema) for _, _, field_code, field_schema in fields]


def _capture_message(
    request: AgentTurnRequest,
    offering_code: str,
    field_code: str,
    field_schema: dict,
    capture: dict,
) -> dict | None:
    input_mode = str(capture.get("inputMode") or "AUTO").upper()
    if input_mode == "AUTO":
        return None
    spanish = request.guest.preferredLanguage.lower().startswith("es")
    title = str(field_schema.get("title") or field_code).strip()
    text = str(
        capture.get("introMessage")
        or field_schema.get("description")
        or (
            f"Por favor, indica {title.lower()}."
            if spanish
            else f"Please provide {title.lower()}."
        )
    ).strip()
    interaction = None

    if input_mode == "SINGLE_SELECT":
        options = _capture_options(capture)
        if not options:
            return None
        interaction_options = [
            {
                "id": f"field:{offering_code}:{field_code}:{option['code']}",
                "label": str(option["label"])[:24],
            }
            for option in options[:10]
        ]
        interaction = {
            "type": "BUTTONS" if len(interaction_options) <= 3 else "LIST",
            "title": title[:60],
            "body": text[:1024],
            "buttonText": "Ver opciones" if spanish else "View options",
            "options": interaction_options,
        }
    elif input_mode == "MULTI_SELECT":
        labels = [str(option["label"]) for option in _capture_options(capture)]
        if labels:
            choices = ", ".join(labels)
            suffix = (
                f" Opciones disponibles: {choices}. Indica todas las que deseas."
                if spanish
                else f" Available options: {choices}. Tell us every option you want."
            )
            text = (text + suffix)[:20000]
    elif input_mode == "CATALOG_ITEMS":
        catalog = capture.get("catalog")
        external_url = catalog.get("externalUrl") if isinstance(catalog, dict) else None
        if isinstance(external_url, str) and external_url.strip() and external_url not in text:
            text = f"{text}\n{external_url.strip()}"

    return {
        "messageDraftId": str(uuid4()),
        "purpose": "CLARIFICATION",
        "text": text,
        "language": request.guest.preferredLanguage,
        "operationIds": [],
        "conversationTaskIds": [],
        "interaction": interaction,
    }


def _capture_options(capture: dict) -> list[dict]:
    catalog = capture.get("catalog")
    raw_options = catalog.get("options") if isinstance(catalog, dict) else None
    if not isinstance(raw_options, list):
        return []
    return [
        option for option in raw_options
        if isinstance(option, dict)
        and isinstance(option.get("code"), str)
        and option["code"].strip()
        and isinstance(option.get("label"), str)
        and option["label"].strip()
    ]


def _capture_option_codes(capture: dict) -> set[str]:
    return {str(option["code"]) for option in _capture_options(capture)}


def _latest_inbound_message(request: AgentTurnRequest):
    return next(
        (
            message
            for message in reversed(request.conversation.recentMessages)
            if message.direction == "INBOUND"
        ),
        None,
    )


def _normalized_phrase(value: str) -> str:
    return " ".join(value.casefold().strip().replace("_", " ").split()).strip("!?., ")


def _remove_maintenance_issue_interactions(
    request: AgentTurnRequest,
    messages: list[dict],
) -> None:
    latest_inbound = next(
        (
            message
            for message in reversed(request.conversation.recentMessages)
            if message.direction == "INBOUND"
        ),
        None,
    )
    if latest_inbound is None:
        return

    reply_id = (latest_inbound.interactionReplyId or "").strip().casefold()
    normalized_text = " ".join(latest_inbound.text.casefold().strip().split()).strip("!?., ")
    selected_maintenance = reply_id == "offering:maintenance" or normalized_text in {
        "maintenance",
        "mantenimiento",
        "servicio de mantenimiento",
    }
    if not selected_maintenance:
        return

    # The issue is an unconstrained guest description. Model-generated categories
    # would discard useful details and can route the request incorrectly.
    for message in messages:
        message["interaction"] = None


def _ensure_personalized_service_menu(request: AgentTurnRequest, messages: list[dict]) -> None:
    if not request.availableOfferings or not _is_greeting_turn(request):
        return
    if not messages:
        messages.append({
            "messageDraftId": str(uuid4()),
            "purpose": "ANSWER",
            "text": "",
            "language": request.guest.preferredLanguage,
            "operationIds": [],
            "conversationTaskIds": [],
            "interaction": None,
        })
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
    started = _successful_started_operations(request)
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
        offering_code = str(operation.get("offeringCode") or "")
        offering_name = offering_names.get(offering_code, offering_code.replace("_", " ").title())
        text = (
            f"La solicitud de {offering_name.lower()} ha sido iniciada con el folio {reference}. "
            "Recibirás actualizaciones por este medio; por favor, mantente atento."
            if spanish
            else f"The {offering_name.lower()} request has been started with reference {reference}. "
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


def _successful_started_operations(request: AgentTurnRequest) -> list[dict]:
    return [
        result.result
        for result in request.previousToolResults
        if result.status == "SUCCEEDED" and result.toolName == DomainToolName.START_SERVICE.value
        and isinstance(result.result, dict) and result.result.get("referenceCode")
    ]


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
