import time
from uuid import UUID

from pydantic import ValidationError

from app.core.config import settings
from app.core.errors import AgentModelError
from app.prompts.v2_turn import build_v2_turn_prompt
from app.schemas.v2_turns import AgentTurnRequest, AgentTurnResponse
from app.schemas.v2_turns import DomainToolName
from app.services.openai_client import call_openai_json_result


def plan_v2_turn(request: AgentTurnRequest) -> AgentTurnResponse:
    started_at = time.perf_counter()
    result = call_openai_json_result(build_v2_turn_prompt(request), purpose="V2_AGENT_TURN")
    payload = dict(result.payload)
    payload["usage"] = {
        "model": settings.openai_model,
        **result.usage.as_api_dict(),
        "latencyMs": round((time.perf_counter() - started_at) * 1000),
    }
    try:
        response = AgentTurnResponse.model_validate(payload)
    except ValidationError as exc:
        raise AgentModelError("OpenAI returned an invalid V2 agent turn") from exc
    _validate_plan(request, response)
    return response


def _validate_plan(request: AgentTurnRequest, response: AgentTurnResponse) -> None:
    if response.agentTurnId != request.agentTurnId:
        raise AgentModelError("Agent turn ID does not match the request")

    allowed_tools = set(request.toolPolicy.allowedTools)
    message_ids = {message.messageId for message in request.conversation.recentMessages}
    operation_ids = {operation.operationId for operation in request.activeOperations}
    tasks_by_id = {
        task.conversationTaskId: task
        for operation in request.activeOperations
        for task in operation.pendingConversationTasks
    }
    task_ids = set(tasks_by_id)
    offerings = {offering.offeringCode: offering for offering in request.availableOfferings}
    operations_by_id = {operation.operationId: operation for operation in request.activeOperations}

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
