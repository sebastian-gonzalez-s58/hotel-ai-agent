"""Offline Python validator coverage; does not claim Spring mutation coverage."""
from copy import deepcopy

from app.agents.v2_turn_planner import _validate_plan
from app.core.errors import AgentModelError
from app.schemas.v2_turns import AgentTurnRequest, AgentTurnResponse


EXPECTED_REASON = {
    "forbidden-tool": "is not allowed",
    "unknown-offering": "unavailable offering",
    "stale-version": "stale operation version",
    "foreign-evidence": "evidence outside",
    "missing-input": "requires a non-empty value",
    "missing-confirmation": "confirmation evidence",
}


def validation_case(case, profile):
    step = case.steps[0]
    data = step.event.data
    payload = deepcopy(profile)
    payload["activeOperations"] = case.initial_operations
    payload["trigger"]["messageId"] = data["request_message"]["messageId"]
    payload["conversation"]["recentMessages"] = [{
        **data["request_message"], "direction": "INBOUND", "actor": "GUEST",
        "createdAt": payload["createdAt"],
    }]
    if "allowed_tools" in data:
        payload["toolPolicy"]["allowedTools"] = data["allowed_tools"]
    request = AgentTurnRequest.model_validate(payload)
    response = AgentTurnResponse.model_validate({
        "schemaVersion": "2.0", "agentTurnId": request.agentTurnId, "disposition": "TOOL_CALLS_REQUIRED",
        "messages": [], "toolCalls": [data["tool_call"]], "warnings": [],
        "usage": {"model": "synthetic", "inputTokens": 0, "cachedInputTokens": 0, "outputTokens": 0,
                  "reasoningTokens": 0, "totalTokens": 0},
    })
    reason = None
    try:
        _validate_plan(request, response)
    except AgentModelError as error:
        reason = str(error)
    passed = reason is not None and EXPECTED_REASON[data["violation"]] in reason
    return {"id": case.id, "scenario_id": case.scenario_id,
            "status": "OFFLINE_PARTIAL_PASS" if passed else "FAILED",
            "coverage": "Python validator only; Spring rejection and mutations unevaluated",
            "turns": [{"id": step.id, "rejection_reason": reason,
                       "errors": [] if passed else [{"expected": EXPECTED_REASON[data["violation"]], "actual": reason}],
                       "checks_pending": [c.model_dump() for c in step.checks if c.path.startswith("/backend/")]}]}
