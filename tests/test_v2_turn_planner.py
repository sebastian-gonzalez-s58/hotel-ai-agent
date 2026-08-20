import unittest
from unittest.mock import patch
from uuid import UUID

from app.core.errors import AgentModelError
from app.agents.v2_turn_planner import _validate_plan
from app.schemas.v2_turns import AgentTurnRequest, AgentTurnResponse
from tests.test_v2_turn_endpoint import MESSAGE_ID, TURN_ID, payload


class V2TurnPlannerTest(unittest.TestCase):
    def test_rejects_tool_not_in_policy(self):
        request = AgentTurnRequest.model_validate(payload())
        response = AgentTurnResponse(
            schemaVersion="2.0",
            agentTurnId=UUID(TURN_ID),
            disposition="TOOL_CALLS_REQUIRED",
            messages=[],
            toolCalls=[{
                "toolCallId": "80000000-0000-0000-0000-000000000001",
                "toolName": "START_SERVICE",
                "arguments": {"offeringCode": "spa", "input": {}},
                "confidence": 1,
                "evidenceMessageIds": [],
            }],
            usage={"model": "test", "inputTokens": 0, "cachedInputTokens": 0,
                   "outputTokens": 0, "reasoningTokens": 0, "totalTokens": 0},
            warnings=[],
        )
        with self.assertRaises(AgentModelError):
            _validate_plan(request, response)

    def test_accepts_start_service_for_an_available_offering_with_confirmation(self):
        request_payload = payload()
        request_payload["availableOfferings"] = [offering()]
        request_payload["toolPolicy"] = {"allowedTools": ["START_SERVICE"], "maxToolCalls": 2}
        request = AgentTurnRequest.model_validate(request_payload)
        response = tool_response({
            "toolCallId": "80000000-0000-0000-0000-000000000002",
            "toolName": "START_SERVICE",
            "arguments": {
                "offeringCode": "ROOM_SERVICE",
                "input": {"items": [{"name": "Hamburguesa", "quantity": 1}]},
                "guestConfirmationEvidenceMessageId": MESSAGE_ID,
            },
            "confidence": 1,
            "evidenceMessageIds": [MESSAGE_ID],
        })

        _validate_plan(request, response)

    def test_rejects_start_service_for_an_unknown_offering(self):
        request_payload = payload()
        request_payload["toolPolicy"] = {"allowedTools": ["START_SERVICE"], "maxToolCalls": 2}
        request = AgentTurnRequest.model_validate(request_payload)
        response = tool_response({
            "toolCallId": "80000000-0000-0000-0000-000000000003",
            "toolName": "START_SERVICE",
            "arguments": {"offeringCode": "INVENTED", "input": {}},
            "confidence": 1,
            "evidenceMessageIds": [MESSAGE_ID],
        })

        with self.assertRaises(AgentModelError):
            _validate_plan(request, response)

    def test_accepts_only_an_advertised_service_action_at_the_current_version(self):
        operation_id = "90000000-0000-0000-0000-000000000001"
        request_payload = payload()
        request_payload["activeOperations"] = [operation(operation_id)]
        request_payload["toolPolicy"] = {
            "allowedTools": ["EXECUTE_SERVICE_ACTION"],
            "maxToolCalls": 2,
        }
        request = AgentTurnRequest.model_validate(request_payload)
        response = tool_response({
            "toolCallId": "80000000-0000-0000-0000-000000000004",
            "toolName": "EXECUTE_SERVICE_ACTION",
            "targetOperationId": operation_id,
            "arguments": {
                "operationId": operation_id,
                "actionCode": "CONFIRM",
                "expectedVersion": 3,
                "input": {"confirmed": True},
                "evidenceMessageIds": [MESSAGE_ID],
            },
            "confidence": 1,
            "evidenceMessageIds": [MESSAGE_ID],
        })

        _validate_plan(request, response)

        response.toolCalls[0].arguments["actionCode"] = "INVENTED"
        with self.assertRaises(AgentModelError):
            _validate_plan(request, response)


def tool_response(tool_call):
    return AgentTurnResponse(
        schemaVersion="2.0",
        agentTurnId=UUID(TURN_ID),
        disposition="TOOL_CALLS_REQUIRED",
        messages=[],
        toolCalls=[tool_call],
        usage={"model": "test", "inputTokens": 0, "cachedInputTokens": 0,
               "outputTokens": 0, "reasoningTokens": 0, "totalTokens": 0},
        warnings=[],
    )


def offering():
    return {
        "offeringCode": "ROOM_SERVICE",
        "name": "Servicio a la habitacion",
        "description": "Alimentos y bebidas",
        "executionMode": "PROCESS",
        "inputSchema": {"type": "object"},
        "requiresExplicitGuestConfirmation": True,
    }


def operation(operation_id):
    return {
        "operationId": operation_id,
        "offeringCode": "ROOM_SERVICE",
        "lifecycle": "WAITING_FOR_GUEST",
        "detailedStatus": "WAITING_FOR_CONFIRMATION",
        "summary": "Pedido de room service",
        "availableActions": [{
            "actionCode": "CONFIRM",
            "description": "Confirmar el pedido",
            "inputSchema": {"type": "object"},
            "requiresExplicitGuestConfirmation": True,
        }],
        "pendingConversationTasks": [],
        "version": 3,
    }


if __name__ == "__main__":
    unittest.main()
