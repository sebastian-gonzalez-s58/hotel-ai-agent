import unittest
from unittest.mock import patch
from uuid import UUID

from app.core.errors import AgentModelError
from app.agents.v2_turn_planner import _validate_plan
from app.schemas.v2_turns import AgentTurnRequest, AgentTurnResponse
from tests.test_v2_turn_endpoint import TURN_ID, payload


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


if __name__ == "__main__":
    unittest.main()
