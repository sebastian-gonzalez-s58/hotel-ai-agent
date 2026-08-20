from datetime import datetime, timezone
import unittest
from unittest.mock import patch
from uuid import UUID

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.schemas.v2_turns import AgentTurnResponse
from app.services.idempotency_cache import v2_turn_idempotency_cache


TURN_ID = "10000000-0000-0000-0000-000000000001"
MESSAGE_ID = "20000000-0000-0000-0000-000000000001"


def payload():
    return {
        "schemaVersion": "2.0",
        "agentTurnId": TURN_ID,
        "traceId": "trace-1",
        "trigger": {"type": "INBOUND_MESSAGE", "messageId": MESSAGE_ID},
        "hotel": {
            "hotelId": "30000000-0000-0000-0000-000000000001",
            "hotelCode": "cristalino",
            "name": "Hotel Cristalino",
            "timeZone": "America/Cancun",
            "defaultLanguage": "es-MX",
        },
        "guest": {
            "guestId": "40000000-0000-0000-0000-000000000001",
            "stayId": "50000000-0000-0000-0000-000000000001",
            "displayName": "Sebastian Gonzalez",
            "roomNumber": "Royal Suite",
            "preferredLanguage": "es-MX",
        },
        "conversation": {
            "conversationId": "60000000-0000-0000-0000-000000000001",
            "conversationRouteId": "70000000-0000-0000-0000-000000000001",
            "channel": "WHATSAPP",
            "status": "OPEN",
            "summary": "",
            "recentMessages": [{
                "messageId": MESSAGE_ID,
                "direction": "INBOUND",
                "actor": "GUEST",
                "text": "Hola",
                "createdAt": datetime.now(timezone.utc).isoformat(),
            }],
        },
        "activeOperations": [],
        "availableOfferings": [],
        "toolPolicy": {"allowedTools": ["LIST_AVAILABLE_OFFERINGS"], "maxToolCalls": 2},
        "previousToolResults": [],
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }


class V2TurnEndpointTest(unittest.TestCase):
    def setUp(self):
        self.old_token = settings.agent_internal_token
        self.old_mode = settings.agent_runtime_mode
        settings.agent_internal_token = "v2-test-token"
        settings.agent_runtime_mode = "v2"
        v2_turn_idempotency_cache.clear()
        self.client = TestClient(app)

    def tearDown(self):
        settings.agent_internal_token = self.old_token
        settings.agent_runtime_mode = self.old_mode
        v2_turn_idempotency_cache.clear()

    def headers(self):
        return {
            "Authorization": "Bearer v2-test-token",
            "X-Request-Id": "request-1",
            "X-ChatbotInn-Timestamp": datetime.now(timezone.utc).isoformat(),
            "Idempotency-Key": TURN_ID,
        }

    @patch("app.main.plan_v2_turn")
    def test_returns_and_replays_same_turn(self, planner):
        planner.return_value = AgentTurnResponse(
            schemaVersion="2.0",
            agentTurnId=UUID(TURN_ID),
            disposition="NO_ACTION",
            messages=[],
            toolCalls=[],
            usage={"model": "test", "inputTokens": 1, "cachedInputTokens": 0,
                   "outputTokens": 1, "reasoningTokens": 0, "totalTokens": 2},
            warnings=[],
        )
        request_payload = payload()
        first = self.client.post("/internal/v2/turns", headers=self.headers(), json=request_payload)
        second = self.client.post("/internal/v2/turns", headers=self.headers(), json=request_payload)
        self.assertEqual(200, first.status_code)
        self.assertEqual(first.json(), second.json())
        planner.assert_called_once()

    @patch("app.main.plan_v2_turn")
    def test_rejects_same_turn_id_with_different_payload(self, planner):
        planner.return_value = AgentTurnResponse(
            schemaVersion="2.0",
            agentTurnId=UUID(TURN_ID),
            disposition="NO_ACTION",
            messages=[],
            toolCalls=[],
            usage={"model": "test", "inputTokens": 1, "cachedInputTokens": 0,
                   "outputTokens": 1, "reasoningTokens": 0, "totalTokens": 2},
            warnings=[],
        )
        first_payload = payload()
        second_payload = dict(first_payload)
        second_payload["traceId"] = "different-trace"

        first = self.client.post("/internal/v2/turns", headers=self.headers(), json=first_payload)
        second = self.client.post("/internal/v2/turns", headers=self.headers(), json=second_payload)

        self.assertEqual(200, first.status_code)
        self.assertEqual(409, second.status_code)
        planner.assert_called_once()

    def test_rejects_mismatched_idempotency_key(self):
        headers = self.headers()
        headers["Idempotency-Key"] = "wrong"
        response = self.client.post("/internal/v2/turns", headers=headers, json=payload())
        self.assertEqual(400, response.status_code)

    def test_v2_endpoint_is_disabled_in_legacy_mode(self):
        settings.agent_runtime_mode = "legacy"
        response = self.client.post("/internal/v2/turns", headers=self.headers(), json=payload())
        self.assertEqual(404, response.status_code)


if __name__ == "__main__":
    unittest.main()
