import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core.agent_tracking import get_agent_tracking_context
from app.core.config import settings
from app.main import app
from app.schemas.tasks import AgentTaskResponse


class AgentTaskEndpointTest(unittest.TestCase):
    def setUp(self):
        self.previous_token = settings.agent_internal_token
        settings.agent_internal_token = "endpoint-test-token"
        self.client = TestClient(app)

    def tearDown(self):
        settings.agent_internal_token = self.previous_token

    def test_capabilities_requires_internal_token(self):
        response = self.client.get("/hotel/capabilities")
        self.assertEqual(401, response.status_code)

    @patch("app.main.execute_agent_task")
    def test_executes_typed_task_without_consuming_request_body(self, execute_task):
        captured_context = {}

        def execute(_request):
            context = get_agent_tracking_context()
            captured_context["purpose"] = context.purpose
            captured_context["operation_id"] = context.operation_id
            return AgentTaskResponse(
                taskType="CLASSIFY_OFFERING",
                status="COMPLETED",
                offeringCode="spa",
                complete=True,
            )

        execute_task.side_effect = execute

        response = self.client.post(
            "/hotel/tasks",
            headers={"Authorization": "Bearer endpoint-test-token"},
            json={
                "taskType": "CLASSIFY_OFFERING",
                "latestMessage": "quiero reservar un masaje",
                "context": {
                    "operationId": "00000000-0000-0000-0000-000000000002",
                },
                "allowedOfferings": [
                    {"offering": {"code": "spa", "name": "SPA"}},
                ],
            },
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual("spa", response.json()["offeringCode"])
        self.assertEqual("CLASSIFY_OFFERING", captured_context["purpose"])
        self.assertEqual(
            "00000000-0000-0000-0000-000000000002",
            captured_context["operation_id"],
        )
        execute_task.assert_called_once()


if __name__ == "__main__":
    unittest.main()
