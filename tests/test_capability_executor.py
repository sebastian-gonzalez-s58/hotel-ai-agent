import unittest
from unittest.mock import patch

from app.agents.capability_executor import execute_agent_task
from app.schemas.tasks import AgentTaskRequest
from app.services.openai_client import OpenAiJsonResult
from app.services.telemetry_client import OpenAiTokenUsage


class CapabilityExecutorTest(unittest.TestCase):
    @patch("app.agents.capability_executor.resolve_authorized_resources")
    @patch("app.agents.capability_executor.call_openai_json_result")
    def test_rejects_offering_outside_allowed_set(self, openai_call, resources):
        resources.return_value = {
            "allowedOfferings": [{"offering": {"code": "spa"}}],
            "offering": None,
            "catalogMatches": [],
            "knowledgeMatches": [],
        }
        openai_call.return_value = _result(
            {
                "status": "COMPLETED",
                "offeringCode": "room-service",
                "confidence": 0.99,
                "complete": True,
            }
        )

        response = execute_agent_task(
            AgentTaskRequest(
                taskType="CLASSIFY_OFFERING",
                latestMessage="quiero comida",
            )
        )

        self.assertEqual("UNRESOLVED", response.status)
        self.assertIsNone(response.offeringCode)
        self.assertFalse(response.complete)

    @patch("app.agents.capability_executor.resolve_authorized_resources")
    @patch("app.agents.capability_executor.call_openai_json_result")
    def test_discards_fields_not_declared_by_offering(self, openai_call, resources):
        offering = {
            "requiredFields": [
                {"code": "reservation_date", "required": True},
            ]
        }
        resources.return_value = {
            "allowedOfferings": [],
            "offering": offering,
            "catalogMatches": [],
            "knowledgeMatches": [],
        }
        openai_call.return_value = _result(
            {
                "status": "COMPLETED",
                "extractedValues": [
                    {"fieldCode": "reservation_date", "value": "2026-08-10"},
                    {"fieldCode": "secret_note", "value": "invented"},
                ],
                "complete": True,
            }
        )

        response = execute_agent_task(
            AgentTaskRequest(
                taskType="EXTRACT_REQUIREMENT_VALUES",
                offering=offering,
                latestMessage="el diez de agosto",
            )
        )

        self.assertEqual(1, len(response.extractedValues))
        self.assertEqual("reservation_date", response.extractedValues[0].fieldCode)

    @patch("app.agents.capability_executor.resolve_authorized_resources")
    @patch("app.agents.capability_executor.call_openai_json_result")
    def test_discards_catalog_ids_not_returned_by_authorized_search(self, openai_call, resources):
        resources.return_value = {
            "allowedOfferings": [],
            "offering": None,
            "catalogMatches": [
                {"item": {"item": {"id": "allowed-item"}}},
            ],
            "knowledgeMatches": [],
        }
        openai_call.return_value = _result(
            {
                "status": "COMPLETED",
                "catalogSelections": [
                    {"itemId": "made-up-item", "quantity": 1},
                ],
                "complete": True,
            }
        )

        response = execute_agent_task(
            AgentTaskRequest(
                taskType="MATCH_CATALOG_ITEMS",
                latestMessage="una hamburguesa",
            )
        )

        self.assertEqual([], response.catalogSelections)
        self.assertFalse(response.complete)


def _result(payload):
    return OpenAiJsonResult(
        payload=payload,
        usage=OpenAiTokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        response_id="resp-test",
    )


if __name__ == "__main__":
    unittest.main()
