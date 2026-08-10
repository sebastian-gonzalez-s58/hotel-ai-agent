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

    @patch("app.agents.capability_executor.resolve_authorized_resources")
    @patch("app.agents.capability_executor.call_openai_json_result")
    def test_builds_active_offering_menu_with_authorized_action_ids(self, openai_call, resources):
        resources.return_value = {
            "allowedOfferings": [
                {"offering": {"code": "FAQ", "name": "Preguntas frecuentes"}},
                {"offering": {"code": "SPA", "name": "Reservación de spa"}},
                {"offering": {"code": "ROOM_SERVICE", "name": "Servicio a la habitación"}},
                {"offering": {"code": "MAINTENANCE", "name": "Mantenimiento"}},
            ],
            "offering": None,
            "catalogMatches": [],
            "knowledgeMatches": [],
        }
        openai_call.return_value = _result(
            {
                "status": "COMPLETED",
                "message": {
                    "text": "Hola, ¿en qué servicio del hotel puedo ayudarte hoy?",
                    "interaction": {"actions": [{"wrong": "shape"}]},
                },
            }
        )

        response = execute_agent_task(
            AgentTaskRequest(
                taskType="GENERATE_CLARIFICATION",
                latestMessage="Hola",
                context={"language": "es-MX"},
                allowedOfferings=resources.return_value["allowedOfferings"],
                taskConfig={
                    "offerActiveOfferings": True,
                    "menuTitle": "Menú principal",
                    "menuButtonText": "Ver opciones",
                },
            )
        )

        self.assertEqual("NEEDS_CLARIFICATION", response.status)
        self.assertEqual("LIST", response.message.interaction.type)
        self.assertEqual(
            ["FAQ", "SPA", "ROOM_SERVICE", "MAINTENANCE"],
            [action.id for action in response.message.interaction.actions],
        )
        self.assertFalse(response.complete)

    @patch("app.agents.capability_executor.resolve_authorized_resources")
    @patch("app.agents.capability_executor.call_openai_json_result")
    def test_normalizes_common_model_shape_errors(self, openai_call, resources):
        resources.return_value = {
            "allowedOfferings": [],
            "offering": None,
            "catalogMatches": [],
            "knowledgeMatches": [],
        }
        openai_call.return_value = _result(
            {
                "status": "not-a-status",
                "confidence": "not-a-number",
                "message": "Necesito un poco más de información.",
                "missingFieldCodes": "question",
                "evidence": [{"title": "missing required type"}],
            }
        )

        response = execute_agent_task(
            AgentTaskRequest(taskType="GENERATE_CLARIFICATION", latestMessage="Hola")
        )

        self.assertEqual("UNRESOLVED", response.status)
        self.assertIsNone(response.confidence)
        self.assertEqual("Necesito un poco más de información.", response.message.text)
        self.assertEqual([], response.missingFieldCodes)
        self.assertEqual([], response.evidence)

    @patch("app.agents.capability_executor.resolve_authorized_resources")
    @patch("app.agents.capability_executor.call_openai_json_result")
    def test_room_service_confirmation_includes_normalized_order(self, openai_call, resources):
        resources.return_value = {
            "allowedOfferings": [],
            "offering": {"code": "ROOM_SERVICE"},
            "catalogMatches": [],
            "knowledgeMatches": [],
        }
        openai_call.return_value = _result(
            {
                "status": "COMPLETED",
                "message": {
                    "text": "Â¿Desea confirmar, cambiar o cancelar su pedido?",
                    "interaction": {
                        "actions": [
                            {"id": "CONFIRM", "label": "Confirmar"},
                            {"id": "CHANGE", "label": "Cambiar"},
                            {"id": "CANCEL", "label": "Cancelar"},
                        ]
                    },
                },
            }
        )

        response = execute_agent_task(
            AgentTaskRequest(
                taskType="GENERATE_GUEST_CONFIRMATION",
                offeringCode="ROOM_SERVICE",
                latestMessage="1 barbacoa ancestral y 1 cerveza indio",
                context={"language": "es-MX"},
                operation={
                    "deliveryLocationLabel": "habitaciÃ³n Royal Suite",
                    "paymentLabel": "cargo a la habitaciÃ³n",
                },
            )
        )

        self.assertIn("1 barbacoa ancestral", response.message.text)
        self.assertIn("1 cerveza indio", response.message.text)
        self.assertIn("Destino: habitaciÃ³n Royal Suite", response.message.text)
        self.assertEqual(
            ["CONFIRM", "CHANGE", "CANCEL"],
            [action.id for action in response.message.interaction.actions],
        )


def _result(payload):
    return OpenAiJsonResult(
        payload=payload,
        usage=OpenAiTokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        response_id="resp-test",
    )


if __name__ == "__main__":
    unittest.main()
