from datetime import datetime, timezone
import json
from pathlib import Path
from string import Formatter
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from pydantic import ValidationError
from jsonschema import Draft202012Validator

from app.core.config import settings
from app.core.errors import AgentModelError
from app.main import app
from app.schemas.localization import LocalizationRequest
from app.services.notification_localization import REGISTRY, localize_notification


class NotificationLocalizationTest(unittest.TestCase):
    def request(self, texts, locale="en"):
        return LocalizationRequest(messageId=uuid4(), namespace="chatbotinn_telware_demo", locale=locale,
                                   texts=[{"text": text, "maxLength": 20000} for text in texts],
                                   protectedValues=["REQ-20260908-ABC12345", "Deep Tissue"])

    @patch("app.services.notification_localization.translate_batch")
    def test_maintenance_confirmation_uses_reviewed_english_without_model(self, translate):
        reference = "REQ-20260908-ABC12345"
        body = REGISTRY["templates"]["maintenance.confirm"]["es"].format(reference=reference)
        response = localize_notification(self.request([body, "S\u00ed, qued\u00f3 resuelto", "No, sigue sin resolver"]))
        self.assertIn(reference, response.texts[0])
        self.assertIn("Please confirm", response.texts[0])
        self.assertEqual(["Yes, resolved", "Not resolved"], response.texts[1:])
        translate.assert_not_called()

    @patch("app.services.notification_localization.translate_batch")
    def test_spa_booking_keeps_service_date_and_time_exact(self, translate):
        values = dict(reference="REQ-20260908-ABC12345", service="Deep Tissue", date="2026-09-09", time="16:30")
        body = REGISTRY["templates"]["spa.alternative"]["es"].format(**values)
        result = localize_notification(self.request([body]))
        for value in values.values():
            self.assertIn(value, result.texts[0])
        self.assertIn("Would you like to accept", result.texts[0])
        translate.assert_not_called()

    @patch("app.services.notification_localization.translate_batch")
    def test_room_service_menu_link_and_reference_are_preserved(self, translate):
        values = dict(reference="REQ-20260908-ABC12345", url="https://example.test/menu")
        text = REGISTRY["templates"]["order.change"]["es"].format(**values)
        result = localize_notification(self.request([text, "Hacer cambios", "Cancelar pedido"]))
        self.assertIn(values["url"], result.texts[0])
        self.assertIn(values["reference"], result.texts[0])
        self.assertEqual(["Make changes", "Cancel order"], result.texts[1:])
        translate.assert_not_called()

    @patch("app.services.notification_localization.translate_batch")
    def test_arbitrary_guest_locale_uses_one_bounded_presentation_batch(self, translate):
        translate.return_value = (["Bonjour", "Oui"], None)
        result = localize_notification(self.request(["Hola", "Si"], "fr-CA"))
        self.assertEqual("fr-CA", result.locale)
        self.assertEqual(["Bonjour", "Oui"], result.texts)
        translate.assert_called_once()
        self.assertEqual(4.0, translate.call_args.kwargs["timeout"])
        self.assertEqual("fr-CA", translate.call_args.args[1])

    @patch("app.services.notification_localization.translate_batch")
    def test_staff_facts_are_translated_without_a_conversation_plan(self, translate):
        translate.return_value = (["The pool closes at 22:00."], None)
        result = localize_notification(self.request(["La alberca cierra a las 22:00."]))
        self.assertEqual(["The pool closes at 22:00."], result.texts)
        self.assertEqual(["La alberca cierra a las 22:00."], translate.call_args.args[0])

    @patch("app.services.notification_localization.translate_batch")
    def test_appended_kitchen_note_is_not_treated_as_part_of_the_menu_url(self, translate):
        body = REGISTRY["templates"]["order.change"]["es"].format(
            reference="REQ-20260908-ABC12345", url="https://example.test/menu") + " No tenemos chilaquiles."
        translate.return_value = (["The kitchen has no chilaquiles. https://example.test/menu"], None)
        localize_notification(self.request([body]))
        translate.assert_called_once()
        self.assertIn("No tenemos chilaquiles.", translate.call_args.args[0][0])

    @patch("app.services.notification_localization.translate_batch")
    def test_spanish_negative_confirmation_fits_whatsapp_without_a_model_call(self, translate):
        request = self.request(["No, sigue sin resolver"], "es-MX")
        request.texts[0].maxLength = 20
        result = localize_notification(request)
        self.assertEqual(["No, sigue pendiente"], result.texts)
        translate.assert_not_called()

    @patch("app.services.notification_localization.translate_batch")
    def test_channel_limit_failure_does_not_return_a_truncated_button(self, translate):
        translate.return_value = (["Much too long for this button"], None)
        request = self.request(["Otra respuesta"], "de")
        request.texts[0].maxLength = 20
        with self.assertRaises(AgentModelError):
            localize_notification(request)

    def test_reviewed_templates_preserve_all_parameters_in_both_languages(self):
        for key, variants in REGISTRY["templates"].items():
            def parameters(text):
                return sorted(name for _, name, _, _ in Formatter().parse(text) if name is not None)
            self.assertEqual(parameters(variants["es"]), parameters(variants["en"]), key)

    def test_contract_rejects_tools_and_unbounded_payloads(self):
        payload = self.request(["Hello"]).model_dump(mode="json")
        payload["toolCalls"] = [{"toolName": "START_SERVICE"}]
        with self.assertRaises(ValidationError):
            LocalizationRequest.model_validate(payload)
        with self.assertRaises(ValidationError):
            self.request(["x" * 20000, "x" * 20000])

    def test_openapi_matches_notification_request_and_response(self):
        contract = json.loads(Path("contracts/v2/agent-runtime.openapi.json").read_text(encoding="utf-8"))
        schemas = contract["components"]["schemas"]
        request = self.request(["Cancelar pedido"])
        Draft202012Validator(schemas["LocalizationRequest"]).validate(request.model_dump(mode="json"))
        response = localize_notification(request)
        Draft202012Validator(schemas["LocalizationResponse"]).validate(response.model_dump(mode="json"))
        self.assertIn("/internal/v2/localizations", contract["paths"])


class NotificationLocalizationEndpointTest(unittest.TestCase):
    def setUp(self):
        self.token = settings.agent_internal_token
        self.mode = settings.agent_runtime_mode
        settings.agent_internal_token = "notification-test-token"
        settings.agent_runtime_mode = "v2"
        self.client = TestClient(app)
        self.message_id = str(uuid4())

    def tearDown(self):
        settings.agent_internal_token = self.token
        settings.agent_runtime_mode = self.mode

    def headers(self):
        return {"Authorization": "Bearer notification-test-token", "X-Request-Id": self.message_id,
                "Idempotency-Key": self.message_id,
                "X-ChatbotInn-Timestamp": datetime.now(timezone.utc).isoformat()}

    def payload(self):
        return {"messageId": self.message_id, "namespace": "chatbotinn_telware_demo", "locale": "en",
                "texts": [{"text": "Cancelar pedido", "maxLength": 20}], "protectedValues": []}

    @patch("app.main.plan_v2_turn")
    def test_authenticated_endpoint_only_localizes_without_running_tools(self, planner):
        result = self.client.post("/internal/v2/localizations", json=self.payload(), headers=self.headers())
        self.assertEqual(200, result.status_code)
        self.assertEqual({"locale": "en", "version": "1", "texts": ["Cancel order"]}, result.json())
        planner.assert_not_called()

    def test_rejects_missing_authentication(self):
        response = self.client.post("/internal/v2/localizations", json=self.payload())
        self.assertEqual(401, response.status_code)

    def test_rejects_mismatched_identity_header(self):
        headers = self.headers()
        headers["Idempotency-Key"] = str(uuid4())
        response = self.client.post("/internal/v2/localizations", json=self.payload(), headers=headers)
        self.assertEqual(400, response.status_code)

    def test_not_available_in_legacy_runtime(self):
        settings.agent_runtime_mode = "legacy"
        response = self.client.post("/internal/v2/localizations", json=self.payload(), headers=self.headers())
        self.assertEqual(404, response.status_code)


if __name__ == "__main__":
    unittest.main()
