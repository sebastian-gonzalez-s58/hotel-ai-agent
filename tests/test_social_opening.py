import unittest
from unittest.mock import patch

from app.agents.social_opening import classify_social_opening
from app.core.errors import AgentDependencyError, AgentModelError, AgentTimeoutError
from app.services.openai_client import OpenAiJsonResult
from app.services.telemetry_client import OpenAiTokenUsage


class SocialOpeningTest(unittest.TestCase):
    @patch("app.agents.social_opening.call_openai_json_result")
    def test_accepts_confident_opening_and_preserves_usage(self, model):
        usage = OpenAiTokenUsage(input_tokens=8, output_tokens=2, total_tokens=10)
        model.return_value = OpenAiJsonResult({"isGreeting": True, "confidence": 0.99}, usage, "social")
        opening, actual_usage = classify_social_opening("Hellow")
        self.assertTrue(opening)
        self.assertEqual(usage, actual_usage)
        self.assertEqual(3, model.call_args.kwargs["timeout_seconds"])
        self.assertTrue(model.call_args.kwargs["strict_schema"])

    @patch("app.agents.social_opening.call_openai_json_result")
    def test_ambiguous_or_non_opening_message_does_not_force_menu(self, model):
        for value in [{"isGreeting": True, "confidence": 0.5}, {"isGreeting": False, "confidence": 1},
                      {"isGreeting": "true", "confidence": 1}, {"isGreeting": True}]:
            model.return_value = OpenAiJsonResult(value, OpenAiTokenUsage(), "social")
            self.assertFalse(classify_social_opening("social message")[0])

    @patch("app.agents.social_opening.call_openai_json_result")
    def test_provider_failure_keeps_original_social_path(self, model):
        for error in [AgentTimeoutError("timeout"), AgentModelError("invalid"), AgentDependencyError("unavailable")]:
            model.side_effect = error
            self.assertFalse(classify_social_opening("Hellow")[0])
