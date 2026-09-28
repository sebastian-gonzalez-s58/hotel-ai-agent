import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.core.config import settings
from app.services.openai_client import _response_format, call_openai_json_result


class OpenAiClientTest(unittest.TestCase):
    def test_uses_json_object_without_a_schema(self):
        self.assertEqual(
            _response_format(None, "unused"),
            {"type": "json_object"},
        )

    def test_uses_supplied_json_schema_for_structured_output(self):
        schema = {
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
        }

        response_format = _response_format(schema, "agent_turn_response_v2")

        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["name"], "agent_turn_response_v2")
        self.assertEqual(response_format["schema"], schema)
        self.assertFalse(response_format["strict"])


    @patch("app.services.openai_client.record_model_call")
    @patch("app.services.openai_client.get_openai_client")
    def test_gpt56_uses_explicit_reasoning_without_sampling_parameters(self, get_client, record):
        response = SimpleNamespace(output_text='{"status":"ok"}', id="resp-test", usage=None)
        get_client.return_value.responses.create.return_value = response
        schema = {"type": "object", "properties": {"status": {"type": "string"}},
                  "required": ["status"], "additionalProperties": False}
        for model in ("gpt-5.6-terra", "gpt-5.6-luna"):
            for effort in ("none", "low", "medium", "high", "xhigh", "max"):
                with self.subTest(model=model, effort=effort), patch.object(
                    settings, "openai_model", model,
                ), patch.object(settings, "openai_reasoning_effort", effort):
                    result = call_openai_json_result("Return JSON", response_schema=schema,
                                                    response_schema_name="result", strict_schema=True)
                args = get_client.return_value.responses.create.call_args.kwargs
                self.assertEqual({"effort": effort}, args["reasoning"])
                self.assertNotIn("temperature", args)
                self.assertNotIn("top_p", args)
                self.assertEqual(model, args["model"])
                self.assertEqual({"format": {"type": "json_schema", "name": "result",
                                             "schema": schema, "strict": True}}, args["text"])
                self.assertEqual({"status": "ok"}, result.payload)
                self.assertEqual("resp-test", result.response_id)

    @patch("app.services.openai_client.record_model_call")
    @patch("app.services.openai_client.get_openai_client")
    def test_gpt41_keeps_existing_sampling_and_json_mode(self, get_client, record):
        get_client.return_value.responses.create.return_value = SimpleNamespace(
            output_text='{"status":"ok"}', id="resp-test", usage=None,
        )
        for model in ("gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"):
            with self.subTest(model=model), patch.object(settings, "openai_model", model), \
                    patch.object(settings, "openai_reasoning_effort", "high"):
                call_openai_json_result("Return JSON")
            args = get_client.return_value.responses.create.call_args.kwargs
            self.assertEqual(0, args["temperature"])
            self.assertNotIn("reasoning", args)
            self.assertEqual({"format": {"type": "json_object"}}, args["text"])

    @patch("app.services.openai_client.record_model_call")
    @patch("app.services.openai_client.get_openai_client")
    def test_reasoning_preserves_short_translation_timeout(self, get_client, record):
        limited_client = MagicMock()
        limited_client.responses.create.return_value = SimpleNamespace(
            output_text='{"texts":{"text_0":"Hello"}}', id="resp-translation", usage=None,
        )
        get_client.return_value.with_options.return_value = limited_client
        with patch.object(settings, "openai_model", "gpt-5.6-terra"), \
                patch.object(settings, "openai_reasoning_effort", "none"):
            call_openai_json_result("Translate as JSON", purpose="LOCALIZATION", timeout_seconds=4)
        get_client.return_value.with_options.assert_called_once_with(timeout=4, max_retries=0)
        self.assertEqual({"effort": "none"}, limited_client.responses.create.call_args.kwargs["reasoning"])
        self.assertEqual("LOCALIZATION", record.call_args.kwargs["purpose"])


if __name__ == "__main__":
    unittest.main()
