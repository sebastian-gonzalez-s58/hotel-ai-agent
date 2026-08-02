import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core.agent_tracking import AgentTrackingContext
from app.services.telemetry_client import extract_openai_usage, record_model_call


class TelemetryTest(unittest.TestCase):
    def test_extracts_responses_api_token_details(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=120,
                output_tokens=30,
                total_tokens=150,
                input_tokens_details=SimpleNamespace(cached_tokens=20),
                output_tokens_details=SimpleNamespace(reasoning_tokens=7),
            )
        )

        usage = extract_openai_usage(response)

        self.assertEqual(120, usage.input_tokens)
        self.assertEqual(20, usage.cached_input_tokens)
        self.assertEqual(30, usage.output_tokens)
        self.assertEqual(7, usage.reasoning_tokens)
        self.assertEqual(150, usage.total_tokens)

    @patch("app.services.telemetry_client.post_ai_model_call")
    def test_reports_usage_with_operation_context(self, post_call):
        context = AgentTrackingContext(
            purpose="EXTRACT_REQUIREMENT_VALUES",
            conversation_id="00000000-0000-0000-0000-000000000001",
            operation_id="00000000-0000-0000-0000-000000000002",
        )
        usage = extract_openai_usage(
            SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=10,
                    output_tokens=4,
                    total_tokens=14,
                )
            )
        )

        record_model_call(
            context=context,
            status="SUCCEEDED",
            usage=usage,
            latency_ms=125,
            response_id="resp-1",
        )

        payload = post_call.call_args.args[0]
        self.assertEqual(10, payload["inputTokens"])
        self.assertEqual(4, payload["outputTokens"])
        self.assertEqual(14, payload["totalTokens"])
        self.assertEqual("00000000-0000-0000-0000-000000000002", payload["operationId"])


if __name__ == "__main__":
    unittest.main()
