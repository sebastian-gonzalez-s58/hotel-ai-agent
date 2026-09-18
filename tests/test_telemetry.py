import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core.agent_tracking import AgentTrackingContext, tracking_context_from_payload
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

    @patch("app.services.telemetry_client.post_ai_model_call")
    def test_v2_localization_logs_usage_without_legacy_callback(self, post_call):
        context = tracking_context_from_payload(
            {"messageId": "v2-message", "texts": [{"text": "Private guest text"}]},
            purpose="LOCALIZATION", request_id="request-1", runtime_version="v2",
        )
        for status in ("SUCCEEDED", "FAILED"):
            with self.subTest(status=status), self.assertLogs("chatbotinn-agent.telemetry", "INFO") as logs:
                record_model_call(context=context, status=status,
                                  usage=extract_openai_usage(None), latency_ms=125,
                                  error_message="Private error detail")
            event = json.loads(logs.records[0].getMessage().split("V2 model call ", 1)[1])
            self.assertEqual("v2-message", event["messageId"])
            self.assertEqual(status, event["status"])
            self.assertEqual(125, event["latencyMs"])
            self.assertNotIn("Private", logs.records[0].getMessage())
        post_call.assert_not_called()

    def test_v2_context_uses_nested_ids_without_guessing_an_operation(self):
        payload = {"agentTurnId": "turn-1", "conversation": {"conversationId": "conversation-1"},
                   "trigger": {"messageId": "message-1"},
                   "activeOperations": [{"operationId": "unrelated-operation"}]}
        context = tracking_context_from_payload(payload, purpose="TURN", request_id="request-1",
                                                runtime_version="v2")
        self.assertEqual("turn-1", context.agent_turn_id)
        self.assertEqual("conversation-1", context.conversation_id)
        self.assertEqual("message-1", context.message_id)
        self.assertIsNone(context.operation_id)
        payload["trigger"]["operationId"] = "operation-1"
        context = tracking_context_from_payload(payload, purpose="TURN", request_id="request-1",
                                                runtime_version="v2")
        self.assertEqual("operation-1", context.operation_id)

    @patch("app.services.telemetry_client.post_ai_model_call")
    def test_v2_turn_usage_never_posts_to_legacy_tables(self, post_call):
        context = AgentTrackingContext(purpose="TURN", runtime_version="v2", agent_turn_id="turn-1")
        with self.assertLogs("chatbotinn-agent.telemetry", "INFO"):
            record_model_call(context=context, status="SUCCEEDED",
                              usage=extract_openai_usage(None), latency_ms=20)
        post_call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
