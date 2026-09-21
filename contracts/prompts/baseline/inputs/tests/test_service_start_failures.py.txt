"""SC-010: terminal failures preserve data and never authorize speculative starts."""
import json
import unittest
from uuid import uuid4

from app.agents.v2_turn_planner import plan_v2_turn, _validate_plan, _preserve_start_failure
from app.core.errors import AgentModelError
from app.schemas.v2_turns import ToolResult
from tests.conversation_regression.library import Event, load_cases, profiles
from tests.conversation_regression.model_runtime import scripted_runtime
from tests.conversation_regression.replay import replay_case
from tests.conversation_regression.session import ConversationSession, structured_summary
from tests.test_conversation_regression_library import response


class ServiceStartFailureTest(unittest.TestCase):
    def setUp(self):
        self.cases = {c.id: c for c in load_cases()}
        self.case = self.cases["SC-010-tool-failure"]
        self.profile = profiles()["hotel"]

    def failed_session(self):
        session = ConversationSession(self.case, self.profile)
        for step in self.case.steps[:3]:
            request = session.receive(step.event)
            with scripted_runtime(step.model_replies):
                result = plan_v2_turn(request)
            session.accept(result)
        return session

    def test_failure_and_old_button_never_announce_success(self):
        result = replay_case(self.case, self.profile)
        self.assertEqual("OFFLINE_PARTIAL_PASS", result["status"], result)
        failed = result["turns"][2]["observed"]
        self.assertIn("No pude confirmar", failed["text"])
        self.assertFalse(failed["state"]["awaitingExplicitConfirmation"])
        self.assertNotIn("roomServiceConfirmation", failed["state"])

    def test_offering_selection_does_not_erase_or_restart_failed_order(self):
        session = self.failed_session()
        before = structured_summary(session.request.conversation.summary)
        request = session.receive(Event(kind="guest", text="Room service", reply_id="offering:ROOM_SERVICE"))
        with scripted_runtime([]):
            result = plan_v2_turn(request)
        self.assertEqual([], result.toolCalls)
        self.assertEqual(before, structured_summary(result.updatedConversationSummary))

    def test_free_text_retry_cannot_start_another_order(self):
        session = self.failed_session()
        step = self.cases["SC-007-language-switch"].steps[1]
        request = session.receive(step.event)
        with scripted_runtime(step.model_replies):
            result = plan_v2_turn(request)
        self.assertEqual([], result.toolCalls)
        self.assertEqual("START_UNCERTAIN", structured_summary(result.updatedConversationSummary)["phase"])

    def test_validator_blocks_model_start_even_when_summary_clears_failure(self):
        session = self.failed_session()
        request = session.receive(Event(kind="guest", text="Confirmar"))
        state = structured_summary(request.conversation.summary)
        proposed = response(request, "{}", disposition="TOOL_CALLS_REQUIRED", toolCalls=[{
            "toolCallId": uuid4(), "toolName": "START_SERVICE", "confidence": 1,
            "evidenceMessageIds": [request.trigger.messageId],
            "arguments": {"offeringCode": "ROOM_SERVICE", "input": state["capturedFields"],
                          "guestConfirmationEvidenceMessageId": str(request.trigger.messageId)},
        }])
        with self.assertRaisesRegex(AgentModelError, "previous service-start failure"):
            _validate_plan(request, proposed)

    def test_independent_summary_cannot_erase_pending_failures(self):
        session = self.failed_session()
        request = session.receive(Event(kind="guest", text="Otra pregunta"))
        old = structured_summary(request.conversation.summary)["serviceStartFailures"]
        draft = response(request, '{"pendingOffering":"FAQ","capturedFields":{"question":"Horario"}}')
        result = _preserve_start_failure(request, draft)
        state = structured_summary(result.updatedConversationSummary)
        self.assertEqual(old, state["serviceStartFailures"])
        self.assertEqual("FAQ", state["pendingOffering"])

    def test_second_failed_service_retains_both_receipts(self):
        session = self.failed_session()
        request = session.request.model_copy(deep=True)
        request.previousToolResults = [ToolResult(toolCallId=uuid4(), toolName="START_SERVICE", status="FAILED",
            error={"code": "TECHNICAL", "message": "Synthetic", "retryable": False, "details": {"offeringCode": "SPA"}})]
        request.trigger.type = "TOOL_RESULTS"
        with scripted_runtime([]):
            result = plan_v2_turn(request)
        receipts = structured_summary(result.updatedConversationSummary)["serviceStartFailures"]
        self.assertEqual({"ROOM_SERVICE", "SPA"}, {r["offeringCode"] for r in receipts.values()})

    def test_live_evaluation_does_not_execute_injected_failure_case(self):
        from tests.conversation_regression.evaluate import live_case
        from tests.test_conversation_evaluation import FakeLiveModel
        result = live_case(self.case, self.profile, FakeLiveModel())
        self.assertEqual("NOT_RUN", result["status"])

    def test_independent_success_is_acknowledged_without_erasing_failed_order(self):
        from tests.test_v2_service_start_acknowledgements import request_for, start_result
        state = structured_summary(self.failed_session().request.conversation.summary)
        success = start_result("MAINTENANCE")
        request = request_for([success], state)
        with scripted_runtime([]):
            result = plan_v2_turn(request)
        self.assertIn(success["result"]["referenceCode"], result.messages[0].text)
        self.assertEqual(state["serviceStartFailures"], structured_summary(result.updatedConversationSummary)["serviceStartFailures"])


if __name__ == "__main__":
    unittest.main()
