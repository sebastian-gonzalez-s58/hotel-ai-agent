"""BC-004: interruption/resumption uses preserved state, never reconstructed model text."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from app.agents.v2_turn_planner import plan_v2_turn, _preserve_room_service_draft
from app.schemas.v2_turns import ToolResult
from tests.conversation_regression.library import Event, load_cases, profiles
from tests.conversation_regression.model_runtime import scripted_runtime
from tests.conversation_regression.replay import replay_case
from tests.conversation_regression.session import ConversationSession, structured_summary
from tests.test_conversation_regression_library import response


class RoomServiceInterruptionTest(unittest.TestCase):
    def setUp(self):
        self.cases = {c.id: c for c in load_cases()}
        self.case = self.cases["SC-006-interrupt-resume"]
        self.profile = profiles()["hotel"]

    def suspended(self, captured=None):
        session = ConversationSession(self.case, self.profile)
        state = {"roomServiceDraft": deepcopy(self.case.initial_summary)}
        state["roomServiceDraft"]["awaitingExplicitConfirmation"] = False
        if captured is not None:
            state["roomServiceDraft"]["capturedFields"] = captured
        session.request.conversation.summary = json.dumps(state)
        return session

    def test_full_interruption_and_partial_resume_conversations(self):
        for name in ("SC-006-interrupt-resume", "SC-006-partial-resume"):
            with self.subTest(name=name):
                result = replay_case(self.cases[name], self.profile)
                self.assertEqual("OFFLINE_PARTIAL_PASS", result["status"], result)

    def test_offering_button_resumes_existing_order_without_model_or_duplicate_draft(self):
        session = self.suspended()
        request = session.receive(Event(kind="guest", text="Room service", reply_id="offering:ROOM_SERVICE"))
        with scripted_runtime([]):
            result = plan_v2_turn(request)
        observed = session.accept(result)
        self.assertEqual(self.case.initial_summary["capturedFields"], observed["state"]["capturedFields"])
        self.assertTrue(observed["state"]["awaitingExplicitConfirmation"])
        self.assertNotIn("roomServiceDraft", observed["state"])
        self.assertEqual([], observed["tools"])

    def test_resume_with_only_location_asks_items_and_cannot_confirm_empty_order(self):
        session = self.suspended({"deliveryLocation": "ROOM"})
        request = session.receive(Event(kind="guest", text="Room service", reply_id="offering:ROOM_SERVICE"))
        with scripted_runtime([]):
            result = plan_v2_turn(request)
        state = structured_summary(result.updatedConversationSummary)
        self.assertEqual({"deliveryLocation": "ROOM"}, state["capturedFields"])
        self.assertFalse(state["awaitingExplicitConfirmation"])
        self.assertEqual("CAPTURING_ITEMS", state["phase"])
        self.assertEqual([], result.toolCalls)

    def test_generic_confirmation_does_not_resume_or_start_suspended_order(self):
        from app.agents.v2_turn_planner import _resume_room_service_draft
        from app.agents.v2_scope_router import ScopeDecision
        session = self.suspended()
        request = session.receive(Event(kind="guest", text="Sí"))
        scope = ScopeDecision(kind="CONTEXT_REPLY", offeringCode="ROOM_SERVICE", relevantText="Sí",
            hasRequestDetails=False, containsUnrelatedTopic=False, confidence=1, replyAction="CONFIRM")
        self.assertIsNone(_resume_room_service_draft(request, scope, 0))
        # Even a planner that proposes the old draft cannot bypass the current-confirmation validator.
        from app.agents.v2_turn_planner import _validate_plan
        from app.core.errors import AgentModelError
        from uuid import uuid4
        proposed = response(request, toolCalls=[{"toolCallId": uuid4(), "toolName": "START_SERVICE",
            "arguments": {"offeringCode": "ROOM_SERVICE", "input": self.case.initial_summary["capturedFields"],
                "guestConfirmationEvidenceMessageId": str(request.trigger.messageId)},
            "confidence": 1, "evidenceMessageIds": [request.trigger.messageId]}], disposition="TOOL_CALLS_REQUIRED")
        from app.services.input_understanding import understanding_turn
        with understanding_turn(request), self.assertRaises(AgentModelError):
            _validate_plan(request, proposed)

    def test_cancelled_order_does_not_reappear_as_suspended(self):
        session = ConversationSession(self.case, self.profile)
        request = session.receive(Event(kind="guest", text="Cancelar", reply_id="confirmation:ROOM_SERVICE:CANCEL"))
        from tests.conversation_regression.room_confirmation_support import use_presented_room_button
        initial = structured_summary(request.conversation.summary)
        initial.update(awaitingExplicitConfirmation=True, phase="AWAITING_CONFIRMATION")
        request.conversation.summary = json.dumps(initial)
        use_presented_room_button(request, "CANCEL")
        with scripted_runtime([]):
            result = plan_v2_turn(request)
        self.assertEqual({}, structured_summary(result.updatedConversationSummary))
        self.assertEqual([], result.toolCalls)

    def test_successful_start_does_not_rearchive_submitted_order(self):
        session = ConversationSession(self.case, self.profile)
        request = session.receive(Event(kind="guest", text="Confirmar"))
        from uuid import uuid4
        request.previousToolResults = [ToolResult(toolCallId=uuid4(), toolName="START_SERVICE", status="SUCCEEDED",
                                                  result={"offeringCode": "ROOM_SERVICE"})]
        result = _preserve_room_service_draft(request, response(request, "{}"))
        self.assertEqual({}, structured_summary(result.updatedConversationSummary))

    def test_other_service_cannot_rewrite_the_suspended_order(self):
        session = self.suspended()
        request = session.receive(Event(kind="guest", text="Recepción"))
        altered = {"roomServiceDraft": {"pendingOffering": "ROOM_SERVICE", "capturedFields": {"items": []}}}
        result = _preserve_room_service_draft(request, response(request, json.dumps(altered)))
        self.assertEqual(self.case.initial_summary["capturedFields"],
                         structured_summary(result.updatedConversationSummary)["roomServiceDraft"]["capturedFields"])

    def test_tool_simulation_requires_actual_outstanding_call_and_consumes_it_once(self):
        session = ConversationSession(self.case, self.profile)
        event = self.case.steps[1].event
        with self.assertRaises(ValueError):
            session.receive(event)
        request = session.receive(self.case.steps[0].event)
        with scripted_runtime(self.case.steps[0].model_replies):
            result = plan_v2_turn(request)
        session.accept(result)
        followup = session.receive(event)
        self.assertEqual(result.toolCalls[0].toolCallId, followup.previousToolResults[0].toolCallId)
        self.assertEqual(request.conversation.recentMessages, followup.conversation.recentMessages)
        with self.assertRaises(ValueError):
            session.receive(event)


if __name__ == "__main__":
    unittest.main()
