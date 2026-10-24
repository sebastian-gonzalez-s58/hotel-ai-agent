import json
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.agents.pending_service_cancellation import plan_pending_cancellation
from app.agents.v2_scope_router import ScopeDecision
from app.agents.v2_turn_planner import plan_v2_turn
from app.schemas.v2_turns import ToolResult
from app.services.telemetry_client import OpenAiTokenUsage
from tests.test_guest_order_sessions import request_for
from tests.test_guest_booking_sessions import request_for as booking_request
from tests.conversation_regression.model_runtime import scripted_runtime


def setup(factory=request_for, text="Cancelar", locale="es-MX"):
    request = factory(locale=locale)
    code = request.availableOfferings[0].offeringCode
    message = request.conversation.recentMessages[-1]
    message.text = text
    message.interactionReplyId = None
    state = {"pendingOffering": code, "phase": request.availableOfferings[0].guestExperience,
             "capturedFields": {}, "spaDraftHistory": {"old": "CANCELLED"}}
    request.conversation.summary = json.dumps(state)
    request.toolPolicy.allowedTools.append("CANCEL_GUEST_WEB_SESSION")
    scope = ScopeDecision(kind="CONTEXT_REPLY", offeringCode=code, relevantText=text,
        hasRequestDetails=False, containsUnrelatedTopic=False, confidence=1,
        replyAction="CANCEL", replyActionEvidence=text, replyActionConfidence=1)
    return request, message, state, scope


class PendingCancellationTest(unittest.TestCase):
    def test_web_cancel_uses_revoke_tool_instead_of_reopening_link(self):
        for factory in (request_for, booking_request):
            for text, locale in (("Cancelar", "es-MX"), ("No deseo reservar", "es-MX"),
                                 ("I do not want to order anymore", "en-US")):
                request, _, _, scope = setup(factory, text, locale)
                with self.subTest(factory=factory, text=text), scripted_runtime([]), patch(
                        "app.agents.v2_turn_planner.classify_hotel_scope", return_value=(scope, OpenAiTokenUsage())):
                    result = plan_v2_turn(request)
                self.assertEqual(["CANCEL_GUEST_WEB_SESSION"], [c.toolName for c in result.toolCalls])
                self.assertEqual([], result.messages)

    def test_cancel_during_maintenance_capture_never_starts_a_folio(self):
        request, message, state, scope = setup()
        request.availableOfferings[0].offeringCode = "MAINTENANCE"
        request.availableOfferings[0].guestExperience = None
        state.update(pendingOffering="MAINTENANCE", phase="CAPTURING")
        request.conversation.summary = json.dumps(state)
        scope.offeringCode = "MAINTENANCE"
        with scripted_runtime([]), patch("app.agents.v2_turn_planner.classify_hotel_scope",
                return_value=(scope, OpenAiTokenUsage())):
            result = plan_v2_turn(request)
        self.assertEqual([], result.toolCalls)
        self.assertNotIn("pendingOffering", json.loads(result.updatedConversationSummary))
        self.assertIn("descart", result.messages[0].text)

    def test_success_clears_pending_state_but_failure_does_not_claim_cancellation(self):
        for status, result_data, cleared in (("SUCCEEDED", {"status": "CANCELLED"}, True),
                ("SUCCEEDED", {"status": "ALREADY_SUBMITTED"}, True), ("FAILED", {}, False)):
            request, _, state, _ = setup(booking_request)
            request.previousToolResults = [ToolResult(toolCallId=uuid4(), toolName="CANCEL_GUEST_WEB_SESSION",
                status=status, result=result_data)]
            with scripted_runtime([]):
                result = plan_v2_turn(request)
            self.assertEqual([], result.toolCalls)
            updated = json.loads(result.updatedConversationSummary)
            self.assertEqual(not cleared, "pendingOffering" in updated)
            self.assertEqual(state["spaDraftHistory"], updated["spaDraftHistory"])
            if result_data.get("status") == "ALREADY_SUBMITTED":
                self.assertIn("No la he cancelado", result.messages[0].text)
            if status == "FAILED":
                self.assertIn("No pude cancelar", result.messages[0].text)

    def test_uncertain_negated_hypothetical_existing_or_staff_replies_are_not_draft_cancellations(self):
        for overrides in ({"replyAction": "NONE"}, {"replyActionConfidence": .5},
                {"replyActionEvidence": "different message"}, {"kind": "STATUS_REQUEST"},
                {"offeringCode": "OTHER"}, {"separateRequest": True}):
            request, message, state, scope = setup()
            scope = scope.model_copy(update=overrides)
            self.assertIsNone(plan_pending_cancellation(request, message, state, scope))
        request, message, state, scope = setup()
        request.conversation.focusedConversationTaskId = uuid4()
        self.assertIsNone(plan_pending_cancellation(request, message, state, scope))


if __name__ == "__main__":
    unittest.main()
