"""Opt-in calendars preserve existing SPA tasks and never submit from chat."""
import json
import unittest
from uuid import uuid4

from app.agents.guest_booking_sessions import plan_guest_booking_session
from app.agents.v2_scope_router import ScopeDecision
from app.agents.v2_turn_planner import plan_v2_turn, _plan_hotel_turn, _validate_lifecycle_call
from app.core.errors import AgentModelError
from app.schemas.v2_turns import DomainToolCall, ToolResult
from tests.test_guest_order_sessions import request_for as order_request
from tests.conversation_regression.model_runtime import scripted_runtime


def request_for(code="SPA", locale="en-US", enabled=True):
    request = order_request(code, locale, enabled=False)
    offering = request.availableOfferings[0]
    offering.name = "Spa booking"
    offering.description = "Reserve a treatment"
    offering.inputSchema = {"type": "object", "required": ["serviceName", "reservationDate", "reservationTime"],
                           "properties": {k: {"type": "string"} for k in
                                          ("serviceName", "reservationDate", "reservationTime")}}
    offering.guestExperience = "WEB_BOOKING" if enabled else None
    request.toolPolicy.allowedTools = ["OPEN_GUEST_BOOKING_SESSION", "START_SERVICE"]
    return request


class GuestBookingSessionTest(unittest.TestCase):
    def amendment(self):
        from tests.test_multilingual_understanding import task_operation
        request = request_for()
        request.toolPolicy.allowedTools.append('COMPLETE_CONVERSATION_TASK')
        operation = task_operation('SPA_RESERVATION_CHANGE_DETAILS')
        operation.offeringCode = 'SPA'
        task = operation.pendingConversationTasks[0]
        task.context = {'guestExperience': 'WEB_BOOKING'}
        task.requiredOutputSchema = {'type': 'object', 'required': ['decision'],
                                    'properties': {'decision': {'enum': ['UPDATE', 'CANCEL']}}}
        request.activeOperations = [operation]
        request.conversation.focusedConversationTaskId = task.conversationTaskId
        request.conversation.recentMessages[-1].interactionReplyId = None
        request.conversation.recentMessages[-1].text = 'Another treatment tomorrow'
        return request, operation, task

    def test_amendment_reopens_calendar_and_never_collects_replacement_fields(self):
        request, operation, task = self.amendment()
        for kind in ('CONTEXT_REPLY', 'SERVICE_REQUEST'):
            scope = ScopeDecision(kind=kind, offeringCode='SPA', relevantText='Another treatment tomorrow',
                hasRequestDetails=True, containsUnrelatedTopic=False, confidence=1)
            with scripted_runtime([]):
                result = _plan_hotel_turn(request, 0, scope)
            self.assertEqual(['OPEN_GUEST_BOOKING_SESSION'], [c.toolName for c in result.toolCalls])
            self.assertEqual(task.conversationTaskId, result.toolCalls[0].targetConversationTaskId)
            _validate_lifecycle_call(result.toolCalls[0], {'SPA': request.availableOfferings[0]}, {operation.operationId: operation})

    def test_amendment_cancel_still_completes_its_task(self):
        request, _, task = self.amendment()
        request.conversation.recentMessages[-1].text = 'Cancel'
        scope = ScopeDecision(kind='CONTEXT_REPLY', offeringCode='SPA', relevantText='Cancel',
                hasRequestDetails=False, containsUnrelatedTopic=False, confidence=1, replyAction='CANCEL')
        with scripted_runtime([]):
            result = _plan_hotel_turn(request, 0, scope)
        self.assertEqual('COMPLETE_CONVERSATION_TASK', result.toolCalls[0].toolName)
        self.assertEqual({'decision': 'CANCEL'}, result.toolCalls[0].arguments['result'])

    def test_amendment_link_preserves_folio_context_and_guest_language(self):
        for locale in ('en-US', 'es-MX'):
            request, _, _ = self.amendment()
            request.guest.preferredLanguage = locale
            request.conversation.summary = '{"existing": "booking"}'
            url = 'https://hotel.example/guest-booking#token=edit-session'
            request.previousToolResults = [ToolResult(toolCallId=uuid4(), toolName='OPEN_GUEST_BOOKING_SESSION',
                status='SUCCEEDED', result={'offeringCode': 'SPA', 'url': url, 'amendment': True})]
            with scripted_runtime([]):
                result = plan_v2_turn(request)
            self.assertIn(url, result.messages[0].text)
            self.assertIn('mismo folio' if locale.startswith('es') else 'same reference', result.messages[0].text)
            self.assertEqual(request.conversation.summary, result.updatedConversationSummary)

    def test_menu_launch_is_generic_localized_and_never_collects_fields(self):
        for code in ("SPA", "TENNIS_LESSON"):
            for locale in ("en-US", "es-MX"):
                with self.subTest(code=code, locale=locale), scripted_runtime([]):
                    request = request_for(code, locale)
                    result = plan_v2_turn(request)
                    self.assertEqual(["OPEN_GUEST_BOOKING_SESSION"], [c.toolName for c in result.toolCalls])
                    self.assertEqual({"offeringCode": code, "language": locale}, result.toolCalls[0].arguments)
                    self.assertEqual([], result.messages)
                    self.assertEqual("WEB_BOOKING", json.loads(result.updatedConversationSummary)["phase"])

    def test_exact_server_link_and_staff_approval_are_preserved(self):
        url = "https://hotel.example/guest-booking#token=opaque_1-2"
        for locale, phrase in (("en-US", "staff for approval"), ("es-MX", "personal para su aprobación")):
            with self.subTest(locale=locale), scripted_runtime([], localize=True):
                request = request_for(locale=locale)
                request.previousToolResults = [ToolResult(toolCallId=uuid4(), toolName="OPEN_GUEST_BOOKING_SESSION",
                    status="SUCCEEDED", result={"offeringCode": "SPA", "url": url})]
                result = plan_v2_turn(request)
                self.assertEqual([], result.toolCalls)
                self.assertIn(phrase, result.messages[0].text)
                self.assertEqual(1, result.messages[0].text.count(url))
                self.assertEqual(locale, result.messages[0].language)

    def test_free_text_request_and_web_draft_confirmation_open_calendar(self):
        for kind in ("SERVICE_REQUEST", "CONTEXT_REPLY"):
            with self.subTest(kind=kind), scripted_runtime([]):
                request = request_for()
                request.conversation.recentMessages[-1].interactionReplyId = None
                request.conversation.summary = json.dumps({"pendingOffering": "SPA", "phase": "WEB_BOOKING"})
                scope = ScopeDecision(kind=kind, offeringCode="SPA", relevantText="Tomorrow at ten",
                                      hasRequestDetails=True, containsUnrelatedTopic=False, confidence=1)
                result = _plan_hotel_turn(request, 0, scope)
                self.assertEqual(["OPEN_GUEST_BOOKING_SESSION"], [c.toolName for c in result.toolCalls])

    def test_disabled_and_status_routes_keep_existing_behavior(self):
        request = request_for(enabled=False)
        self.assertIsNone(plan_guest_booking_session(request, request.conversation.recentMessages[-1], {}, selected_code="SPA"))
        request = request_for()
        for kind in ("STATUS_REQUEST", "NAVIGATION"):
            scope = ScopeDecision(kind=kind, offeringCode="SPA", relevantText="My booking",
                                  hasRequestDetails=False, containsUnrelatedTopic=False, confidence=1)
            self.assertIsNone(plan_guest_booking_session(request, request.conversation.recentMessages[-1],
                {"pendingOffering": "SPA", "phase": "WEB_BOOKING"}, scope=scope))

    def test_pending_staff_alternative_remains_in_chat(self):
        from tests.test_multilingual_understanding import task_operation
        request = request_for()
        operation = task_operation("SPA_ALTERNATIVE_DECISION")
        operation.offeringCode = "SPA"
        request.activeOperations = [operation]
        scope = ScopeDecision(kind="CONTEXT_REPLY", offeringCode="SPA", relevantText="Yes",
                              hasRequestDetails=False, containsUnrelatedTopic=False, confidence=1)
        self.assertIsNone(plan_guest_booking_session(request, request.conversation.recentMessages[-1],
            {"pendingOffering": "SPA", "phase": "WEB_BOOKING"}, scope=scope))

    def test_start_service_and_cross_purpose_launch_are_rejected(self):
        request = request_for()
        evidence = request.conversation.recentMessages[-1].messageId
        call = DomainToolCall(toolCallId=uuid4(), toolName="START_SERVICE",
            arguments={"offeringCode": "SPA", "input": {"serviceName": "Massage", "reservationDate": "2026-10-03",
                "reservationTime": "10:00"}, "guestConfirmationEvidenceMessageId": str(evidence)},
            confidence=1, evidenceMessageIds=[evidence])
        with self.assertRaisesRegex(AgentModelError, "WEB_BOOKING"):
            _validate_lifecycle_call(call, {"SPA": request.availableOfferings[0]}, {})
        call.toolName = "OPEN_GUEST_ORDER_SESSION"
        with self.assertRaises(AgentModelError):
            _validate_lifecycle_call(call, {"SPA": request.availableOfferings[0]}, {})
        call.toolName = "OPEN_GUEST_BOOKING_SESSION"
        _validate_lifecycle_call(call, {"SPA": request.availableOfferings[0]}, {})
        call.targetOperationId = uuid4()
        with self.assertRaises(AgentModelError):
            _validate_lifecycle_call(call, {"SPA": request.availableOfferings[0]}, {})

    def test_failed_or_invalid_launch_never_loops_or_announces_booking(self):
        for status, url, code in (("FAILED", None, "SPA"), ("SUCCEEDED", "javascript:alert(1)", "SPA"),
                                  ("SUCCEEDED", "https://user:pass@hotel.example", "SPA"),
                                  ("SUCCEEDED", "https://hotel.example/#token=x", "OTHER")):
            with self.subTest(status=status, url=url, code=code), scripted_runtime([]):
                request = request_for()
                request.previousToolResults = [ToolResult(toolCallId=uuid4(), toolName="OPEN_GUEST_BOOKING_SESSION",
                    status=status, result={"offeringCode": code, "url": url})]
                result = plan_v2_turn(request)
                self.assertEqual([], result.toolCalls)
                self.assertIn("no reservation has been submitted", result.messages[0].text)

    def test_booking_state_preserves_independent_order_draft(self):
        request = request_for()
        state = {"roomServiceDraft": {"capturedFields": {"items": [{"name": "Coffee", "quantity": 2}]}},
                 "catalogPending": {"stale": True}}
        result = plan_guest_booking_session(request, request.conversation.recentMessages[-1], state, selected_code="SPA")
        updated = json.loads(result["updated_summary"])
        self.assertEqual(state["roomServiceDraft"], updated["roomServiceDraft"])
        self.assertNotIn("catalogPending", updated)
        self.assertFalse(updated["readyToStart"])


if __name__ == "__main__":
    unittest.main()
