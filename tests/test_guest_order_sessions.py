"""Offline contract: enabled web orders launch once and cannot become chat submissions."""
import json
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.agents.guest_order_sessions import plan_guest_order_session
from app.agents.v2_scope_router import ScopeDecision
from app.agents.v2_turn_planner import plan_v2_turn, _plan_hotel_turn, _validate_lifecycle_call
from app.core.errors import AgentModelError
from app.schemas.v2_turns import AgentTurnRequest, DomainToolCall, ToolResult
from tests.test_v2_turn_endpoint import payload
from tests.conversation_regression.model_runtime import scripted_runtime


def request_for(code="ROOM_SERVICE", locale="es-MX", enabled=True):
    data = payload()
    data["guest"]["preferredLanguage"] = locale
    data["availableOfferings"] = [{"offeringCode": code, "name": "Order", "description": "Order from the catalog",
        "executionMode": "PROCESS", "inputSchema": {"type": "object", "required": ["items"],
        "properties": {"items": {"type": "array"}}}, "requiresExplicitGuestConfirmation": True}]
    if enabled:
        data["availableOfferings"][0]["guestExperience"] = "WEB_ORDER"
    data["toolPolicy"]["allowedTools"] = ["OPEN_GUEST_ORDER_SESSION", "START_SERVICE"]
    data["conversation"]["recentMessages"][0].update(text="Order", interactionReplyId="offering:" + code)
    return AgentTurnRequest.model_validate(data)


class GuestOrderSessionTest(unittest.TestCase):
    def web_amendment(self):
        from tests.test_multilingual_understanding import task_operation
        request = request_for()
        request.toolPolicy.allowedTools.append("COMPLETE_CONVERSATION_TASK")
        operation = task_operation("ROOM_SERVICE_ORDER_CHANGE_DETAILS")
        task = operation.pendingConversationTasks[0]
        task.context = {"guestExperience": "WEB_ORDER"}
        task.requiredOutputSchema = {"type": "object", "properties": {"items": {"type": "array"}, "roomServiceChangeCancelled": {"const": True}},
                                     "oneOf": [{"required": ["items"]}, {"required": ["roomServiceChangeCancelled"]}]}
        request.activeOperations = [operation]
        request.conversation.focusedConversationTaskId = task.conversationTaskId
        request.conversation.recentMessages[-1].interactionReplyId = None
        request.conversation.recentMessages[-1].text = "Mejor quiero una hamburguesa"
        return request, operation, task

    def test_web_amendment_reopens_same_task_instead_of_chat_capture_or_new_order(self):
        request, operation, task = self.web_amendment()
        for kind in ("CONTEXT_REPLY", "SERVICE_REQUEST"):
            scope = ScopeDecision(kind=kind, offeringCode="ROOM_SERVICE", relevantText="Una hamburguesa", hasRequestDetails=True,
                                  containsUnrelatedTopic=False, confidence=1)
            with scripted_runtime([]):
                response = _plan_hotel_turn(request, 0, scope)
            self.assertEqual(["OPEN_GUEST_ORDER_SESSION"], [c.toolName for c in response.toolCalls])
            self.assertEqual(task.conversationTaskId, response.toolCalls[0].targetConversationTaskId)
            self.assertEqual(operation.operationId, response.toolCalls[0].targetOperationId)
            _validate_lifecycle_call(response.toolCalls[0], {o.offeringCode:o for o in request.availableOfferings}, {operation.operationId:operation})

    def test_web_amendment_can_still_cancel_existing_order(self):
        request, operation, task = self.web_amendment()
        request.conversation.recentMessages[-1].text = "Cancelar pedido"
        scope = ScopeDecision(kind="CONTEXT_REPLY", offeringCode="ROOM_SERVICE", relevantText="Cancelar pedido", hasRequestDetails=False,
                              containsUnrelatedTopic=False, confidence=1, replyAction="CANCEL")
        with scripted_runtime([]):
            response = _plan_hotel_turn(request, 0, scope)
        self.assertEqual(["COMPLETE_CONVERSATION_TASK"], [c.toolName for c in response.toolCalls])
        self.assertEqual({"roomServiceChangeCancelled": True}, response.toolCalls[0].arguments["result"])

    def test_offering_selection_launches_generic_orders_without_model_or_capture(self):
        for code in ("ROOM_SERVICE", "POOL_BAR"):
            for locale in ("es-MX", "en-US"):
                with self.subTest(code=code, locale=locale), scripted_runtime([]):
                    request = request_for(code, locale)
                    response = plan_v2_turn(request)
                    self.assertEqual([], response.messages)
                    self.assertEqual(["OPEN_GUEST_ORDER_SESSION"], [c.toolName for c in response.toolCalls])
                    self.assertEqual({"offeringCode": code, "language": locale}, response.toolCalls[0].arguments)
                    self.assertEqual("WEB_ORDER", json.loads(response.updatedConversationSummary)["phase"])

    def test_result_preserves_exact_link_in_english_and_spanish_without_submitting(self):
        url = "https://hotel.example/guest/order#token=opaque_TEST-23"
        for locale, phrase in (("en-US", "choose your items"), ("es-MX", "elegir tus productos")):
            with self.subTest(locale=locale), scripted_runtime([], localize=True):
                request = request_for(locale=locale)
                request.previousToolResults = [ToolResult(toolCallId=uuid4(), toolName="OPEN_GUEST_ORDER_SESSION",
                    status="SUCCEEDED", result={"offeringCode": "ROOM_SERVICE", "url": url})]
                response = plan_v2_turn(request)
                self.assertEqual([], response.toolCalls)
                self.assertIn(phrase, response.messages[0].text)
                self.assertEqual(1, response.messages[0].text.count(url))
                self.assertEqual(locale, response.messages[0].language)

    def test_request_details_and_chat_confirmation_reopen_page_not_start_order(self):
        request = request_for()
        request.conversation.recentMessages[-1].interactionReplyId = None
        for kind in ("SERVICE_REQUEST", "CONTEXT_REPLY"):
            request.conversation.summary = json.dumps({"pendingOffering": "ROOM_SERVICE", "phase": "WEB_ORDER"})
            scope = ScopeDecision(kind=kind, offeringCode="ROOM_SERVICE", relevantText="Sí", hasRequestDetails=True,
                                  containsUnrelatedTopic=False, confidence=1, replyAction="CONFIRM")
            with self.subTest(kind=kind), scripted_runtime([]):
                response = _plan_hotel_turn(request, 0, scope)
                self.assertEqual(["OPEN_GUEST_ORDER_SESSION"], [c.toolName for c in response.toolCalls])

    def test_disabled_offering_and_existing_kitchen_task_keep_old_flow(self):
        request = request_for(enabled=False)
        self.assertIsNone(plan_guest_order_session(request, request.conversation.recentMessages[-1], {}, selected_code="ROOM_SERVICE"))
        request = request_for()
        request.conversation.focusedConversationTaskId = uuid4()
        scope = ScopeDecision(kind="CONTEXT_REPLY", offeringCode=None, relevantText="Sí", hasRequestDetails=False,
                              containsUnrelatedTopic=False, confidence=1)
        self.assertIsNone(plan_guest_order_session(request, request.conversation.recentMessages[-1],
                          {"pendingOffering": "ROOM_SERVICE"}, scope=scope))

    def test_enabled_offering_rejects_model_start_but_disabled_contract_still_validates(self):
        request = request_for()
        evidence = request.conversation.recentMessages[-1].messageId
        call = DomainToolCall(toolCallId=uuid4(), toolName="START_SERVICE", arguments={"offeringCode": "ROOM_SERVICE",
            "input": {"items": [{}]}, "guestConfirmationEvidenceMessageId": str(evidence)},
            confidence=1, evidenceMessageIds=[evidence])
        with self.assertRaisesRegex(AgentModelError, "WEB_ORDER"):
            _validate_lifecycle_call(call, {o.offeringCode: o for o in request.availableOfferings}, {})
        request.availableOfferings[0].guestExperience = None
        _validate_lifecycle_call(call, {o.offeringCode: o for o in request.availableOfferings}, {})

    def test_failed_launch_does_not_fall_back_to_chat_checkout_or_relaunch_loop(self):
        request = request_for()
        request.previousToolResults = [ToolResult(toolCallId=uuid4(), toolName="OPEN_GUEST_ORDER_SESSION", status="FAILED")]
        with scripted_runtime([]):
            response = plan_v2_turn(request)
        self.assertEqual([], response.toolCalls)
        self.assertIn("no se ha enviado", response.messages[0].text)

    def test_unfocused_kitchen_cancellation_completes_existing_task_without_launch(self):
        from tests.test_multilingual_understanding import task_operation
        request = request_for()
        request.toolPolicy.allowedTools.append("COMPLETE_CONVERSATION_TASK")
        request.conversation.recentMessages[-1].text = "Cancelar pedido"
        request.conversation.recentMessages[-1].interactionReplyId = "room-service-change:CANCEL"
        request.activeOperations = [task_operation("ROOM_SERVICE_KITCHEN_CHANGE_DECISION")]
        request.conversation.focusedConversationTaskId = None
        request.conversation.summary = json.dumps({"pendingOffering": "ROOM_SERVICE", "phase": "WEB_ORDER"})
        scope = ScopeDecision(kind="CONTEXT_REPLY", offeringCode="ROOM_SERVICE", relevantText="Cancelar pedido",
                              hasRequestDetails=False, containsUnrelatedTopic=False, confidence=1, replyAction="CANCEL")
        with scripted_runtime([]):
            response = _plan_hotel_turn(request, 0, scope)
        self.assertEqual(["COMPLETE_CONVERSATION_TASK"], [c.toolName for c in response.toolCalls])
        self.assertEqual({"decision": "CANCEL"}, response.toolCalls[0].arguments["result"])

    def test_status_route_and_old_delivered_cancel_button_do_not_open_cart(self):
        request = request_for()
        scope = ScopeDecision(kind="STATUS_REQUEST", offeringCode="ROOM_SERVICE", relevantText="Where is my order?",
                              hasRequestDetails=False, containsUnrelatedTopic=False, confidence=1)
        self.assertIsNone(plan_guest_order_session(request, request.conversation.recentMessages[-1],
                          {"pendingOffering": "ROOM_SERVICE", "phase": "WEB_ORDER"}, scope=scope))
        from tests.test_room_service_status import RoomServiceStatusTest
        fixture = RoomServiceStatusTest()
        request = fixture.request(reply="confirmation:ROOM_SERVICE:" + "a" * 32 + ":CANCEL")
        request.availableOfferings[0].guestExperience = "WEB_ORDER"
        operation = fixture.bind(request)
        with scripted_runtime([]):
            response = plan_v2_turn(request)
        self.assertEqual([], response.toolCalls)
        self.assertIn(operation.referenceCode, response.messages[0].text)
        self.assertIn("ya fue entregado", response.messages[0].text)


if __name__ == "__main__":
    unittest.main()
