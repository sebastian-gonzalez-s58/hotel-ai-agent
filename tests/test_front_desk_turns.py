import json
import time
import unittest
from unittest.mock import patch
from uuid import UUID

from app.agents.v2_scope_router import ScopeDecision
from app.agents.v2_turn_planner import _front_desk_start_plan, plan_v2_turn
from app.schemas.v2_turns import AgentTurnRequest
from app.services.telemetry_client import OpenAiTokenUsage
from tests.test_spa_turns import spa_operation
from tests.test_v2_service_start_acknowledgements import start_result
from tests.test_v2_turn_endpoint import payload


def front_desk_request():
    data = payload()
    data["hotel"]["hotelCode"] = "telware"
    data["conversation"]["recentMessages"][0].update(
        text="Contacto con recepcion", interactionReplyId="offering:FRONT_DESK")
    data["toolPolicy"] = {"allowedTools": ["START_SERVICE"], "maxToolCalls": 2}
    data["availableOfferings"] = [{
        "offeringCode": "FRONT_DESK", "name": "Recepcion", "description": "Contacto directo",
        "executionMode": "PROCESS", "requiresExplicitGuestConfirmation": False,
        "inputSchema": {"type": "object", "required": [], "additionalProperties": False,
                        "properties": {"handoffReason": {"type": "string"}}},
    }]
    return AgentTurnRequest.model_validate(data)


class FrontDeskTurnsTest(unittest.TestCase):
    def setUp(self):
        self.model = patch("app.agents.v2_turn_planner.call_openai_json_result",
                           side_effect=AssertionError("Unexpected planner call"))
        self.model.start()
        self.addCleanup(self.model.stop)
        self.scope = patch("app.agents.v2_turn_planner.classify_hotel_scope",
                           side_effect=AssertionError("Unexpected scope call"))
        self.scope_mock = self.scope.start()
        self.addCleanup(self.scope.stop)
        self.extract = patch("app.agents.spa_turns._extract", side_effect=AssertionError("Unexpected SPA extraction"))
        self.extract.start()
        self.addCleanup(self.extract.stop)

    def assert_immediate_start(self, request):
        before = request.model_dump()
        response = plan_v2_turn(request)
        self.assertEqual("TOOL_CALLS_REQUIRED", response.disposition)
        self.assertEqual([], response.messages)
        self.assertEqual(1, len(response.toolCalls))
        call = response.toolCalls[0]
        self.assertEqual("START_SERVICE", call.toolName)
        self.assertEqual({"offeringCode": "FRONT_DESK", "input": {}}, call.arguments)
        self.assertEqual([request.trigger.messageId], call.evidenceMessageIds)
        self.assertIsNone(call.targetOperationId)
        self.assertIsNone(call.targetConversationTaskId)
        self.assertEqual(before, request.model_dump())
        return response

    def test_menu_choice_starts_without_capture_confirmation_or_model(self):
        response = self.assert_immediate_start(front_desk_request())
        self.assertEqual(0, response.usage.totalTokens)

    def test_exact_offering_name_also_starts_immediately(self):
        request = front_desk_request()
        request.conversation.recentMessages[0].interactionReplyId = None
        request.conversation.recentMessages[0].text = "Recepcion"
        self.assert_immediate_start(request)

    def test_explicit_natural_language_request_only_needs_scope_detection(self):
        for details in (False, True):
            with self.subTest(details=details):
                request = front_desk_request()
                latest = request.conversation.recentMessages[0]
                latest.interactionReplyId = None
                latest.text = "Quiero que recepcion me contacte" + (" sobre mi factura" if details else "")
                self.scope_mock.side_effect = None
                self.scope_mock.return_value = (ScopeDecision(
                    kind="SERVICE_REQUEST", offeringCode="FRONT_DESK", relevantText=latest.text,
                    hasRequestDetails=details, containsUnrelatedTopic=False, confidence=1.0,
                ), OpenAiTokenUsage(input_tokens=2, output_tokens=1, total_tokens=3))
                response = self.assert_immediate_start(request)
                self.assertEqual(3, response.usage.totalTokens)

    def test_other_draft_and_active_tasks_survive_selection_and_acknowledgement(self):
        request = front_desk_request()
        state = {"pendingOffering": "ROOM_SERVICE", "phase": "AWAITING_CONFIRMATION",
                 "capturedFields": {"items": [{"name": "Tacos", "quantity": 2}]},
                 "spaDraft": {"capturedFields": {"serviceName": "Masaje"}},
                 "spaTasks": {"task": {"capturedFields": {"reservationDate": "2026-09-02"}}}}
        request.conversation.summary = json.dumps(state)
        request.activeOperations = [spa_operation()]
        request.conversation.focusedConversationTaskId = request.activeOperations[0].pendingConversationTasks[0].conversationTaskId
        operations = [o.model_dump() for o in request.activeOperations]
        first = self.assert_immediate_start(request)
        self.assertEqual(state, json.loads(first.updatedConversationSummary))
        request.conversation.summary = first.updatedConversationSummary
        request.previousToolResults = [self.result()]
        second = plan_v2_turn(request)
        self.assertEqual(state, json.loads(second.updatedConversationSummary))
        self.assertEqual(operations, [o.model_dump() for o in request.activeOperations])
        self.assertEqual([], second.toolCalls)

    def test_success_has_one_linked_direct_contact_notice_and_no_extra_model_call(self):
        for language in ("es-MX", "en-US"):
            request = front_desk_request()
            self.assert_immediate_start(request)
            result = self.result()
            request.previousToolResults = [result]
            request.guest.preferredLanguage = language
            response = plan_v2_turn(request)
            self.assertEqual("RESPONSE_READY", response.disposition)
            self.assertEqual([], response.toolCalls)
            self.assertEqual(0, response.usage.totalTokens)
            self.assertEqual(1, len(response.messages))
            message = response.messages[0]
            self.assertEqual([UUID(result.result["operationId"])], message.operationIds)
            self.assertEqual([], message.conversationTaskIds)
            self.assertIsNone(message.interaction)
            self.assertIn(result.result["referenceCode"], message.text)
            self.assertIn("contacto contigo directamente" if language == "es-MX" else "contact you directly", message.text)

    def test_success_clears_legacy_front_desk_capture_without_erasing_other_state(self):
        request = front_desk_request()
        request.conversation.summary = json.dumps({
            "pendingOffering": "FRONT_DESK", "capturedFields": {}, "readyToStart": False,
            "spaDraft": {"id": "another-request"},
        })
        request.previousToolResults = [self.result()]
        response = plan_v2_turn(request)
        self.assertEqual({"spaDraft": {"id": "another-request"}}, json.loads(response.updatedConversationSummary))

    def test_old_capture_does_not_start_on_greeting_status_or_unrelated_reply(self):
        for text in ("Hola", "Ya quedo resuelto", "Quiero dos tacos", "Como va recepcion?"):
            request = front_desk_request()
            request.conversation.summary = '{"pendingOffering":"FRONT_DESK"}'
            request.conversation.recentMessages[0].text = text
            request.conversation.recentMessages[0].interactionReplyId = None
            self.assertIsNone(_front_desk_start_plan(request, time.perf_counter(), None))

    def test_does_not_bypass_offering_contract_or_tool_policy(self):
        for changed in ("unavailable", "required", "confirmation", "mode", "tool", "limit", "trigger", "field"):
            with self.subTest(changed=changed):
                request = front_desk_request()
                if changed == "unavailable":
                    request.availableOfferings = []
                elif changed == "required":
                    request.availableOfferings[0].inputSchema["required"] = ["handoffReason"]
                elif changed == "confirmation":
                    request.availableOfferings[0].requiresExplicitGuestConfirmation = True
                elif changed == "mode":
                    request.availableOfferings[0].executionMode = "KNOWLEDGE"
                elif changed == "tool":
                    request.toolPolicy.allowedTools = []
                elif changed == "limit":
                    request.toolPolicy.maxToolCalls = 0
                elif changed == "trigger":
                    request.trigger.type = "TOOL_RESULTS"
                else:
                    request.conversation.recentMessages[0].interactionReplyId = "field:ROOM_SERVICE:deliveryLocation:ROOM"
                self.assertIsNone(_front_desk_start_plan(request, time.perf_counter(), None))

    @staticmethod
    def result():
        from app.schemas.v2_turns import ToolResult
        return ToolResult.model_validate(start_result("FRONT_DESK"))
