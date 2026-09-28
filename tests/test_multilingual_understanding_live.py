"""Opt-in model checks: synthetic inputs only, no tools executed or WhatsApp messages sent."""
import json
import os
import unittest
from uuid import uuid4

from app.agents.v2_turn_planner import plan_v2_turn
from app.agents.v2_scope_router import classify_hotel_scope
from app.schemas.v2_turns import OfferingCapability
from tests.test_multilingual_understanding import enable, room_request, task_operation
from tests.test_spa_turns import follow_up, request_for
from tests.test_v2_scope_router import maintenance_offering
from tests.test_v2_turn_planner import guided_room_service_offering


def next_turn(request, response, text):
    updated = follow_up(request, response, text)
    for message in response.messages:
        outbound = updated.conversation.recentMessages[-1].model_copy(update={
            "messageId": uuid4(), "direction": "OUTBOUND", "text": message.text,
        })
        updated.conversation.recentMessages.insert(-1, outbound)
    return updated


@unittest.skipUnless(os.getenv("RUN_LIVE_MULTILINGUAL_EVALS") == "1", "Opt-in model evaluation")
class LiveEnglishUnderstandingTest(unittest.TestCase):
    def test_explicit_separate_order_is_still_allowed(self):
        request = room_request("Start a separate second room-service order, please")
        request.guest.preferredLanguage = "en"
        message = request.conversation.recentMessages[-1]
        result, _ = classify_hotel_scope(request, message, json.loads(request.conversation.summary))
        self.assertEqual("SERVICE_REQUEST", result.kind)
        self.assertEqual("ROOM_SERVICE", result.offeringCode)
        self.assertTrue(result.separateRequest)

    def test_polite_order_edit_and_confirmation(self):
        request = room_request("Hello, I would like two burgers without cheese, please")
        request.guest.preferredLanguage = "en"
        captured = plan_v2_turn(request)
        self.assertEqual([], captured.toolCalls)
        state = json.loads(captured.updatedConversationSummary)
        self.assertTrue(state["awaitingExplicitConfirmation"])
        self.assertEqual(2, state["capturedFields"]["items"][0]["quantity"])
        self.assertIn("without cheese", state["capturedFields"]["items"][0]["modifications"])

        request = next_turn(request, captured, "Yes, but without onions as well")
        changed = plan_v2_turn(request)
        self.assertEqual([], changed.toolCalls)
        item = json.loads(changed.updatedConversationSummary)["capturedFields"]["items"][0]
        self.assertEqual(2, item["quantity"])
        self.assertIn("without cheese", item["modifications"])
        self.assertTrue(any("without onions" in value for value in item["modifications"]))

        request = next_turn(request, changed, "Yes, I confirm the order")
        confirmed = plan_v2_turn(request)
        self.assertEqual(1, len(confirmed.toolCalls))
        self.assertEqual("START_SERVICE", confirmed.toolCalls[0].toolName)
        self.assertEqual("ROOM_SERVICE", confirmed.toolCalls[0].arguments["offeringCode"])
        self.assertEqual([item], confirmed.toolCalls[0].arguments["input"]["items"])

    def test_kitchen_replacement_completes_existing_task(self):
        request = enable(request_for("My complete new order is two sandwiches without mustard"))
        request.guest.preferredLanguage = "en"
        request.availableOfferings = [OfferingCapability.model_validate(guided_room_service_offering())]
        request.activeOperations = [task_operation("ROOM_SERVICE_ORDER_CHANGE_DETAILS")]
        request.conversation.focusedConversationTaskId = request.activeOperations[0].pendingConversationTasks[0].conversationTaskId
        response = plan_v2_turn(request)
        self.assertEqual(1, len(response.toolCalls))
        call = response.toolCalls[0]
        self.assertEqual("COMPLETE_CONVERSATION_TASK", call.toolName)
        self.assertEqual(request.activeOperations[0].operationId, call.targetOperationId)
        self.assertEqual(2, call.arguments["result"]["items"][0]["quantity"])
        self.assertIn("without mustard", call.arguments["result"]["items"][0]["modifications"])

    def test_maintenance_negation_does_not_confirm_resolution(self):
        request = enable(request_for("No, the air conditioning is still not working"))
        request.guest.preferredLanguage = "en"
        request.availableOfferings = [OfferingCapability.model_validate(maintenance_offering())]
        request.activeOperations = [task_operation("MAINTENANCE_RESOLUTION_CONFIRMATION")]
        request.conversation.focusedConversationTaskId = request.activeOperations[0].pendingConversationTasks[0].conversationTaskId
        response = plan_v2_turn(request)
        self.assertEqual(1, len(response.toolCalls))
        self.assertEqual("COMPLETE_CONVERSATION_TASK", response.toolCalls[0].toolName)
        self.assertIs(False, response.toolCalls[0].arguments["result"]["resolved"])
