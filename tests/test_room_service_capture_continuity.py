import json
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.agents.spa_turns import summary_state
from app.agents.v2_turn_planner import plan_v2_turn
from app.schemas.v2_turns import OfferingCapability, OperationSnapshot
from app.services.telemetry_client import OpenAiTokenUsage
from tests.test_v2_scope_router import decision, maintenance_offering, request_for
from tests.test_v2_service_start_acknowledgements import request_for as result_request, start_result
from tests.test_v2_turn_planner import operation
from tests.conversation_regression.room_confirmation_support import use_presented_room_button


def draft(items=None):
    fields = {"deliveryLocation": "ROOM"}
    if items:
        fields["items"] = items
    return {"pendingOffering": "ROOM_SERVICE", "capturedFields": fields,
            "awaitingExplicitConfirmation": bool(items), "readyToStart": False}


class RoomServiceCaptureContinuityTest(unittest.TestCase):
    def setUp(self):
        classifier = patch("app.agents.v2_turn_planner.classify_hotel_scope")
        self.classifier = classifier.start()
        self.addCleanup(classifier.stop)
        for target in ("app.agents.v2_turn_planner.call_openai_json_result",
                       "app.agents.v2_scope_router.call_openai_json_result",
                       "app.agents.spa_turns.call_openai_json_result"):
            model = patch(target, side_effect=AssertionError("Unexpected model call"))
            model.start()
            self.addCleanup(model.stop)

    def turn(self, text, state=None, *, kind="SERVICE_REQUEST", details=False, reply_id=None):
        request = request_for(text)
        request.conversation.summary = json.dumps(state if state is not None else draft())
        request.conversation.recentMessages[0].interactionReplyId = reply_id
        if reply_id == "current-cancel":
            use_presented_room_button(request, "CANCEL")
        self.classifier.return_value = (decision(kind, text, "ROOM_SERVICE", details=details),
                                        OpenAiTokenUsage())
        return plan_v2_turn(request)

    def test_alejandro_order_preserves_room_even_when_scope_calls_it_a_new_service(self):
        for details in (True, False):
            with self.subTest(details=details):
                response = self.turn("Traeme unas gorditas de pulpo", details=details)
                state = summary_state(response.updatedConversationSummary)
                self.assertEqual("ROOM", state["capturedFields"]["deliveryLocation"])
                self.assertEqual("gorditas de pulpo", state["capturedFields"]["items"][0]["name"])
                self.assertTrue(state["awaitingExplicitConfirmation"])
                self.assertEqual("BUTTONS", response.messages[0].interaction.type)
                self.assertNotIn("menudigitalonline", response.messages[0].text)
                self.assertEqual([], response.toolCalls)

    def test_snapshot_history_uses_latest_captured_location(self):
        request = request_for("2 tacos")
        request.conversation.summary = '{}\n{"pendingOffering":"ROOM_SERVICE","capturedFields":{}}\n' + json.dumps(draft())
        before = request.model_dump()
        self.classifier.return_value = (decision("SERVICE_REQUEST", "2 tacos", "ROOM_SERVICE", details=False),
                                        OpenAiTokenUsage())
        response = plan_v2_turn(request)
        self.assertEqual("ROOM", summary_state(response.updatedConversationSummary)["capturedFields"]["deliveryLocation"])
        self.assertEqual(before, request.model_dump())

    def test_existing_order_can_be_changed_without_resetting_location(self):
        response = self.turn("Mejor 2 tacos", draft([{"name": "pozole", "quantity": 1}]))
        fields = summary_state(response.updatedConversationSummary)["capturedFields"]
        self.assertEqual("ROOM", fields["deliveryLocation"])
        self.assertEqual(["tacos"], [item["name"] for item in fields["items"]])
        self.assertEqual(2, fields["items"][0]["quantity"])

    def test_delivery_change_keeps_items_and_returns_confirmation(self):
        items = [{"name": "pozole", "quantity": 2}]
        for reply_id in (None, "field:ROOM_SERVICE:deliveryLocation:DOCK_1"):
            with self.subTest(reply_id=reply_id):
                response = self.turn("Mejor entregalo en muelle 1", draft(items), reply_id=reply_id)
                state = summary_state(response.updatedConversationSummary)
                self.assertEqual("DOCK_1", state["capturedFields"]["deliveryLocation"])
                self.assertEqual(items, state["capturedFields"]["items"])
                self.assertTrue(state["awaitingExplicitConfirmation"])
                self.assertNotIn("menudigitalonline", response.messages[0].text)

    def test_reselecting_service_does_not_clear_draft(self):
        state = draft([{"name": "pozole", "quantity": 1}])
        response = self.turn("Servicio a la habitacion", state, reply_id="offering:ROOM_SERVICE")
        self.assertEqual(state["capturedFields"], summary_state(response.updatedConversationSummary)["capturedFields"])
        self.assertEqual("BUTTONS", response.messages[0].interaction.type)

    def test_clear_separate_order_request_starts_new_capture(self):
        for text in ("Quiero hacer otro pedido aparte", "I want another order"):
            for details in (True, False):
                with self.subTest(text=text, details=details):
                    response = self.turn(text, details=details)
                    state = summary_state(response.updatedConversationSummary)
                    self.assertEqual({}, state["capturedFields"])
                    self.assertEqual("ROOM_SERVICE", state["pendingOffering"])
                    self.assertEqual("LIST", response.messages[0].interaction.type)

    def test_collecting_location_after_items_keeps_the_items(self):
        response = self.turn("2 tacos", {"pendingOffering": "ROOM_SERVICE", "capturedFields": {}})
        state = summary_state(response.updatedConversationSummary)
        self.assertEqual("tacos", state["capturedFields"]["items"][0]["name"])
        self.assertEqual("LIST", response.messages[0].interaction.type)
        response = self.turn("Habitacion", state, reply_id="field:ROOM_SERVICE:deliveryLocation:ROOM")
        fields = summary_state(response.updatedConversationSummary)["capturedFields"]
        self.assertEqual("ROOM", fields["deliveryLocation"])
        self.assertEqual("tacos", fields["items"][0]["name"])
        self.assertEqual("BUTTONS", response.messages[0].interaction.type)

    def test_new_order_without_draft_is_not_blocked_by_active_operations(self):
        request = request_for("Quiero hacer un pedido")
        request.conversation.summary = "{}"
        request.activeOperations = [OperationSnapshot.model_validate(operation(str(uuid4())))]
        self.classifier.return_value = (decision("SERVICE_REQUEST", "Quiero hacer un pedido", "ROOM_SERVICE", details=False),
                                        OpenAiTokenUsage())
        response = plan_v2_turn(request)
        self.assertEqual("ROOM_SERVICE", summary_state(response.updatedConversationSummary)["pendingOffering"])
        self.assertEqual("LIST", response.messages[0].interaction.type)

    def test_cancellation_does_not_restore_cancelled_draft(self):
        for reply_id in (None, "current-cancel"):
            with self.subTest(reply_id=reply_id):
                response = self.turn("Cancelar", draft([{"name": "tacos", "quantity": 2}]), reply_id=reply_id)
                state = summary_state(response.updatedConversationSummary)
                self.assertNotIn("pendingOffering", state)
                self.assertNotIn("roomServiceDraft", state)
                self.assertEqual([], response.toolCalls)

    def test_legacy_cancel_button_does_not_cancel_current_draft(self):
        state = draft([{"name": "tacos", "quantity": 2}])
        response = self.turn("Cancelar", state, reply_id="confirmation:ROOM_SERVICE:CANCEL")
        self.assertEqual(state["capturedFields"], summary_state(response.updatedConversationSummary)["capturedFields"])
        self.assertEqual([], response.toolCalls)

    def test_another_service_preserves_draft_through_start_and_resume(self):
        request = request_for("Tambien hay una fuga en el lavabo")
        request.conversation.summary = json.dumps(draft())
        request.availableOfferings.append(OfferingCapability.model_validate(maintenance_offering()))
        self.classifier.return_value = (decision("SERVICE_REQUEST", request.conversation.recentMessages[0].text,
                                                 "MAINTENANCE", details=True), OpenAiTokenUsage())
        response = plan_v2_turn(request)
        self.assertEqual("START_SERVICE", response.toolCalls[0].toolName)
        self.assertEqual("MAINTENANCE", response.toolCalls[0].arguments["offeringCode"])
        state = summary_state(response.updatedConversationSummary)
        self.assertEqual(draft(), state["roomServiceDraft"])
        ack = plan_v2_turn(result_request([start_result("MAINTENANCE")], state))
        state = summary_state(ack.updatedConversationSummary)
        self.assertEqual(draft(), state["roomServiceDraft"])
        self.assertNotIn("pendingOffering", state)
        resumed = self.turn("Servicio a la habitacion", state, reply_id="offering:ROOM_SERVICE")
        state = summary_state(resumed.updatedConversationSummary)
        self.assertEqual("ROOM", state["capturedFields"]["deliveryLocation"])
        self.assertNotIn("roomServiceDraft", state)
        self.assertIn("https://hotel.example/menu", resumed.messages[0].text)

    def test_resuming_with_items_uses_saved_location(self):
        response = self.turn("Traeme 2 tacos", {"roomServiceDraft": draft()})
        fields = summary_state(response.updatedConversationSummary)["capturedFields"]
        self.assertEqual("ROOM", fields["deliveryLocation"])
        self.assertEqual("tacos", fields["items"][0]["name"])

    def test_cancelling_saved_order_does_not_resurrect_it(self):
        response = self.turn("Cancela mi pedido", {"roomServiceDraft": draft()})
        state = summary_state(response.updatedConversationSummary)
        self.assertNotIn("pendingOffering", state)
        self.assertNotIn("roomServiceDraft", state)

    def test_uncertain_message_preserves_captured_data(self):
        response = self.turn("No se", kind="UNCLEAR")
        self.assertEqual(draft(), summary_state(response.updatedConversationSummary))
        self.assertEqual([], response.toolCalls)

    def test_successful_room_service_start_clears_draft(self):
        state = draft([{"name": "pozole", "quantity": 1}])
        state["phase"] = "STARTING"
        response = plan_v2_turn(result_request([start_result()], state))
        self.assertNotIn("pendingOffering", summary_state(response.updatedConversationSummary))
        self.assertNotIn("roomServiceDraft", summary_state(response.updatedConversationSummary))

    def test_unrelated_topic_does_not_consume_order_fields(self):
        response = self.turn("Explica sliding windows", kind="OUT_OF_SCOPE")
        self.assertEqual(draft(), summary_state(response.updatedConversationSummary))
        self.assertEqual([], response.toolCalls)

    def test_hotel_question_still_uses_faq_instead_of_order_capture(self):
        request = request_for("A que hora cierra la alberca?")
        request.conversation.summary = json.dumps(draft())
        self.classifier.return_value = (decision("HOTEL_QUESTION", request.conversation.recentMessages[0].text, "FAQ"),
                                        OpenAiTokenUsage())
        response = plan_v2_turn(request)
        self.assertEqual("SEARCH_KNOWLEDGE", response.toolCalls[0].toolName)
        self.assertEqual(draft(), summary_state(response.updatedConversationSummary)["roomServiceDraft"])
