from tests.conversation_regression.room_confirmation_support import current_room_button, use_presented_room_button
import json
import os
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.agents.v2_scope_router import classify_hotel_scope
from app.agents.v2_turn_planner import plan_v2_turn, _latest_capture_state
from app.services.catalog_selection import pending_catalog_selection
from app.schemas.v2_turns import OfferingCapability
from tests.test_multilingual_understanding import room_request, scope_for, result, task_operation, order, edit
from tests.test_v2_scope_router import maintenance_offering
from tests.test_spa_turns import follow_up


def selection_request(text, captured=None):
    request = room_request(text)
    request.guest.preferredLanguage = "en"
    request.conversation.summary = json.dumps({
        "pendingOffering": "ROOM_SERVICE", "phase": "CAPTURING_LOCATION",
        "capturedFields": captured or {}, "awaitingExplicitConfirmation": False,
    })
    return request


class CatalogSelectionTest(unittest.TestCase):
    def setUp(self):
        for target, name in [
            ("app.agents.v2_turn_planner.classify_hotel_scope", "scope"),
            ("app.agents.v2_turn_planner.call_openai_json_result", "planner"),
            ("app.agents.v2_turn_planner.localize_response", "localize"),
        ]:
            mock = patch(target)
            setattr(self, name, mock.start())
            self.addCleanup(mock.stop)
        self.planner.side_effect = AssertionError("Catalog replies must not use the general planner")
        self.localize.side_effect = lambda request, response, started: response

    def route(self, request, code=None, **changes):
        message = request.conversation.recentMessages[-1]
        self.scope.return_value = (scope_for(message, selectionCode=code,
                selectionAttempted=True, selectionEvidence=message.text, selectionConfidence=1, **changes), result({}).usage)
        return plan_v2_turn(request)

    def test_typed_codes_and_labels_follow_the_exact_button_path(self):
        for text, code in [("Pool 1", "POOL_1"), ("pool 2", "POOL_2"), ("POOL_1", "POOL_1"),
                           ("Alberca 1", "POOL_1"), ("Muelle 2", "DOCK_2"),
                           ("dock 1", "DOCK_1"), ("Habitaci\u00f3n", "ROOM"), ("room", "ROOM")]:
            with self.subTest(text=text):
                request = selection_request(text)
                button = request.model_copy(deep=True)
                button.conversation.recentMessages[-1].interactionReplyId = f"field:ROOM_SERVICE:deliveryLocation:{code}"
                typed_response, button_response = plan_v2_turn(request), plan_v2_turn(button)
                self.assertEqual(button_response.updatedConversationSummary, typed_response.updatedConversationSummary)
                self.assertEqual(button_response.messages[0].text, typed_response.messages[0].text)
                self.assertIn("https://hotel.example/menu", typed_response.messages[0].text)
                self.assertEqual(code, json.loads(typed_response.updatedConversationSummary.splitlines()[-1])["capturedFields"]["deliveryLocation"])
                self.assertFalse(typed_response.toolCalls)
                self.assertIsNone(request.conversation.recentMessages[-1].interactionReplyId)
                self.assertEqual(text, request.conversation.recentMessages[-1].text)
        self.scope.assert_not_called()

    def test_approved_translations_match_without_a_model_call(self):
        for locale, text in [("fr", "Piscine 1"), ("de", "Schwimmbecken 1"), ("ja", "\u30d7\u30fc\u30eb1")]:
            request = selection_request(text)
            request.guest.preferredLanguage = locale
            request.availableOfferings[0].inputSchema["x-chatbotinn-localization"] = {
                "variants": {locale: {"Alberca 1": text}}}
            response = plan_v2_turn(request)
            self.assertIn('"deliveryLocation":"POOL_1"', response.updatedConversationSummary)
        self.scope.assert_not_called()

    def test_translated_or_polite_text_uses_bounded_scope_result(self):
        for text in ["Please deliver to Pool 1", "\u00c0 la piscine 1, s'il vous pla\u00eet", "Am Schwimmbecken 1 bitte",
                     "\u30d7\u30fc\u30eb1\u306b\u304a\u9858\u3044\u3057\u307e\u3059"]:
            with self.subTest(text=text):
                request = selection_request(text)
                response = self.route(request, "POOL_1")
                self.assertIn('"deliveryLocation":"POOL_1"', response.updatedConversationSummary)
                self.assertIn("https://hotel.example/menu", response.messages[0].text)
                self.assertEqual(15, response.usage.totalTokens)
                self.assertFalse(response.toolCalls)

    def test_existing_order_is_preserved_and_confirmed_not_requested_again(self):
        items = [{"name": "burgers", "quantity": 2, "modifications": ["without onions"]}]
        for text, semantic in [("Pool 1", False), ("At Pool 1 please", True)]:
            request = selection_request(text, {"items": items})
            response = self.route(request, "POOL_1") if semantic else plan_v2_turn(request)
            state = json.loads(response.updatedConversationSummary)
            self.assertEqual(items, state["capturedFields"]["items"])
            self.assertEqual("POOL_1", state["capturedFields"]["deliveryLocation"])
            self.assertTrue(state["awaitingExplicitConfirmation"])
            self.assertIn("https://hotel.example/menu", response.messages[0].text)
            self.assertFalse(response.toolCalls)

    def test_location_edit_preserves_exact_items_and_requires_new_confirmation(self):
        items = [{"name": "Pechuga  Herbal", "quantity": 2, "modifications": ["sin  sal"]}]
        for awaiting in [False, True]:
            for text, semantic in [("ROOM", False), ("I want to receive in my room", True)]:
                with self.subTest(awaiting=awaiting, text=text):
                    request = selection_request(text, {"deliveryLocation": "POOL_1", "items": items})
                    state = json.loads(request.conversation.summary)
                    state.update(awaitingExplicitConfirmation=awaiting,
                                 phase="AWAITING_CONFIRMATION" if awaiting else "CAPTURING_ITEMS")
                    request.conversation.summary = json.dumps(state)
                    original = request.model_dump(mode="json")
                    with patch("app.services.input_understanding.call_openai_json_result",
                               side_effect=AssertionError("Location is not an item edit")):
                        response = self.route(request, "ROOM") if semantic else plan_v2_turn(request)
                    captured = json.loads(response.updatedConversationSummary)
                    self.assertEqual({"deliveryLocation": "ROOM", "items": items}, captured["capturedFields"])
                    self.assertTrue(captured["awaitingExplicitConfirmation"])
                    self.assertEqual("CONFIRMATION", response.messages[0].purpose)
                    self.assertFalse(response.toolCalls)
                    self.assertEqual(original, request.model_dump(mode="json"))
                    confirm = follow_up(request, response, "Confirm", current_room_button(response))
                    confirmed = plan_v2_turn(confirm)
                    self.assertEqual(captured["capturedFields"], confirmed.toolCalls[0].arguments["input"])

    def test_ambiguous_location_edit_keeps_draft_but_disarms_confirmation(self):
        fields = {"deliveryLocation": "ROOM", "items": [
            {"name": "soup", "quantity": 1, "modifications": ["without salt"]}]}
        for text, code, changes in [
            ("Pool 1 or Pool 2", None, {}), ("Not Pool 1", None, {}),
            ("Pool 3", "POOL_1", {}), ("At Pool 1", "POOL_99", {}),
            ("Maybe Pool 1", "POOL_1", {"selectionConfidence": .5}),
            ("Not Pool 1", "POOL_1", {"selectionEvidence": "Pool 1"}),
        ]:
            with self.subTest(text=text):
                request = selection_request(text, fields)
                state = json.loads(request.conversation.summary)
                state["awaitingExplicitConfirmation"] = True
                request.conversation.summary = json.dumps(state)
                message = request.conversation.recentMessages[-1]
                scope = scope_for(message, selectionCode=code, selectionAttempted=True,
                                  selectionEvidence=text, selectionConfidence=1).model_copy(update=changes)
                self.scope.return_value = (scope, result({}).usage)
                response = plan_v2_turn(request)
                state = json.loads(response.updatedConversationSummary)
                self.assertEqual(fields, state["capturedFields"])
                self.assertFalse(state["awaitingExplicitConfirmation"])
                self.assertEqual(5, len(response.messages[0].interaction.options))
                self.assertFalse(response.toolCalls)
                confirm = follow_up(request, response, "Confirm", "confirmation:ROOM_SERVICE:CONFIRM")
                self.assertFalse(plan_v2_turn(confirm).toolCalls)
                clarified = follow_up(request, response, "Pool 2", "field:ROOM_SERVICE:deliveryLocation:POOL_2")
                recovered = plan_v2_turn(clarified)
                self.assertEqual(fields["items"], json.loads(recovered.updatedConversationSummary)["capturedFields"]["items"])
                self.assertTrue(json.loads(recovered.updatedConversationSummary)["awaitingExplicitConfirmation"])

    def test_item_edit_does_not_resolve_an_ambiguous_location(self):
        fields = {"deliveryLocation": "ROOM", "items": [
            {"name": "soup", "quantity": 1, "modifications": ["without salt"]}]}
        request = selection_request("Pool 1 or Pool 2", fields)
        response = self.route(request)
        request = follow_up(request, response, "Make it two soups")
        message = request.conversation.recentMessages[-1]
        self.scope.return_value = (scope_for(message), result({}).usage)
        extracted = order(edit(message.text, quantity=2, quantity_quote="two",
                               action="UPDATE", existingItemIndex=0), mode="PATCH")
        with patch("app.services.input_understanding.call_openai_json_result", return_value=result(extracted)):
            response = plan_v2_turn(request)
        state = json.loads(response.updatedConversationSummary)
        self.assertFalse(state["awaitingExplicitConfirmation"])
        self.assertEqual(2, state["capturedFields"]["items"][0]["quantity"])
        self.assertEqual(["without salt"], state["capturedFields"]["items"][0]["modifications"])
        self.assertFalse(response.toolCalls)
        for text, action in [("Confirm", "CONFIRM"), ("Change", "CHANGE")]:
            request = follow_up(request, response, text, f"confirmation:ROOM_SERVICE:{action}")
            response = plan_v2_turn(request)
            self.assertFalse(json.loads(response.updatedConversationSummary)["awaitingExplicitConfirmation"])
            self.assertEqual(5, len(response.messages[0].interaction.options))
            self.assertFalse(response.toolCalls)

    def test_ambiguous_unavailable_and_invalid_evidence_preserve_draft(self):
        for text, code, changes in [
            ("Pool 1 or Pool 2", None, {}), ("Not Pool 1", None, {}), ("Pool 3", "POOL_1", {}),
            ("Pool 1 or Pool 2", None, {"replyAction": "AMBIGUOUS"}),
            ("At the pool", None, {}), ("Maybe Pool 1", "POOL_1", {"selectionConfidence": .5}),
            ("At Pool 1", "POOL_99", {}), ("Not Pool 1", "POOL_1", {"selectionEvidence": "Pool 1"}),
        ]:
            with self.subTest(text=text, changes=changes):
                request = selection_request(text, {"items": [{"name": "soup", "quantity": 1, "modifications": []}]})
                message = request.conversation.recentMessages[-1]
                scope = scope_for(message, selectionCode=code, selectionAttempted=True, selectionEvidence=text, selectionConfidence=1)
                scope = scope.model_copy(update=changes)
                self.scope.return_value = (scope, result({}).usage)
                response = plan_v2_turn(request)
                self.assertEqual(request.conversation.summary, response.updatedConversationSummary)
                self.assertEqual(5, len(response.messages[0].interaction.options))
                self.assertFalse(response.toolCalls)

    def test_duplicate_translated_labels_are_not_silently_resolved(self):
        request = selection_request("Pool")
        request.availableOfferings[0].inputSchema["x-chatbotinn-localization"] = {
            "variants": {"en": {"Alberca 1": "Pool", "Alberca 2": "Pool"}}}
        selection = pending_catalog_selection(request, _latest_capture_state(request))
        self.assertIsNone(selection.exact_code("Pool"))
        message = request.conversation.recentMessages[-1]
        self.assertIsNone(selection.semantic_code(scope_for(message, selectionCode="POOL_1",
                        selectionEvidence="Pool", selectionConfidence=1), "Pool"))

    def test_capture_selection_excludes_non_draft_and_non_guest_context(self):
        request = selection_request("Pool 1")
        state = _latest_capture_state(request)
        for changes in [{"pendingOffering": "MAINTENANCE"}, {"awaitingExplicitConfirmation": True},
                        {"readyToStart": True}, {"phase": "STARTING"}]:
            self.assertIsNone(pending_catalog_selection(request, dict(state, **changes)))
        request.trigger.conversationTaskId = uuid4()
        self.assertIsNone(pending_catalog_selection(request, state))

        request.trigger.conversationTaskId = None
        request.conversation.recentMessages[-1].conversationTaskIds = [uuid4()]
        self.assertIsNone(pending_catalog_selection(request, state))

    def test_editable_location_is_bounded_to_multilingual_room_draft(self):
        request = selection_request("ROOM", {"deliveryLocation": "POOL_1"})
        state = _latest_capture_state(request)
        self.assertEqual("POOL_1", pending_catalog_selection(request, state).current_value)
        for changes in [{"readyToStart": True}, {"phase": "STARTING"}]:
            self.assertIsNone(pending_catalog_selection(request, dict(state, **changes)))
        request.trigger.conversationTaskId = uuid4()
        self.assertIsNone(pending_catalog_selection(request, state))
        request.trigger.conversationTaskId = None
        request.conversation.recentMessages[-1].conversationTaskIds = [uuid4()]
        self.assertIsNone(pending_catalog_selection(request, state))
        request.conversation.recentMessages[-1].conversationTaskIds = []
        request.availableOfferings[0].offeringCode = "DINING"
        self.assertIsNone(pending_catalog_selection(request, dict(state, pendingOffering="DINING")))
        request.availableOfferings[0].offeringCode = "ROOM_SERVICE"
        request.trigger.eventPayload.pop("languageContext", None)
        self.assertIsNone(pending_catalog_selection(request, state))

    def test_pending_maintenance_does_not_block_room_service_selection(self):
        request = selection_request("Pool 1")
        request.activeOperations = [task_operation("MAINTENANCE_RESOLUTION")]
        request.conversation.focusedConversationTaskId = request.activeOperations[0].pendingConversationTasks[0].conversationTaskId
        response = plan_v2_turn(request)
        self.assertIn('"deliveryLocation":"POOL_1"', response.updatedConversationSummary)
        self.assertIn("https://hotel.example/menu", response.messages[0].text)
        self.assertFalse(response.toolCalls)

    def test_new_maintenance_request_is_not_consumed_as_a_location(self):
        request = selection_request("Please send maintenance to pool 1")
        request.availableOfferings.append(OfferingCapability.model_validate(maintenance_offering()))
        response = self.route(request, "POOL_1", kind="SERVICE_REQUEST", offeringCode="MAINTENANCE", hasRequestDetails=False)
        self.assertIn("MAINTENANCE", response.updatedConversationSummary)
        self.assertNotIn("https://hotel.example/menu", response.messages[0].text)
        self.assertNotIn('"deliveryLocation":"POOL_1"', response.updatedConversationSummary)

    def test_generic_catalog_does_not_depend_on_room_service_codes(self):
        request = selection_request("Terrace")
        offering = request.availableOfferings[0]
        offering.offeringCode = "DINING"
        catalog = offering.inputSchema["properties"]["deliveryLocation"]["x-chatbotinn-capture"]["catalog"]
        catalog["options"] = [{"code": "TERRACE", "label": "Terraza"}]
        request.conversation.summary = json.dumps({"pendingOffering": "DINING", "capturedFields": {"notes": "allergy"}})
        response = plan_v2_turn(request)
        state = json.loads(response.updatedConversationSummary.splitlines()[-1])
        self.assertEqual({"notes": "allergy", "deliveryLocation": "TERRACE"}, state["capturedFields"])


class CatalogScopeContractTest(unittest.TestCase):
    def test_scope_receives_current_location_and_configured_alternatives_for_edit(self):
        request = selection_request("I want to receive in my room", {"deliveryLocation": "POOL_1"})
        state = _latest_capture_state(request)
        state["awaitingExplicitConfirmation"] = True
        message = request.conversation.recentMessages[-1]
        decision = scope_for(message, selectionCode="ROOM", selectionAttempted=True,
                             selectionEvidence=message.text, selectionConfidence=1)
        with patch("app.agents.v2_scope_router.call_openai_json_result", return_value=result(decision.model_dump())) as call:
            resolved, _ = classify_hotel_scope(request, message, state)
        context = json.loads(call.call_args.args[0].split("Context:\n")[1])
        self.assertEqual("POOL_1", context["pendingSelection"]["currentValue"])
        self.assertEqual("ROOM", resolved.selectionCode)
        self.assertEqual([None, "ROOM", "DOCK_1", "DOCK_2", "POOL_1", "POOL_2"],
                         call.call_args.kwargs["response_schema"]["properties"]["selectionCode"]["enum"])

    def test_scope_gets_only_available_choices_and_current_evidence(self):
        request = selection_request("At Pool 1 please")
        message = request.conversation.recentMessages[-1]
        decision = scope_for(message, selectionCode="POOL_1", selectionEvidence=message.text, selectionConfidence=1)
        with patch("app.agents.v2_scope_router.call_openai_json_result", return_value=result(decision.model_dump())) as call:
            resolved, _ = classify_hotel_scope(request, message, _latest_capture_state(request))
        schema = call.call_args.kwargs["response_schema"]
        self.assertEqual([None, "ROOM", "DOCK_1", "DOCK_2", "POOL_1", "POOL_2"], schema["properties"]["selectionCode"]["enum"])
        self.assertIn('"pendingSelection":', call.call_args.args[0])
        self.assertIn('"Alberca 1"', call.call_args.args[0])
        self.assertEqual("POOL_1", resolved.selectionCode)


@unittest.skipUnless(os.getenv("RUN_LIVE_CATALOG_EVALS") == "1", "Opt-in synthetic OpenAI catalog evaluation")
class LiveCatalogSelectionTest(unittest.TestCase):
    def test_multilingual_choices_and_nonchoices(self):
        for text, expected in [
            ("Please deliver to Pool 1", "POOL_1"),
            ("A la piscine 2, s'il vous plait", "POOL_2"),
            ("Am Schwimmbecken 1 bitte", "POOL_1"),
            ("\u30d7\u30fc\u30eb1\u306b\u304a\u9858\u3044\u3057\u307e\u3059", "POOL_1"),
            ("Pool 1 or Pool 2", None),
            ("Not Pool 1", None),
            ("What time does Pool 1 close?", None),
            ("Please deliver to Pool 3", None),
            ("Two burgers", None),
        ]:
            with self.subTest(text=text):
                request = selection_request(text)
                request.agentTurnId = uuid4()
                request.guest.displayName = "Test Guest"
                request.guest.roomNumber = "TEST"
                request.hotel.name = "Synthetic Hotel"
                request.hotel.hotelCode = "SYNTHETIC_TEST"
                state = _latest_capture_state(request)
                message = request.conversation.recentMessages[-1]
                scope, _ = classify_hotel_scope(request, message, state)
                selection = pending_catalog_selection(request, state)
                self.assertEqual(expected, selection.semantic_code(scope, text))
                if text == "Two burgers":
                    self.assertFalse(scope.selectionAttempted)
                elif text in {"Pool 1 or Pool 2", "Not Pool 1", "Please deliver to Pool 3"}:
                    self.assertTrue(scope.selectionAttempted)
