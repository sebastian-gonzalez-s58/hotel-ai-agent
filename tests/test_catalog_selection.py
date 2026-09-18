import json
import os
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.agents.v2_scope_router import classify_hotel_scope
from app.agents.v2_turn_planner import plan_v2_turn, _latest_capture_state
from app.services.catalog_selection import pending_catalog_selection
from app.schemas.v2_turns import OfferingCapability
from tests.test_multilingual_understanding import room_request, scope_for, result
from tests.test_v2_scope_router import maintenance_offering


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
                selectionEvidence=message.text, selectionConfidence=1, **changes), result({}).usage)
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
                scope = scope_for(message, selectionCode=code, selectionEvidence=text, selectionConfidence=1)
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

    def test_only_current_unfilled_single_select_field_is_resolved(self):
        request = selection_request("Pool 1")
        state = _latest_capture_state(request)
        for changes in [{"pendingOffering": "MAINTENANCE"}, {"awaitingExplicitConfirmation": True},
                        {"capturedFields": {"deliveryLocation": "ROOM"}}, {"readyToStart": True}, {"phase": "STARTING"}]:
            self.assertIsNone(pending_catalog_selection(request, dict(state, **changes)))
        request.conversation.focusedConversationTaskId = uuid4()
        self.assertIsNone(pending_catalog_selection(request, state))
        request.conversation.focusedConversationTaskId = None
        request.conversation.recentMessages[-1].conversationTaskIds = [uuid4()]
        self.assertIsNone(pending_catalog_selection(request, state))

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
