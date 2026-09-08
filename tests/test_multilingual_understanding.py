import copy
import json
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.agents import spa_turns
from app.agents.v2_scope_router import ScopeDecision
from app.agents.v2_turn_planner import plan_v2_turn, _maintenance_resolution_value, _validate_plan
from app.core.errors import AgentModelError, AgentTimeoutError
from app.schemas.v2_turns import OfferingCapability, OperationSnapshot
from app.services.input_understanding import (
    OrderExtraction, apply_order_extraction, understanding_turn, understanding_enabled,
    record_scope_action, semantic_action, understand_order,
)
from app.services.localized_content import localize_response, _cache
from app.services.openai_client import OpenAiJsonResult
from app.services.telemetry_client import OpenAiTokenUsage
from tests.test_spa_turns import request_for, follow_up, spa_operation
from tests.test_v2_turn_planner import guided_room_service_offering, operation, conversation_task
from tests.test_multilingual_foundation import response_for


def enable(request):
    request.trigger.eventPayload["languageContext"] = {"version": 1, "explicit": True}
    return request


def scope_for(message, action="NONE", **changes):
    return ScopeDecision(**{
        "kind": "CONTEXT_REPLY", "offeringCode": None, "relevantText": message.text,
        "hasRequestDetails": action == "NONE", "containsUnrelatedTopic": False, "confidence": 1,
        "replyAction": action, "replyActionEvidence": message.text, "replyActionConfidence": 1,
        **changes,
    })


def result(data):
    return OpenAiJsonResult(data, OpenAiTokenUsage(input_tokens=10, output_tokens=5, total_tokens=15), "extraction-test")


def edit(text, name=None, quantity=None, quantity_quote=None, modifiers=None, **changes):
    return {"action": "ADD", "existingItemIndex": None, "name": name, "nameEvidence": name,
            "quantity": quantity, "quantityEvidence": quantity_quote,
            "modifications": ([{"text": modifier, "evidence": modifier} for modifier in modifiers]
                              if modifiers is not None else None),
            "modificationAction": "APPEND" if modifiers is not None else "KEEP",
            "evidence": text, **changes}


def order(*edits, mode="REPLACE", status="RESOLVED", confidence=1):
    return {"status": status, "mode": mode, "edits": list(edits), "confidence": confidence}


def room_request(text, items=None, awaiting=False):
    request = enable(request_for(text))
    request.availableOfferings = [OfferingCapability.model_validate(guided_room_service_offering())]
    request.availableOfferings[0].requiresExplicitGuestConfirmation = True
    fields = {"deliveryLocation": "ROOM"}
    if items:
        fields["items"] = items
    request.conversation.summary = json.dumps({"pendingOffering": "ROOM_SERVICE", "capturedFields": fields,
                                               "awaitingExplicitConfirmation": awaiting})
    return request


def task_operation(kind):
    op = operation(str(uuid4()))
    op["offeringCode"] = "MAINTENANCE" if kind.startswith("MAINTENANCE") else "ROOM_SERVICE"
    task = conversation_task(str(uuid4()), op["operationId"])
    task["taskType"] = kind
    task["requiredOutputSchema"] = (
        {"type": "object", "required": ["resolved"], "properties": {"resolved": {"type": "boolean"}}}
        if kind.startswith("MAINTENANCE") else
        {"type": "object", "required": ["decision"], "properties": {"decision": {"enum": ["CHANGE", "CANCEL"]}}}
        if kind.endswith("DECISION") else
        {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "minItems": 1}}})
    op["pendingConversationTasks"] = [task]
    return OperationSnapshot.model_validate(op)


class OrderEvidenceTest(unittest.TestCase):
    def test_multilingual_names_quantities_and_negations_remain_original(self):
        samples = [
            ("deux soupes sans sel", "soupes", 2, "deux", "sans sel"),
            ("zwei Kaffee ohne Milch", "Kaffee", 2, "zwei", "ohne Milch"),
            ("dois sucos sem gelo", "sucos", 2, "dois", "sem gelo"),
            ("寿司を二つ、わさび抜き", "寿司", 2, "二つ", "わさび抜き"),
            ("٢ قهوة بدون سكر", "قهوة", 2, "٢", "بدون سكر"),
            ("两份面条，不要花生", "面条", 2, "两份", "不要花生"),
            ("２ tacos sin queso", "tacos", 2, "２", "sin queso"),
        ]
        for text, name, qty, quote, modifier in samples:
            with self.subTest(text=text):
                data = OrderExtraction.model_validate(order(edit(text, name, qty, quote, [modifier])))
                items = apply_order_extraction(data, text, [])
                self.assertEqual([{"name": name, "quantity": qty, "modifications": [modifier]}], items)

    def test_rejects_invented_names_counts_and_negation_changes(self):
        text = "٢ قهوة بدون سكر"
        original = order(edit(text, "قهوة", 2, "٢", ["بدون سكر"]))
        changes = [
            {"quantity": 3}, {"quantityEvidence": "3"}, {"name": "coffee"},
            {"modifications": [{"text": "سكر", "evidence": "بدون سكر"}]},
            {"evidence": "old message"}, {"quantity": None, "quantityEvidence": None},
        ]
        for change in changes:
            with self.subTest(change=change):
                data = copy.deepcopy(original)
                data["edits"][0].update(change)
                with self.assertRaises(ValueError):
                    apply_order_extraction(OrderExtraction.model_validate(data), text, [])

    def test_uncertain_ranges_and_decimal_counts_never_default_to_one(self):
        for text, quote in [("2 o 3 tacos", "2 o 3"), ("1.5 tacos", "1.5"), ("-2 tacos", "-2"), ("tacos", None)]:
            data = order(edit(text, "tacos", 2 if quote else None, quote))
            with self.subTest(text=text), self.assertRaises(ValueError):
                apply_order_extraction(OrderExtraction.model_validate(data), text, [])
        with self.assertRaises(ValueError):
            apply_order_extraction(OrderExtraction.model_validate(order(status="AMBIGUOUS")), "a few tacos", [])

    def test_partial_edits_keep_untouched_items_and_allergies(self):
        original = [{"name": "soupe", "quantity": 1, "modifications": ["sans arachides"]},
                    {"name": "cafe", "quantity": 2, "modifications": []}]
        text = "deux soupes, sans sel"
        data = order(edit(text, quantity=2, quantity_quote="deux", modifiers=["sans sel"],
                          action="UPDATE", existingItemIndex=0), mode="PATCH")
        updated = apply_order_extraction(OrderExtraction.model_validate(data), text, original)
        self.assertEqual(["sans arachides", "sans sel"], updated[0]["modifications"])
        self.assertEqual(2, updated[0]["quantity"])
        self.assertEqual(original[1], updated[1])
        self.assertEqual(1, original[0]["quantity"])

    def test_remove_one_item_is_not_cancelling_the_request(self):
        original = [{"name": name, "quantity": 1, "modifications": []} for name in ("cafe", "soupe")]
        data = order(edit("retirez le cafe", action="REMOVE", existingItemIndex=0), mode="PATCH")
        updated = apply_order_extraction(OrderExtraction.model_validate(data), "retirez le cafe", original)
        self.assertEqual([original[1]], updated)
        with self.assertRaises(ValueError):
            apply_order_extraction(OrderExtraction.model_validate(data), "retirez le cafe", original[:1])

    def test_kitchen_replacement_rejects_partial_edits(self):
        data = order(edit("deux", quantity=2, quantity_quote="deux", action="UPDATE", existingItemIndex=0), mode="PATCH")
        with self.assertRaises(ValueError):
            apply_order_extraction(OrderExtraction.model_validate(data), "deux", [{"name": "cafe", "quantity": 1}], True)

    @patch("app.services.input_understanding.call_openai_json_result")
    def test_one_bounded_call_per_identical_turn_input_and_no_context_leak(self, model):
        request = room_request("deux soupes")
        message = request.conversation.recentMessages[-1]
        model.return_value = result(order(edit(message.text, "soupes", 2, "deux")))
        with understanding_turn(request) as state:
            self.assertTrue(understanding_enabled())
            first = understand_order(request, message, [])
            self.assertEqual(first, understand_order(request, message, []))
            self.assertEqual(15, state.usage["totalTokens"])
        self.assertFalse(understanding_enabled())
        self.assertEqual(1, model.call_count)
        self.assertLessEqual(model.call_args.kwargs["timeout_seconds"], 8)
        with understanding_turn(request):
            understand_order(request, message, [])
        self.assertEqual(2, model.call_count)


class UnderstandingFlowTest(unittest.TestCase):
    def setUp(self):
        self.action = "NONE"
        self.kind = "CONTEXT_REPLY"
        self.offering = None
        for target, name in [
            ("app.agents.v2_turn_planner.classify_hotel_scope", "scope"),
            ("app.agents.v2_turn_planner.call_openai_json_result", "planner"),
            ("app.services.input_understanding.call_openai_json_result", "order_model"),
            ("app.agents.spa_turns.call_openai_json_result", "spa_model"),
            ("app.agents.v2_turn_planner.localize_response", "localizer"),
        ]:
            patcher = patch(target)
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)
        self.scope.side_effect = lambda r, m, s: (scope_for(m, self.action, kind=self.kind, offeringCode=self.offering), OpenAiTokenUsage())
        self.localizer.side_effect = lambda request, response, started: response
        self.planner.side_effect = AssertionError("Unexpected general planner fallback")
        self.order_model.side_effect = AssertionError("Unexpected order extraction")
        self.spa_model.side_effect = AssertionError("Unexpected SPA extraction")

    def test_order_capture_edit_then_confirmation_preserves_current_evidence(self):
        request = room_request("deux soupes sans sel")
        before = request.model_dump(mode="json")
        self.order_model.side_effect = None
        self.order_model.return_value = result(order(edit("deux soupes sans sel", "soupes", 2, "deux", ["sans sel"])))
        response = plan_v2_turn(request)
        self.assertEqual([], response.toolCalls)
        self.assertIn("sans sel", response.messages[0].text)
        self.assertEqual(before, request.model_dump(mode="json"))
        self.assertEqual(15, response.usage.totalTokens)
        request = follow_up(request, response, "trois soupes")
        self.order_model.return_value = result(order(edit("trois soupes", quantity=3, quantity_quote="trois",
                                                        action="UPDATE", existingItemIndex=0), mode="PATCH"))
        changed = plan_v2_turn(request)
        request = follow_up(request, changed, "oui, je confirme")
        self.action = "CONFIRM"
        confirmed = plan_v2_turn(request)
        self.assertEqual("START_SERVICE", confirmed.toolCalls[0].toolName)
        self.assertEqual([request.trigger.messageId], confirmed.toolCalls[0].evidenceMessageIds)
        self.assertEqual([{ "name": "soupes", "quantity": 3, "modifications": ["sans sel"]}],
                         confirmed.toolCalls[0].arguments["input"]["items"])
        self.assertEqual(2, self.order_model.call_count)

    def test_order_details_in_first_message_survive_location_selection(self):
        request = room_request("deux soupes sans sel")
        request.conversation.summary = "{}"
        self.kind, self.offering = "SERVICE_REQUEST", "ROOM_SERVICE"
        self.order_model.side_effect = None
        self.order_model.return_value = result(order(edit("deux soupes sans sel", "soupes", 2, "deux", ["sans sel"])))
        response = plan_v2_turn(request)
        self.assertEqual([], response.toolCalls)
        self.assertEqual("soupes", json.loads(response.updatedConversationSummary)["capturedFields"]["items"][0]["name"])
        request = follow_up(request, response, "Habitacion", "field:ROOM_SERVICE:deliveryLocation:ROOM")
        selected = plan_v2_turn(request)
        captured = json.loads(selected.updatedConversationSummary)
        self.assertTrue(captured["awaitingExplicitConfirmation"])
        self.assertIn("sans sel", selected.messages[0].text)
        self.assertIn("https://", selected.messages[0].text)
        self.assertEqual(1, self.order_model.call_count)

    def test_long_order_confirmation_never_truncates_restrictions_in_buttons(self):
        name = "soupe"
        modifier = "sans " + "sel " * 85
        text = "deux soupes " + modifier
        request = room_request(text)
        self.order_model.side_effect = None
        self.order_model.return_value = result(order(*[
            edit(text, name, 2, "deux", [modifier.strip()]) for _ in range(4)]))
        response = plan_v2_turn(request)
        self.assertIsNone(response.messages[0].interaction)
        self.assertEqual(4, response.messages[0].text.count(modifier.strip()))

    def test_wrong_resolution_cannot_bypass_validation(self):
        request = enable(request_for("non, ce n'est pas résolu"))
        request.activeOperations = [task_operation("MAINTENANCE_RESOLUTION_CONFIRMATION")]
        self.action = "NOT_RESOLVED"
        response = plan_v2_turn(request)
        response.toolCalls[0].arguments["result"]["resolved"] = True
        with understanding_turn(request):
            message = request.conversation.recentMessages[-1]
            record_scope_action(message, scope_for(message, "NOT_RESOLVED"))
            with self.assertRaises(AgentModelError):
                _validate_plan(request, response)

    def test_timeout_returns_clarification_and_disarms_old_confirmation(self):
        request = room_request("changez le plat", [{"name": "soupe", "quantity": 1, "modifications": []}], True)
        self.order_model.side_effect = AgentTimeoutError("slow model")
        response = plan_v2_turn(request)
        self.assertEqual([], response.toolCalls)
        self.assertFalse(json.loads(response.updatedConversationSummary)["awaitingExplicitConfirmation"])
        self.assertEqual(1, self.order_model.call_count)

    def test_multilingual_maintenance_decisions_and_negations(self):
        for text, action, expected in [("Ja, es funktioniert wieder", "RESOLVED", True),
                                       ("Non, ce n'est pas encore reparé", "NOT_RESOLVED", False),
                                       ("まだ直っていません", "NOT_RESOLVED", False),
                                       ("تم حل المشكلة", "RESOLVED", True)]:
            with self.subTest(text=text):
                request = enable(request_for(text))
                request.activeOperations = [task_operation("MAINTENANCE_RESOLUTION_CONFIRMATION")]
                self.action = action
                response = plan_v2_turn(request)
                self.assertIs(response.toolCalls[0].arguments["result"]["resolved"], expected)

    def test_ambiguous_resolution_never_completes(self):
        request = enable(request_for("Oui, mais le problème continue"))
        request.activeOperations = [task_operation("MAINTENANCE_RESOLUTION_CONFIRMATION")]
        self.action = "AMBIGUOUS"
        response = plan_v2_turn(request)
        self.assertEqual([], response.toolCalls)
        self.assertEqual(2, len(response.messages[0].interaction.options))

    def test_multiple_pending_tasks_require_target_for_generic_yes(self):
        request = enable(request_for("ja"))
        request.activeOperations = [task_operation("MAINTENANCE_RESOLUTION_CONFIRMATION"), spa_operation()]
        self.action = "CONFIRM"
        response = plan_v2_turn(request)
        self.assertEqual([], response.toolCalls)
        self.assertIn("folio", response.messages[0].text)

    def test_room_draft_confirmation_does_not_complete_pending_maintenance(self):
        request = room_request("sim, confirmo", [{"name": "sopa", "quantity": 2, "modifications": []}], True)
        request.activeOperations = [task_operation("MAINTENANCE_RESOLUTION_CONFIRMATION")]
        self.action = "CONFIRM"
        response = plan_v2_turn(request)
        self.assertEqual("START_SERVICE", response.toolCalls[0].toolName)
        self.assertEqual("ROOM_SERVICE", response.toolCalls[0].arguments["offeringCode"])

    def test_new_spa_request_is_allowed_while_maintenance_waits(self):
        request = enable(request_for("Je voudrais réserver au spa"))
        request.activeOperations = [task_operation("MAINTENANCE_RESOLUTION_CONFIRMATION")]
        self.kind, self.offering = "SERVICE_REQUEST", "SPA"
        self.scope.side_effect = lambda r, m, s: (scope_for(m, kind="SERVICE_REQUEST", offeringCode="SPA", hasRequestDetails=False), OpenAiTokenUsage())
        response = plan_v2_turn(request)
        self.assertEqual([], response.toolCalls)
        self.assertEqual("SPA", json.loads(response.updatedConversationSummary)["pendingOffering"])

    def test_kitchen_complete_replacement_is_extracted_once(self):
        request = enable(request_for("zwei Suppen ohne Salz"))
        request.activeOperations = [task_operation("ROOM_SERVICE_ORDER_CHANGE_DETAILS")]
        self.order_model.side_effect = None
        self.order_model.return_value = result(order(edit("zwei Suppen ohne Salz", "Suppen", 2, "zwei", ["ohne Salz"])))
        response = plan_v2_turn(request)
        self.assertEqual(1, self.order_model.call_count)
        self.assertEqual(1, len(response.toolCalls))
        self.assertEqual([request.trigger.messageId], response.toolCalls[0].evidenceMessageIds)
        self.assertEqual("Suppen", response.toolCalls[0].arguments["result"]["items"][0]["name"])

    def test_kitchen_change_decision_in_portuguese(self):
        request = enable(request_for("quero alterar o pedido"))
        request.activeOperations = [task_operation("ROOM_SERVICE_KITCHEN_CHANGE_DECISION")]
        self.action = "CHANGE"
        response = plan_v2_turn(request)
        self.assertEqual({"decision": "CHANGE"}, response.toolCalls[0].arguments["result"])

    def test_focused_spa_decision_is_not_consumed_by_maintenance(self):
        request = enable(request_for("j'accepte"))
        spa = spa_operation()
        request.activeOperations = [task_operation("MAINTENANCE_RESOLUTION_CONFIRMATION"), spa]
        request.conversation.focusedConversationTaskId = spa.pendingConversationTasks[0].conversationTaskId
        self.action = "CONFIRM"
        response = plan_v2_turn(request)
        self.assertEqual(spa.operationId, response.toolCalls[0].targetOperationId)
        self.assertEqual({"decision": "ACCEPT"}, response.toolCalls[0].arguments["result"])

    def test_spa_multilingual_capture_still_requires_current_confirmation(self):
        request = enable(request_for("massage le 2026-09-02 à 17:00"))
        self.kind, self.offering = "SERVICE_REQUEST", "SPA"
        self.spa_model.side_effect = None
        self.spa_model.return_value = result({key: {"status": "RESOLVED", "value": value,
            "evidence": value, "confidence": 1} for key, value in {
                "serviceName": "massage", "reservationDate": "2026-09-02", "reservationTime": "17:00"}.items()})
        response = plan_v2_turn(request)
        self.assertEqual([], response.toolCalls)
        self.assertTrue(json.loads(response.updatedConversationSummary)["spaDraft"]["awaitingConfirmation"])
        request = follow_up(request, response, "je confirme")
        self.action, self.kind, self.offering = "CONFIRM", "CONTEXT_REPLY", None
        response = plan_v2_turn(request)
        self.assertEqual("START_SERVICE", response.toolCalls[0].toolName)
        self.assertEqual("SPA", response.toolCalls[0].arguments["offeringCode"])
        self.assertEqual([request.trigger.messageId], response.toolCalls[0].evidenceMessageIds)


class SemanticDecisionTest(unittest.TestCase):
    def test_full_message_evidence_confidence_and_pure_intent_required(self):
        request = enable(request_for("do not confirm"))
        message = request.conversation.recentMessages[-1]
        for changes in [{"replyActionEvidence": "confirm"}, {"replyActionConfidence": 0.7},
                        {"kind": "SERVICE_REQUEST"}, {"containsUnrelatedTopic": True}]:
            with self.subTest(changes=changes), understanding_turn(request):
                record_scope_action(message, scope_for(message, "CONFIRM", **changes))
                self.assertIsNone(semantic_action(message.text))
                self.assertIsNone(_maintenance_resolution_value(message))
        self.assertIsNone(semantic_action(message.text))

    def test_legacy_without_handshake_keeps_its_existing_path(self):
        request = request_for("ya quedó resuelto")
        with understanding_turn(request):
            self.assertFalse(understanding_enabled())
            self.assertTrue(_maintenance_resolution_value(request.conversation.recentMessages[-1]))


class SpaMultilingualExtractionTest(unittest.TestCase):
    def extract(self, text, values, evidence, confidence=1, captured=None):
        request = enable(request_for(text))
        data = {key: {"status": "RESOLVED" if key in values else "UNCHANGED", "value": values.get(key),
                      "evidence": evidence.get(key), "confidence": confidence} for key in spa_turns.FIELDS}
        with understanding_turn(request), patch("app.agents.spa_turns.call_openai_json_result", return_value=result(data)) as model:
            response = spa_turns._extract(request, request.conversation.recentMessages[-1], captured or {}, {})
        self.assertEqual(1, model.call_count)
        self.assertLessEqual(model.call_args.kwargs["timeout_seconds"], 8)
        return response

    def test_relative_dates_use_hotel_clock_and_original_treatment(self):
        text = "Je voudrais un massage demain à dix-sept heures"
        fields, unresolved, _, _ = self.extract(text,
            {"serviceName": "massage", "reservationDate": "2026-08-31", "reservationTime": "17:00"},
            {"serviceName": "massage", "reservationDate": "demain", "reservationTime": "dix-sept heures"})
        self.assertEqual("2026-08-31", fields["reservationDate"])
        self.assertEqual("massage", fields["serviceName"])
        self.assertEqual({}, unresolved)

    def test_ambiguous_numeric_date_and_bare_time_clear_stale_values(self):
        fields, unresolved, _, _ = self.extract("03/04 5:00",
            {"reservationDate": "2027-04-03", "reservationTime": "17:00"},
            {"reservationDate": "03/04", "reservationTime": "5:00"},
            captured={"reservationDate": "2026-09-02", "reservationTime": "18:00"})
        self.assertEqual({}, fields)
        self.assertEqual({"reservationDate", "reservationTime"}, set(unresolved))

    def test_explicit_iso_and_non_latin_digits_are_preserved(self):
        fields, unresolved, _, _ = self.extract("٢٠٢٦-٠٩-٠٢ ١٧:٣٠",
            {"reservationDate": "2026-09-02", "reservationTime": "17:30"},
            {"reservationDate": "٢٠٢٦-٠٩-٠٢", "reservationTime": "١٧:٣٠"})
        self.assertEqual("2026-09-02", fields["reservationDate"])
        self.assertEqual("17:30", fields["reservationTime"])
        self.assertEqual({}, unresolved)

    def test_changed_numeric_date_and_missing_evidence_require_clarification(self):
        fields, unresolved, _, _ = self.extract("2026-09-02",
            {"reservationDate": "2026-09-03", "serviceName": "massage"},
            {"reservationDate": "2026-09-02", "serviceName": "old message"},
            captured={"serviceName": "sauna"})
        self.assertEqual({}, fields)
        self.assertEqual({"reservationDate", "serviceName"}, set(unresolved))

    def test_low_confidence_does_not_resolve_and_timeout_does_not_raise(self):
        fields, unresolved, _, _ = self.extract("demain", {"reservationDate": "2026-08-31"},
                                              {"reservationDate": "demain"}, confidence=0.6)
        self.assertEqual({}, fields)
        self.assertIn("reservationDate", unresolved)
        request = enable(request_for("demain"))
        with understanding_turn(request), patch("app.agents.spa_turns.call_openai_json_result", side_effect=AgentTimeoutError("timeout")):
            fields, unresolved, usage, changed = spa_turns._extract(request, request.conversation.recentMessages[-1], {}, {})
        self.assertFalse(changed)


class OriginalDataLocalizationTest(unittest.TestCase):
    def test_order_names_and_restrictions_are_masked_before_translation(self):
        request = room_request("deux soupes sans sel")
        request.guest.preferredLanguage = "fr"
        response = response_for(request, "Order: 2 x soupes (sans sel)")
        response.updatedConversationSummary = json.dumps({"capturedFields": {"items": [
            {"name": "soupes", "quantity": 2, "modifications": ["sans sel"]}]}})
        _cache.clear()
        def translate(prompt, **kwargs):
            self.assertNotIn("soupes", prompt)
            self.assertNotIn("sans sel", prompt)
            texts = json.loads(prompt.split("\n", 1)[1])
            return result({"texts": [text.replace("Order", "Commande") for text in texts]})
        with patch("app.services.localized_content.call_openai_json_result", side_effect=translate):
            localized = localize_response(request, response, time.perf_counter())
        self.assertEqual("Commande: 2 x soupes (sans sel)", localized.messages[0].text)
        self.assertEqual(response.updatedConversationSummary, localized.updatedConversationSummary)


if __name__ == "__main__":
    unittest.main()
