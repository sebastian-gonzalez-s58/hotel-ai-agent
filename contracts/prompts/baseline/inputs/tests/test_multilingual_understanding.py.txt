from tests.conversation_regression.room_confirmation_support import current_room_button, use_presented_room_button
import copy
import json
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.agents import spa_turns
from app.agents.v2_scope_router import ScopeDecision
from app.agents.v2_turn_planner import plan_v2_turn, _maintenance_resolution_value, _validate_plan, _coerce_order_items
from app.core.errors import AgentDependencyError, AgentModelError, AgentTimeoutError
from app.prompts.input_policy import ORDER_QUANTITY_POLICY
from app.prompts.v2_turn import build_v2_turn_prompt
from app.schemas.v2_turns import OfferingCapability, OperationSnapshot
from app.services.input_understanding import (
    OrderExtraction, apply_order_extraction, understanding_turn, understanding_enabled,
    record_scope_action, semantic_action, understand_order,
    order_understanding_issue, order_understanding_failed,
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
    def test_partial_capture_keeps_complete_and_incomplete_items_but_kitchen_requires_all(self):
        text = "tacos sin sal y una sopa"
        data = OrderExtraction.model_validate(order(
            edit("tacos sin sal", "tacos", modifiers=["sin sal"]),
            edit("una sopa", "sopa", 1, "una")))
        items = apply_order_extraction(data, text, [], allow_partial=True)
        self.assertEqual([{"name": "tacos", "modifications": ["sin sal"]},
                          {"name": "sopa", "quantity": 1, "modifications": []}], items)
        with self.assertRaises(ValueError):
            apply_order_extraction(data, text, [], require_full=True, allow_partial=True)

    def test_short_quantity_can_only_fill_one_unambiguous_pending_item(self):
        items = [{"name": "sopa", "quantity": 1, "modifications": []},
                 {"name": "tacos", "modifications": ["sin sal"]}]
        def extraction(index):
            return OrderExtraction.model_validate(order(edit("Uno", quantity=1, quantity_quote="Uno",
                action="UPDATE", existingItemIndex=index), mode="PATCH"))
        updated = apply_order_extraction(extraction(1), "Uno", items, allow_partial=True)
        self.assertEqual(1, updated[1]["quantity"])
        self.assertEqual(items[0], updated[0])
        self.assertNotIn("quantity", items[1])
        for existing, index in [(items, 0), ([{"name": "sopa", "modifications": []}, items[1]], 1)]:
            with self.assertRaises(ValueError):
                apply_order_extraction(extraction(index), "Uno", existing, allow_partial=True)

    def test_product_quote_may_include_its_quantity_without_rejecting_the_product(self):
        text = "Hello, I would like two burgers without cheese, please"
        data = order(edit("two burgers without cheese", "burgers", 2, "two", ["without cheese"],
                          nameEvidence="two burgers"))
        items = apply_order_extraction(OrderExtraction.model_validate(data), text, [])
        self.assertEqual([{"name": "burgers", "quantity": 2, "modifications": ["without cheese"]}], items)
        data["edits"][0]["name"] = "sandwiches"
        with self.assertRaises(ValueError):
            apply_order_extraction(OrderExtraction.model_validate(data), text, [])

    def test_echoed_unchanged_values_do_not_require_new_evidence_but_edits_do(self):
        text = "Yes, but without onions as well"
        existing = [{"name": "burgers", "quantity": 2, "modifications": ["without cheese"]}]
        data = order(edit(text, "burgers", 2, None, ["without onions as well"],
                          action="UPDATE", existingItemIndex=0), mode="PATCH")
        items = apply_order_extraction(OrderExtraction.model_validate(data), text, existing)
        self.assertEqual(["without cheese", "without onions as well"], items[0]["modifications"])
        self.assertEqual(2, items[0]["quantity"])
        for changes in [{"quantity": 3}, {"name": "sandwiches", "nameEvidence": "sandwiches"}]:
            invalid = copy.deepcopy(data)
            invalid["edits"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                apply_order_extraction(OrderExtraction.model_validate(invalid), text, existing)
        wrong_count = order(edit("make it 3", quantity=2, quantity_quote="3", action="UPDATE", existingItemIndex=0), mode="PATCH")
        with self.assertRaises(ValueError):
            apply_order_extraction(OrderExtraction.model_validate(wrong_count), "make it 3", existing)

    def test_new_item_keep_label_cannot_discard_evidenced_restrictions(self):
        text = "My complete new order is two sandwiches without mustard"
        data = order(edit(text, "sandwiches", 2, "two", ["without mustard"], modificationAction="KEEP"))
        items = apply_order_extraction(OrderExtraction.model_validate(data), text, [], True)
        self.assertEqual([{"name": "sandwiches", "quantity": 2, "modifications": ["without mustard"]}], items)
        data["mode"] = "PATCH"
        data["edits"][0].update(action="UPDATE", existingItemIndex=0)
        with self.assertRaises(ValueError):
            apply_order_extraction(OrderExtraction.model_validate(data), text, items)

    def test_multilingual_capture_never_uses_legacy_quantity_defaults(self):
        request = room_request("burgers")
        valid = {"name": "soup", "quantity": 2, "modifications": []}
        with understanding_turn(request):
            self.assertEqual([], _coerce_order_items("burgers"))
            self.assertEqual([], _coerce_order_items([valid, {"name": "burger"}]))
            self.assertEqual([valid], _coerce_order_items([valid]))
        self.assertEqual(1, _coerce_order_items([{"name": "burger"}])[0]["quantity"])

    @patch("app.services.input_understanding.call_openai_json_result")
    def test_planner_and_extractor_share_quantity_policy(self, model):
        request = room_request("a burger without cheese")
        message = request.conversation.recentMessages[-1]
        model.return_value = result(order(edit(message.text, "burger", 1, "a", ["without cheese"])))
        with understanding_turn(request):
            self.assertEqual(1, understand_order(request, message, [])[0]["quantity"])
        self.assertIn(ORDER_QUANTITY_POLICY, build_v2_turn_prompt(request))
        self.assertIn(ORDER_QUANTITY_POLICY, model.call_args.args[0])
        self.assertIn("initial order uses REPLACE", model.call_args.args[0])
        self.assertNotIn("use quantity 1 instead", build_v2_turn_prompt(request))

    @patch("app.services.input_understanding.call_openai_json_result")
    def test_cached_failures_keep_reason_without_logging_guest_content(self, model):
        request = room_request("two burgers SECRET_GUEST_NOTE")
        message = request.conversation.recentMessages[-1]
        model.side_effect = AgentTimeoutError("SECRET_PROVIDER_PAYLOAD")
        with understanding_turn(request), self.assertLogs("chatbotinn-agent.input-understanding") as logs:
            self.assertIsNone(understand_order(request, message, []))
            self.assertIsNone(understand_order(request, message, []))
            self.assertEqual("TIMEOUT", order_understanding_issue())
            self.assertTrue(order_understanding_failed())
        self.assertIsNone(order_understanding_issue())
        self.assertEqual(1, model.call_count)
        self.assertIn("reason=TIMEOUT", " ".join(logs.output))
        self.assertNotIn("SECRET", " ".join(logs.output))

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

    def test_removal_with_contradictory_change_action_still_applies_the_specific_edit(self):
        items = [{"name": "hamburguesa", "quantity": 2, "modifications": ["sin queso"]},
                 {"name": "sopa", "quantity": 1, "modifications": []}]
        request = room_request("Quita la sopa", items, True)
        # Reproduce the actual classifier response: CHANGE plus concrete request details.
        self.scope.side_effect = lambda r, m, s: (scope_for(m, "CHANGE", hasRequestDetails=True), OpenAiTokenUsage())
        self.order_model.side_effect = None
        self.order_model.return_value = result(order(edit("Quita la sopa", action="REMOVE", existingItemIndex=1), mode="PATCH"))
        response = plan_v2_turn(request)
        fields = json.loads(response.updatedConversationSummary)["capturedFields"]
        self.assertEqual({"deliveryLocation": "ROOM", "items": items[:1]}, fields)
        self.assertTrue(json.loads(response.updatedConversationSummary)["awaitingExplicitConfirmation"])
        self.assertEqual([], response.toolCalls)
        self.assertEqual(1, self.order_model.call_count)
        confirm = follow_up(request, response, "Confirm", current_room_button(response))
        self.assertEqual(fields, plan_v2_turn(confirm).toolCalls[0].arguments["input"])

    def test_generic_change_without_details_still_asks_what_to_edit(self):
        items = [{"name": "sopa", "quantity": 1, "modifications": ["sin sal"]}]
        request = room_request("Quiero cambiar el pedido", items, True)
        self.action = "CHANGE"
        response = plan_v2_turn(request)
        fields = json.loads(response.updatedConversationSummary)["capturedFields"]
        self.assertEqual({"deliveryLocation": "ROOM", "items": items}, fields)
        self.assertFalse(json.loads(response.updatedConversationSummary)["awaitingExplicitConfirmation"])
        self.assertFalse(response.toolCalls)
        self.order_model.assert_not_called()

    def test_removing_last_item_does_not_cancel_or_confirm_the_order(self):
        items = [{"name": "sopa", "quantity": 1, "modifications": []}]
        request = room_request("Quita la sopa", items, True)
        self.order_model.side_effect = None
        self.order_model.return_value = result(order(edit("Quita la sopa", action="REMOVE", existingItemIndex=0), mode="PATCH"))
        response = plan_v2_turn(request)
        state = json.loads(response.updatedConversationSummary)
        self.assertEqual(items, state["capturedFields"]["items"])
        self.assertFalse(state["awaitingExplicitConfirmation"])
        self.assertFalse(response.toolCalls)

    def test_confirmation_preserves_exact_captured_order(self):
        items = [{"name": "Pechuga  Herbal", "quantity": 1,
                  "modifications": ["sin  sal"]}]
        request = room_request("Confirm", items, True)
        use_presented_room_button(request)
        before = request.model_dump(mode="json")
        response = plan_v2_turn(request)
        self.assertEqual(json.loads(request.conversation.summary)["capturedFields"],
                         response.toolCalls[0].arguments["input"])
        self.assertEqual(before, request.model_dump(mode="json"))
        self.order_model.assert_not_called()

        # BC-006 / BC-013: repairing formatting must not authorize a changed order.
        variants = [
            {"deliveryLocation": "POOL_1", "items": items},
            {"deliveryLocation": "ROOM", "items": [{**items[0], "quantity": 2}]},
            {"deliveryLocation": "ROOM", "items": [{**items[0], "name": "Herbal Chicken"}]},
            {"deliveryLocation": "ROOM", "items": [{**items[0], "modifications": []}]},
        ]
        for changed in variants:
            with self.subTest(changed=changed), understanding_turn(request):
                invalid = response.model_copy(deep=True)
                invalid.toolCalls[0].arguments["input"] = changed
                with self.assertRaisesRegex(AgentModelError, "current captured order"):
                    _validate_plan(request, invalid)

        for change in ("not_awaiting", "no_confirmation", "old_evidence"):
            with self.subTest(change=change):
                invalid_request = request.model_copy(deep=True)
                invalid_response = response.model_copy(deep=True)
                if change == "not_awaiting":
                    state = json.loads(invalid_request.conversation.summary)
                    state["awaitingExplicitConfirmation"] = False
                    invalid_request.conversation.summary = json.dumps(state)
                elif change == "no_confirmation":
                    invalid_request.conversation.recentMessages[-1].interactionReplyId = None
                    invalid_request.conversation.recentMessages[-1].text = "Change"
                else:
                    invalid_response.toolCalls[0].evidenceMessageIds = [uuid4()]
                with understanding_turn(invalid_request), self.assertRaises(AgentModelError):
                    _validate_plan(invalid_request, invalid_response)

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
        self.assertEqual("INPUT_RETRY", json.loads(response.updatedConversationSummary)["phase"])
        self.assertIn("ORDER_INPUT_TIMEOUT", response.warnings)
        self.assertEqual(1, self.order_model.call_count)

    def test_english_order_then_conditional_edit_never_confirms_old_order(self):
        text = "Hello, I would like two burgers without cheese"
        request = room_request(text)
        request.guest.preferredLanguage = "en"
        self.order_model.side_effect = None
        self.order_model.return_value = result(order(edit(text, "burgers", 2, "two", ["without cheese"])))
        captured = plan_v2_turn(request)
        self.assertEqual([], captured.toolCalls)
        self.assertTrue(json.loads(captured.updatedConversationSummary)["awaitingExplicitConfirmation"])
        self.assertIn("Order confirmation", captured.messages[0].text)
        request = follow_up(request, captured, "Yes, but without onions")
        self.order_model.return_value = result(order(edit("Yes, but without onions",
                modifiers=["without onions"], action="UPDATE", existingItemIndex=0), mode="PATCH"))
        changed = plan_v2_turn(request)
        self.assertEqual([], changed.toolCalls)
        request = follow_up(request, changed, "Yes, confirm")
        self.action = "CONFIRM"
        confirmed = plan_v2_turn(request)
        self.assertEqual(1, len(confirmed.toolCalls))
        self.assertEqual("START_SERVICE", confirmed.toolCalls[0].toolName)
        self.assertEqual([{"name": "burgers", "quantity": 2,
                           "modifications": ["without cheese", "without onions"]}],
                         confirmed.toolCalls[0].arguments["input"]["items"])
        self.assertEqual(2, self.order_model.call_count)

    def test_technical_failure_keeps_details_and_cannot_confirm_failed_edit(self):
        for error, reason in [(AgentTimeoutError("slow"), "TIMEOUT"),
                              (AgentDependencyError("unavailable"), "DEPENDENCY_ERROR"),
                              (AgentModelError("invalid"), "INVALID_RESPONSE")]:
            with self.subTest(reason=reason):
                self.action = "NONE"
                request = room_request("Make those three burgers", [
                    {"name": "burgers", "quantity": 2, "modifications": ["without cheese"]}], True)
                request.guest.preferredLanguage = "en"
                previous_fields = json.loads(request.conversation.summary)["capturedFields"]
                self.order_model.side_effect = error
                response = plan_v2_turn(request)
                state = json.loads(response.updatedConversationSummary)
                self.assertEqual(previous_fields, state["capturedFields"])
                self.assertEqual("INPUT_RETRY", state["phase"])
                self.assertFalse(state["awaitingExplicitConfirmation"])
                self.assertIn("temporary error", response.messages[0].text)
                self.assertIn("ORDER_INPUT_" + reason, response.warnings)
                request = follow_up(request, response, "Yes, confirm")
                self.action = "CONFIRM"
                self.assertEqual([], plan_v2_turn(request).toolCalls)

    def test_missing_quantity_is_not_reported_as_technical_error_and_can_recover(self):
        request = room_request("burgers without cheese")
        request.guest.preferredLanguage = "en"
        self.order_model.side_effect = None
        self.order_model.return_value = result(order(edit("burgers without cheese", "burgers",
                                                         modifiers=["without cheese"])))
        response = plan_v2_turn(request)
        self.assertIn("exact quantity", response.messages[0].text)
        self.assertNotIn("temporary error", response.messages[0].text)
        self.assertIn("ORDER_INPUT_MISSING_QUANTITY", response.warnings)
        request = follow_up(request, response, "two burgers without cheese")
        self.order_model.return_value = result(order(edit("two burgers without cheese", "burgers", 2,
                                                         "two", ["without cheese"])))
        response = plan_v2_turn(request)
        self.assertTrue(json.loads(response.updatedConversationSummary)["awaitingExplicitConfirmation"])
        self.assertEqual([], response.warnings)

    def test_product_then_short_quantity_preserves_restrictions_and_cannot_start_early(self):
        request = room_request("Gorditas de pulpo sin cebolla")
        request.guest.preferredLanguage = "en"
        self.order_model.side_effect = None
        self.order_model.return_value = result(order(edit(request.conversation.recentMessages[-1].text,
            "Gorditas de pulpo", modifiers=["sin cebolla"])))
        response = plan_v2_turn(request)
        state = json.loads(response.updatedConversationSummary)
        self.assertEqual([{"name": "Gorditas de pulpo", "modifications": ["sin cebolla"]}], state["capturedFields"]["items"])
        self.assertFalse(state["awaitingExplicitConfirmation"])
        self.assertIn("Gorditas de pulpo", response.messages[0].text)
        self.assertFalse(response.toolCalls)
        early = follow_up(request, response, "Confirm", "confirmation:ROOM_SERVICE:CONFIRM")
        self.assertFalse(plan_v2_turn(early).toolCalls)
        request = follow_up(request, response, "Uno")
        self.order_model.return_value = result(order(edit("Uno", quantity=1, quantity_quote="Uno",
            action="UPDATE", existingItemIndex=0), mode="PATCH"))
        response = plan_v2_turn(request)
        captured = json.loads(response.updatedConversationSummary)
        self.assertTrue(captured["awaitingExplicitConfirmation"])
        self.assertEqual([{"name": "Gorditas de pulpo", "quantity": 1, "modifications": ["sin cebolla"]}], captured["capturedFields"]["items"])
        request = follow_up(request, response, "Confirm", current_room_button(response))
        self.assertEqual(captured["capturedFields"], plan_v2_turn(request).toolCalls[0].arguments["input"])

    def test_location_selection_keeps_incomplete_item_and_asks_only_its_quantity(self):
        request = room_request("Room", [{"name": "sopa", "quantity": 1, "modifications": []},
                                         {"name": "tacos", "modifications": ["sin sal"]}])
        request.conversation.recentMessages[-1].interactionReplyId = "field:ROOM_SERVICE:deliveryLocation:ROOM"
        response = plan_v2_turn(request)
        state = json.loads(response.updatedConversationSummary)
        self.assertEqual(json.loads(request.conversation.summary)["capturedFields"], state["capturedFields"])
        self.assertFalse(state["awaitingExplicitConfirmation"])
        self.assertIn("tacos", response.messages[0].text)
        self.assertNotIn("sopa", response.messages[0].text)
        self.assertFalse(response.toolCalls)

    def test_first_order_timeout_does_not_ask_location_or_erase_another_draft(self):
        request = room_request("I would like two burgers")
        request.conversation.summary = '{"pendingOffering":"SPA","capturedFields":{"service":"massage"}}'
        request.guest.preferredLanguage = "en"
        self.kind, self.offering = "SERVICE_REQUEST", "ROOM_SERVICE"
        self.order_model.side_effect = AgentTimeoutError("slow")
        response = plan_v2_turn(request)
        self.assertEqual(request.conversation.summary, response.updatedConversationSummary)
        self.assertIn("temporary error", response.messages[0].text)
        self.assertIsNone(response.messages[0].interaction)
        self.assertEqual([], response.toolCalls)

    def test_english_kitchen_replacement_failure_then_success_keeps_operation(self):
        request = enable(request_for("two burgers without cheese"))
        request.guest.preferredLanguage = "en"
        request.activeOperations = [task_operation("ROOM_SERVICE_ORDER_CHANGE_DETAILS")]
        task = request.activeOperations[0].pendingConversationTasks[0]
        self.order_model.side_effect = AgentTimeoutError("slow")
        failed = plan_v2_turn(request)
        self.assertEqual([task.operationId], failed.messages[0].operationIds)
        self.assertEqual(request.conversation.summary, failed.updatedConversationSummary)
        self.assertIn("temporary error", failed.messages[0].text)
        request = follow_up(request, failed, "two burgers without cheese")
        self.order_model.side_effect = None
        self.order_model.return_value = result(order(edit("two burgers without cheese", "burgers", 2,
                                                         "two", ["without cheese"])))
        completed = plan_v2_turn(request)
        self.assertEqual(1, len(completed.toolCalls))
        self.assertEqual("COMPLETE_CONVERSATION_TASK", completed.toolCalls[0].toolName)
        self.assertEqual(task.conversationTaskId, completed.toolCalls[0].targetConversationTaskId)
        self.assertEqual([request.trigger.messageId], completed.toolCalls[0].evidenceMessageIds)

    def test_english_kitchen_decision_uses_templates_and_configured_menu(self):
        request = room_request("What can I do?")
        request.conversation.summary = "{}"
        request.guest.preferredLanguage = "en"
        request.activeOperations = [task_operation("ROOM_SERVICE_KITCHEN_CHANGE_DECISION")]
        catalog = request.availableOfferings[0].inputSchema["properties"]["items"]["x-chatbotinn-capture"]["catalog"]
        catalog["externalUrl"] = "https://sandbox.example/menu"
        response = plan_v2_turn(request)
        message = response.messages[0]
        self.assertIn("The kitchen has requested", message.text)
        self.assertIn("https://sandbox.example/menu", message.text)
        self.assertNotIn("hotelcristalino", message.text)
        self.assertEqual(["room-service-change:CHANGE", "room-service-change:CANCEL"],
                         [option.id for option in message.interaction.options])
        self.assertEqual(["Make changes", "Cancel order"], [option.label for option in message.interaction.options])

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
    def test_treatment_inside_longer_original_quote_is_preserved(self):
        text = "Masaje herbal el 2026-09-02"
        fields, unresolved, _, _ = self.extract(text,
            {"serviceName": "Masaje herbal", "reservationDate": "2026-09-02"},
            {"serviceName": text, "reservationDate": text})
        self.assertEqual({"serviceName": "Masaje herbal", "reservationDate": "2026-09-02"}, fields)
        self.assertEqual({}, unresolved)
        fields, unresolved, _, _ = self.extract(text,
            {"serviceName": "Herbal massage", "reservationDate": "2026-09-03"},
            {"serviceName": text, "reservationDate": text})
        self.assertEqual({}, fields)
        self.assertEqual(set(spa_turns.FIELDS[:2]), set(unresolved))

    def test_explicit_time_reply_preserves_other_fields_without_model_interpretation(self):
        captured = {"serviceName": "Masaje herbal", "reservationDate": "2026-09-02"}
        for text, expected in [("6pm", "18:00"), ("12am", "00:00"), ("12 PM", "12:00"),
                               ("6:30 p.m.", "18:30"), ("١٨:٣٠", "18:30")]:
            with self.subTest(text=text):
                request = enable(request_for(text))
                with understanding_turn(request), patch("app.agents.spa_turns.call_openai_json_result") as model:
                    fields, unresolved, usage, changed = spa_turns._extract(
                        request, request.conversation.recentMessages[-1], captured, {})
                self.assertEqual({**captured, "reservationTime": expected}, fields)
                self.assertEqual({}, unresolved)
                self.assertTrue(changed)
                self.assertEqual({}, usage)
                model.assert_not_called()

    def test_ambiguous_clarifications_use_guest_labels_and_only_relevant_question(self):
        request = enable(request_for("unclear"))
        request.guest.preferredLanguage = "en"
        for field, wanted, forbidden in [
            ("serviceName", "treatment", ["date", "time"]),
            ("reservationDate", "date", ["treatment", "24-hour"]),
            ("reservationTime", "time", ["treatment", "date"]),
        ]:
            with self.subTest(field=field):
                text = spa_turns._prompt(request, None, {}, {field: "unclear"})
                self.assertIn(wanted, text)
                for value in [*spa_turns.FIELDS, *forbidden]:
                    self.assertNotIn(value, text)

    def test_explicit_time_parser_does_not_guess_or_accept_invalid_clock_values(self):
        for text in ["6", "6:00", "not 6pm", "6pm or 7pm", "6pm if available", "25:00", "13pm", "6:99pm"]:
            with self.subTest(text=text):
                self.assertIsNone(spa_turns._explicit_local_time(text))
        for evidence, value in [("6pm", "06:00"), ("13pm", "13:00"), ("6:99pm", "18:59")]:
            slot = spa_turns.MultilingualField(status="RESOLVED", evidence=evidence, value=value, confidence=1)
            self.assertFalse(spa_turns._safe_multilingual_field("reservationTime", slot))

    def test_explicit_time_does_not_clear_other_ambiguities_or_validate_a_past_date(self):
        request = enable(request_for("6pm"))
        captured = {"serviceName": "Masaje herbal", "reservationDate": "2020-01-01"}
        with understanding_turn(request), patch("app.agents.spa_turns.call_openai_json_result") as model:
            fields, unresolved, _, _ = spa_turns._extract(request, request.conversation.recentMessages[-1],
                                                         captured, {"serviceName": "which massage"})
        self.assertEqual("which massage", unresolved["serviceName"])
        self.assertIn("reservationDate", unresolved)
        self.assertNotIn("reservationDate", fields)
        self.assertEqual("Masaje herbal", fields["serviceName"])
        self.assertEqual("18:00", fields["reservationTime"])
        model.assert_not_called()

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
            return result({"texts": {item["id"]: item["text"].replace("Order", "Commande") for item in texts}})
        with patch("app.services.localized_content.call_openai_json_result", side_effect=translate):
            localized = localize_response(request, response, time.perf_counter())
        self.assertEqual("Commande: 2 x soupes (sans sel)", localized.messages[0].text)
        self.assertEqual(response.updatedConversationSummary, localized.updatedConversationSummary)


if __name__ == "__main__":
    unittest.main()
