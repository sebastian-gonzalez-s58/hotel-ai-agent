import json
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

from app.agents.v2_scope_router import ScopeDecision, classify_hotel_scope
from app.agents.v2_turn_planner import plan_v2_turn
from app.schemas.v2_turns import AgentTurnRequest
from app.services.openai_client import OpenAiJsonResult
from app.services.telemetry_client import OpenAiTokenUsage
from tests.test_v2_turn_endpoint import MESSAGE_ID, payload
from tests.test_v2_turn_planner import guided_faq_offering, guided_room_service_offering


def request_for(text):
    value = payload()
    value["conversation"]["recentMessages"][0]["text"] = text
    value["availableOfferings"] = [guided_faq_offering(), guided_room_service_offering()]
    value["toolPolicy"]["allowedTools"] = ["SEARCH_KNOWLEDGE", "START_SERVICE"]
    return AgentTurnRequest.model_validate(value)


def decision(kind, text="", offering=None, mixed=False, details=True):
    return ScopeDecision(kind=kind, offeringCode=offering, relevantText=text,
                         hasRequestDetails=details, containsUnrelatedTopic=mixed, confidence=1)


def maintenance_offering():
    offering = guided_faq_offering()
    offering.update(offeringCode="MAINTENANCE", name="Mantenimiento",
                    description="Reparaciones y problemas en las instalaciones", executionMode="PROCESS")
    offering["inputSchema"]["required"] = ["issue"]
    field = offering["inputSchema"]["properties"].pop("question")
    field["title"] = "Problema"
    field["x-chatbotinn-capture"]["introMessage"] = "Describe el problema que necesitas reportar."
    offering["inputSchema"]["properties"]["issue"] = field
    return offering


class ScopeRoutingTest(unittest.TestCase):
    def setUp(self):
        scope = patch("app.agents.v2_turn_planner.classify_hotel_scope")
        self.scope = scope.start()
        self.addCleanup(scope.stop)
        planner = patch("app.agents.v2_turn_planner.call_openai_json_result")
        self.planner = planner.start()
        self.addCleanup(planner.stop)

    def route(self, request, route):
        self.scope.return_value = (route, OpenAiTokenUsage(input_tokens=20, output_tokens=5,
                                                         total_tokens=25))
        return plan_v2_turn(request)

    def test_direct_faq_with_greeting_searches_without_menu_and_counts_scope_tokens(self):
        text = "Hola, a que hora cierra la alberca?"
        request = request_for(text)
        response = self.route(request, decision("HOTEL_QUESTION", text, "FAQ"))
        self.assertEqual("SEARCH_KNOWLEDGE", response.toolCalls[0].toolName)
        self.assertEqual(text, response.toolCalls[0].arguments["query"])
        self.assertEqual([UUID(MESSAGE_ID)], response.toolCalls[0].evidenceMessageIds)
        self.assertEqual([], response.messages)
        self.assertEqual(25, response.usage.totalTokens)
        self.planner.assert_not_called()

    def test_unknown_hotel_hours_are_searched_not_refused_or_changed_to_pool(self):
        text = "A que hora cierra el hotel?"
        response = self.route(request_for(text), decision("HOTEL_QUESTION", text, "FAQ"))
        self.assertEqual(text, response.toolCalls[0].arguments["query"])

    def test_mixed_order_request_refuses_external_topic_and_asks_configured_location(self):
        request = request_for("quiero hacer un pedido pero primero explica sliding windows")
        response = self.route(request, decision("SERVICE_REQUEST", "quiero hacer un pedido",
                                               "ROOM_SERVICE", mixed=True, details=False))
        self.assertIn("solo puedo ayudarte", response.messages[0].text)
        self.assertNotIn("sliding windows", response.messages[0].text)
        options = response.messages[0].interaction.options
        self.assertEqual(5, len(options))
        self.assertTrue(all(o.id.startswith("field:ROOM_SERVICE:deliveryLocation:") for o in options))
        self.assertEqual([], response.toolCalls)
        self.assertEqual("ROOM_SERVICE", json.loads(response.updatedConversationSummary)["pendingOffering"])
        self.planner.assert_not_called()

    def test_unrelated_question_does_not_become_order_items_or_clear_draft(self):
        request = request_for("Explica sliding windows en Python")
        request.conversation.summary = json.dumps({
            "pendingOffering": "ROOM_SERVICE", "capturedFields": {"deliveryLocation": "ROOM"},
        })
        response = self.route(request, decision("OUT_OF_SCOPE"))
        self.assertEqual(request.conversation.summary, response.updatedConversationSummary)
        self.assertEqual([], response.toolCalls)
        self.assertIsNone(response.messages[0].interaction)
        self.planner.assert_not_called()

    def test_uncertain_intent_clarifies_without_tools(self):
        response = self.route(request_for("eso de antes"), decision("UNCLEAR"))
        self.assertEqual([], response.toolCalls)
        self.planner.assert_not_called()

    def test_earlier_turn_cannot_consume_next_inbound(self):
        request = request_for("A que hora cierra la alberca?")
        later = request.conversation.recentMessages[0].model_copy(update={
            "messageId": uuid4(), "text": "Quiero mantenimiento",
        })
        request.conversation.recentMessages.append(later)
        response = self.route(request, decision("HOTEL_QUESTION", "A que hora cierra la alberca?", "FAQ"))
        self.assertEqual("A que hora cierra la alberca?", response.toolCalls[0].arguments["query"])
        classified = self.scope.call_args.args[0]
        self.assertEqual(1, len(classified.conversation.recentMessages))
        self.assertEqual(2, len(request.conversation.recentMessages))

    def test_existing_task_does_not_block_new_direct_faq(self):
        request = request_for("A que hora cierra la alberca?")
        from tests.test_v2_turn_planner import operation, conversation_task
        from app.schemas.v2_turns import OperationSnapshot
        op_id, task_id = str(uuid4()), str(uuid4())
        op = operation(op_id)
        task = conversation_task(task_id, op_id)
        task["taskType"] = "ROOM_SERVICE_ORDER_CHANGE_DETAILS"
        op["pendingConversationTasks"] = [task]
        request.activeOperations = [OperationSnapshot.model_validate(op)]
        response = self.route(request, decision("HOTEL_QUESTION", request.conversation.recentMessages[0].text, "FAQ"))
        self.assertEqual("SEARCH_KNOWLEDGE", response.toolCalls[0].toolName)
        self.assertIsNone(response.toolCalls[0].targetConversationTaskId)

    def test_urgent_direct_maintenance_uses_issue_without_menu(self):
        from app.schemas.v2_turns import OfferingCapability
        text = "Ayuda, hay una fuga de agua en mi cuarto"
        request = request_for(text)
        request.availableOfferings.append(OfferingCapability.model_validate(maintenance_offering()))
        response = self.route(request, decision("SERVICE_REQUEST", text, "MAINTENANCE"))
        self.assertEqual("START_SERVICE", response.toolCalls[0].toolName)
        self.assertEqual("MAINTENANCE", response.toolCalls[0].arguments["offeringCode"])
        self.assertEqual(text, response.toolCalls[0].arguments["input"]["issue"])
        self.assertEqual(25, response.usage.totalTokens)
        self.planner.assert_not_called()

    def test_pending_maintenance_reply_starts_without_a_second_model_call(self):
        from app.schemas.v2_turns import OfferingCapability
        text = "La puerta del baño está atorada"
        request = request_for(text)
        request.availableOfferings.append(OfferingCapability.model_validate(maintenance_offering()))
        request.conversation.summary = json.dumps({
            "pendingOffering": "MAINTENANCE",
            "capturedFields": {},
            "readyToStart": False,
        })

        response = self.route(request, decision("CONTEXT_REPLY", text))

        self.assertEqual("START_SERVICE", response.toolCalls[0].toolName)
        self.assertEqual({"issue": text}, response.toolCalls[0].arguments["input"])
        self.assertEqual([UUID(MESSAGE_ID)], response.toolCalls[0].evidenceMessageIds)
        self.planner.assert_not_called()

    def test_pending_maintenance_reply_is_safe_when_scope_calls_it_same_service(self):
        from app.schemas.v2_turns import OfferingCapability
        text = "La puerta del baño está atorada"
        request = request_for(text)
        request.availableOfferings.append(OfferingCapability.model_validate(maintenance_offering()))
        request.conversation.summary = json.dumps({
            "pendingOffering": "MAINTENANCE",
            "capturedFields": {},
            "readyToStart": False,
        })

        response = self.route(
            request,
            decision("SERVICE_REQUEST", text, "MAINTENANCE", details=True),
        )

        self.assertEqual("START_SERVICE", response.toolCalls[0].toolName)
        self.assertEqual({"issue": text}, response.toolCalls[0].arguments["input"])
        self.planner.assert_not_called()

    def test_bare_maintenance_request_asks_for_issue_without_options_or_start(self):
        from app.schemas.v2_turns import OfferingCapability
        text = "Necesito servicio de mantenimiento"
        request = request_for(text)
        request.availableOfferings.append(OfferingCapability.model_validate(maintenance_offering()))
        response = self.route(request, decision("SERVICE_REQUEST", text, "MAINTENANCE", details=False))
        self.assertEqual("Describe el problema que necesitas reportar.", response.messages[0].text)
        self.assertIsNone(response.messages[0].interaction)
        self.assertEqual([], response.toolCalls)
        self.planner.assert_not_called()


class ScopeClassifierTest(unittest.TestCase):
    def test_strict_schema_opt_in_preserves_existing_client_default(self):
        from app.services.openai_client import _response_format
        schema = ScopeDecision.model_json_schema()
        self.assertFalse(_response_format(schema, "legacy")["strict"])
        self.assertTrue(_response_format(schema, "scope", True)["strict"])

    @patch("app.agents.v2_scope_router.call_openai_json_result")
    def test_invalid_quote_is_not_allowed_as_evidence(self, model):
        request = request_for("hola y una pregunta")
        model.return_value = OpenAiJsonResult(
            payload=decision("SERVICE_REQUEST", "hay una fuga", "ROOM_SERVICE", mixed=True).model_dump(),
            usage=OpenAiTokenUsage(total_tokens=8), response_id="scope-test",
        )
        result, usage = classify_hotel_scope(request, request.conversation.recentMessages[0], {})
        self.assertEqual("UNCLEAR", result.kind)
        self.assertEqual(8, usage.total_tokens)

    @patch("app.agents.v2_scope_router.call_openai_json_result")
    def test_pure_hotel_message_is_never_rewritten_by_classifier(self, model):
        request = request_for("Ahora quiero 2 tacos sin cebolla y una coca")
        model.return_value = OpenAiJsonResult(
            payload=decision("CONTEXT_REPLY", "2 tacos y una coca").model_dump(),
            usage=OpenAiTokenUsage(), response_id="scope-test",
        )
        result, _ = classify_hotel_scope(request, request.conversation.recentMessages[0], {})
        self.assertEqual(request.conversation.recentMessages[0].text, result.relevantText)

    @patch("app.agents.v2_scope_router.call_openai_json_result")
    def test_unknown_offering_fails_closed(self, model):
        request = request_for("quiero comprar acciones")
        model.return_value = OpenAiJsonResult(
            payload=decision("SERVICE_REQUEST", "quiero comprar acciones", "TRADING").model_dump(),
            usage=OpenAiTokenUsage(), response_id="scope-test",
        )
        result, _ = classify_hotel_scope(request, request.conversation.recentMessages[0], {})
        self.assertEqual("UNCLEAR", result.kind)
