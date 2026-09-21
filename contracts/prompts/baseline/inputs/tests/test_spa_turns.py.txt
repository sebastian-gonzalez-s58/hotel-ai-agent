import json
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.agents.spa_turns import _original_fields, _prompt, summary_state
from app.agents.v2_scope_router import ScopeDecision
from app.agents.v2_turn_planner import _validate_plan, plan_v2_turn
from app.core.errors import AgentModelError
from app.schemas.v2_turns import AgentTurnRequest, OfferingCapability, OperationSnapshot, TurnTriggerType
from app.services.openai_client import OpenAiJsonResult
from app.services.telemetry_client import OpenAiTokenUsage
from tests.test_v2_turn_endpoint import payload
from tests.test_v2_turn_planner import conversation_task, guided_room_service_offering, guided_spa_offering, operation


VALUES = {"serviceName": "Masaje relajante", "reservationDate": "2026-09-02", "reservationTime": "17:00"}


def spa_schema():
    return {
        "type": "object", "required": ["decision"], "additionalProperties": False,
        "properties": {
            "decision": {"enum": ["UPDATE", "CANCEL"]},
            "serviceName": {"type": "string", "minLength": 1, "maxLength": 200},
            "reservationDate": {"type": "string", "format": "date"},
            "reservationTime": {"type": "string", "pattern": "^([01][0-9]|2[0-3]):[0-5][0-9]$"},
        },
        "oneOf": [
            {"properties": {"decision": {"const": "UPDATE"}}, "required": list(VALUES)},
            {"properties": {"decision": {"const": "CANCEL"}}},
        ],
    }


def spa_operation(task_type="SPA_ALTERNATIVE_DECISION", original=None):
    op_id, task_id = str(uuid4()), str(uuid4())
    op = operation(op_id)
    op["offeringCode"] = "SPA"
    task = conversation_task(task_id, op_id)
    task["taskType"] = task_type
    task["requiredOutputSchema"] = spa_schema() if task_type.endswith("DETAILS") else {
        "type": "object", "required": ["decision"], "additionalProperties": False,
        "properties": {"decision": {"type": "string", "enum": ["ACCEPT", "CHANGE", "CANCEL"]}},
    }
    task["context"] = {
        **(original or VALUES),
        "proposedServiceName": "Masaje deportivo", "proposedReservationDate": "2026-09-03",
        "proposedReservationTime": "18:00", "spaStaffResponse": "Podemos ofrecer esta alternativa.",
    }
    op["pendingConversationTasks"] = [task]
    return OperationSnapshot.model_validate(op)


def request_for(text, task_type=None):
    data = payload()
    data["createdAt"] = "2026-08-31T04:30:00Z"
    data["hotel"]["timeZone"] = "America/Mexico_City"
    data["availableOfferings"] = [guided_spa_offering()]
    data["availableOfferings"][0]["requiresExplicitGuestConfirmation"] = True
    data["toolPolicy"] = {"allowedTools": ["START_SERVICE", "COMPLETE_CONVERSATION_TASK"], "maxToolCalls": 2}
    data["conversation"]["recentMessages"][0]["text"] = text
    data["conversation"]["recentMessages"][0]["createdAt"] = data["createdAt"]
    request = AgentTurnRequest.model_validate(data)
    if task_type:
        request.activeOperations = [spa_operation(task_type)]
        request.conversation.focusedConversationTaskId = request.activeOperations[0].pendingConversationTasks[0].conversationTaskId
    return request


def follow_up(request, response, text, button=None):
    request = request.model_copy(deep=True)
    request.conversation.summary = response.updatedConversationSummary or ""
    message = request.conversation.recentMessages[-1].model_copy(update={
        "messageId": uuid4(), "text": text, "interactionReplyId": button,
        "conversationTaskIds": [], "operationIds": [],
    })
    request.conversation.recentMessages.append(message)
    request.trigger.type = TurnTriggerType.INBOUND_MESSAGE
    request.trigger.messageId = message.messageId
    request.trigger.conversationTaskId = None
    request.previousToolResults = []
    return request


def extraction(text, values=None, ambiguous=None):
    values, ambiguous = values or {}, ambiguous or {}
    return OpenAiJsonResult(payload={
        key: {"status": "AMBIGUOUS" if key in ambiguous else "RESOLVED" if key in values else "UNCHANGED",
              "value": values.get(key), "evidence": ambiguous.get(key) or (text if key in values else None)}
        for key in VALUES
    }, usage=OpenAiTokenUsage(input_tokens=20, cached_input_tokens=3, output_tokens=10,
                             reasoning_tokens=2, total_tokens=30), response_id="spa-test")


class SpaTurnsTest(unittest.TestCase):
    def setUp(self):
        for target, attribute in [
            ("app.agents.v2_turn_planner.classify_hotel_scope", "scope"),
            ("app.agents.spa_turns.call_openai_json_result", "extract"),
            ("app.agents.v2_turn_planner.call_openai_json_result", "planner"),
        ]:
            mock = patch(target)
            setattr(self, attribute, mock.start())
            self.addCleanup(mock.stop)
        self.scope.side_effect = lambda r, m, s: (ScopeDecision(
            kind="CONTEXT_REPLY", offeringCode=None, relevantText=m.text,
            hasRequestDetails=True, containsUnrelatedTopic=False, confidence=1,
        ), OpenAiTokenUsage(input_tokens=4, output_tokens=1, total_tokens=5))
        self.planner.side_effect = AssertionError("Unexpected general planner call")
        self.extract.side_effect = AssertionError("Unexpected extraction call")

    def capture(self, request, values=VALUES, ambiguous=None):
        self.extract.side_effect = None
        self.extract.return_value = extraction(request.conversation.recentMessages[-1].text, values, ambiguous)
        return plan_v2_turn(request)

    def new_request(self, text="Quiero masaje relajante el 2 de septiembre a las 5 de la tarde"):
        request = request_for(text)
        self.scope.side_effect = lambda r, m, s: (ScopeDecision(
            kind="SERVICE_REQUEST", offeringCode="SPA", relevantText=m.text,
            hasRequestDetails=True, containsUnrelatedTopic=False, confidence=1,
        ), OpenAiTokenUsage(total_tokens=5))
        return request

    def contextual(self):
        self.scope.side_effect = lambda r, m, s: (ScopeDecision(
            kind="CONTEXT_REPLY", offeringCode=None, relevantText=m.text,
            hasRequestDetails=True, containsUnrelatedTopic=False, confidence=1,
        ), OpenAiTokenUsage(total_tokens=5))

    def test_all_fields_together_then_explicit_confirmation_starts(self):
        request = self.new_request()
        response = self.capture(request)
        self.assertEqual([], response.toolCalls)
        self.assertIn("2026-09-02", response.messages[0].text)
        self.assertIn("17:00", response.messages[0].text)
        self.assertEqual(35, response.usage.totalTokens)
        self.assertEqual(3, response.usage.cachedInputTokens)
        self.assertEqual(2, response.usage.reasoningTokens)
        self.assertEqual(["Confirmar", "Cambiar", "Cancelar"], [o.label for o in response.messages[0].interaction.options])
        request = follow_up(request, response, "Confirmar", response.messages[0].interaction.options[0].id)
        response = plan_v2_turn(request)
        self.assertEqual(VALUES, response.toolCalls[0].arguments["input"])
        self.assertEqual(str(request.trigger.messageId), response.toolCalls[0].arguments["guestConfirmationEvidenceMessageId"])
        self.assertEqual([request.trigger.messageId], response.toolCalls[0].evidenceMessageIds)
        self.assertEqual([], response.messages)

    def test_initial_catalog_uses_configured_link_without_extraction(self):
        request = request_for("SPA")
        request.conversation.recentMessages[0].interactionReplyId = "offering:SPA"
        response = plan_v2_turn(request)
        self.assertIn("https://spa.example/catalog", response.messages[0].text)
        self.assertEqual([], response.toolCalls)
        self.extract.assert_not_called()

    def test_clock_uses_request_in_hotel_timezone(self):
        request = self.new_request("Masaje manana a las 5 de la tarde")
        response = self.capture(request, {**VALUES, "reservationDate": "2026-08-31"})
        prompt = self.extract.call_args.args[0]
        self.assertIn('"hotelLocalNow": "2026-08-30T22:30:00-06:00"', prompt)
        self.assertIn('"requestClock": "2026-08-31T04:30:00+00:00"', prompt)
        self.assertTrue(self.extract.call_args.kwargs["strict_schema"])
        self.assertIn("2026-08-31", response.messages[0].text)

    def test_ambiguous_time_clarifies_preserving_date_and_service(self):
        request = self.new_request("Masaje el 2 de septiembre a las 5")
        response = self.capture(request, {k: v for k, v in VALUES.items() if k != "reservationTime"},
                                {"reservationTime": "a las 5"})
        self.assertIn("24 horas", response.messages[0].text)
        self.assertEqual([], response.toolCalls)
        self.contextual()
        request = follow_up(request, response, "de la tarde")
        response = self.capture(request, {"reservationTime": "17:00"})
        self.assertEqual(VALUES, summary_state(response.updatedConversationSummary)["spaDraft"]["capturedFields"])
        self.assertIn('"reservationTime": "a las 5"', self.extract.call_args.args[0])

    def test_raw_invalid_and_past_values_never_reach_start(self):
        for field, value in [("reservationDate", "30 de agosto"), ("reservationDate", "2026-02-30"),
                             ("reservationDate", "2026-08-01"), ("reservationTime", "5 de la tarde"),
                             ("reservationTime", "25:00"), ("reservationTime", "17:00:01"),
                             ("serviceName", "x" * 201)]:
            with self.subTest(field=field, value=value):
                request = self.new_request("Quiero reservar este tratamiento con estos datos")
                response = self.capture(request, {**VALUES, field: value})
                self.assertEqual([], response.toolCalls)
                self.assertNotIn(field, summary_state(response.updatedConversationSummary)["spaDraft"]["capturedFields"])

    def test_ambiguous_numeric_date_is_not_defaulted(self):
        request = self.new_request("Masaje el 03/04 a las 17:00")
        response = self.capture(request, {"serviceName": "Masaje", "reservationTime": "17:00"},
                                {"reservationDate": "03/04"})
        self.assertIn("YYYY-MM-DD", response.messages[0].text)
        self.assertEqual([], response.toolCalls)

    def test_invalid_extraction_schema_retains_usage_and_requires_clarification(self):
        request = self.new_request()
        self.extract.side_effect = None
        self.extract.return_value = OpenAiJsonResult(payload={}, usage=OpenAiTokenUsage(total_tokens=9), response_id="bad")
        response = plan_v2_turn(request)
        self.assertEqual(23, response.usage.totalTokens)
        self.assertEqual(2, self.extract.call_count)
        self.assertEqual([], response.toolCalls)

    def test_extraction_retry_counts_invalid_and_successful_attempts(self):
        request = self.new_request()
        self.extract.side_effect = [
            OpenAiJsonResult(payload={}, usage=OpenAiTokenUsage(input_tokens=8, cached_input_tokens=2,
                            output_tokens=3, reasoning_tokens=1, total_tokens=11), response_id="invalid"),
            extraction(request.conversation.recentMessages[0].text, VALUES),
        ]
        response = plan_v2_turn(request)
        self.assertEqual(2, self.extract.call_count)
        self.assertEqual(28, response.usage.inputTokens)
        self.assertEqual(5, response.usage.cachedInputTokens)
        self.assertEqual(13, response.usage.outputTokens)
        self.assertEqual(3, response.usage.reasoningTokens)
        self.assertEqual(46, response.usage.totalTokens)
        self.assertEqual([], response.toolCalls)

    def test_summary_uses_guest_timezone_label_and_does_not_promise_availability(self):
        request = self.new_request()
        response = self.capture(request)
        self.assertIn("(hora del hotel)", response.messages[0].text)
        self.assertNotIn("America/Mexico_City", response.messages[0].text)
        self.assertIn("solicitud", response.messages[0].text)
        self.assertIn("disponibilidad está pendiente", response.messages[0].text)
        request = self.new_request("SPA treatment tomorrow at 5 pm")
        request.guest.preferredLanguage = "en"
        response = self.capture(request)
        self.assertIn("(hotel local time)", response.messages[0].text)
        self.assertNotIn("America/Mexico_City", response.messages[0].text)

    def test_spanish_fallback_copy_has_accents_and_question_marks(self):
        request = request_for("SPA")
        self.assertEqual("¿Qué tratamiento de SPA deseas reservar?", _prompt(request, None, {}, {}))
        self.assertEqual("¿Para qué fecha deseas hacer la reservación?", _prompt(request, None, {"serviceName": "Masaje"}, {}))
        self.assertEqual("¿A qué hora deseas la reservación?", _prompt(request, None, {k: v for k, v in VALUES.items() if k != "reservationTime"}, {}))
        self.assertIn("día, mes y año", _prompt(request, None, {}, {"reservationDate": "03/04"}))
        self.assertIn("mañana", _prompt(request, None, {}, {"reservationTime": "a las 5"}))
        self.assertIn("Conservaré los demás datos", _prompt(request, None, VALUES, {}))

    def test_extraction_cannot_use_evidence_outside_trigger(self):
        request = self.new_request("Masaje")
        self.extract.side_effect = None
        self.extract.return_value = extraction("an older unrelated message", VALUES)
        response = plan_v2_turn(request)
        self.assertEqual({}, summary_state(response.updatedConversationSummary)["spaDraft"]["capturedFields"])

    def test_initial_change_preserves_unchanged_fields_and_reconfirms(self):
        request = self.new_request()
        response = self.capture(request)
        self.contextual()
        request = follow_up(request, response, "Cambiar", response.messages[0].interaction.options[1].id)
        response = plan_v2_turn(request)
        self.assertEqual(VALUES, summary_state(response.updatedConversationSummary)["spaDraft"]["capturedFields"])
        request = follow_up(request, response, "Mejor a las 18:00")
        response = self.capture(request, {"reservationTime": "18:00"})
        self.assertEqual([], response.toolCalls)
        request = follow_up(request, response, "si confirmo")
        response = plan_v2_turn(request)
        self.assertEqual({**VALUES, "reservationTime": "18:00"}, response.toolCalls[0].arguments["input"])

    def test_cancel_initial_does_not_start_or_cancel_existing_operations(self):
        request = self.new_request()
        request.activeOperations = [spa_operation()]
        response = self.capture(request)
        request = follow_up(request, response, "Cancelar", response.messages[0].interaction.options[2].id)
        response = plan_v2_turn(request)
        self.assertEqual([], response.toolCalls)
        self.assertIsNone(summary_state(response.updatedConversationSummary)["spaDraft"])
        self.assertEqual(1, len(request.activeOperations))

    def test_alternative_buttons_are_exact_and_show_structured_proposal(self):
        request = request_for("Old guest message", "SPA_ALTERNATIVE_DECISION")
        task = request.activeOperations[0].pendingConversationTasks[0]
        request.trigger.type = TurnTriggerType.CONVERSATION_TASK_CREATED
        request.trigger.conversationTaskId = task.conversationTaskId
        response = plan_v2_turn(request)
        self.assertEqual([f"spa:{task.conversationTaskId}:{action}" for action in ("ACCEPT", "CHANGE", "CANCEL")],
                         [o.id for o in response.messages[0].interaction.options])
        self.assertIn("Masaje deportivo", response.messages[0].text)
        self.assertIn("2026-09-03", response.messages[0].text)
        self.assertIn("18:00", response.messages[0].text)
        self.assertIn("Podemos ofrecer", response.messages[0].text)
        self.assertNotIn("Masaje relajante", response.messages[0].text)
        self.assertEqual([], response.toolCalls)

    def test_alternative_accept_change_cancel_complete_only_decision(self):
        for action in ("ACCEPT", "CHANGE", "CANCEL"):
            with self.subTest(action=action):
                request = request_for(action, "SPA_ALTERNATIVE_DECISION")
                task = request.activeOperations[0].pendingConversationTasks[0]
                request.conversation.recentMessages[0].interactionReplyId = f"spa:{task.conversationTaskId}:{action}"
                response = plan_v2_turn(request)
                call = response.toolCalls[0]
                self.assertEqual({"decision": action}, call.arguments["result"])
                self.assertEqual(task.version, call.arguments["expectedVersion"])
                self.assertEqual(task.conversationTaskId, call.targetConversationTaskId)
                self.assertEqual([], response.messages)
        self.extract.assert_not_called()

    def test_incomplete_proposal_cannot_be_accepted_using_original_values(self):
        request = request_for("Aceptar", "SPA_ALTERNATIVE_DECISION")
        task = request.activeOperations[0].pendingConversationTasks[0]
        task.context.pop("proposedReservationTime")
        request.conversation.recentMessages[0].interactionReplyId = f"spa:{task.conversationTaskId}:ACCEPT"
        response = plan_v2_turn(request)
        self.assertEqual([], response.toolCalls)
        self.assertEqual(2, len(response.messages[0].interaction.options))

    def test_stale_button_is_not_redirected_to_focused_task_or_room(self):
        request = request_for("Cancelar", "SPA_ALTERNATIVE_DECISION")
        request.activeOperations.append(spa_operation("SPA_RESERVATION_CHANGE_DETAILS"))
        request.conversation.recentMessages[0].interactionReplyId = f"spa:{uuid4()}:CANCEL"
        response = plan_v2_turn(request)
        self.assertEqual([], response.toolCalls)
        self.planner.assert_not_called()
        self.extract.assert_not_called()

    def test_concurrent_task_button_overrides_focus_without_cross_completion(self):
        request = request_for("Aceptar", "SPA_ALTERNATIVE_DECISION")
        request.activeOperations.append(spa_operation())
        selected = request.activeOperations[1].pendingConversationTasks[0]
        request.conversation.recentMessages[0].interactionReplyId = f"spa:{selected.conversationTaskId}:ACCEPT"
        response = plan_v2_turn(request)
        self.assertEqual(1, len(response.toolCalls))
        self.assertEqual(selected.conversationTaskId, response.toolCalls[0].targetConversationTaskId)

    def test_concurrent_unfocused_tasks_ask_which_request(self):
        request = request_for("si", "SPA_ALTERNATIVE_DECISION")
        request.activeOperations.append(spa_operation())
        request.conversation.focusedConversationTaskId = None
        response = plan_v2_turn(request)
        self.assertEqual([], response.toolCalls)
        self.assertIn("¿A qué solicitud", response.messages[0].text)
        self.assertEqual(2, len(response.messages[0].interaction.options))

    def test_detail_creation_prompts_without_consuming_old_guest_and_has_cancel(self):
        request = request_for("cancelar", "SPA_RESERVATION_CHANGE_DETAILS")
        task = request.activeOperations[0].pendingConversationTasks[0]
        request.trigger.type = TurnTriggerType.CONVERSATION_TASK_CREATED
        request.trigger.conversationTaskId = task.conversationTaskId
        response = plan_v2_turn(request)
        self.assertEqual([], response.toolCalls)
        self.assertEqual(f"spa:{task.conversationTaskId}:CANCEL", response.messages[0].interaction.options[0].id)
        self.assertIn("cambiar", response.messages[0].text)

    def test_spa_task_creation_is_not_suppressed_by_an_old_greeting(self):
        request = request_for("Hola", "SPA_ALTERNATIVE_DECISION")
        request.trigger.type = TurnTriggerType.CONVERSATION_TASK_CREATED
        request.trigger.conversationTaskId = request.conversation.focusedConversationTaskId
        response = plan_v2_turn(request)
        self.assertEqual([], response.toolCalls)
        self.assertIn("El SPA propone", response.messages[0].text)
        self.planner.assert_not_called()

    def test_non_spa_task_creation_is_left_to_existing_handler(self):
        from app.agents.spa_turns import plan_spa_turn
        for offering, kind in [("ROOM_SERVICE", "ROOM_SERVICE_ORDER_CHANGE_DETAILS"),
                               ("MAINTENANCE", "MAINTENANCE_GUEST_RESOLUTION_CONFIRMATION")]:
            request = request_for("Hola", "SPA_ALTERNATIVE_DECISION")
            request.activeOperations[0].offeringCode = offering
            request.activeOperations[0].pendingConversationTasks[0].taskType = kind
            request.trigger.type = TurnTriggerType.CONVERSATION_TASK_CREATED
            request.trigger.conversationTaskId = request.conversation.focusedConversationTaskId
            self.assertIsNone(plan_spa_turn(request))

    def test_stale_spa_button_with_greeting_label_still_cannot_fall_through(self):
        request = request_for("Hola", "SPA_ALTERNATIVE_DECISION")
        request.conversation.recentMessages[0].interactionReplyId = f"spa:{uuid4()}:CANCEL"
        response = plan_v2_turn(request)
        self.assertEqual([], response.toolCalls)
        self.assertIn("ya no está pendiente", response.messages[0].text)
        self.planner.assert_not_called()

    def test_detail_partial_edit_preserves_original_then_confirms_update(self):
        request = request_for("Solo cambia la hora a las 18:30", "SPA_RESERVATION_CHANGE_DETAILS")
        response = self.capture(request, {"reservationTime": "18:30"})
        self.assertEqual([], response.toolCalls)
        self.assertIn("Masaje relajante", response.messages[0].text)
        self.assertIn("2026-09-02", response.messages[0].text)
        request = follow_up(request, response, "Confirmar cambios", response.messages[0].interaction.options[0].id)
        response = plan_v2_turn(request)
        self.assertEqual({"decision": "UPDATE", **VALUES, "reservationTime": "18:30"}, response.toolCalls[0].arguments["result"])
        self.assertEqual([request.trigger.messageId], response.toolCalls[0].evidenceMessageIds)

    def test_real_flat_context_takes_precedence_over_legacy_nested_input(self):
        task = spa_operation("SPA_RESERVATION_CHANGE_DETAILS").pendingConversationTasks[0]
        self.assertNotIn("serviceInputJson", task.context)
        self.assertEqual(VALUES, _original_fields(task))
        task.context["serviceInputJson"] = json.dumps({**VALUES, "serviceName": "Old service"})
        self.assertEqual(VALUES, _original_fields(task))
        task.partialResult = {"reservationTime": "19:00"}
        self.assertEqual({**VALUES, "reservationTime": "19:00"}, _original_fields(task))
        task.context["reservationDate"] = "raw invalid date"
        self.assertNotIn("reservationDate", _original_fields(task))

    def test_legacy_nested_input_remains_supported(self):
        task = spa_operation("SPA_RESERVATION_CHANGE_DETAILS").pendingConversationTasks[0]
        for original in (VALUES, json.dumps(VALUES)):
            task.context = {"serviceInputJson": original}
            self.assertEqual(VALUES, _original_fields(task))

    def test_detail_all_fields_and_partial_result_are_supported(self):
        request = request_for("Facial el 4 de septiembre a las 10:00", "SPA_RESERVATION_CHANGE_DETAILS")
        task = request.activeOperations[0].pendingConversationTasks[0]
        task.context["serviceInputJson"] = VALUES
        task.partialResult = {"serviceName": "Facial"}
        response = self.capture(request, {"reservationDate": "2026-09-04", "reservationTime": "10:00"})
        request = follow_up(request, response, "confirmo")
        response = plan_v2_turn(request)
        self.assertEqual({"decision": "UPDATE", "serviceName": "Facial", "reservationDate": "2026-09-04",
                          "reservationTime": "10:00"}, response.toolCalls[0].arguments["result"])

    def test_detail_cancel_works_without_any_reservation_fields(self):
        for use_button in (True, False):
            request = request_for("cancela mi reserva", "SPA_RESERVATION_CHANGE_DETAILS")
            task = request.activeOperations[0].pendingConversationTasks[0]
            task.context = {}
            if use_button:
                request.conversation.recentMessages[0].interactionReplyId = f"spa:{task.conversationTaskId}:CANCEL"
            response = plan_v2_turn(request)
            self.assertEqual({"decision": "CANCEL"}, response.toolCalls[0].arguments["result"])

    def test_old_confirmation_after_edit_or_version_change_cannot_complete(self):
        request = request_for("a las 18:30", "SPA_RESERVATION_CHANGE_DETAILS")
        response = self.capture(request, {"reservationTime": "18:30"})
        old_button = response.messages[0].interaction.options[0].id
        request = follow_up(request, response, "mejor a las 19:00")
        response = self.capture(request, {"reservationTime": "19:00"})
        request = follow_up(request, response, "Confirmar cambios", old_button)
        response = plan_v2_turn(request)
        self.assertEqual([], response.toolCalls)
        request = follow_up(request, response, "Confirmar cambios", response.messages[0].interaction.options[0].id)
        request.activeOperations[0].pendingConversationTasks[0].version += 1
        response = plan_v2_turn(request)
        self.assertEqual([], response.toolCalls)

    def test_both_task_completions_end_no_action(self):
        for kind in ("SPA_ALTERNATIVE_DECISION", "SPA_RESERVATION_CHANGE_DETAILS"):
            request = request_for("confirmo")
            request.trigger.type = TurnTriggerType.TOOL_RESULTS
            data = {"taskType": kind, "conversationTaskId": str(uuid4())}
            request = AgentTurnRequest.model_validate({**request.model_dump(), "previousToolResults": [{
                "toolCallId": uuid4(), "toolName": "COMPLETE_CONVERSATION_TASK",
                "status": "SUCCEEDED", "result": data,
            }]})
            response = plan_v2_turn(request)
            self.assertEqual("NO_ACTION", response.disposition)
            self.assertEqual([], response.messages)
            self.assertEqual([], response.toolCalls)

    def test_successful_spa_start_uses_standard_ack_with_folio_and_no_model(self):
        request = request_for("Confirmar")
        existing_task = str(uuid4())
        request.conversation.summary = json.dumps({"pendingOffering": "SPA", "capturedFields": VALUES,
                                                  "spaDraft": {"submitted": True},
                                                  "spaTasks": {existing_task: {"capturedFields": VALUES}}})
        op_id = uuid4()
        request = AgentTurnRequest.model_validate({**request.model_dump(), "trigger": {"type": "TOOL_RESULTS"},
            "previousToolResults": [{"toolCallId": uuid4(), "toolName": "START_SERVICE", "status": "SUCCEEDED",
                                     "result": {"operationId": str(op_id), "offeringCode": "SPA", "referenceCode": "SPA-0042"}}]})
        response = plan_v2_turn(request)
        self.assertEqual("RESPONSE_READY", response.disposition)
        self.assertEqual("STATUS_UPDATE", response.messages[0].purpose)
        self.assertIn("folio SPA-0042", response.messages[0].text)
        self.assertIn("Recibirás actualizaciones", response.messages[0].text)
        self.assertEqual([op_id], response.messages[0].operationIds)
        self.assertNotIn("Masaje relajante", response.messages[0].text)
        self.assertEqual([], response.toolCalls)
        state = summary_state(response.updatedConversationSummary)
        self.assertIsNone(state["spaDraft"])
        self.assertIn(existing_task, state["spaTasks"])
        self.planner.assert_not_called()
        self.extract.assert_not_called()

    def test_failed_start_is_not_treated_as_successful_task_completion(self):
        from app.agents.spa_turns import _completion_result
        request = request_for("Confirmar")
        request = AgentTurnRequest.model_validate({**request.model_dump(), "previousToolResults": [{
            "toolCallId": uuid4(), "toolName": "START_SERVICE", "status": "FAILED", "result": {"offeringCode": "SPA"},
        }]})
        self.assertIsNone(_completion_result(request, {}))

    def test_tool_success_identified_by_removed_task_id(self):
        request = request_for("a las 18:30", "SPA_RESERVATION_CHANGE_DETAILS")
        response = self.capture(request, {"reservationTime": "18:30"})
        task_id = request.activeOperations[0].pendingConversationTasks[0].conversationTaskId
        request = follow_up(request, response, "confirmo")
        response = plan_v2_turn(request)
        request = follow_up(request, response, "confirmo")
        request.activeOperations = []
        request = AgentTurnRequest.model_validate({**request.model_dump(), "previousToolResults": [{
            "toolCallId": uuid4(), "toolName": "COMPLETE_CONVERSATION_TASK", "status": "SUCCEEDED",
            "result": {"conversationTaskId": str(task_id)},
        }]})
        response = plan_v2_turn(request)
        self.assertEqual("NO_ACTION", response.disposition)
        self.assertEqual({}, summary_state(response.updatedConversationSummary)["spaTasks"])

    def test_later_inbound_cannot_supply_confirmation_for_current_trigger(self):
        request = self.new_request()
        response = self.capture(request)
        self.contextual()
        request = follow_up(request, response, "Cambiar", response.messages[0].interaction.options[1].id)
        later = request.conversation.recentMessages[-1].model_copy(update={
            "messageId": uuid4(), "text": "confirmo", "interactionReplyId": None,
        })
        request.conversation.recentMessages.append(later)
        response = plan_v2_turn(request)
        self.assertEqual([], response.toolCalls)
        self.assertFalse(summary_state(response.updatedConversationSummary)["spaDraft"]["awaitingConfirmation"])

    def test_missing_trigger_and_outbound_evidence_are_rejected(self):
        request = self.new_request()
        response = self.capture(request)
        request = follow_up(request, response, "Confirmar", response.messages[0].interaction.options[0].id)
        response = plan_v2_turn(request)
        request.trigger.messageId = uuid4()
        with self.assertRaises(AgentModelError):
            _validate_plan(request, response)

    def test_disallowed_tools_and_zero_budget_do_not_mutate(self):
        for allowed, limit in [([], 2), (["COMPLETE_CONVERSATION_TASK"], 0)]:
            request = request_for("Cancelar", "SPA_ALTERNATIVE_DECISION")
            task = request.activeOperations[0].pendingConversationTasks[0]
            request.conversation.recentMessages[0].interactionReplyId = f"spa:{task.conversationTaskId}:CANCEL"
            request.toolPolicy.allowedTools = allowed
            request.toolPolicy.maxToolCalls = limit
            self.assertEqual([], plan_v2_turn(request).toolCalls)

    def test_off_topic_text_does_not_touch_spa_draft(self):
        request = request_for("Escribe un algoritmo")
        request.conversation.summary = json.dumps({"spaDraft": {"id": str(uuid4()), "capturedFields": VALUES}})
        self.scope.side_effect = None
        self.scope.return_value = (ScopeDecision(kind="OUT_OF_SCOPE", offeringCode=None, relevantText="",
                                                hasRequestDetails=False, containsUnrelatedTopic=False, confidence=1), OpenAiTokenUsage())
        response = plan_v2_turn(request)
        self.assertEqual(request.conversation.summary, response.updatedConversationSummary)
        self.assertEqual([], response.toolCalls)
        self.extract.assert_not_called()

    def test_two_detail_drafts_keep_independent_fields(self):
        request = request_for("a las 18:30", "SPA_RESERVATION_CHANGE_DETAILS")
        second = spa_operation("SPA_RESERVATION_CHANGE_DETAILS", {**VALUES, "serviceName": "Facial"})
        request.activeOperations.append(second)
        first_id = str(request.conversation.focusedConversationTaskId)
        response = self.capture(request, {"reservationTime": "18:30"})
        request = follow_up(request, response, "a las 19:00")
        second_id = second.pendingConversationTasks[0].conversationTaskId
        request.conversation.recentMessages[-1].conversationTaskIds = [second_id]
        response = self.capture(request, {"reservationTime": "19:00"})
        drafts = summary_state(response.updatedConversationSummary)["spaTasks"]
        self.assertEqual("18:30", drafts[first_id]["capturedFields"]["reservationTime"])
        self.assertEqual("Facial", drafts[str(second_id)]["capturedFields"]["serviceName"])

    def test_initial_spa_capture_continues_despite_older_focused_task(self):
        request = self.new_request()
        request.activeOperations = [spa_operation()]
        request.conversation.focusedConversationTaskId = request.activeOperations[0].pendingConversationTasks[0].conversationTaskId
        response = self.capture(request, {"serviceName": VALUES["serviceName"]})
        self.contextual()
        request = follow_up(request, response, "El 2 de septiembre a las 5 de la tarde")
        response = self.capture(request, {"reservationDate": VALUES["reservationDate"], "reservationTime": VALUES["reservationTime"]})
        self.assertEqual([], response.toolCalls)
        self.assertEqual(VALUES, summary_state(response.updatedConversationSummary)["spaDraft"]["capturedFields"])

    def test_task_selection_can_be_followed_by_unfocused_free_text(self):
        request = request_for("Cambiar", "SPA_RESERVATION_CHANGE_DETAILS")
        request.activeOperations.append(spa_operation("SPA_RESERVATION_CHANGE_DETAILS"))
        request.conversation.focusedConversationTaskId = None
        response = plan_v2_turn(request)
        request = follow_up(request, response, "SPA 2", response.messages[0].interaction.options[1].id)
        response = plan_v2_turn(request)
        request = follow_up(request, response, "a las 19:00")
        response = self.capture(request, {"reservationTime": "19:00"})
        self.assertIn(str(request.activeOperations[1].operationId), [str(i) for i in response.messages[0].operationIds])

    def test_selected_b_cancel_overrides_backend_default_focus_a(self):
        request = request_for("Cambiar", "SPA_RESERVATION_CHANGE_DETAILS")
        request.activeOperations.append(spa_operation("SPA_RESERVATION_CHANGE_DETAILS"))
        task_a = request.activeOperations[0].pendingConversationTasks[0]
        task_b = request.activeOperations[1].pendingConversationTasks[0]
        request.conversation.recentMessages[0].interactionReplyId = f"spa:{task_b.conversationTaskId}:CHANGE"
        selected = plan_v2_turn(request)
        self.assertEqual(str(task_b.conversationTaskId), summary_state(selected.updatedConversationSummary)["spaTaskFocus"])
        request = follow_up(request, selected, "cancelar")
        self.assertEqual(task_a.conversationTaskId, request.conversation.focusedConversationTaskId)
        response = plan_v2_turn(request)
        self.assertEqual(task_b.conversationTaskId, response.toolCalls[0].targetConversationTaskId)
        self.assertEqual(task_b.operationId, response.toolCalls[0].targetOperationId)
        self.assertEqual({"decision": "CANCEL"}, response.toolCalls[0].arguments["result"])
        self.extract.assert_not_called()

    def test_selected_b_time_edit_and_confirm_override_backend_default_focus_a(self):
        request = request_for("Cambiar", "SPA_RESERVATION_CHANGE_DETAILS")
        request.activeOperations.append(spa_operation("SPA_RESERVATION_CHANGE_DETAILS", {**VALUES, "serviceName": "Facial"}))
        task_a = request.activeOperations[0].pendingConversationTasks[0]
        task_b = request.activeOperations[1].pendingConversationTasks[0]
        request.conversation.recentMessages[0].interactionReplyId = f"spa:{task_b.conversationTaskId}:CHANGE"
        selected = plan_v2_turn(request)
        request = follow_up(request, selected, "Mejor a las 19:00")
        self.assertEqual(task_a.conversationTaskId, request.conversation.focusedConversationTaskId)
        summary = self.capture(request, {"reservationTime": "19:00"})
        self.assertEqual([task_b.operationId], summary.messages[0].operationIds)
        self.assertIn("Facial", summary.messages[0].text)
        request = follow_up(request, summary, "sí confirmo")
        self.assertEqual(task_a.conversationTaskId, request.conversation.focusedConversationTaskId)
        response = plan_v2_turn(request)
        self.assertEqual(task_b.conversationTaskId, response.toolCalls[0].targetConversationTaskId)
        self.assertEqual({"decision": "UPDATE", **VALUES, "serviceName": "Facial", "reservationTime": "19:00"},
                         response.toolCalls[0].arguments["result"])

    def test_explicit_new_room_selection_releases_saved_spa_focus(self):
        request = request_for("Cambiar", "SPA_RESERVATION_CHANGE_DETAILS")
        request.activeOperations.append(spa_operation("SPA_RESERVATION_CHANGE_DETAILS"))
        task_b = request.activeOperations[1].pendingConversationTasks[0]
        request.conversation.recentMessages[0].interactionReplyId = f"spa:{task_b.conversationTaskId}:CHANGE"
        selected = plan_v2_turn(request)
        request = follow_up(request, selected, "Room service", "offering:ROOM_SERVICE")
        request.availableOfferings.append(OfferingCapability.model_validate(guided_room_service_offering()))
        response = plan_v2_turn(request)
        state = summary_state(response.updatedConversationSummary)
        self.assertEqual("ROOM_SERVICE", state["pendingOffering"])
        self.assertIsNone(state["spaTaskFocus"])
        self.assertIsNone(state["spaOperationFocus"])
        self.assertIn(str(task_b.conversationTaskId), state["spaTasks"])
        request = follow_up(request, response, "Mi habitación", "field:ROOM_SERVICE:deliveryLocation:ROOM")
        response = plan_v2_turn(request)
        request = follow_up(request, response, "2 hamburguesas")
        response = plan_v2_turn(request)
        state = summary_state(response.updatedConversationSummary)
        self.assertEqual("ROOM_SERVICE", state["pendingOffering"])
        self.assertEqual(2, state["capturedFields"]["items"][0]["quantity"])
        self.assertIsNone(state["spaTaskFocus"])
        self.assertIsNone(state["spaOperationFocus"])
        self.assertIn(str(task_b.conversationTaskId), state["spaTasks"])
        self.assertEqual([], response.toolCalls)
        self.extract.assert_not_called()

    def test_navigation_releases_spa_selection_but_keeps_independent_draft(self):
        request = request_for("Cambiar", "SPA_RESERVATION_CHANGE_DETAILS")
        task = request.activeOperations[0].pendingConversationTasks[0]
        request.conversation.recentMessages[0].interactionReplyId = f"spa:{task.conversationTaskId}:CHANGE"
        selected = plan_v2_turn(request)
        request = follow_up(request, selected, "Muéstrame los servicios del hotel")
        self.scope.side_effect = lambda r, m, s: (ScopeDecision(kind="NAVIGATION", offeringCode=None,
            relevantText=m.text, hasRequestDetails=False, containsUnrelatedTopic=False, confidence=1), OpenAiTokenUsage())
        response = plan_v2_turn(request)
        state = summary_state(response.updatedConversationSummary)
        self.assertIsNone(state["spaTaskFocus"])
        self.assertIsNone(state["spaOperationFocus"])
        self.assertIn(str(task.conversationTaskId), state["spaTasks"])
        self.assertEqual([], response.toolCalls)

    def test_change_transition_keeps_operation_b_until_new_details_task_arrives(self):
        request = request_for("Cambiar", "SPA_RESERVATION_CHANGE_DETAILS")
        request.activeOperations.append(spa_operation("SPA_ALTERNATIVE_DECISION"))
        task_a = request.activeOperations[0].pendingConversationTasks[0]
        op_b = request.activeOperations[1]
        task_b1 = op_b.pendingConversationTasks[0]
        request.conversation.recentMessages[0].interactionReplyId = f"spa:{task_b1.conversationTaskId}:CHANGE"
        change = plan_v2_turn(request)
        self.assertEqual({"decision": "CHANGE"}, change.toolCalls[0].arguments["result"])
        request = follow_up(request, change, "Cambiar")
        request.activeOperations[1].pendingConversationTasks = []
        request = AgentTurnRequest.model_validate({**request.model_dump(), "trigger": {"type": "TOOL_RESULTS"},
            "previousToolResults": [{"toolCallId": change.toolCalls[0].toolCallId,
                "toolName": "COMPLETE_CONVERSATION_TASK", "status": "SUCCEEDED",
                "result": {"conversationTaskId": str(task_b1.conversationTaskId), "operationId": str(op_b.operationId),
                           "taskType": "SPA_ALTERNATIVE_DECISION"}}]})
        completed = plan_v2_turn(request)
        self.assertEqual("NO_ACTION", completed.disposition)
        self.assertEqual(str(op_b.operationId), summary_state(completed.updatedConversationSummary)["spaOperationFocus"])
        self.assertIsNone(summary_state(completed.updatedConversationSummary)["spaTaskFocus"])
        request = follow_up(request, completed, "Mejor a las 19:00")
        waiting = plan_v2_turn(request)
        self.assertEqual([], waiting.toolCalls)
        self.assertEqual(str(op_b.operationId), summary_state(waiting.updatedConversationSummary)["spaOperationFocus"])
        task_b2 = spa_operation("SPA_RESERVATION_CHANGE_DETAILS").pendingConversationTasks[0]
        task_b2.operationId = op_b.operationId
        request = follow_up(request, waiting, "Mejor a las 19:00")
        request.activeOperations[1].pendingConversationTasks = [task_b2]
        self.assertEqual(task_a.conversationTaskId, request.conversation.focusedConversationTaskId)
        summary = self.capture(request, {"reservationTime": "19:00"})
        self.assertEqual([op_b.operationId], summary.messages[0].operationIds)
        self.assertEqual(str(task_b2.conversationTaskId), summary_state(summary.updatedConversationSummary)["spaTaskFocus"])
        request = follow_up(request, summary, "sí confirmo")
        update = plan_v2_turn(request)
        self.assertEqual(task_b2.conversationTaskId, update.toolCalls[0].targetConversationTaskId)
        self.assertEqual({"decision": "UPDATE", **VALUES, "reservationTime": "19:00"}, update.toolCalls[0].arguments["result"])

    def test_accept_cancel_and_update_success_release_selected_operation(self):
        for kind, decision in [("SPA_ALTERNATIVE_DECISION", "ACCEPT"), ("SPA_ALTERNATIVE_DECISION", "CANCEL"),
                               ("SPA_RESERVATION_CHANGE_DETAILS", "UPDATE"), ("SPA_RESERVATION_CHANGE_DETAILS", "CANCEL")]:
            request = request_for("confirmo", kind)
            task = request.activeOperations[0].pendingConversationTasks[0]
            request.conversation.summary = json.dumps({"spaTaskFocus": str(task.conversationTaskId),
                "spaOperationFocus": str(task.operationId), "spaTasks": {str(task.conversationTaskId): {
                    "operationId": str(task.operationId), "taskType": kind, "submittedDecision": decision}}})
            request = AgentTurnRequest.model_validate({**request.model_dump(), "trigger": {"type": "TOOL_RESULTS"},
                "previousToolResults": [{"toolCallId": uuid4(), "toolName": "COMPLETE_CONVERSATION_TASK", "status": "SUCCEEDED",
                    "result": {"conversationTaskId": str(task.conversationTaskId), "operationId": str(task.operationId), "taskType": kind}}]})
            response = plan_v2_turn(request)
            self.assertEqual("NO_ACTION", response.disposition)
            state = summary_state(response.updatedConversationSummary)
            self.assertIsNone(state["spaTaskFocus"])
            self.assertIsNone(state["spaOperationFocus"])

    def test_new_room_order_and_capture_preserve_parked_spa_draft(self):
        request = self.new_request()
        response = self.capture(request)
        spa_draft = summary_state(response.updatedConversationSummary)["spaDraft"]
        request = follow_up(request, response, "Quiero hacer un pedido")
        request.availableOfferings.append(OfferingCapability.model_validate(guided_room_service_offering()))
        self.scope.side_effect = lambda r, m, s: (ScopeDecision(kind="SERVICE_REQUEST", offeringCode="ROOM_SERVICE",
            relevantText=m.text, hasRequestDetails=False, containsUnrelatedTopic=False, confidence=1), OpenAiTokenUsage())
        response = plan_v2_turn(request)
        self.assertEqual("ROOM_SERVICE", summary_state(response.updatedConversationSummary)["pendingOffering"])
        self.assertEqual(spa_draft, summary_state(response.updatedConversationSummary)["spaDraft"])
        request = follow_up(request, response, "Mi habitación", "field:ROOM_SERVICE:deliveryLocation:ROOM")
        response = plan_v2_turn(request)
        self.contextual()
        request = follow_up(request, response, "2 hamburguesas")
        response = plan_v2_turn(request)
        self.assertEqual("ROOM_SERVICE", summary_state(response.updatedConversationSummary)["pendingOffering"])
        self.assertEqual(spa_draft, summary_state(response.updatedConversationSummary)["spaDraft"])
        self.assertEqual([], response.toolCalls)

    def test_new_maintenance_request_is_not_consumed_by_spa_task(self):
        from tests.test_v2_scope_router import maintenance_offering
        request = request_for("Hay una fuga en el baño", "SPA_RESERVATION_CHANGE_DETAILS")
        request.conversation.summary = json.dumps({"spaTaskFocus": str(request.conversation.focusedConversationTaskId)})
        request.availableOfferings.append(OfferingCapability.model_validate(maintenance_offering()))
        self.scope.side_effect = lambda r, m, s: (ScopeDecision(kind="SERVICE_REQUEST", offeringCode="MAINTENANCE",
            relevantText=m.text, hasRequestDetails=True, containsUnrelatedTopic=False, confidence=1), OpenAiTokenUsage())
        self.planner.side_effect = None
        self.planner.return_value = OpenAiJsonResult(payload={
            "disposition": "TOOL_CALLS_REQUIRED", "messages": [], "toolCalls": [{
                "toolName": "START_SERVICE", "arguments": {"offeringCode": "MAINTENANCE", "input": {"issue": "fuga en el baño"}},
                "confidence": 1, "evidenceMessageIds": [str(request.trigger.messageId)],
            }],
        }, usage=OpenAiTokenUsage(), response_id="maintenance-test")
        response = plan_v2_turn(request)
        self.assertEqual("MAINTENANCE", response.toolCalls[0].arguments["offeringCode"])
        self.assertIsNone(response.toolCalls[0].targetConversationTaskId)
        self.assertIsNone(summary_state(response.updatedConversationSummary)["spaTaskFocus"])
        self.extract.assert_not_called()

    def test_mixed_scope_extracts_only_current_hotel_portion(self):
        hotel_text = "Quiero masaje mañana a las 17:00"
        request = request_for(hotel_text + " y explica programación")
        self.scope.side_effect = lambda r, m, s: (ScopeDecision(kind="SERVICE_REQUEST", offeringCode="SPA",
            relevantText=hotel_text, hasRequestDetails=True, containsUnrelatedTopic=True, confidence=1), OpenAiTokenUsage())
        self.extract.side_effect = None
        self.extract.return_value = extraction(hotel_text, {**VALUES, "reservationDate": "2026-08-31"})
        response = plan_v2_turn(request)
        self.assertNotIn("programación", self.extract.call_args.args[0])
        self.assertNotIn("programación", response.updatedConversationSummary)
        self.assertIn("solo puedo ayudarte", response.messages[0].text)
        self.assertEqual([], response.toolCalls)

    def test_dst_gap_and_fold_require_another_unambiguous_time(self):
        from datetime import datetime
        for clock, day in [("2026-03-07T12:00:00+00:00", "2026-03-08"), ("2026-10-31T12:00:00+00:00", "2026-11-01")]:
            request = self.new_request("Masaje en la fecha y hora indicada")
            request.hotel.timeZone = "America/New_York"
            request.createdAt = datetime.fromisoformat(clock)
            hour = "02:30" if day == "2026-03-08" else "01:30"
            response = self.capture(request, {**VALUES, "reservationDate": day, "reservationTime": hour})
            self.assertEqual([], response.toolCalls)
            self.assertNotIn("reservationTime", summary_state(response.updatedConversationSummary)["spaDraft"]["capturedFields"])


if __name__ == "__main__":
    unittest.main()
