import json
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

from app.agents.spa_turns import summary_state
from app.agents.v2_turn_planner import (
    _ensure_service_start_acknowledgements,
    _normalize_guest_experience,
    plan_v2_turn,
)
from app.schemas.v2_turns import AgentTurnRequest, TurnTriggerType
from app.services.openai_client import OpenAiJsonResult
from app.services.telemetry_client import OpenAiTokenUsage
from tests.test_spa_turns import VALUES, spa_operation
from tests.test_v2_turn_endpoint import payload
from tests.test_v2_turn_planner import guided_faq_offering, guided_room_service_offering, guided_spa_offering


def start_result(code="ROOM_SERVICE", status="SUCCEEDED"):
    return {
        "toolCallId": str(uuid4()), "toolName": "START_SERVICE", "status": status,
        "result": {"offeringCode": code, "operationId": str(uuid4()),
                   "referenceCode": f"{code}-0042", "lifecycle": "ACTIVE"},
    }


def request_for(results, state=None):
    data = payload()
    data["hotel"]["hotelCode"] = "telware"
    data["trigger"] = {"type": "TOOL_RESULTS"}
    data["conversation"]["recentMessages"][0]["text"] = "Confirmar"
    data["conversation"]["summary"] = json.dumps(state or {})
    data["availableOfferings"] = [guided_room_service_offering(), guided_spa_offering(), guided_faq_offering()]
    for code, name in [("MAINTENANCE", "Mantenimiento"), ("FRONT_DESK", "Recepcion"),
                       ("AIRPORT_TRANSFER", "Traslado al aeropuerto")]:
        data["availableOfferings"].append({
            "offeringCode": code, "name": name, "description": name,
            "executionMode": "PROCESS", "inputSchema": {"type": "object"},
        })
    data["toolPolicy"] = {"allowedTools": ["START_SERVICE", "SEARCH_KNOWLEDGE"], "maxToolCalls": 2}
    data["previousToolResults"] = results
    return AgentTurnRequest.model_validate(data)


class ServiceStartAcknowledgementsTest(unittest.TestCase):
    def setUp(self):
        self.model_calls = []
        for target in (
            "app.agents.v2_turn_planner.call_openai_json_result",
            "app.agents.v2_turn_planner.classify_hotel_scope",
            "app.agents.v2_scope_router.call_openai_json_result",
            "app.agents.spa_turns._extract",
            "app.agents.spa_turns.call_openai_json_result",
        ):
            mock = patch(target, side_effect=AssertionError(f"Unexpected model path: {target}"))
            self.model_calls.append(mock.start())
            self.addCleanup(mock.stop)
        self.planner = self.model_calls[0]

    def assert_zero_model_usage(self, response):
        for mock in self.model_calls:
            mock.assert_not_called()
        for field in ("inputTokens", "cachedInputTokens", "outputTokens", "reasoningTokens", "totalTokens"):
            self.assertEqual(0, getattr(response.usage, field), field)
        self.assertGreaterEqual(response.usage.latencyMs, 0)
        self.assertEqual("RESPONSE_READY", response.disposition)
        self.assertEqual([], response.toolCalls)

    def model_failure_response(self):
        self.planner.side_effect = None
        self.planner.return_value = OpenAiJsonResult(
            payload={"disposition": "RESPONSE_READY", "messages": [{
                "purpose": "CLARIFICATION", "text": "No se pudo completar la otra solicitud.",
                "language": "es-MX", "operationIds": [], "conversationTaskIds": [],
            }], "updatedConversationSummary": "Retain this failure context."},
            usage=OpenAiTokenUsage(input_tokens=3, output_tokens=2, total_tokens=5),
            response_id="offline-failure",
        )

    def test_all_service_offerings_acknowledge_without_model_scope_or_extraction(self):
        for code in ("ROOM_SERVICE", "MAINTENANCE", "FRONT_DESK", "SPA", "AIRPORT_TRANSFER"):
            for language in ("es-MX", "en-US"):
                for trigger in ("INBOUND_MESSAGE", "TOOL_RESULTS"):
                    with self.subTest(code=code, language=language, trigger=trigger):
                        result = start_result(code)
                        request = request_for([result])
                        request.guest.preferredLanguage = language
                        request.trigger = request.trigger.model_copy(update={
                            "type": TurnTriggerType(trigger),
                            "messageId": request.conversation.recentMessages[0].messageId,
                        })
                        before = request.model_dump()
                        response = plan_v2_turn(request)
                        self.assert_zero_model_usage(response)
                        self.assertEqual(before, request.model_dump())
                        self.assertEqual(1, len(response.messages))
                        message = response.messages[0]
                        self.assertEqual("STATUS_UPDATE", message.purpose)
                        self.assertEqual([UUID(result["result"]["operationId"])], message.operationIds)
                        self.assertEqual([], message.conversationTaskIds)
                        self.assertIsNone(message.interaction)
                        self.assertEqual(language, message.language)
                        self.assertIn(result["result"]["referenceCode"], message.text)
                        self.assertIn("Recibir" if language == "es-MX" else "We will send updates", message.text)

    def test_future_offering_does_not_require_catalog_presence(self):
        request = request_for([start_result("FUTURE_SERVICE")])
        request.availableOfferings = []
        response = plan_v2_turn(request)
        self.assert_zero_model_usage(response)
        self.assertIn("future service", response.messages[0].text)

    def test_fast_path_uses_existing_renderer_before_capture_planners(self):
        request = request_for([start_result()])
        with patch("app.agents.v2_turn_planner._ensure_service_start_acknowledgements",
                   wraps=_ensure_service_start_acknowledgements) as render, \
                patch("app.agents.v2_turn_planner.plan_spa_turn", side_effect=AssertionError("SPA planner")), \
                patch("app.agents.v2_turn_planner._room_service_operation_task_plan",
                      side_effect=AssertionError("Task planner")):
            response = plan_v2_turn(request)
        render.assert_called_once()
        self.assert_zero_model_usage(response)

    def test_completed_process_still_acknowledges_authoritative_creation(self):
        result = start_result()
        result["result"].update(lifecycle="COMPLETED", detailedStatus="PROCESS_COMPLETED")
        request = request_for([result])
        response = plan_v2_turn(request)
        self.assert_zero_model_usage(response)
        self.assertIn(result["result"]["referenceCode"], response.messages[0].text)

    def test_clears_only_submitted_capture_preserving_spa_and_multiple_operations(self):
        operations = [spa_operation(), spa_operation("SPA_RESERVATION_CHANGE_DETAILS")]
        tasks = {str(o.pendingConversationTasks[0].conversationTaskId): {
            "operationId": str(o.operationId), "capturedFields": VALUES,
        } for o in operations}
        independent = {
            "spaDraft": {"id": str(uuid4()), "capturedFields": VALUES, "submitted": False},
            "spaTasks": tasks, "spaTaskFocus": next(iter(tasks)),
            "spaOperationFocus": str(operations[0].operationId),
            "operationNotes": {str(o.operationId): "pending guest response" for o in operations},
        }
        for code in ("ROOM_SERVICE", "MAINTENANCE", "FRONT_DESK", "AIRPORT_TRANSFER"):
            with self.subTest(code=code):
                state = {**independent, "pendingOffering": code, "capturedFields": {"details": "submitted"},
                         "phase": "STARTING", "readyToStart": True, "awaitingExplicitConfirmation": True}
                request = request_for([start_result(code)], state)
                request.activeOperations = operations
                request.conversation.summary = "Guest history.\n" + request.conversation.summary + "\nKeep this note."
                before = request.model_dump()
                response = plan_v2_turn(request)
                self.assert_zero_model_usage(response)
                self.assertEqual(independent, summary_state(response.updatedConversationSummary))
                self.assertTrue(response.updatedConversationSummary.startswith("Guest history.\n"))
                self.assertTrue(response.updatedConversationSummary.endswith("\nKeep this note."))
                self.assertEqual(before, request.model_dump())

    def test_does_not_clear_other_offering_or_unsubmitted_same_offering_capture(self):
        for code, submitted in (("MAINTENANCE", True), ("ROOM_SERVICE", False)):
            with self.subTest(code=code, submitted=submitted):
                request = request_for([start_result()], {
                    "pendingOffering": code, "readyToStart": submitted,
                    "phase": "CAPTURING_ITEMS", "capturedFields": {"details": "another request"},
                })
                response = plan_v2_turn(request)
                self.assert_zero_model_usage(response)
                self.assertEqual(request.conversation.summary, response.updatedConversationSummary)

    def test_preserves_unstructured_summary_verbatim(self):
        for summary in ("", "Guest needs two independent services.", "History.\n{invalid}", "{}"):
            with self.subTest(summary=summary):
                request = request_for([start_result()])
                request.conversation.summary = summary
                response = plan_v2_turn(request)
                self.assert_zero_model_usage(response)
                self.assertEqual(summary, response.updatedConversationSummary)

    def test_pretty_printed_submitted_capture_preserves_independent_state(self):
        request = request_for([start_result()])
        request.conversation.summary = json.dumps({
            "pendingOffering": "ROOM_SERVICE", "phase": "STARTING",
            "capturedFields": {"items": []}, "note": "keep", "otherDrafts": [{}],
        }, indent=2)
        response = plan_v2_turn(request)
        self.assert_zero_model_usage(response)
        self.assertEqual({"note": "keep", "otherDrafts": [{}]}, summary_state(response.updatedConversationSummary))

    def test_malformed_capture_marker_cannot_block_authoritative_acknowledgement(self):
        for pending in (None, [], {}, 42):
            with self.subTest(pending=pending):
                request = request_for([start_result()], {"pendingOffering": pending, "readyToStart": True})
                response = plan_v2_turn(request)
                self.assert_zero_model_usage(response)
                self.assertEqual(request.conversation.summary, response.updatedConversationSummary)

    def test_spa_start_clears_draft_but_preserves_existing_tasks_and_focus(self):
        independent = {"spaTasks": {str(uuid4()): {"capturedFields": VALUES}},
                       "spaTaskFocus": str(uuid4()), "spaOperationFocus": str(uuid4())}
        request = request_for([start_result("SPA")], {
            **independent, "pendingOffering": "SPA", "capturedFields": VALUES,
            "awaitingExplicitConfirmation": True, "readyToStart": True, "spaDraft": {"submitted": True},
        })
        response = plan_v2_turn(request)
        self.assert_zero_model_usage(response)
        self.assertEqual({**independent, "spaDraft": None}, summary_state(response.updatedConversationSummary))

    def test_multiple_successful_starts_keep_each_folio_and_operation_link(self):
        for codes in (("ROOM_SERVICE", "MAINTENANCE", "FRONT_DESK"),
                      ("ROOM_SERVICE", "SPA"), ("ROOM_SERVICE", "ROOM_SERVICE")):
            with self.subTest(codes=codes):
                results = [start_result(code) for code in codes]
                for index, result in enumerate(results):
                    result["result"]["referenceCode"] += f"-{index}"
                request = request_for(results, {"pendingOffering": "ROOM_SERVICE", "phase": "STARTING",
                                                "spaDraft": {"submitted": True}, "note": "keep"})
                response = plan_v2_turn(request)
                self.assert_zero_model_usage(response)
                self.assertEqual(len(results), len(response.messages))
                for result, message in zip(results, response.messages):
                    self.assertIn(result["result"]["referenceCode"], message.text)
                    self.assertEqual([UUID(result["result"]["operationId"])], message.operationIds)
                state = summary_state(response.updatedConversationSummary)
                self.assertNotIn("pendingOffering", state)
                self.assertEqual("keep", state["note"])
                self.assertEqual(None if "SPA" in codes else {"submitted": True}, state["spaDraft"])

    def test_successful_catalog_lookup_does_not_prevent_start_acknowledgement(self):
        for tool in ("SEARCH_CATALOG", "LIST_AVAILABLE_OFFERINGS", "GET_OFFERING_DEFINITION"):
            with self.subTest(tool=tool):
                lookup = {"toolCallId": str(uuid4()), "toolName": tool, "status": "SUCCEEDED", "result": {}}
                response = plan_v2_turn(request_for([lookup, start_result()]))
                self.assert_zero_model_usage(response)
                self.assertEqual(1, len(response.messages))

    def test_duplicate_successes_acknowledge_each_authoritative_operation_once(self):
        first, second = start_result(), start_result()
        duplicate = {**first, "toolCallId": str(uuid4()),
                     "result": {**first["result"], "operationId": first["result"]["operationId"].upper()}}
        for results in ([first, duplicate], [first, second, duplicate], [first, *([duplicate] * 12)]):
            with self.subTest(result_count=len(results)):
                response = plan_v2_turn(request_for(results))
                self.assert_zero_model_usage(response)
                expected = [first, second] if second in results else [first]
                self.assertEqual(len(expected), len(response.messages))
                for result, message in zip(expected, response.messages):
                    self.assertEqual([UUID(result["result"]["operationId"])], message.operationIds)
                    self.assertIn(result["result"]["referenceCode"], message.text)

    def test_duplicate_operation_does_not_bypass_batch_error_or_faq_exclusion(self):
        self.model_failure_response()
        first = start_result()
        for changes in ({"status": "FAILED"}, {"status": "REJECTED"},
                        {"result": {**first["result"], "offeringCode": "FAQ"}}):
            with self.subTest(changes=changes):
                duplicate = {**first, "toolCallId": str(uuid4()), **changes}
                response = plan_v2_turn(request_for([first, duplicate]))
                self.assertEqual("CLARIFICATION", response.messages[0].purpose)
                self.assertEqual(5, response.usage.totalTokens)
        self.assertEqual(3, self.planner.call_count)

    def test_failed_and_rejected_starts_are_not_acknowledged(self):
        self.model_failure_response()
        for code in ("ROOM_SERVICE", "MAINTENANCE", "FRONT_DESK", "SPA", "AIRPORT_TRANSFER"):
            for status in ("FAILED", "REJECTED"):
                with self.subTest(code=code, status=status):
                    response = plan_v2_turn(request_for([start_result(code, status)]))
                    self.assertEqual("CLARIFICATION", response.messages[0].purpose)
                    self.assertNotIn("0042", response.messages[0].text)
                    self.assertEqual(5, response.usage.totalTokens)
        self.assertEqual(10, self.planner.call_count)

    def test_mixed_failures_are_not_overwritten_by_success_acknowledgements(self):
        self.model_failure_response()
        for code in ("ROOM_SERVICE", "SPA"):
            for tool in ("START_SERVICE", "SEARCH_CATALOG", "COMPLETE_CONVERSATION_TASK"):
                for status in ("FAILED", "REJECTED"):
                    with self.subTest(code=code, tool=tool, status=status):
                        failed = start_result("MAINTENANCE", status)
                        failed["toolName"] = tool
                        request = request_for([start_result(code), failed])
                        response = plan_v2_turn(request)
                        self.assertEqual("CLARIFICATION", response.messages[0].purpose)
                        self.assertEqual("Retain this failure context.", response.updatedConversationSummary)
                        self.assertEqual(5, response.usage.totalTokens)
        self.assertEqual(12, self.planner.call_count)

    def test_incomplete_or_conflicting_success_results_stay_on_normal_path(self):
        self.model_failure_response()
        invalid = [None, {}, {"operationId": "not-a-uuid"}, {"referenceCode": " "},
                   {"referenceCode": 42}, {"operationId": None}, {"offeringCode": None}]
        for fields in invalid:
            for mixed in (False, True):
                with self.subTest(fields=fields, mixed=mixed):
                    result = start_result()
                    if fields:
                        result["result"].update(fields)
                    else:
                        result["result"] = fields
                    results = [start_result("MAINTENANCE"), result] if mixed else [result]
                    response = plan_v2_turn(request_for(results))
                    self.assertEqual("CLARIFICATION", response.messages[0].purpose)
                    self.assertEqual(5, response.usage.totalTokens)
        result = start_result()
        result["error"] = {"code": "CONFLICT", "message": "Conflicting result", "retryable": False}
        self.assertEqual("CLARIFICATION", plan_v2_turn(request_for([result])).messages[0].purpose)
        self.assertEqual(15, self.planner.call_count)

    def test_other_mutation_or_oversized_batch_is_not_silently_acknowledged(self):
        self.model_failure_response()
        for tool in ("COMPLETE_CONVERSATION_TASK", "SAVE_CONVERSATION_TASK_PROGRESS", "EXECUTE_SERVICE_ACTION"):
            with self.subTest(tool=tool):
                other = {"toolCallId": str(uuid4()), "toolName": tool, "status": "SUCCEEDED", "result": {}}
                response = plan_v2_turn(request_for([start_result(), other]))
                self.assertEqual("CLARIFICATION", response.messages[0].purpose)
        response = plan_v2_turn(request_for([start_result() for _ in range(11)]))
        self.assertEqual("CLARIFICATION", response.messages[0].purpose)
        self.assertEqual(4, self.planner.call_count)

    def test_normalization_uses_same_safe_capture_cleanup(self):
        request = request_for([start_result()], {
            "pendingOffering": "ROOM_SERVICE", "readyToStart": True, "capturedFields": {"items": []},
            "spaDraft": {"capturedFields": VALUES}, "note": "keep",
        })
        normalized = _normalize_guest_experience(request, {
            "disposition": "TOOL_CALLS_REQUIRED", "messages": [], "toolCalls": [{"toolName": "START_SERVICE"}],
        })
        self.assertEqual("RESPONSE_READY", normalized["disposition"])
        self.assertEqual([], normalized["toolCalls"])
        self.assertEqual({"spaDraft": {"capturedFields": VALUES}, "note": "keep"},
                         summary_state(normalized["updatedConversationSummary"]))

    def test_faq_keeps_process_driven_answer_or_handoff_without_folio(self):
        for matched in (True, False):
            with self.subTest(matched=matched):
                search = {"toolCallId": str(uuid4()), "toolName": "SEARCH_KNOWLEDGE", "status": "SUCCEEDED",
                          "result": {"query": "When does the pool close?", "confidence": 1.0 if matched else 0.0,
                                     "matchStatus": "EXACT_MATCH" if matched else "NO_MATCH",
                                     "matches": [{"answer": "The pool closes at 22:00."}] if matched else []}}
                request = request_for([search])
                first = plan_v2_turn(request)
                self.assertEqual("TOOL_CALLS_REQUIRED", first.disposition)
                self.assertEqual("FAQ", first.toolCalls[0].arguments["offeringCode"])
                self.assertEqual("AUTOMATIC" if matched else "HUMAN_REQUIRED",
                                 first.toolCalls[0].arguments["input"]["resolutionMode"])
                result = start_result("FAQ")
                response = plan_v2_turn(request_for([search, result]))
                self.assert_zero_model_usage(response)
                self.assertEqual("ANSWER" if matched else "HANDOFF", response.messages[0].purpose)
                self.assertNotIn(result["result"]["referenceCode"], response.messages[0].text)
                self.assertEqual([UUID(result["result"]["operationId"])], response.messages[0].operationIds)
                # A non-FAQ start must not swallow the process-driven FAQ response either.
                response = plan_v2_turn(request_for([search, result, start_result()]))
                self.assert_zero_model_usage(response)
                self.assertEqual("ANSWER" if matched else "HANDOFF", response.messages[0].purpose)
                self.assertNotIn(result["result"]["referenceCode"], response.messages[0].text)

    def test_faq_start_without_search_context_is_not_a_generic_folio_ack(self):
        self.model_failure_response()
        for code in ("FAQ", "faq", " FAQ "):
            response = plan_v2_turn(request_for([start_result(code)]))
            self.assertEqual("CLARIFICATION", response.messages[0].purpose)
            self.assertNotIn("0042", response.messages[0].text)
        self.assertEqual(3, self.planner.call_count)


if __name__ == "__main__":
    unittest.main()
