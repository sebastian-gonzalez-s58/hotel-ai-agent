import copy
import json
import time
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

from app.agents.v2_turn_planner import plan_v2_turn, _faq_knowledge_lookup_plan
from app.core.config import settings
from app.core.errors import AgentDependencyError, AgentModelError, AgentTimeoutError
from app.schemas.v2_turns import AgentTurnRequest
from app.services.faq_grounding import (
    FaqSelection, FaqVerification, resolve_semantic_faq, restore_grounded_faq,
)
from app.services.openai_client import OpenAiJsonResult
from app.services.telemetry_client import OpenAiTokenUsage
from tests.test_v2_turn_endpoint import payload, MESSAGE_ID
from tests.test_v2_turn_planner import guided_faq_offering


SOURCE_ID = "90000000-0000-0000-0000-000000000150"
SOURCE_ANSWER = "La alberca abre todos los días de 8:00 a 22:00 y cierra a las 22:00."
ANSWER = "The pool closes every day at 22:00. Can I help you with anything else?"


def search_for(question="What time does the pool close?"):
    return {"query": question, "matchStatus": "SEMANTIC_REVIEW_REQUIRED", "confidence": 0.0,
            "retrievalMode": "SEMANTIC_V1", "candidateSetComplete": True, "matches": [{
                "catalogItemId": SOURCE_ID, "question": "¿Cuál es el horario de la alberca?",
                "answer": SOURCE_ANSWER, "sourceReference": None, "confidence": 0.0}]}


def request_for(search=None, language="en"):
    search = search if search is not None else search_for()
    data = payload()
    data["hotel"]["hotelCode"] = "TELWARE_DEMO"
    data["guest"]["preferredLanguage"] = language
    data["trigger"]["eventPayload"] = {"languageContext": {"version": 1, "explicit": True}}
    data["availableOfferings"] = [guided_faq_offering()]
    data["toolPolicy"] = {"allowedTools": ["SEARCH_KNOWLEDGE", "START_SERVICE"], "maxToolCalls": 2}
    data["conversation"]["recentMessages"][0]["text"] = search["query"]
    data["previousToolResults"] = [{"toolCallId": str(uuid4()), "toolName": "SEARCH_KNOWLEDGE",
                                    "status": "SUCCEEDED", "result": search}]
    return AgentTurnRequest.model_validate(data)


def selection(answer=ANSWER, **changes):
    return {"status": "ANSWERABLE", "sourceId": SOURCE_ID, "sameSubject": True,
            "fullyAnswers": True, "conflictingSources": False, "confidence": 0.98,
            "supportingQuotes": ["22:00"], "answer": answer, **changes}


def verification(**changes):
    return {"supported": True, "sameSubject": True, "fullyAnswers": True,
            "noConflictingSources": True,
            "preservesNegations": True, "correctLanguage": True, "naturalAndNonredundant": True,
            "hasBriefHelpOffer": True, "confidence": 0.98, **changes}


def result(data):
    return OpenAiJsonResult(data, OpenAiTokenUsage(input_tokens=10, output_tokens=5, total_tokens=15), "test")


def start_receipt(request, response):
    operation_id = str(uuid4())
    data = request.model_dump(mode="json")
    data["conversation"]["summary"] = response.updatedConversationSummary
    data["previousToolResults"].append({"toolCallId": str(response.toolCalls[0].toolCallId),
        "toolName": "START_SERVICE", "status": "SUCCEEDED", "result": {
            "offeringCode": "FAQ", "operationId": operation_id, "referenceCode": "FAQ-TEST-123"}})
    return AgentTurnRequest.model_validate(data), operation_id


class MultilingualFaqGroundingTest(unittest.TestCase):
    @patch("app.services.faq_grounding.call_openai_json_result")
    def test_cross_language_candidates_preserve_original_evidence(self, model):
        cases = [
            ("en", "What time does the pool close?", ANSWER),
            ("fr", "À quelle heure ferme la piscine?", "La piscine ferme à 22:00. Puis-je vous aider autrement?"),
            ("de", "Wann schließt der Pool?", "Der Pool schließt um 22:00. Kann ich Ihnen sonst helfen?"),
            ("pt", "A que horas fecha a piscina?", "A piscina fecha às 22:00. Posso ajudar em mais alguma coisa?"),
            ("ja", "プールは何時に閉まりますか？", "プールは22:00に閉まります。他にお手伝いできますか？"),
            ("zh", "泳池几点关闭？", "泳池在22:00关闭。还有什么可以帮您的吗？"),
            ("ar", "متى يغلق المسبح؟", "يغلق المسبح الساعة 22:00. هل يمكنني مساعدتك بشيء آخر؟"),
        ]
        for language, question, answer in cases:
            with self.subTest(language=language):
                model.reset_mock()
                model.side_effect = [result(selection(answer)), result(verification())]
                search = search_for(question)
                request = request_for(search, language)
                snapshot, usage = resolve_semantic_faq(request, search, time.perf_counter())
                restored = restore_grounded_faq(request, search, snapshot)
                self.assertEqual(answer, restored["guestAnswer"])
                self.assertEqual(SOURCE_ANSWER, restored["answer"])
                self.assertEqual(30, usage["totalTokens"])
                self.assertIn(question, model.call_args_list[0].args[0])
                self.assertIn(SOURCE_ANSWER, model.call_args_list[0].args[0])
                self.assertEqual(2, model.call_count)
                self.assertTrue(all(0 < call.kwargs["timeout_seconds"] <= 6 for call in model.call_args_list))
                self.assertTrue(all(call.kwargs["strict_schema"] for call in model.call_args_list))

    @patch("app.services.faq_grounding.call_openai_json_result")
    def test_wrong_subject_partial_question_and_conflicts_require_staff(self, model):
        cases = [
            ("When does the HOTEL close?", {"sameSubject": False}),
            ("When does the pool close and what does admission cost?", {"fullyAnswers": False}),
            ("When does it close?", {"status": "AMBIGUOUS"}),
            ("When does the pool close?", {"conflictingSources": True}),
            ("Is the restaurant available?", {"status": "NO_MATCH"}),
            ("When does the pool close?", {"confidence": 0.89}),
        ]
        for question, changes in cases:
            with self.subTest(question=question, changes=changes):
                model.reset_mock()
                model.return_value = result(selection(**changes))
                search = search_for(question)
                response = plan_v2_turn(request_for(search))
                self.assertEqual("HUMAN_REQUIRED", response.toolCalls[0].arguments["input"]["resolutionMode"])
                self.assertEqual(question, response.toolCalls[0].arguments["input"]["question"])
                self.assertEqual([], response.messages)
                self.assertEqual(1, model.call_count)

    @patch("app.services.faq_grounding.call_openai_json_result")
    def test_unknown_source_fabricated_quote_and_changed_literals_are_rejected(self, model):
        for changes in ({"sourceId": str(uuid4())}, {"supportingQuotes": ["24:00"]},
                        {"supportingQuotes": []}, {"answer": ANSWER.replace("22:00", "23:00")},
                        {"answer": ANSWER + " See https://unapproved.example/info"}, {"answer": "  "}):
            with self.subTest(changes=changes):
                model.reset_mock()
                model.return_value = result(selection(**changes))
                snapshot, _ = resolve_semantic_faq(request_for(), search_for(), time.perf_counter())
                self.assertIsNone(snapshot)
                self.assertEqual(1, model.call_count)

    @patch("app.services.faq_grounding.call_openai_json_result")
    def test_independent_verification_can_reject_every_safety_dimension(self, model):
        for field in verification():
            with self.subTest(field=field):
                model.side_effect = [result(selection()), result(verification(**{field: 0.94 if field == "confidence" else False}))]
                snapshot, usage = resolve_semantic_faq(request_for(), search_for(), time.perf_counter())
                self.assertIsNone(snapshot)
                self.assertEqual(30, usage["totalTokens"])

    @patch("app.services.faq_grounding.call_openai_json_result")
    def test_empty_incomplete_and_oversized_catalogs_never_call_model(self, model):
        cases = [dict(search_for(), matches=[]), dict(search_for(), candidateSetComplete=False),
                 dict(search_for(), matches=[search_for()["matches"][0]] * 65)]
        too_long = search_for()
        too_long["matches"][0]["answer"] = "a" * 40001
        cases.append(too_long)
        for search in cases:
            with self.subTest(complete=search["candidateSetComplete"], count=len(search["matches"])):
                response = plan_v2_turn(request_for(search))
                self.assertEqual("HUMAN_REQUIRED", response.toolCalls[0].arguments["input"]["resolutionMode"])
        model.assert_not_called()

    @patch("app.services.faq_grounding.call_openai_json_result")
    def test_invalid_model_output_and_dependencies_fall_back_without_retry_loop(self, model):
        for error in (AgentTimeoutError("timeout"), AgentDependencyError("unavailable"), AgentModelError("invalid")):
            for stage in (1, 2):
                with self.subTest(error=type(error).__name__, stage=stage):
                    model.reset_mock()
                    model.side_effect = [error] if stage == 1 else [result(selection()), error]
                    response = plan_v2_turn(request_for())
                    self.assertEqual("HUMAN_REQUIRED", response.toolCalls[0].arguments["input"]["resolutionMode"])
                    self.assertEqual(stage, model.call_count)
        model.side_effect = [result({"answer": "missing schema"})]
        self.assertEqual("HUMAN_REQUIRED", plan_v2_turn(request_for()).toolCalls[0].arguments["input"]["resolutionMode"])

    @patch("app.services.faq_grounding.call_openai_json_result")
    def test_exhausted_budget_does_not_start_another_model_request(self, model):
        snapshot, usage = resolve_semantic_faq(request_for(), search_for(),
                                              time.perf_counter() - settings.request_timeout_seconds)
        self.assertIsNone(snapshot)
        self.assertEqual({}, usage)
        model.assert_not_called()

    @patch("app.services.faq_grounding.call_openai_json_result")
    def test_answer_snapshot_is_bound_to_query_trigger_hotel_language_and_source(self, model):
        model.side_effect = [result(selection()), result(verification())]
        request, search = request_for(), search_for()
        snapshot, _ = resolve_semantic_faq(request, search, time.perf_counter())
        for field in ("query", "message", "hotel", "language", "source"):
            with self.subTest(field=field):
                changed, candidate = request.model_copy(deep=True), copy.deepcopy(search)
                if field == "query": candidate["query"] = "When does the hotel close?"
                if field == "message": changed.trigger.messageId = uuid4()
                if field == "hotel": changed.hotel.hotelId = uuid4()
                if field == "language": changed.guest.preferredLanguage = "de"
                if field == "source": candidate["matches"][0]["answer"] = "Different answer"
                self.assertIsNone(restore_grounded_faq(changed, candidate, snapshot))

    def test_structured_schemas_require_every_field_and_forbid_extra_data(self):
        for schema in (FaqSelection, FaqVerification):
            data = schema.model_json_schema()
            self.assertFalse(data["additionalProperties"])
            self.assertEqual(set(data["properties"]), set(data["required"]))

    @patch("app.services.faq_grounding.call_openai_json_result")
    def test_verifier_sees_conflicting_sources_even_if_selector_misses_them(self, model):
        search = search_for()
        search["matches"].append({**search["matches"][0], "catalogItemId": str(uuid4()),
                                  "answer": "La alberca cierra a las 20:00."})
        model.side_effect = [result(selection()), result(verification(noConflictingSources=False))]
        snapshot, _ = resolve_semantic_faq(request_for(search), search, time.perf_counter())
        self.assertIsNone(snapshot)
        self.assertIn("20:00", model.call_args_list[1].args[0])


class MultilingualFaqProcessTest(unittest.TestCase):
    def test_lookup_negotiates_semantic_mode_without_rewriting_guest_question(self):
        request = request_for(search_for("プールは何時に閉まりますか？"), "ja")
        request.previousToolResults = []
        request.conversation.summary = json.dumps({"pendingOffering": "FAQ", "capturedFields": {}})
        response = _faq_knowledge_lookup_plan(request, time.perf_counter(), direct_question=True)
        self.assertTrue(response.toolCalls[0].arguments["semantic"])
        self.assertEqual(request.conversation.recentMessages[0].text, response.toolCalls[0].arguments["query"])
        self.assertEqual([UUID(MESSAGE_ID)], response.toolCalls[0].evidenceMessageIds)
        self.assertEqual([], response.messages)
        request.trigger.eventPayload = {}
        legacy = _faq_knowledge_lookup_plan(request, time.perf_counter(), direct_question=True)
        self.assertNotIn("semantic", legacy.toolCalls[0].arguments)

    @patch("app.agents.v2_turn_planner.localize_response", side_effect=lambda request, response, started: response)
    @patch("app.services.faq_grounding.call_openai_json_result")
    def test_verified_answer_always_starts_faq_before_reply_and_is_not_retranslated(self, model, localize):
        model.side_effect = [result(selection()), result(verification())]
        request = request_for()
        start = plan_v2_turn(request)
        self.assertEqual("TOOL_CALLS_REQUIRED", start.disposition)
        self.assertEqual([], start.messages)
        tool = start.toolCalls[0]
        self.assertEqual("START_SERVICE", tool.toolName)
        self.assertEqual("FAQ", tool.arguments["offeringCode"])
        self.assertEqual("AUTOMATIC", tool.arguments["input"]["resolutionMode"])
        self.assertEqual(SOURCE_ANSWER, tool.arguments["input"]["knowledgeAnswer"])
        self.assertEqual(SOURCE_ID, tool.arguments["input"]["knowledgeItemId"])
        self.assertEqual(30, start.usage.totalTokens)
        receipt, operation_id = start_receipt(request, start)
        localize.reset_mock()
        finish = plan_v2_turn(receipt)
        self.assertEqual("RESPONSE_READY", finish.disposition)
        self.assertEqual([], finish.toolCalls)
        self.assertEqual(ANSWER, finish.messages[0].text)
        self.assertEqual("en", finish.messages[0].language)
        self.assertEqual([UUID(operation_id)], finish.messages[0].operationIds)
        self.assertNotIn("FAQ-TEST-123", finish.messages[0].text)
        self.assertEqual("{}", finish.updatedConversationSummary)
        self.assertEqual(0, finish.usage.totalTokens)
        self.assertEqual(2, model.call_count)
        localize.assert_not_called()

    @patch("app.agents.v2_turn_planner.localize_response", side_effect=lambda request, response, started: response)
    @patch("app.services.faq_grounding.call_openai_json_result")
    def test_unknown_answer_starts_human_faq_and_preserves_localized_handoff(self, model, localize):
        model.return_value = result(selection(status="NO_MATCH", sourceId=None, answer=None, supportingQuotes=[]))
        request = request_for(search_for("When does the hotel close?"))
        start = plan_v2_turn(request)
        self.assertEqual({"question": "When does the hotel close?", "resolutionMode": "HUMAN_REQUIRED"},
                         start.toolCalls[0].arguments["input"])
        receipt, operation_id = start_receipt(request, start)
        localize.reset_mock()
        finish = plan_v2_turn(receipt)
        self.assertEqual("HANDOFF", finish.messages[0].purpose)
        self.assertEqual([UUID(operation_id)], finish.messages[0].operationIds)
        self.assertNotIn("22:00", finish.messages[0].text)
        self.assertEqual([], finish.toolCalls)
        localize.assert_called_once()

    @patch("app.services.faq_grounding.call_openai_json_result")
    def test_source_revoked_before_start_retries_once_as_human_without_model(self, model):
        data = request_for().model_dump(mode="json")
        data["previousToolResults"].append({"toolCallId": str(uuid4()), "toolName": "START_SERVICE",
            "status": "REJECTED", "error": {"code": "FAQ_KNOWLEDGE_STALE", "message": "Source changed", "retryable": False}})
        response = plan_v2_turn(AgentTurnRequest.model_validate(data))
        self.assertEqual("HUMAN_REQUIRED", response.toolCalls[0].arguments["input"]["resolutionMode"])
        self.assertNotIn("knowledgeAnswer", response.toolCalls[0].arguments["input"])
        self.assertIsNone(json.loads(response.updatedConversationSummary)["faqGrounding"])
        model.assert_not_called()

    @patch("app.services.faq_grounding.call_openai_json_result")
    def test_legacy_exact_search_keeps_existing_behavior(self, model):
        search = search_for()
        search.pop("retrievalMode")
        search["matchStatus"] = "EXACT_MATCH"
        search["confidence"] = 1.0
        response = plan_v2_turn(request_for(search))
        self.assertEqual("AUTOMATIC", response.toolCalls[0].arguments["input"]["resolutionMode"])
        model.assert_not_called()

    @patch("app.agents.v2_turn_planner.localize_response", side_effect=lambda request, response, started: response)
    @patch("app.services.faq_grounding.call_openai_json_result")
    def test_legacy_stale_result_does_not_reappear_after_human_process_start(self, model, localize):
        search = search_for()
        search.pop("retrievalMode")
        search["matchStatus"] = "EXACT_MATCH"
        data = request_for(search).model_dump(mode="json")
        data["previousToolResults"].append({"toolCallId": str(uuid4()), "toolName": "START_SERVICE",
            "status": "REJECTED", "error": {"code": "FAQ_KNOWLEDGE_STALE", "message": "Source changed", "retryable": False}})
        request = AgentTurnRequest.model_validate(data)
        response = plan_v2_turn(request)
        self.assertEqual("HUMAN_REQUIRED", response.toolCalls[0].arguments["input"]["resolutionMode"])
        receipt, _ = start_receipt(request, response)
        finish = plan_v2_turn(receipt)
        self.assertEqual("HANDOFF", finish.messages[0].purpose)
        self.assertNotIn("22:00", finish.messages[0].text)
        model.assert_not_called()
