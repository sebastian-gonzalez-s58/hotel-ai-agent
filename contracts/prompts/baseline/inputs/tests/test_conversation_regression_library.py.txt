"""Tests of fixture integrity and replay honesty, not product acceptance."""
import json
import socket
import unittest
from copy import deepcopy
from unittest.mock import patch
from uuid import uuid4

from pydantic import ValidationError

from app.schemas.v2_turns import AgentTurnResponse
from tests.conversation_regression.library import Case, Check, Event, ModelReply, load_cases, profiles
from tests.conversation_regression.replay import ReplayBoundaryError, ScriptedModel, replay_case
from tests.conversation_regression.session import ConversationSession, check_observation


def response(request, summary=None, **overrides):
    payload = {
        "schemaVersion": "2.0", "agentTurnId": request.agentTurnId,
        "disposition": "RESPONSE_READY", "messages": [], "toolCalls": [],
        "updatedConversationSummary": summary,
        "usage": {"model": "synthetic", "inputTokens": 0, "cachedInputTokens": 0,
                  "outputTokens": 0, "reasoningTokens": 0, "totalTokens": 0},
        "warnings": [],
    }
    payload.update(overrides)
    return AgentTurnResponse.model_validate(payload)


class ConversationRegressionLibraryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {case.id: case for case in load_cases()}
        cls.profile = profiles()["hotel"]

    def session(self):
        return ConversationSession(self.cases["SC-005-edit-items"], self.profile)

    def test_every_contract_scenario_has_cases(self):
        self.assertEqual({f"SC-{i:03}" for i in range(1, 22)}, {c.scenario_id for c in self.cases.values()})
        self.assertTrue(any(c.origin == "successful_path" for c in self.cases.values()))

    def test_fixture_cannot_inject_expected_state_between_turns(self):
        case = self.cases["SC-005-edit-items"].model_dump()
        case["steps"][1]["expected_state_as_input"] = {"items": []}
        with self.assertRaises(ValidationError):
            Case.model_validate(case)

    def test_duplicate_step_ids_are_rejected(self):
        case = self.cases["SC-005-edit-items"].model_dump()
        case["steps"][1]["id"] = case["steps"][0]["id"]
        with self.assertRaises(ValidationError):
            Case.model_validate(case)

    def test_guest_cannot_modify_backend_data(self):
        with self.assertRaises(ValidationError):
            Event(kind="guest", text="Hola", data={"summary": "expected"})

    def test_actual_summary_and_assistant_message_reach_next_turn(self):
        session = self.session()
        first = session.receive(Event(kind="guest", text="Cambiar pedido"))
        session.accept(response(first, '{"actual": "retained"}', messages=[{
            "messageDraftId": uuid4(), "purpose": "CLARIFICATION", "text": "¿Qué cambiamos?",
            "language": "es-MX", "operationIds": [], "conversationTaskIds": [],
        }]))
        second = session.receive(Event(kind="guest", text="Dos"))
        self.assertEqual({"actual": "retained"}, json.loads(second.conversation.summary))
        self.assertEqual(["GUEST", "ASSISTANT", "GUEST"], [m.actor for m in second.conversation.recentMessages])
        self.assertEqual("¿Qué cambiamos?", second.conversation.recentMessages[1].text)
        self.assertEqual(second.trigger.messageId, second.conversation.recentMessages[-1].messageId)
        self.assertNotEqual(first.trigger.messageId, second.trigger.messageId)

    def test_profile_and_fixture_are_isolated_between_sessions(self):
        original = deepcopy(self.profile)
        session = self.session()
        session.request.guest.displayName = "Modified"
        session.request.activeOperations.clear()
        self.assertEqual(original, self.profile)
        self.assertEqual("Test Guest", self.session().request.guest.displayName)

    def test_request_copy_cannot_rewrite_persisted_session(self):
        session = self.session()
        request = session.receive(Event(kind="guest", text="Hola"))
        request.conversation.summary = '{"injected":true}'
        self.assertNotIn("injected", session.request.conversation.summary)

    def test_null_and_blank_summaries_preserve_state_but_empty_object_clears(self):
        session = self.session()
        for summary in [None, " ", "{}"]:
            before = session.request.conversation.summary
            request = session.receive(Event(kind="guest", text="Hola"))
            session.accept(response(request, summary))
            self.assertEqual("{}" if summary == "{}" else before, session.request.conversation.summary)

    def test_wrong_turn_response_is_rejected(self):
        session = self.session()
        request = session.receive(Event(kind="guest", text="Hola"))
        with self.assertRaises(ValueError):
            session.accept(response(request, agentTurnId=uuid4()))
        self.assertTrue(session.pending)

    def test_unfinished_turn_or_tool_effect_blocks_next_guest(self):
        session = self.session()
        request = session.receive(Event(kind="guest", text="Recepción"))
        with self.assertRaises(RuntimeError):
            session.receive(Event(kind="guest", text="Otra cosa"))
        session.accept(response(request, disposition="TOOL_CALLS_REQUIRED", toolCalls=[{
            "toolCallId": uuid4(), "toolName": "START_SERVICE", "arguments": {},
            "confidence": 1, "evidenceMessageIds": [request.trigger.messageId],
        }]))
        with self.assertRaises(RuntimeError):
            session.receive(Event(kind="guest", text="Otra cosa"))

    def test_language_evidence_is_validated_before_any_state_mutation(self):
        session = self.session()
        request = session.receive(Event(kind="guest", text="English please"))
        before = session.request.conversation.summary
        decision = {"locale": "en", "source": "EXPLICIT", "confidence": 1, "messageId": uuid4()}
        with self.assertRaises(ValueError):
            session.accept(response(request, "{}", languageDecision=decision))
        self.assertEqual(before, session.request.conversation.summary)
        self.assertTrue(session.pending)
        decision["messageId"] = request.trigger.messageId
        session.accept(response(request, languageDecision=decision))
        next_request = session.receive(Event(kind="guest", text="Hello"))
        self.assertEqual("en", next_request.guest.preferredLanguage)
        self.assertEqual("EXPLICIT", next_request.trigger.eventPayload["languageContext"]["source"])

    def test_missing_value_cannot_satisfy_negative_check(self):
        errors = check_observation({}, [Check(path="/missing", op="not_contains", value="bad")])
        self.assertEqual(1, len(errors))

    def test_type_mismatch_and_invalid_contains_fail_without_crashing(self):
        checks = [Check(path="/value", op="equals", value=True),
                  Check(path="/text", op="not_contains", value={})]
        self.assertEqual(2, len(check_observation({"value": 1, "text": "ok"}, checks)))

    def test_json_pointer_escapes_and_indexes(self):
        checks = [Check(path="/a~1b/0/x~0y", op="equals", value=2)]
        self.assertEqual([], check_observation({"a/b": [{"x~y": 2}]}, checks))

    def test_model_script_mismatch_and_unused_replies_are_harness_errors(self):
        model = ScriptedModel([ModelReply(purpose="EXPECTED", payload={})])
        with self.assertRaises(ReplayBoundaryError):
            model.assert_consumed()
        with self.assertRaises(ReplayBoundaryError):
            model("prompt", purpose="OTHER")
        with self.assertRaises(ReplayBoundaryError):
            model("prompt", purpose="EXPECTED")

    def test_unready_fixture_is_not_run_not_passed(self):
        case = self.cases["SC-013-route-renewal"]
        with patch("tests.conversation_regression.replay.plan_v2_turn") as planner:
            self.assertEqual("NOT_RUN", replay_case(case, self.profile)["status"])
            planner.assert_not_called()

    def test_backend_checks_are_not_fabricated_by_offline_replay(self):
        case = self.cases["SC-010-duplicate-event"]
        result = replay_case(case, self.profile)
        self.assertEqual("OFFLINE_PARTIAL_PASS", result["status"])
        self.assertEqual(["duplicate"], result["not_run_steps"])
        self.assertNotIn("duplicate", [t["id"] for t in result["turns"]])
        self.assertFalse(case.agent_adapter_ready, "Live runner must not simulate backend replay")

    def test_replay_checks_actual_output_and_stops_on_failure(self):
        case = self.cases["SC-005-edit-items"].model_copy(deep=True)
        for step in case.steps:
            step.model_replies = []
        with patch("tests.conversation_regression.replay.plan_v2_turn",
                   side_effect=lambda req: response(req, '{"actual":"unexpected"}')) as planner:
            result = replay_case(case, self.profile)
        self.assertEqual("FAILED", result["status"])
        self.assertEqual(1, planner.call_count)
        self.assertEqual({"actual": "unexpected"}, result["turns"][0]["observed"]["state"])

    def test_replay_blocks_external_connections(self):
        def attempts_network(request):
            with socket.socket() as connection:
                connection.connect(("hotel.example", 443))
        case = self.cases["SC-018-direct-front-desk"]
        with patch("tests.conversation_regression.replay.plan_v2_turn", side_effect=attempts_network):
            result = replay_case(case, self.profile)
        self.assertEqual("HARNESS_ERROR", result["status"])
        self.assertIn("External effects", result["reason"])


if __name__ == "__main__":
    unittest.main()
