"""BC-005/013: language preferences never replace pending business data."""
from copy import deepcopy
import json
import unittest

from app.agents.v2_turn_planner import plan_v2_turn
from tests.conversation_regression.library import Event, load_cases, profiles
from tests.conversation_regression.model_runtime import scripted_runtime
from tests.conversation_regression.session import ConversationSession, structured_summary


class PendingLanguageSwitchTest(unittest.TestCase):
    def setUp(self):
        self.cases = {case.id: case for case in load_cases()}
        self.case = self.cases["SC-007-language-partial"]
        self.profile = profiles()["hotel"]

    def test_language_only_change_keeps_other_service_tasks_and_original_order(self):
        case = self.case.model_copy(deep=True)
        case.initial_operations = deepcopy(self.cases["SC-015-ambiguous-tasks"].initial_operations)
        session = ConversationSession(case, self.profile)
        request = session.receive(case.steps[0].event)
        before = request.model_dump(mode="json")
        with scripted_runtime(case.steps[0].model_replies):
            response = plan_v2_turn(request)
        self.assertEqual(before, request.model_dump(mode="json"), "Input snapshot must not be mutated")
        self.assertEqual(case.initial_summary, structured_summary(response.updatedConversationSummary))
        self.assertEqual([], response.toolCalls, "Language change must not complete any service task")
        self.assertEqual("EXPLICIT", response.languageDecision.source)
        self.assertEqual("en", response.messages[0].language)

    def test_spanish_quantity_keeps_explicit_english_and_does_not_translate_product(self):
        session = ConversationSession(self.case, self.profile)
        request = session.receive(self.case.steps[0].event)
        with scripted_runtime(self.case.steps[0].model_replies):
            session.accept(plan_v2_turn(request))
        request = session.receive(Event(kind="guest", text="Dos"))
        self.assertTrue(request.trigger.eventPayload["languageContext"]["explicit"])
        replies = [r.model_copy(deep=True) for r in self.case.steps[1].model_replies]
        replies[0].payload.update(relevantText="Dos", detectedLanguage="es", languageConfidence=1)
        replies[1].payload["edits"][0].update(quantityEvidence="Dos", evidence="Dos")
        with scripted_runtime(replies):
            response = plan_v2_turn(request)
        observed = session.accept(response)
        self.assertIsNone(response.languageDecision)
        self.assertEqual("en", observed["guest"]["preferredLanguage"])
        self.assertEqual("en", response.messages[0].language)
        self.assertEqual({"deliveryLocation": "ROOM", "items": [{"name": "sopa", "quantity": 2, "modifications": []}]},
                         observed["state"]["capturedFields"])
        self.assertTrue(observed["state"]["awaitingExplicitConfirmation"])
        self.assertEqual([], response.toolCalls)

    def test_second_explicit_preference_changes_language_without_erasing_restrictions(self):
        session = ConversationSession(self.case, self.profile)
        initial = deepcopy(self.case.initial_summary)
        initial["capturedFields"]["items"][0]["modifications"] = ["sin crema"]
        session.request.conversation.summary = json.dumps(initial)
        request = session.receive(self.case.steps[0].event)
        with scripted_runtime(self.case.steps[0].model_replies):
            session.accept(plan_v2_turn(request))
        request = session.receive(Event(kind="guest", text="Mejor en español"))
        replies = [r.model_copy(deep=True) for r in self.case.steps[0].model_replies]
        replies[0].payload.update(relevantText="Mejor en español", requestedLanguage="es-MX")
        with scripted_runtime(replies):
            response = plan_v2_turn(request)
        observed = session.accept(response)
        self.assertEqual("EXPLICIT", response.languageDecision.source)
        self.assertEqual("es-MX", observed["guest"]["preferredLanguage"])
        self.assertEqual(initial, observed["state"])
        self.assertEqual([], response.toolCalls)


if __name__ == "__main__":
    unittest.main()
