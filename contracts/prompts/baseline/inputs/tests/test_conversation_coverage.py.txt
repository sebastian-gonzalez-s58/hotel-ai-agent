import json
import unittest
from unittest.mock import patch

from app.agents.v2_turn_planner import plan_v2_turn, _validate_plan
from app.core.errors import AgentModelError
from tests.test_multilingual_understanding import room_request, scope_for, result
from tests.conversation_regression.library import load_cases, profiles
from tests.conversation_regression.model_runtime import scripted_runtime
from tests.conversation_regression.replay import replay_case
from tests.conversation_regression.session import ConversationSession
from app.services.input_understanding import understanding_turn, record_scope_action, semantic_action_offering


class ExpandedConversationCoverageTest(unittest.TestCase):
    def room_confirmation(self, queued):
        request = room_request('Confirm', [{'name': 'sopa', 'quantity': 2, 'modifications': []}], awaiting=True)
        request.trigger.eventPayload['confirmationQueuedBeforePrompt'] = queued
        message = request.conversation.recentMessages[-1]
        with patch('app.agents.v2_turn_planner.classify_hotel_scope', return_value=(scope_for(message, action='CONFIRM', hasRequestDetails=False), result({}).usage)), \
                patch('app.agents.v2_turn_planner.localize_response', side_effect=lambda req, res, started: res):
            return request, plan_v2_turn(request)

    def test_queued_confirmation_reissues_summary_without_starting_order(self):
        _, response = self.room_confirmation(True)
        self.assertFalse(response.toolCalls)
        self.assertEqual('CONFIRMATION', response.messages[0].purpose)
        self.assertEqual(2, json.loads(response.updatedConversationSummary)['capturedFields']['items'][0]['quantity'])

    def test_fresh_confirmation_still_starts_order(self):
        _, response = self.room_confirmation(False)
        self.assertEqual(['START_SERVICE'], [call.toolName.value for call in response.toolCalls])

    def test_validator_blocks_model_proposal_for_queued_confirmation(self):
        request, response = self.room_confirmation(False)
        request.trigger.eventPayload['confirmationQueuedBeforePrompt'] = True
        with self.assertRaisesRegex(AgentModelError, 'predates'):
            _validate_plan(request, response)

    def test_all_current_cases_have_explicit_spring_adapters_and_scripts(self):
        for case in load_cases():
            self.assertTrue(case.integration_ready, case.id)
            self.assertTrue(all(step.model_replies is not None for step in case.steps), case.id)

    def test_backend_only_case_does_not_claim_offline_pass(self):
        case = next(c for c in load_cases() if c.id == 'SC-021-queued-messages')
        report = replay_case(case, profiles()[case.profile])
        self.assertEqual('NOT_RUN', report['status'])
        self.assertEqual([], report['turns'])

    def test_explicit_maintenance_reply_selects_only_maintenance_among_two_tasks(self):
        case = next(c for c in load_cases() if c.id == 'SC-015-ambiguous-tasks')
        session = ConversationSession(case, profiles()[case.profile])
        for index, step in enumerate(case.steps):
            request = session.receive(step.event)
            with scripted_runtime(step.model_replies):
                response = plan_v2_turn(request)
            session.accept(response)
            if index == 0:
                self.assertFalse(response.toolCalls)
            else:
                self.assertEqual('COMPLETE_CONVERSATION_TASK', response.toolCalls[0].toolName.value)
                self.assertEqual(case.initial_operations[0]['operationId'], str(response.toolCalls[0].targetOperationId))
                self.assertFalse(response.toolCalls[0].arguments['result']['resolved'])

    def test_offline_task_checks_remain_explicitly_pending(self):
        case = next(c for c in load_cases() if c.id == 'SC-015-ambiguous-tasks')
        report = replay_case(case, profiles()[case.profile])
        self.assertEqual('OFFLINE_PARTIAL_PASS', report['status'])
        self.assertTrue(report['turns'][1]['checks_pending'])

    def test_current_service_evidence_overrides_old_focus_but_not_explicit_reply_target(self):
        from app.agents.v2_turn_planner import _select_understood_task
        case = next(c for c in load_cases() if c.id == 'SC-015-ambiguous-tasks')
        request = ConversationSession(case, profiles()[case.profile]).receive(case.steps[1].event)
        maintenance = request.activeOperations[0].pendingConversationTasks[0]
        spa = request.activeOperations[1].pendingConversationTasks[0]
        request.conversation.focusedConversationTaskId = spa.conversationTaskId
        with scripted_runtime(case.steps[1].model_replies):
            response = plan_v2_turn(request)
        self.assertEqual(maintenance.conversationTaskId, response.toolCalls[0].targetConversationTaskId)
        message = request.conversation.recentMessages[-1]
        message.conversationTaskIds = [spa.conversationTaskId]
        with understanding_turn(request):
            record_scope_action(message, scope_for(message, action='NOT_RESOLVED', offeringCode='MAINTENANCE'))
            self.assertIsNone(_select_understood_task(request, message, [maintenance]))
            self.assertEqual(spa, _select_understood_task(request, message, [spa]))

    def test_task_service_hint_requires_exact_current_evidence_and_confidence(self):
        request = room_request('The air conditioner is still broken')
        message = request.conversation.recentMessages[-1]
        for changes in ({'replyActionEvidence': 'broken'}, {'confidence': 0.5}, {'containsUnrelatedTopic': True}):
            with self.subTest(changes=changes), understanding_turn(request):
                scope = scope_for(message, action='NOT_RESOLVED', offeringCode='MAINTENANCE', **changes)
                record_scope_action(message, scope)
                self.assertIsNone(semantic_action_offering(message.text))

    def test_service_hint_does_not_leak_into_a_later_turn(self):
        request = room_request('It is still broken')
        message = request.conversation.recentMessages[-1]
        with understanding_turn(request):
            record_scope_action(message, scope_for(message, action='NOT_RESOLVED', offeringCode='MAINTENANCE'))
            self.assertEqual('MAINTENANCE', semantic_action_offering(message.text))
        with understanding_turn(request):
            self.assertIsNone(semantic_action_offering(message.text))


if __name__ == '__main__':
    unittest.main()
