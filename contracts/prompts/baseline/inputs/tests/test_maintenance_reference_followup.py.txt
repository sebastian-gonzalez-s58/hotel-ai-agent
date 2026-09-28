import json
import unittest
from uuid import uuid4

from app.agents.v2_turn_planner import plan_v2_turn
from tests import test_maintenance_new_request as new_tests
from tests.conversation_regression.library import ModelReply
from tests.conversation_regression.model_runtime import scripted_runtime


def follow(request, response, text):
    request = request.model_copy(deep=True)
    request.conversation.summary = response.updatedConversationSummary
    previous = request.conversation.recentMessages[-1]
    for message in response.messages:
        outbound = previous.model_copy(deep=True)
        outbound.messageId = uuid4()
        outbound.direction = 'OUTBOUND'
        outbound.actor = 'AGENT'
        outbound.text = message.text
        outbound.interactionReplyId = None
        request.conversation.recentMessages.append(outbound)
    inbound = previous.model_copy(deep=True)
    inbound.messageId = uuid4()
    inbound.text = text
    inbound.interactionReplyId = None
    request.conversation.recentMessages.append(inbound)
    request.trigger.messageId = inbound.messageId
    return request


class MaintenanceReferenceFollowupTest(unittest.TestCase):
    def run_turn(self, request, scope=None):
        original = request.model_dump()
        replies = [] if scope is None else [ModelReply(purpose='V2_HOTEL_SCOPE', payload=scope.model_dump())]
        with scripted_runtime(replies):
            response = plan_v2_turn(request)
        self.assertEqual(original, request.model_dump())
        return response

    def start(self, locale='es-MX'):
        request = new_tests.new_request('El problema sigue persistiendo' if locale.startswith('es')
                                        else 'The problem is still happening', locale, False)
        request.recentOperations[0].input = {'issue': 'The air conditioner does not cool'}
        request.recentOperations[1].input = {'issue': 'The shower leaks'}
        response = self.run_turn(request, new_tests.MaintenanceNewRequestTest().scope(request, 'RECURRENCE'))
        self.assertEqual('AWAITING_REFERENCE', json.loads(response.updatedConversationSummary)['maintenanceRecurrence']['phase'])
        return request, response

    def test_reference_reply_prompts_confirmation_then_starts_original_issue(self):
        for locale in ['es-MX', 'en']:
            with self.subTest(locale=locale):
                request, question = self.start(locale)
                source = request.recentOperations[0]
                selected = follow(request, question, source.referenceCode.lower())
                confirmation = self.run_turn(selected)
                self.assertEqual([], confirmation.toolCalls)
                self.assertIn(source.input['issue'], confirmation.messages[0].text)
                self.assertIsNotNone(confirmation.messages[0].interaction)
                result = self.run_turn(follow(selected, confirmation, 'yes'))
                self.assertEqual(1, len(result.toolCalls))
                self.assertEqual(source.input, result.toolCalls[0].arguments['input'])
                self.assertEqual(str(source.operationId), result.toolCalls[0].arguments['recurrenceSourceOperationId'])

    def test_unknown_reference_can_be_corrected_without_restarting_issue_capture(self):
        request, question = self.start()
        for text in ['REQ-UNKNOWN', 'TEST-OLD-20', 'TEST-OLD-2 TEST-OP-1', 'yes']:
            with self.subTest(text=text):
                unknown = follow(request, question, text)
                retry = self.run_turn(unknown)
                self.assertEqual([], retry.toolCalls)
                self.assertIsNone(retry.messages[0].interaction)
                corrected = self.run_turn(follow(unknown, retry, 'test-old-2'))
                self.assertIn('The shower leaks', corrected.messages[0].text)
                self.assertIsNotNone(corrected.messages[0].interaction)

    def test_cancel_before_selection_clears_pending_state(self):
        request, question = self.start()
        response = self.run_turn(follow(request, question, 'cancelar'))
        self.assertEqual([], response.toolCalls)
        self.assertNotIn('maintenanceRecurrence', json.loads(response.updatedConversationSummary))

    def test_legacy_question_without_summary_recovers_existing_conversation(self):
        for locale in ['es-MX', 'en']:
            request, question = self.start(locale)
            request = follow(request, question, 'test-old-2')
            request.conversation.summary = '{}'
            response = self.run_turn(request)
            self.assertIn('The shower leaks', response.messages[0].text)
            self.assertIsNotNone(response.messages[0].interaction)

    def test_reference_uses_current_authorized_snapshot_and_candidate_set(self):
        for change in ['missing', 'wrong-service', 'new-candidate', 'active']:
            with self.subTest(change=change):
                request, question = self.start()
                request = follow(request, question, 'test-old-2')
                operation = request.recentOperations[1]
                if change == 'missing': request.recentOperations.pop()
                if change == 'wrong-service': operation.offeringCode = 'SPA'
                if change == 'new-candidate': operation.operationId = uuid4()
                if change == 'active': operation.lifecycle = 'WAITING_FOR_STAFF'
                response = self.run_turn(request)
                self.assertEqual([], response.toolCalls)
                self.assertIsNone(response.messages[0].interaction)

    def test_new_fault_is_still_independent_while_awaiting_reference(self):
        request, question = self.start()
        request = follow(request, question, 'La puerta no abre')
        response = self.run_turn(request, new_tests.MaintenanceNewRequestTest().scope(request))
        self.assertEqual('START_SERVICE', response.toolCalls[0].toolName.value)
        self.assertNotIn('recurrenceSourceOperationId', response.toolCalls[0].arguments)

    def test_reference_selection_does_not_intercept_after_unrelated_outbound(self):
        from app.agents.maintenance_recurrence import plan_maintenance_recurrence
        request, question = self.start()
        request = follow(request, question, 'test-old-2')
        request.conversation.recentMessages[-2].text = 'Please choose a spa treatment.'
        self.assertIsNone(plan_maintenance_recurrence(request, json.loads(request.conversation.summary),
                                                     request.conversation.recentMessages[-1]))


if __name__ == '__main__':
    unittest.main()
