from datetime import datetime, timezone
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.agents.maintenance_questions import plan_question, TASK_TYPE
from app.agents.v2_turn_planner import plan_v2_turn, _validate_plan
from app.agents.v2_scope_router import ScopeDecision
from app.core.errors import AgentModelError
from app.schemas.v2_turns import AgentTurnRequest
from app.services.telemetry_client import OpenAiTokenUsage
from tests.test_v2_turn_planner import maintenance_resolution_payload


def request_for_question(text='Después de las 5, por favor', question='¿A qué hora podemos pasar?', locale='es-MX'):
    data, op_id, task_id = maintenance_resolution_payload()
    data['guest']['preferredLanguage'] = locale
    data['conversation']['recentMessages'][0]['text'] = text
    data['conversation']['recentMessages'][0]['createdAt'] = '2026-09-28T20:05:00Z'
    task = data['activeOperations'][0]['pendingConversationTasks'][0]
    task.update(taskType=TASK_TYPE, context={'question': question}, createdAt='2026-09-28T20:00:00Z', expiresAt=None,
                requiredOutputSchema={'type': 'object', 'required': ['maintenanceGuestAnswer'], 'additionalProperties': False,
                                      'properties': {'maintenanceGuestAnswer': {'type': 'string'}}})
    data['activeOperations'][0]['referenceCode'] = 'REQ-20260928-TEST0001'
    return AgentTurnRequest.model_validate(data)


def scope(request, **changes):
    return ScopeDecision(**dict(kind='CONTEXT_REPLY', offeringCode='MAINTENANCE', relevantText=request.conversation.recentMessages[0].text,
                                hasRequestDetails=True, containsUnrelatedTopic=False, confidence=1) | changes)


class MaintenanceQuestionsTest(unittest.TestCase):
    def run_plan(self, request, decision=None):
        with patch('app.agents.v2_turn_planner.classify_hotel_scope', return_value=(decision or scope(request), OpenAiTokenUsage())), \
                patch('app.agents.v2_turn_planner.call_openai_json_result') as model:
            result = plan_v2_turn(request)
            model.assert_not_called()
            return result

    def test_verbatim_schedule_reply_completes_only_the_question(self):
        request = request_for_question('Después de las 5; no entren antes.\nGracias.')
        result = self.run_plan(request)
        self.assertEqual(1, len(result.toolCalls))
        call = result.toolCalls[0]
        self.assertEqual('COMPLETE_CONVERSATION_TASK', call.toolName.value)
        self.assertEqual({'maintenanceGuestAnswer': request.conversation.recentMessages[0].text}, call.arguments['result'])
        self.assertEqual([request.conversation.recentMessages[0].messageId], call.evidenceMessageIds)
        self.assertEqual(request.activeOperations[0].operationId, call.targetOperationId)

    def test_fault_details_and_fixed_statement_are_not_new_folios_or_resolution(self):
        for text in ['Solo falla el agua caliente', 'The AC turns on but does not cool', 'Ya funciona, gracias', 'No puedo recibirlos hoy']:
            with self.subTest(text=text):
                result = self.run_plan(request_for_question(text))
                self.assertEqual({'maintenanceGuestAnswer': text}, result.toolCalls[0].arguments['result'])

    def test_original_words_survive_scope_relevant_text_filter(self):
        request = request_for_question('Pueden pasar a las 5, pero antes avísenme.')
        result = self.run_plan(request, scope(request, relevantText='a las 5'))
        self.assertEqual(request.conversation.recentMessages[0].text, result.toolCalls[0].arguments['result']['maintenanceGuestAnswer'])

    def test_other_requests_questions_language_and_buttons_are_not_answers(self):
        request = request_for_question()
        latest = request.conversation.recentMessages[0]
        for changes in [dict(kind='SERVICE_REQUEST', offeringCode='ROOM_SERVICE'), dict(kind='HOTEL_QUESTION'),
                        dict(kind='SOCIAL'), dict(languageChangeOnly=True), dict(separateRequest=True),
                        dict(containsUnrelatedTopic=True), dict(confidence=.5)]:
            with self.subTest(changes=changes):
                self.assertIsNone(plan_question(request, latest, scope(request, **changes)))
        latest.interactionReplyId = 'maintenance-resolution:old:RESOLVED'
        self.assertIsNone(plan_question(request, latest, scope(request)))

    def test_multiple_questions_require_reference_even_with_historical_focus(self):
        request = request_for_question()
        second = request.activeOperations[0].model_copy(deep=True)
        second.operationId = uuid4(); second.referenceCode = 'REQ-20260928-TEST0002'
        second.pendingConversationTasks[0].operationId = second.operationId
        second.pendingConversationTasks[0].conversationTaskId = uuid4()
        request.activeOperations.append(second)
        result = self.run_plan(request)
        self.assertEqual([], result.toolCalls)
        self.assertIn('TEST0001', result.messages[0].text)
        self.assertIn('TEST0002', result.messages[0].text)
        request.conversation.recentMessages[0].text = 'REQ-20260928-TEST0002: después de las 5'
        result = self.run_plan(request)
        self.assertEqual(second.operationId, result.toolCalls[0].targetOperationId)

    def test_unrelated_service_task_does_not_block_the_only_maintenance_answer(self):
        request = request_for_question('Yes please')
        unrelated = request.activeOperations[0].model_copy(deep=True)
        unrelated.operationId = uuid4()
        unrelated.referenceCode = 'REQ-20260928-SPA00001'
        unrelated.offeringCode = 'SPA'
        unrelated.pendingConversationTasks[0].operationId = unrelated.operationId
        unrelated.pendingConversationTasks[0].conversationTaskId = uuid4()
        unrelated.pendingConversationTasks[0].taskType = 'SPA_ALTERNATIVE_DECISION'
        request.activeOperations.append(unrelated)

        result = self.run_plan(request)

        self.assertEqual(1, len(result.toolCalls))
        self.assertEqual(request.activeOperations[0].operationId, result.toolCalls[0].targetOperationId)
        self.assertEqual({'maintenanceGuestAnswer': 'Yes please'}, result.toolCalls[0].arguments['result'])

    def test_stale_unrelated_task_correlation_does_not_block_the_only_question(self):
        request = request_for_question('Yes please')
        unrelated = request.activeOperations[0].model_copy(deep=True)
        unrelated.operationId = uuid4()
        unrelated.offeringCode = 'SPA'
        unrelated.pendingConversationTasks[0].operationId = unrelated.operationId
        unrelated.pendingConversationTasks[0].conversationTaskId = uuid4()
        unrelated.pendingConversationTasks[0].taskType = 'SPA_ALTERNATIVE_DECISION'
        request.activeOperations.append(unrelated)
        request.conversation.recentMessages[0].conversationTaskIds = [
            unrelated.pendingConversationTasks[0].conversationTaskId
        ]

        result = self.run_plan(request)

        self.assertEqual(request.activeOperations[0].operationId, result.toolCalls[0].targetOperationId)

    def test_unknown_or_retired_reference_never_answers_current_question(self):
        request = request_for_question('REQ-20260928-UNKNOWN: a las 5')
        self.assertEqual([], self.run_plan(request).toolCalls)
        request = request_for_question()
        request.trigger.conversationTaskId = uuid4()
        self.assertEqual([], self.run_plan(request).toolCalls)

    def test_reply_queued_before_question_cannot_answer_it(self):
        request = request_for_question()
        request.conversation.recentMessages[0].createdAt = datetime(2026, 9, 28, 19, 59, tzinfo=timezone.utc)
        self.assertEqual([], self.run_plan(request).toolCalls)

    def test_validation_rejects_rewritten_or_resolved_answers(self):
        request = request_for_question()
        result = self.run_plan(request)
        for value in [{'maintenanceGuestAnswer': 'Appointment confirmed at 5'}, {'resolved': True}]:
            tampered = result.model_copy(deep=True)
            tampered.toolCalls[0].arguments['result'] = value
            with self.assertRaises(AgentModelError): _validate_plan(request, tampered)

    def test_success_acknowledges_sharing_without_confirming_appointment(self):
        request = request_for_question(locale='en')
        data = request.model_dump(mode='json')
        task = request.activeOperations[0].pendingConversationTasks[0]
        data['previousToolResults'] = [{'toolCallId': str(uuid4()), 'toolName': 'COMPLETE_CONVERSATION_TASK', 'status': 'SUCCEEDED',
                                      'result': {'conversationTaskId': str(task.conversationTaskId), 'operationId': str(task.operationId),
                                                 'taskType': TASK_TYPE, 'status': 'COMPLETED', 'completionResult': {'maintenanceGuestAnswer': 'After 5'}}}]
        result = self.run_plan(AgentTurnRequest.model_validate(data))
        self.assertEqual([], result.toolCalls)
        self.assertIn('shared your reply', result.messages[0].text)
        self.assertNotIn('confirmed', result.messages[0].text)


if __name__ == '__main__': unittest.main()
