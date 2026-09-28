import json
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.agents.v2_scope_router import ScopeDecision, classify_hotel_scope
from app.agents.v2_turn_planner import plan_v2_turn
from app.services.openai_client import OpenAiJsonResult
from app.services.telemetry_client import OpenAiTokenUsage
from tests import test_maintenance_recurrence as recurrence_tests
from tests.conversation_regression.library import ModelReply
from tests.conversation_regression.model_runtime import scripted_runtime


def new_request(text='El ac no funciona', locale='es-MX', pending=True):
    request=recurrence_tests.MaintenanceRecurrenceTest().request(locale)
    request.trigger.eventPayload.pop('maintenanceButtonContext',None)
    source=request.recentOperations[0]
    source.input={'issue':text}
    second=source.model_copy(deep=True)
    second.operationId=uuid4();second.referenceCode='TEST-OLD-2'
    request.recentOperations.append(second)
    message=request.conversation.recentMessages[-1]
    message.text=text;message.interactionReplyId=None
    if pending:
        request.conversation.summary=json.dumps({'pendingOffering':'MAINTENANCE','capturedFields':{},'readyToStart':False})
        prompt=message.model_copy(deep=True)
        prompt.messageId=uuid4();prompt.direction='OUTBOUND';prompt.actor='AGENT'
        prompt.text='Por favor, describe el problema de mantenimiento.' if locale.startswith('es') else 'Please describe the maintenance problem.'
        request.conversation.recentMessages.insert(0,prompt)
    else:
        request.conversation.summary='{}'
    return request


class MaintenanceNewRequestTest(unittest.TestCase):
    def scope(self,request,followup='NONE'):
        text=request.conversation.recentMessages[-1].text
        return ScopeDecision(kind='CONTEXT_REPLY' if 'pendingOffering' in request.conversation.summary else 'SERVICE_REQUEST',
            offeringCode='MAINTENANCE',relevantText=text,hasRequestDetails=True,containsUnrelatedTopic=False,confidence=1,
            maintenanceFollowUp=followup,maintenanceFollowUpEvidence=text if followup=='RECURRENCE' else None,
            maintenanceFollowUpConfidence=1 if followup=='RECURRENCE' else 0)

    def test_new_fault_with_identical_history_starts_independent_request(self):
        for locale,text in [('es-MX','El ac no funciona'),('en','The AC is not working')]:
            for pending in [False,True]:
                with self.subTest(locale=locale,pending=pending):
                    request=new_request(text,locale,pending)
                    original=request.model_dump()
                    with scripted_runtime([ModelReply(purpose='V2_HOTEL_SCOPE',payload=self.scope(request).model_dump())]):
                        response=plan_v2_turn(request)
                    self.assertEqual(1,len(response.toolCalls))
                    call=response.toolCalls[0]
                    self.assertEqual('START_SERVICE',call.toolName.value)
                    self.assertEqual({'issue':text},call.arguments['input'])
                    self.assertNotIn('recurrenceSourceOperationId',call.arguments)
                    self.assertIsNone(call.targetOperationId)
                    self.assertEqual(original,request.model_dump())

    def test_closed_fault_history_cannot_change_scope_input(self):
        request=new_request()
        scope=self.scope(request)
        result=OpenAiJsonResult(payload=scope.model_dump(),usage=OpenAiTokenUsage(),response_id='synthetic')
        with patch('app.agents.v2_scope_router.call_openai_json_result',return_value=result) as model:
            classify_hotel_scope(request,request.conversation.recentMessages[-1],json.loads(request.conversation.summary))
            with_history=model.call_args.args[0]
            request.recentOperations=[]
            classify_hotel_scope(request,request.conversation.recentMessages[-1],json.loads(request.conversation.summary))
            self.assertEqual(with_history,model.call_args.args[0])

    def test_explicit_recurrence_still_asks_which_previous_request(self):
        for locale,text in [('es-MX','El ac volvió a fallar'),('en','The AC stopped working again')]:
            request=new_request(text,locale)
            with scripted_runtime([ModelReply(purpose='V2_HOTEL_SCOPE',payload=self.scope(request,'RECURRENCE').model_dump())]):
                response=plan_v2_turn(request)
            self.assertEqual([],response.toolCalls)
            self.assertIn('folio' if locale.startswith('es') else 'Which maintenance request',response.messages[0].text)

    def test_explicit_recurrence_with_one_source_still_requires_confirmation(self):
        request=new_request('El problema anterior no quedó resuelto')
        request.recentOperations=request.recentOperations[:1]
        with scripted_runtime([ModelReply(purpose='V2_HOTEL_SCOPE',payload=self.scope(request,'RECURRENCE').model_dump())]):
            response=plan_v2_turn(request)
        self.assertEqual([],response.toolCalls)
        self.assertIsNotNone(response.messages[0].interaction)

    def test_hotel_question_cannot_be_rerouted_by_conflicting_recurrence_flag(self):
        for locale,text in [('es-MX','¿Qué hago si el aire vuelve a fallar?'),('en','What should I do if the AC fails again?')]:
            request=new_request(text,locale,False)
            scope=self.scope(request,'RECURRENCE')
            scope.kind='HOTEL_QUESTION';scope.offeringCode='FAQ';scope.hasRequestDetails=False
            with scripted_runtime([ModelReply(purpose='V2_HOTEL_SCOPE',payload=scope.model_dump())]):
                response=plan_v2_turn(request)
            self.assertEqual(['SEARCH_KNOWLEDGE'],[call.toolName.value for call in response.toolCalls])
            self.assertNotIn('maintenanceRecurrence',response.updatedConversationSummary)


if __name__=='__main__':unittest.main()
