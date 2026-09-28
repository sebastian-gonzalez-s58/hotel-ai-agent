import json
import unittest
from uuid import uuid4
from app.agents.v2_turn_planner import plan_v2_turn
from app.agents.v2_scope_router import ScopeDecision
from tests.conversation_regression.library import Event, ModelReply, load_cases, profiles
from tests.conversation_regression.model_runtime import scripted_runtime
from tests.conversation_regression.session import ConversationSession


class MaintenanceRecurrenceTest(unittest.TestCase):
    def request(self, locale='es-MX'):
        case=next(c for c in load_cases() if c.id=='SC-009-delivered-buttons-es')
        session=ConversationSession(case, profiles()['hotel'])
        request=session.receive(Event(kind='guest',text='Not resolved',
                                      reply_id=f'maintenance-resolution:{uuid4()}:NOT_RESOLVED'))
        request.guest.preferredLanguage=locale
        source=request.recentOperations[0]
        source.offeringCode='MAINTENANCE'
        source.detailedStatus='RESOLUTION_CONFIRMED'
        source.input={'issue':'El aire acondicionado no enfría'}
        request.trigger.eventPayload['maintenanceButtonContext']={'source':source.model_dump(mode='json'), 'taskStatus':'COMPLETED'}
        return request

    def run_turn(self, request, scope=None):
        replies=[] if scope is None else [ModelReply(purpose='V2_HOTEL_SCOPE', payload=scope.model_dump())]
        before=request.model_dump()
        with scripted_runtime(replies):response=plan_v2_turn(request)
        self.assertEqual(before,request.model_dump())
        return response

    def next_request(self,request,response,action='CONFIRM',typed=None):
        request=request.model_copy(deep=True)
        request.conversation.summary=response.updatedConversationSummary
        message=request.conversation.recentMessages[-1]
        message.messageId=uuid4(); request.trigger.messageId=message.messageId
        message.text=typed or ('Sí, abrir folio' if action=='CONFIRM' else 'Ahora no')
        message.interactionReplyId=None if typed else next(o.id for o in response.messages[0].interaction.options if o.id.endswith(':'+action))
        return request

    def test_old_negative_button_requires_fresh_confirmation_in_both_languages(self):
        for locale,phrase in [('es-MX','nuevo folio'),('en','new request')]:
            request=self.request(locale);response=self.run_turn(request)
            self.assertEqual([],response.toolCalls)
            self.assertIn(phrase,response.messages[0].text)
            self.assertIn('aire acondicionado',response.messages[0].text)
            self.assertEqual(2,len(response.messages[0].interaction.options))

    def test_confirmation_retains_original_issue_and_source(self):
        request=self.request();prompt=self.run_turn(request)
        response=self.run_turn(self.next_request(request,prompt))
        call=response.toolCalls[0]
        self.assertEqual('START_SERVICE',call.toolName.value)
        self.assertEqual({'issue':'El aire acondicionado no enfría'},call.arguments['input'])
        self.assertEqual(str(request.recentOperations[0].operationId),call.arguments['recurrenceSourceOperationId'])
        self.assertNotIn('maintenanceRecurrence',json.loads(response.updatedConversationSummary))

    def test_typed_yes_requires_a_pending_recurrence(self):
        request=self.request();prompt=self.run_turn(request)
        response=self.run_turn(self.next_request(request,prompt,typed='sí'))
        self.assertEqual(1,len(response.toolCalls))

    def test_cancel_invalidates_previous_confirmation(self):
        request=self.request();prompt=self.run_turn(request)
        cancelled=self.run_turn(self.next_request(request,prompt,'CANCEL'))
        request=self.next_request(request,prompt)
        request.conversation.summary=cancelled.updatedConversationSummary
        response=self.run_turn(request)
        self.assertEqual([],response.toolCalls)
        self.assertIn('ya no corresponde',response.messages[0].text)

    def test_replaced_prompt_does_not_confirm_newer_prompt(self):
        request=self.request();first=self.run_turn(request);second=self.run_turn(request)
        request=self.next_request(request,first);request.conversation.summary=second.updatedConversationSummary
        self.assertEqual([],self.run_turn(request).toolCalls)

    def test_existing_recurrence_does_not_start_another(self):
        request=self.request()
        existing=request.recentOperations[0].model_dump(mode='json')
        existing.update(operationId=str(uuid4()),referenceCode='REQ-FOLLOWUP',lifecycle='WAITING_FOR_STAFF')
        request.trigger.eventPayload['maintenanceButtonContext']['recurrence']=existing
        response=self.run_turn(request)
        self.assertEqual([],response.toolCalls)
        self.assertIn('REQ-FOLLOWUP',response.messages[0].text)

    def test_timeout_never_claims_guest_confirmed_resolution(self):
        request=self.request()
        request.trigger.eventPayload['maintenanceButtonContext']['source']['detailedStatus']='CLOSED_NO_GUEST_CONFIRMATION'
        response=self.run_turn(request)
        self.assertIn('sin tu confirmación',response.messages[0].text)

    def test_unknown_task_and_malformed_button_cannot_become_new_issue(self):
        for malformed in [False,True]:
            request=self.request();request.trigger.eventPayload.pop('maintenanceButtonContext')
            if malformed:request.conversation.recentMessages[-1].interactionReplyId='maintenance-resolution:invalid:NOT_RESOLVED'
            response=self.run_turn(request)
            self.assertEqual([],response.toolCalls)
            self.assertIn('identificar',response.messages[0].text)

    def test_repeated_positive_confirmation_has_no_effect(self):
        request=self.request();message=request.conversation.recentMessages[-1]
        message.interactionReplyId=message.interactionReplyId.replace('NOT_RESOLVED','RESOLVED')
        response=self.run_turn(request)
        self.assertEqual([],response.toolCalls)
        self.assertIn('ya estaba registrada',response.messages[0].text)

    def test_an_old_broken_issue_description_is_not_reused(self):
        request=self.request();request.trigger.eventPayload['maintenanceButtonContext']['source']['input']={'issue':'Not resolved'}
        response=self.run_turn(request)
        self.assertIsNone(response.messages[0].interaction)
        self.assertEqual([],response.toolCalls)

    def test_other_service_draft_is_preserved(self):
        request=self.request();draft={'pendingOffering':'ROOM_SERVICE','capturedFields':{'items':[{'name':'sopa'}]}}
        request.conversation.summary=json.dumps(draft)
        response=self.run_turn(request)
        state=json.loads(response.updatedConversationSummary)
        self.assertEqual(draft['capturedFields'],state['capturedFields'])
        self.assertEqual('ROOM_SERVICE',state['pendingOffering'])

    def test_written_recurrence_is_a_confirmation_not_a_start(self):
        request=self.request();message=request.conversation.recentMessages[-1]
        message.interactionReplyId=None;message.text='El aire volvió a fallar'
        scope=ScopeDecision(kind='SERVICE_REQUEST',offeringCode='MAINTENANCE',relevantText=message.text,
                            hasRequestDetails=True,containsUnrelatedTopic=False,confidence=1,
                            maintenanceFollowUp='RECURRENCE',maintenanceFollowUpEvidence=message.text,
                            maintenanceFollowUpConfidence=1)
        response=self.run_turn(request,scope)
        self.assertEqual([],response.toolCalls)
        self.assertIsNotNone(response.messages[0].interaction)

    def test_recurrence_preserves_prose_and_independent_state(self):
        request=self.request()
        request.conversation.summary='Guest asked to keep messages short.\n{"independentNote":"keep"}'
        prompt=self.run_turn(request)
        self.assertTrue(prompt.updatedConversationSummary.startswith('Guest asked to keep messages short.\n'))
        self.assertEqual('keep',json.loads(prompt.updatedConversationSummary.splitlines()[-1])['independentNote'])

    def test_uncertain_prior_start_cannot_be_retried_through_old_button(self):
        request=self.request()
        request.conversation.summary=json.dumps({'serviceStartFailures':{'prior':{'offeringCode':'MAINTENANCE'}}})
        response=self.run_turn(request)
        self.assertEqual([],response.toolCalls)
        self.assertIsNone(response.messages[0].interaction)
        self.assertIn('sin confirmar',response.messages[0].text)
