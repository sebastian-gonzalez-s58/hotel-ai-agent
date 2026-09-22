import json
import unittest
from uuid import uuid4
from app.agents.v2_turn_planner import plan_v2_turn
from app.agents.v2_scope_router import ScopeDecision
from app.schemas.v2_turns import DomainToolName, AvailableAction
from tests import test_spa_turns as spa_tests
from tests.conversation_regression.model_runtime import scripted_runtime
from tests.conversation_regression.library import ModelReply


class ReservationActionsTest(unittest.TestCase):
    def request(self, locale='es-MX'):
        request=spa_tests.request_for('Cancelar')
        request.guest.preferredLanguage=locale
        operation=spa_tests.spa_operation()
        operation.pendingConversationTasks=[]
        operation.lifecycle='ACTIVE';operation.detailedStatus='RESERVATION_CONFIRMED'
        operation.input=dict(spa_tests.VALUES)
        operation.availableActions=[AvailableAction(actionCode='RESERVATION_'+action,description=action,
            inputSchema={'type':'object','additionalProperties':False},requiresExplicitGuestConfirmation=True) for action in ['CHANGE','CANCEL']]
        request.activeOperations=[operation]
        request.toolPolicy.allowedTools.append(DomainToolName.EXECUTE_SERVICE_ACTION)
        request.trigger.eventPayload['reservationButtonContext']={'operationId':str(operation.operationId)}
        request.conversation.recentMessages[-1].interactionReplyId=f'reservation:{operation.operationId}:CANCEL'
        return request

    def run_turn(self,request,scope=None):
        before=request.model_dump()
        with scripted_runtime([] if scope is None else [ModelReply(purpose='V2_HOTEL_SCOPE',payload=scope.model_dump())]):response=plan_v2_turn(request)
        self.assertEqual(before,request.model_dump())
        return response

    def follow(self,request,prompt,action='CANCEL'):
        button=next(o.id for o in prompt.messages[0].interaction.options if o.id.endswith(':'+action))
        return spa_tests.follow_up(request,prompt,'Confirmar',button)

    def test_old_menu_only_prompts_and_fresh_confirmation_executes_current_version(self):
        for locale in ['es-MX','en']:
            request=self.request(locale);prompt=self.run_turn(request)
            self.assertEqual([],prompt.toolCalls)
            self.assertIn('17:00',prompt.messages[0].text)
            confirmed=self.run_turn(self.follow(request,prompt))
            self.assertEqual('EXECUTE_SERVICE_ACTION',confirmed.toolCalls[0].toolName.value)
            self.assertEqual(request.activeOperations[0].version,confirmed.toolCalls[0].arguments['expectedVersion'])

    def test_change_explains_staff_approval_and_preserves_current_reservation(self):
        request=self.request();request.conversation.recentMessages[-1].interactionReplyId=request.conversation.recentMessages[-1].interactionReplyId.replace('CANCEL','CHANGE')
        prompt=self.run_turn(request)
        self.assertIn('se conserva',prompt.messages[0].text)
        self.assertIn('aprobación',prompt.messages[0].text)
        self.assertEqual('RESERVATION_CHANGE',self.run_turn(self.follow(request,prompt,'CHANGE')).toolCalls[0].arguments['actionCode'])

    def test_version_changes_replaced_or_consumed_prompts_cannot_mutate(self):
        for kind in ['version','replaced','consumed']:
            request=self.request();prompt=self.run_turn(request);follow=self.follow(request,prompt)
            if kind=='version': follow.activeOperations[0].version+=1
            if kind=='replaced': follow.conversation.summary=self.run_turn(request).updatedConversationSummary
            if kind=='consumed': follow.conversation.summary='{}'
            self.assertEqual([],self.run_turn(follow).toolCalls)

    def test_cancelled_and_served_reservations_do_not_restart(self):
        for lifecycle,status,phrase in [('CANCELLED','CANCELLED_BY_GUEST','ya está cancelada'),('COMPLETED','SERVICE_COMPLETED','ya se realizó')]:
            request=self.request();request.activeOperations[0].lifecycle=lifecycle;request.activeOperations[0].detailedStatus=status
            response=self.run_turn(request)
            self.assertEqual([],response.toolCalls);self.assertIn(phrase,response.messages[0].text)

    def test_historical_draft_confirm_does_not_start_another_reservation(self):
        request=self.request();request.conversation.recentMessages[-1].interactionReplyId=f'spa-draft:{uuid4()}:CONFIRM:{uuid4().hex}'
        response=self.run_turn(request)
        self.assertEqual([],response.toolCalls)
        self.assertIn('está confirmada',response.messages[0].text)

    def test_unknown_or_mismatched_binding_never_selects_most_recent_reservation(self):
        for kind in ['missing','mismatched']:
            request=self.request()
            if kind=='missing':request.trigger.eventPayload.pop('reservationButtonContext')
            else:request.conversation.recentMessages[-1].interactionReplyId=f'reservation:{uuid4()}:CANCEL'
            response=self.run_turn(request)
            self.assertEqual([],response.toolCalls);self.assertIsNone(response.messages[0].interaction)

    def test_written_intent_requires_unambiguous_reservation_and_current_evidence(self):
        request=self.request();message=request.conversation.recentMessages[-1];message.interactionReplyId=None;message.text='Quiero cancelar mi reserva de spa'
        request.trigger.eventPayload={}
        scope=ScopeDecision(kind='STATUS_REQUEST',offeringCode='SPA',relevantText=message.text,hasRequestDetails=False,containsUnrelatedTopic=False,confidence=1,
            replyAction='CANCEL',replyActionEvidence=message.text,replyActionConfidence=1)
        response=self.run_turn(request,scope)
        self.assertEqual([],response.toolCalls);self.assertIsNotNone(response.messages[0].interaction)
        second=request.activeOperations[0].model_copy(deep=True);second.operationId=uuid4();second.referenceCode='TEST-SECOND';request.activeOperations.append(second)
        response=self.run_turn(request,scope)
        self.assertEqual(2,len(response.messages[0].interaction.options))
        self.assertTrue(all(o.id.startswith('reservation:') for o in response.messages[0].interaction.options))

    def test_decline_prevents_later_confirmation(self):
        request=self.request();prompt=self.run_turn(request)
        decline=self.run_turn(self.follow(request,prompt,'DECLINE'))
        follow=self.follow(request,prompt);follow.conversation.summary=decline.updatedConversationSummary
        self.assertEqual([],self.run_turn(follow).toolCalls)

    def test_explicit_existing_reference_is_not_consumed_by_another_pending_booking(self):
        request=self.request()
        request.conversation.summary=json.dumps({'spaDraft':{'id':str(uuid4()),'capturedFields':{}},'pendingOffering':'SPA'})
        target=request.activeOperations[0]
        target.referenceCode='REQ-SELECTED-BOOKING'
        other=spa_tests.spa_operation()
        other.referenceCode='REQ-OTHER-BOOKING'
        request.activeOperations.append(other)
        message=request.conversation.recentMessages[-1]
        message.interactionReplyId=None
        message.text=f'Quiero cancelar la reserva de spa {target.referenceCode}'
        scope=ScopeDecision(kind='STATUS_REQUEST',offeringCode='SPA',relevantText=message.text,
            hasRequestDetails=False,containsUnrelatedTopic=False,confidence=1,
            replyAction='CANCEL',replyActionEvidence=message.text,replyActionConfidence=1)
        response=self.run_turn(request,scope)
        self.assertTrue(response.messages[0].interaction.options[0].id.startswith(f'reservation-confirm:{target.operationId}:'))
        self.assertIn('spaDraft',json.loads(response.updatedConversationSummary))

    def test_cancelled_draft_buttons_explain_cancellation_without_restarting(self):
        from app.agents.spa_turns import summary_state
        request=spa_tests.request_for('SPA')
        request.conversation.recentMessages[-1].interactionReplyId='offering:SPA'
        draft=self.run_turn(request)
        draft_id=summary_state(draft.updatedConversationSummary)['spaDraft']['id']
        cancel=spa_tests.follow_up(request,draft,'Cancelar',f'spa-draft:{draft_id}:CANCEL')
        cancelled=self.run_turn(cancel)
        for action in ['CONFIRM','CHANGE','CANCEL']:
            old=spa_tests.follow_up(cancel,cancelled,action,f'spa-draft:{draft_id}:{action}')
            response=self.run_turn(old)
            self.assertEqual([],response.toolCalls)
            self.assertIn('ya fue cancelada',response.messages[0].text)
            self.assertIsNone(summary_state(response.updatedConversationSummary)['spaDraft'])

    def test_replaced_draft_buttons_keep_current_draft_untouched(self):
        from app.agents.spa_turns import summary_state
        request=spa_tests.request_for('SPA')
        request.conversation.recentMessages[-1].interactionReplyId='offering:SPA'
        first=self.run_turn(request)
        original=summary_state(first.updatedConversationSummary)['spaDraft']['id']
        fresh=spa_tests.follow_up(request,first,'SPA','offering:SPA')
        second=self.run_turn(fresh)
        current=summary_state(second.updatedConversationSummary)['spaDraft']
        self.assertNotEqual(original,current['id'])
        for action in ['CONFIRM','CHANGE','CANCEL']:
            old=spa_tests.follow_up(fresh,second,action,f'spa-draft:{original}:{action}')
            response=self.run_turn(old)
            self.assertEqual([],response.toolCalls)
            self.assertIn('solicitud de SPA anterior',response.messages[0].text)
            self.assertEqual(current,summary_state(response.updatedConversationSummary)['spaDraft'])


if __name__=='__main__':unittest.main()
