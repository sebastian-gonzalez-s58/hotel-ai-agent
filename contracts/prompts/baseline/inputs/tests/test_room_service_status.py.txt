import json
import unittest
from uuid import uuid4

from app.agents.room_service_status import existing_order_message
from app.agents.v2_scope_router import ScopeDecision
from app.agents.v2_turn_planner import plan_v2_turn
from app.schemas.v2_turns import OperationSnapshot
from tests.conversation_regression.library import Event, ModelReply, load_cases, profiles
from tests.conversation_regression.model_runtime import scripted_runtime
from tests.conversation_regression.session import ConversationSession


class RoomServiceStatusTest(unittest.TestCase):
    def request(self, text='Cambiar', reply='confirmation:ROOM_SERVICE:' + 'a' * 32 + ':CHANGE', locale='es-MX'):
        case = next(c for c in load_cases() if c.id == 'SC-009-delivered-buttons-es')
        session = ConversationSession(case, profiles()['hotel'])
        session.request.guest.preferredLanguage = locale
        request = session.receive(Event(kind='guest', text=text, reply_id=reply))
        return request

    def bind(self, request):
        request.trigger.eventPayload['roomServiceButtonContext'] = {'operationId': str(request.recentOperations[0].operationId)}
        return request.recentOperations[0]

    def scope(self, text, action='CHANGE', **changes):
        value = dict(kind='CONTEXT_REPLY', offeringCode='ROOM_SERVICE', relevantText=text,
                     hasRequestDetails=False, containsUnrelatedTopic=False, confidence=1,
                     existingOrderAction=action, existingOrderEvidence=text, existingOrderConfidence=1)
        value.update(changes)
        return ScopeDecision(**value)

    def run_turn(self, request, scope=None):
        scripts = [] if scope is None else [ModelReply(purpose='V2_HOTEL_SCOPE', payload=scope.model_dump())]
        before = request.model_dump()
        with scripted_runtime(scripts):
            response = plan_v2_turn(request)
        self.assertEqual(before, request.model_dump())
        self.assertEqual([], response.toolCalls)
        return response

    def test_bound_delivered_buttons_explain_action_in_both_languages(self):
        for locale, phrase in [('es-MX', 'ya fue entregado'), ('en', 'already been delivered')]:
            for action in ['CHANGE', 'CANCEL', 'CONFIRM']:
                with self.subTest(locale=locale, action=action):
                    request = self.request(reply='confirmation:ROOM_SERVICE:' + 'a' * 32 + ':' + action, locale=locale)
                    operation = self.bind(request)
                    response = self.run_turn(request)
                    self.assertIn(phrase, response.messages[0].text)
                    self.assertIn(operation.referenceCode, response.messages[0].text)
                    self.assertEqual([operation.operationId], response.messages[0].operationIds)
                    self.assertEqual('{}', response.updatedConversationSummary)

    def test_kitchen_review_is_not_preparation(self):
        request = self.request()
        operation = self.bind(request)
        operation.lifecycle, operation.detailedStatus = 'WAITING_FOR_STAFF', 'KITCHEN_REVIEW'
        text = self.run_turn(request).messages[0].text
        self.assertIn('revisión de cocina', text)
        self.assertNotIn('preparando', text)
        self.assertIn('no puedo modificarlo desde este chat', text)

    def test_accepted_order_is_pending_delivery_without_inventing_preparation(self):
        request = self.request()
        operation = self.bind(request)
        operation.lifecycle, operation.detailedStatus = 'WAITING_FOR_STAFF', 'AWAITING_DELIVERY'
        text = self.run_turn(request).messages[0].text
        self.assertIn('aceptado por cocina', text)
        self.assertIn('pendiente de entrega', text)
        self.assertNotIn('ya fue entregado', text)

    def test_cancelled_order_does_not_claim_a_new_cancellation(self):
        for status, reason in [('CANCELLED_BY_KITCHEN', 'por cocina'), ('CANCELLED_BY_GUEST', 'a tu solicitud'),
                               ('CANCELLED_GUEST_TIMEOUT', 'plazo de respuesta')]:
            request = self.request()
            operation = self.bind(request)
            operation.lifecycle, operation.detailedStatus = 'CANCELLED', status
            text = self.run_turn(request).messages[0].text
            self.assertIn('ya estaba cancelado', text)
            self.assertIn(reason, text)

    def test_unbound_button_never_chooses_a_recent_order(self):
        request = self.request()
        request.recentOperations.append(request.recentOperations[0].model_copy(update={'operationId': uuid4(), 'referenceCode': 'TEST-OTHER'}))
        response = self.run_turn(request)
        self.assertIn('No puedo identificar', response.messages[0].text)
        self.assertEqual([], response.messages[0].operationIds)
        self.assertNotIn('TEST-DELIVERED', response.messages[0].text)

    def test_foreign_binding_does_not_leak_or_select_another_order(self):
        request = self.request()
        request.trigger.eventPayload['roomServiceButtonContext'] = {'operationId': str(uuid4())}
        response = self.run_turn(request)
        self.assertEqual([], response.messages[0].operationIds)

    def test_bound_old_order_does_not_change_new_draft(self):
        request = self.request()
        self.bind(request)
        draft = {'pendingOffering': 'ROOM_SERVICE', 'phase': 'COLLECTING', 'capturedFields': {'items': [{'name': 'té', 'quantity': 2}]}}
        request.conversation.summary = json.dumps(draft)
        response = self.run_turn(request)
        self.assertIn('ya fue entregado', response.messages[0].text)
        self.assertEqual(draft, json.loads(response.updatedConversationSummary))

    def test_written_change_and_cancel_use_actual_delivered_state(self):
        for text, action in [('Quiero cambiar mi pedido', 'CHANGE'), ('Cancela mi pedido', 'CANCEL'),
                             ('Quita la sopa de mi pedido', 'CHANGE')]:
            request = self.request(text=text, reply=None)
            response = self.run_turn(request, self.scope(text, action))
            self.assertIn('ya fue entregado', response.messages[0].text)

    def test_explicit_reference_selects_old_order_among_multiple(self):
        text = 'Quiero cambiar TEST-DELIVERED'
        request = self.request(text=text, reply=None)
        request.activeOperations.append(request.recentOperations[0].model_copy(update={'operationId': uuid4(),
            'referenceCode': 'TEST-NEW', 'lifecycle': 'ACTIVE', 'detailedStatus': 'KITCHEN_REVIEW'}))
        response = self.run_turn(request, self.scope(text))
        self.assertIn('TEST-DELIVERED', response.messages[0].text)
        self.assertNotIn('TEST-NEW', response.messages[0].text)

    def test_unknown_or_ambiguous_reference_does_not_guess(self):
        for text, multiple in [('Cancela REQ-UNKNOWN', False), ('Cancela mi pedido', True)]:
            request = self.request(text=text, reply=None)
            if multiple:
                request.recentOperations.append(request.recentOperations[0].model_copy(update={'operationId': uuid4(), 'referenceCode': 'TEST-OTHER'}))
            response = self.run_turn(request, self.scope(text, 'CANCEL'))
            self.assertIn('folio', response.messages[0].text)
            self.assertEqual([], response.messages[0].operationIds)

    def test_new_order_negation_and_unreliable_evidence_do_not_trigger_rejection(self):
        request = self.request(text='No canceles mi pedido', reply=None)
        latest = request.conversation.recentMessages[-1]
        for changes in [dict(existingOrderAction='NONE'), dict(existingOrderEvidence='canceles'),
                        dict(existingOrderConfidence=.4), dict(separateRequest=True)]:
            self.assertIsNone(existing_order_message(request, latest, self.scope(latest.text, **changes), {}))

    def test_pending_draft_and_allowed_kitchen_task_remain_available(self):
        request = self.request(text='Quiero cambiar mi pedido', reply=None)
        latest = request.conversation.recentMessages[-1]
        scope = self.scope(latest.text)
        self.assertIsNone(existing_order_message(request, latest, scope, {'pendingOffering': 'ROOM_SERVICE'}))
        operation = request.recentOperations[0]
        operation.lifecycle = 'WAITING_FOR_GUEST'
        operation.availableActions = [{'actionCode': 'CHANGE'}]  # No mutation is authorized by this helper.
        self.assertIsNone(existing_order_message(request, latest, scope, {}))
