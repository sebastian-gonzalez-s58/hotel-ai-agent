import json
import unittest
from copy import deepcopy
from unittest.mock import patch
from uuid import uuid4
from app.agents import v2_turn_planner as planner
from app.agents.kitchen_changes import validate_scope
from app.core.errors import AgentModelError
from app.services.catalog_orders import normalize_order
from app.services.telemetry_client import OpenAiTokenUsage
from tests.test_catalog_orders import offering, item, catalog
from tests.test_multilingual_understanding import room_request, task_operation, scope_for
from tests.test_spa_turns import follow_up


def fixture(text='pollo', locale='es-MX', multiple=False):
    sauce = {'id': 'sauce', 'name': 'Salsa', 'minimumSelections': 1, 'maximumSelections': 1, 'required': True,
             'options': [{'id': 'roja', 'name': 'Roja', 'available': True, 'priceAdjustment': 0}, {'id': 'verde', 'name': 'Verde', 'available': True, 'priceAdjustment': 0}]}
    extra = {'id': 'extra', 'name': 'Complemento', 'minimumSelections': 0, 'maximumSelections': 3 if multiple else 1, 'required': False, 'offerToGuest': True,
             'options': [{'id': i, 'name': n, 'priceAdjustment': p, 'available': True} for i,n,p in [('chorizo','Chorizo',60),('pollo','Pechuga de Pollo',50),('arrachera','Arrachera',140),('huevo','Huevo',30)]]}
    off = offering([item('chilaquiles', 'Chilaquiles Clásicos', price=200, groups=[sauce, extra])])
    rows = [{'name':'Chilaquiles Clásicos','quantity':1,'modifications':['sin crema','alergia a nueces'],
             'catalogSelection': {'itemId':'chilaquiles','requestedName':'Chilaquiles Clásicos','name':'Chilaquiles Clásicos',
             'optionIds':['roja','chorizo'] + (['huevo'] if multiple else []), 'sourceModifications':['sin crema','alergia a nueces']}}]
    rows, pending = normalize_order(off, rows, semantic=False)
    assert pending is None
    other = deepcopy(rows[0]);other['catalogSelection']['optionIds']=['verde','arrachera']; other['quantity']=2
    other = normalize_order(off, [other], semantic=False)[0][0]
    rows.append(other)
    choices = [{'id': o['id'], 'label':o['name'],'priceAdjustment':o['priceAdjustment'],'currency':'MXN'} for o in extra['options'] if o['id'] not in rows[0]['catalogSelection']['optionIds']]
    choices.append({'id':'__skip__','label':'Sin este complemento','action':'SKIP'})
    group = {'kind':'OPTION','itemIndex':0,'itemId':'chilaquiles','requestedName':'Chilaquiles Clásicos','groupId':'extra','groupName':'Complemento',
             'minimumSelections':0,'maximumSelections':2 if multiple else 1,'optionalOffer':True,'page':0,'token':str(uuid4()),
             'blockedOptionIds':['chorizo'],'groupOptionIds':[o['id'] for o in extra['options'] if not multiple or o['id']!='huevo'],'choices':choices}
    request=room_request(text);request.guest.preferredLanguage=locale;request.conversation.summary='{}';request.availableOfferings=[off]
    request.activeOperations=[task_operation('ROOM_SERVICE_ORDER_CHANGE_DETAILS')]
    task=request.activeOperations[0].pendingConversationTasks[0]
    task.context={'kitchenChange': {'optionsOnly':True,'baseInput':{'deliveryLocation':'DOCK_1','items':rows},'groups':[group]}}
    task.requiredOutputSchema={'type':'object','oneOf':[{'required':['items']},{'required':['roomServiceChangeCancelled']}],
        'properties':{'items':{'type':'array','minItems':1},'roomServiceChangeCancelled':{'const':True}},'additionalProperties':False}
    request.conversation.focusedConversationTaskId=task.conversationTaskId
    return request, task, rows


class KitchenChangeTest(unittest.TestCase):
    def setUp(self):
        for path, action in [('app.agents.v2_turn_planner.classify_hotel_scope', lambda r,m,s:(scope_for(m),OpenAiTokenUsage())),
                             ('app.agents.v2_turn_planner.localize_response', lambda r,o,t:o)]:
            p=patch(path,side_effect=action);p.start();self.addCleanup(p.stop)
        for path in ['app.services.input_understanding.call_openai_json_result','app.services.catalog_orders.call_openai_json_result','app.agents.v2_turn_planner.call_openai_json_result']:
            p=patch(path,side_effect=AssertionError('Unexpected model call '+path));p.start();self.addCleanup(p.stop)

    def test_real_incident_changes_only_chorizo_and_requires_updated_confirmation(self):
        for locale in ['es-MX','en']:
            request,task,original=fixture(locale=locale)
            response=planner.plan_v2_turn(request)
            self.assertFalse(response.toolCalls)
            self.assertEqual(locale,response.messages[0].language)
            state=json.loads(response.updatedConversationSummary)['catalogReplacementTasks'][str(task.conversationTaskId)]
            changed=state['items']
            self.assertEqual(original[1],changed[1])
            self.assertEqual(original[0]['modifications'],changed[0]['modifications'])
            self.assertEqual(['roja','pollo'],changed[0]['catalogSelection']['optionIds'])
            self.assertEqual('250.00',changed[0]['catalogSelection']['lineTotal'])
            request=follow_up(request,response,'Confirm',response.messages[0].interaction.options[0].id)
            confirmed=planner.plan_v2_turn(request)
            self.assertEqual(changed,confirmed.toolCalls[0].arguments['result']['items'])
            self.assertNotIn('deliveryLocation',confirmed.toolCalls[0].arguments['result'])

    def test_initial_server_button_optional_skip_and_multiple_choices(self):
        request,task,_=fixture()
        request.conversation.recentMessages[-1].interactionReplyId=f"catalog-choice:{task.context['kitchenChange']['groups'][0]['token']}:__skip__"
        response=planner.plan_v2_turn(request)
        changed=json.loads(response.updatedConversationSummary)['catalogReplacementTasks'][str(task.conversationTaskId)]['items']
        self.assertEqual(['roja'],changed[0]['catalogSelection']['optionIds'])
        request,task,_=fixture('¿qué opciones hay?',multiple=True)
        response=planner.plan_v2_turn(request)
        self.assertIsNone(response.messages[0].interaction)
        request=follow_up(request,response,'pollo y arrachera')
        response=planner.plan_v2_turn(request)
        changed=json.loads(response.updatedConversationSummary)['catalogReplacementTasks'][str(task.conversationTaskId)]['items']
        self.assertEqual({'roja','huevo','pollo','arrachera'},set(changed[0]['catalogSelection']['optionIds']))

    def test_unavailable_choice_and_old_buttons_never_change_or_complete_order(self):
        request,task,original=fixture('chorizo')
        response=planner.plan_v2_turn(request)
        self.assertFalse(response.toolCalls)
        self.assertNotIn('Chorizo',response.messages[0].text)
        self.assertIn('Pechuga de Pollo',response.messages[0].text)
        request=follow_up(request,response,'pollo')
        response=planner.plan_v2_turn(request)
        old_confirm=response.messages[0].interaction.options[0].id
        request=follow_up(request,response,'arrachera')
        response=planner.plan_v2_turn(request)
        request=follow_up(request,response,'Confirmar',old_confirm)
        response=planner.plan_v2_turn(request)
        self.assertFalse(response.toolCalls)

    def test_confirmation_revalidates_price_and_cancellation_has_current_guest_evidence(self):
        request,task,_=fixture()
        response=planner.plan_v2_turn(request)
        request=follow_up(request,response,'Confirmar',response.messages[0].interaction.options[0].id)
        catalog(request.availableOfferings[0])['options'][0]['optionGroups'][1]['options'][1]['priceAdjustment']=55
        response=planner.plan_v2_turn(request)
        self.assertFalse(response.toolCalls)
        self.assertIn('255.00',response.messages[0].text)
        request=follow_up(request,response,'Cancelar pedido',response.messages[0].interaction.options[1].id)
        cancelled=planner.plan_v2_turn(request)
        self.assertEqual({'roomServiceChangeCancelled':True},cancelled.toolCalls[0].arguments['result'])

    def test_scope_validation_rejects_unrelated_changes_and_retaining_blocked_option(self):
        request,task,rows=fixture()
        context=task.context['kitchenChange']
        with self.assertRaises(AgentModelError):validate_scope(context,rows)
        rows[0]['catalogSelection']['optionIds']=['roja','pollo']
        validate_scope(context,rows)
        for field,value in [('quantity',2),('name','Otra cosa'),('modifications',[])]:
            changed=deepcopy(rows);changed[0][field]=value
            with self.assertRaises(AgentModelError):validate_scope(context,changed)

    def test_sauce_category_words_are_accepted_but_unknown_additions_are_not(self):
        request,_,_=fixture()
        off=request.availableOfferings[0]
        rows,pending=normalize_order(off,[{'name':'Chilaquiles Clásicos','quantity':1,'modifications':['con salsa roja','con pechuga de pollo']}],semantic=False)
        self.assertIsNone(pending)
        self.assertEqual(['roja','pollo'],rows[0]['catalogSelection']['optionIds'])
        _,pending=normalize_order(off,[{'name':'Chilaquiles Clásicos','quantity':1,'modifications':['con salsa roja y langosta','con pechuga de pollo']}],semantic=False)
        self.assertEqual('MODIFIER',pending['kind'])


if __name__=='__main__':unittest.main()
