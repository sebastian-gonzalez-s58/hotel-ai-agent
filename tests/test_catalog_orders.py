"""Strict catalog journeys: canonical summaries, explicit choices and live revalidation."""
from copy import deepcopy
import json
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.agents import v2_turn_planner as planner, spa_turns
from app.services.catalog_orders import (normalize_order, resolve_selection, apply_choice, pending_choice,
                                         display_service, clarification, _semantic)
from app.services.openai_client import OpenAiJsonResult
from app.services.telemetry_client import OpenAiTokenUsage
from app.schemas.v2_turns import OfferingCapability, AgentTurnResponse, AgentMessage
from app.core.errors import AgentModelError
from tests.test_spa_turns import request_for, follow_up, extraction, spa_operation
from tests.test_v2_turn_planner import guided_room_service_offering, guided_spa_offering


def item(identity='burger', name='Hamburguesa Clásica', category='Hamburguesas', price=100, groups=None):
    return {'id': identity, 'code': identity.upper(), 'label': name, 'categoryLabel': category,
            'description': '', 'prices': [{'priceType': 'BASE', 'amount': price, 'currency': 'MXN', 'active': True}],
            'optionGroups': groups or []}


def group(required=True):
    return {'id': 'size', 'name': 'Tamaño', 'required': required, 'minimumSelections': int(required), 'maximumSelections': 1,
            'options': [{'id': 'small', 'name': 'Chico', 'available': True, 'priceAdjustment': 0},
                        {'id': 'large', 'name': 'Grande', 'available': True, 'priceAdjustment': 25}]}


def offering(options=None, spa=False):
    data = guided_spa_offering() if spa else guided_room_service_offering()
    field = 'serviceName' if spa else 'items'
    data['inputSchema']['properties'][field].setdefault('x-chatbotinn-capture', {})['catalog'] = {
        'selectionPolicy': 'ACTIVE_ITEMS_ONLY', 'options': options if options is not None else [item()],
        'externalUrl': 'https://example.invalid/catalog'}
    data['requiresExplicitGuestConfirmation'] = True
    return OfferingCapability.model_validate(data)


def row(name='hamburguesa clasica', quantity=2, notes=None):
    return {'name': name, 'quantity': quantity, 'modifications': notes or []}


def catalog(off):
    field = 'serviceName' if off.offeringCode == 'SPA' else 'items'
    return off.inputSchema['properties'][field]['x-chatbotinn-capture']['catalog']


class CatalogOrdersTest(unittest.TestCase):
    def setUp(self):
        self.model = patch('app.services.catalog_orders.call_openai_json_result', side_effect=AssertionError('Unexpected model call'))
        self.mock_model = self.model.start()
        self.addCleanup(self.model.stop)

    def test_normalizes_accents_case_and_prices_without_losing_restrictions(self):
        original = [row(notes=['sin cebolla', 'alergia a nueces'])]
        output, pending = normalize_order(offering(), original, semantic=False)
        self.assertIsNone(pending)
        self.assertEqual('Hamburguesa Clásica', output[0]['name'])
        self.assertEqual(original[0]['modifications'], output[0]['modifications'])
        self.assertEqual('hamburguesa clasica', output[0]['catalogSelection']['requestedName'])
        self.assertEqual('200.00', output[0]['catalogSelection']['lineTotal'])
        self.assertNotIn('catalogSelection', original[0])

    def test_no_catalog_policy_retains_legacy_contract(self):
        raw = [row('tea')]
        out, pending = normalize_order(OfferingCapability.model_validate(guided_room_service_offering()), raw)
        self.assertEqual(raw, out)
        self.assertIsNone(pending)

    def test_unknown_or_empty_catalog_never_confirms_and_keeps_other_rows(self):
        for off in [offering(), offering([])]:
            out, pending = normalize_order(off, [row('langosta'), row()], semantic=False)
            self.assertEqual('UNAVAILABLE', pending['kind'])
            self.assertEqual(2, len(out))

    def test_unavailable_clarification_satisfies_agent_message_contract(self):
        request = request_for('pozole')
        request.guest.preferredLanguage = 'en'
        off = offering([item('burger', 'Hamburguesa Delux')])
        _, pending = normalize_order(off, [row('pozole')], semantic=False)

        message = clarification(request, off, 'items', pending)
        parsed = AgentMessage.model_validate(message)

        self.assertIsNotNone(parsed.messageDraftId)
        self.assertEqual('CLARIFICATION', parsed.purpose)
        self.assertEqual('Cancel order', parsed.interaction.options[-1].label)

        request.conversation.recentMessages[-1].interactionReplyId = (
            'catalog-choice:' + pending['token'] + ':__remove__'
        )
        self.assertEqual('__remove__', pending_choice(pending, request.conversation.recentMessages[-1])['id'])

    def test_unavailable_first_item_does_not_hide_later_valid_item(self):
        off = offering([item('burger', 'Hamburguesa Delux')])
        out, pending = normalize_order(off, [row('pozole'), row('hamburguesa delux', 1)], semantic=False)
        self.assertEqual('UNAVAILABLE', pending['kind'])
        self.assertEqual(0, pending['itemIndex'])
        self.assertEqual('Hamburguesa Delux', out[1]['name'])
        self.assertEqual('burger', out[1]['catalogSelection']['itemId'])

    def test_duplicate_name_requires_category_choice(self):
        off = offering([item('frozen', 'Fresa', 'Frozen'), item('smoothie', 'Fresa', 'Smoothies')])
        out, pending = normalize_order(off, [row('fresa')])
        self.assertEqual('ITEM', pending['kind'])
        self.assertEqual(2, len(pending['choices']))
        out[0] = apply_choice(out[0], pending, pending['choices'][1])
        out, pending = normalize_order(off, out, semantic=False)
        self.assertIsNone(pending)
        self.assertEqual('smoothie', out[0]['catalogSelection']['itemId'])

    def test_required_variant_selection_is_explicit_and_priced(self):
        off = offering([item(groups=[group()])])
        out, pending = normalize_order(off, [row()], semantic=False)
        self.assertEqual('OPTION', pending['kind'])
        out[0] = apply_choice(out[0], pending, pending['choices'][1])
        out, pending = normalize_order(off, out, semantic=False)
        self.assertIsNone(pending)
        self.assertEqual(['Grande'], out[0]['catalogSelection']['optionNames'])
        self.assertEqual('250.00', out[0]['catalogSelection']['lineTotal'])

    def test_negative_modifier_does_not_select_extra(self):
        off = offering([item(groups=[group(False)])])
        out, pending = normalize_order(off, [row(notes=['sin Grande'])], semantic=False)
        self.assertIsNone(pending)
        self.assertEqual([], out[0]['catalogSelection']['optionIds'])

    def test_only_required_variant_is_included(self):
        g = group(); g['options'] = g['options'][:1]
        out, pending = normalize_order(offering([item(groups=[g])]), [row()], semantic=False)
        self.assertIsNone(pending)
        self.assertEqual(['small'], out[0]['catalogSelection']['optionIds'])

    def test_removed_identity_not_substituted_by_same_name(self):
        off = offering(); out, _ = normalize_order(off, [row()], semantic=False)
        catalog(off)['options'] = [item('replacement')]
        checked, pending = normalize_order(off, out, semantic=False)
        self.assertEqual('UNAVAILABLE', pending['kind'])
        self.assertEqual('burger', checked[0]['catalogSelection']['itemId'])

    def test_unavailable_selected_option_blocks_confirmation(self):
        off = offering([item(groups=[group()])])
        out, _ = normalize_order(off, [row(notes=['Grande'])], semantic=False)
        catalog(off)['options'][0]['optionGroups'][0]['options'][1]['available'] = False
        _, pending = normalize_order(off, out, semantic=False)
        self.assertIsNotNone(pending)

    def test_stale_catalog_button_cannot_select_new_draft(self):
        _, pending = normalize_order(offering([item(groups=[group()])]), [row()], semantic=False)
        msg = request_for('Grande').conversation.recentMessages[-1]
        msg.interactionReplyId = 'catalog-choice:old:large'
        self.assertIsNone(pending_choice(pending, msg))
        msg.interactionReplyId = 'catalog-choice:' + pending['token'] + ':large'
        self.assertEqual('large', pending_choice(pending, msg)['id'])

    def test_price_change_rebuilds_summary_and_requires_confirmation(self):
        request = request_for('Confirmar'); off = offering(); request.availableOfferings = [off]
        captured = {'items': [row()], 'deliveryLocation': 'ROOM'}
        first = planner._room_service_capture_output(request, off, captured)
        state = json.loads(first['updated_summary'])
        request.conversation.summary = first['updated_summary']
        catalog(off)['options'][0]['prices'][0]['amount'] = 150
        response = planner._room_service_draft_plan(request, time.perf_counter())
        self.assertFalse(response.toolCalls)
        after = json.loads(response.updatedConversationSummary)
        self.assertEqual('300.00', after['capturedFields']['items'][0]['catalogSelection']['lineTotal'])
        self.assertIn('300.00', response.messages[0].text)

    def test_canonical_summary_preserves_qty_options_notes_location(self):
        request = request_for('pedido'); off = offering([item(groups=[group()])])
        out = planner._room_service_capture_output(request, off, {'items': [row(notes=['Grande', 'sin cebolla'])], 'deliveryLocation': 'ROOM'})
        text = out['messages'][0]['text']
        for value in ('2 x Hamburguesa Clásica', 'Grande', 'sin cebolla', '250.00'):
            self.assertIn(value, text)

    def test_semantic_translation_uses_only_offered_identity_with_evidence(self):
        self.mock_model.side_effect = None
        payload = {'status': 'MATCH', 'candidateId': 'burger', 'candidateIds': ['burger'], 'evidence': 'classic burger', 'confidence': .99}
        self.mock_model.return_value = OpenAiJsonResult(payload, OpenAiTokenUsage(), 'test')
        selected, _ = _semantic('classic burger', [item()])
        self.assertEqual('burger', selected['id'])
        for bad in [{'candidateId': 'injected'}, {'evidence': 'other'}, {'confidence': .2}, {'candidateIds': ['foreign']}]:
            self.mock_model.return_value = OpenAiJsonResult({**payload, **bad}, OpenAiTokenUsage(), 'test')
            self.assertIsNone(_semantic('classic burger', [item()])[0])

    def test_persisted_translation_resolves_without_model_call(self):
        translated = item('aromatic', 'Balance aromático')
        translated['translations'] = {'en': 'Aromatic Balance'}
        selected, pending = resolve_selection({'selectionPolicy': 'ACTIVE_ITEMS_ONLY', 'options': [translated]},
                                              'Aromatic Balance', [], semantic=True)
        self.assertIsNone(pending)
        self.assertEqual('aromatic', selected['itemId'])

    def test_semantic_matching_sends_only_relevant_candidates(self):
        options = [item('burger', 'Hamburguesa Clásica'), item('soup', 'Sopa de pollo'),
                   item('salad', 'Ensalada verde')]
        options[1]['translations'] = {'en': 'Chicken Soup'}
        payload = {'status': 'MATCH', 'candidateId': 'soup', 'candidateIds': ['soup'],
                   'evidence': 'chicken soup', 'confidence': .99}
        self.mock_model.side_effect = None
        self.mock_model.return_value = OpenAiJsonResult(payload, OpenAiTokenUsage(), 'test')
        selected, _ = _semantic('chicken soup', options)
        self.assertEqual('soup', selected['id'])
        candidates = json.loads(self.mock_model.call_args.args[0].split('Context:\n', 1)[1])['candidates']
        self.assertEqual(['soup'], [candidate['id'] for candidate in candidates])

    def test_spa_variant_round_trip_and_unavailable_treatment(self):
        off = offering([item('massage', 'Masaje Relajante', groups=[group()])], spa=True)
        selected, pending = resolve_selection(catalog(off), 'Masaje Relajante (Grande)', [], semantic=False)
        self.assertIsNone(pending)
        self.assertEqual('Masaje Relajante (Grande)', display_service(selected))
        self.assertIsNotNone(resolve_selection(catalog(off), 'Masaje Acuático', [], semantic=False)[1])

    def test_spa_catalog_correction_keeps_date_time_and_asks_only_variant(self):
        request = request_for('Masaje Relajante'); off = offering([item('massage', 'Masaje Relajante', groups=[group()])], spa=True)
        request.availableOfferings = [off]
        draft = {'id': str(uuid4()), 'capturedFields': {'serviceName': 'Masaje Relajante', 'reservationDate': '2026-09-02', 'reservationTime': '17:00'}}
        state = {'spaDraft': draft}
        response = spa_turns._capture(request, state, draft, None, None, off, None)
        self.assertEqual('OPTION', draft['catalogPending']['kind'])
        message = request.conversation.recentMessages[-1]
        message.interactionReplyId = response['messages'][0]['interaction']['options'][1]['id']
        response = spa_turns._capture(request, state, draft, message, None, off, None)
        self.assertEqual('2026-09-02', draft['capturedFields']['reservationDate'])
        self.assertEqual('17:00', draft['capturedFields']['reservationTime'])
        self.assertTrue(draft['awaitingConfirmation'])
        self.assertIn('Masaje Relajante (Grande)', response['messages'][0]['text'])

    def test_spa_price_change_blocks_old_confirm(self):
        request = request_for('Confirm'); off = offering([item('massage', 'Masaje Relajante')], spa=True)
        request.availableOfferings = [off]
        draft = {'id': str(uuid4()), 'capturedFields': {'serviceName': 'Masaje Relajante', 'reservationDate': '2026-09-02', 'reservationTime': '17:00'}}
        state = {'spaDraft': draft}
        spa_turns._capture(request, state, draft, None, None, off, None)
        old = draft['confirmationToken']
        catalog(off)['options'][0]['prices'][0]['amount'] = 150
        response = spa_turns._capture(request, state, draft, request.conversation.recentMessages[-1], ('spa-draft', draft['id'], 'CONFIRM', old), off, None)
        self.assertFalse(response.get('toolCalls'))
        self.assertNotEqual(old, draft['confirmationToken'])
        self.assertIn('150.00', response['messages'][0]['text'])

class CatalogConversationTest(unittest.TestCase):
    def setUp(self):
        from tests.test_multilingual_understanding import scope_for
        for path, action in [
                ('app.agents.v2_turn_planner.classify_hotel_scope', lambda r,m,s:(scope_for(m), OpenAiTokenUsage())),
                ('app.agents.v2_turn_planner.localize_response', lambda r,o,t:o)]:
            p=patch(path,side_effect=action);p.start();self.addCleanup(p.stop)
        for path,attr in [('app.services.input_understanding.call_openai_json_result','extract'),
                          ('app.services.catalog_orders.call_openai_json_result','match'),
                          ('app.agents.v2_turn_planner.call_openai_json_result','general')]:
            p=patch(path,side_effect=AssertionError('Unexpected model call: '+path));setattr(self,attr,p.start());self.addCleanup(p.stop)

    def capture(self, off, text='dos hamburguesas', name='hamburguesas', notes=None):
        from tests.test_multilingual_understanding import room_request, result, order, edit
        request=room_request(text);request.availableOfferings=[off]
        self.extract.side_effect=None;self.extract.return_value=result(order(edit(text,name,2,'dos',notes)))
        self.match.side_effect=None;self.match.return_value=OpenAiJsonResult({'status':'MATCH','candidateId':'burger','candidateIds':['burger'],
                   'evidence':name,'confidence':.99},OpenAiTokenUsage(),'test')
        return request,planner.plan_v2_turn(request)

    def test_whole_order_variant_choice_then_versioned_confirmation(self):
        from tests.conversation_regression.room_confirmation_support import current_room_button
        off=offering([item(groups=[group()])]);request,response=self.capture(off)
        self.assertFalse(response.toolCalls)
        request=follow_up(request,response,'Grande',response.messages[0].interaction.options[1].id)
        response=planner.plan_v2_turn(request)
        self.assertIn('250.00',response.messages[0].text)
        request=follow_up(request,response,'Confirmar',current_room_button(response))
        response=planner.plan_v2_turn(request)
        self.assertEqual('START_SERVICE',response.toolCalls[0].toolName.value)
        self.assertEqual('burger',response.toolCalls[0].arguments['input']['items'][0]['catalogSelection']['itemId'])

    def test_whole_order_unavailable_after_summary_does_not_start(self):
        from tests.conversation_regression.room_confirmation_support import current_room_button
        off=offering();request,response=self.capture(off)
        request=follow_up(request,response,'Confirmar',current_room_button(response))
        catalog(request.availableOfferings[0])['options']=[]
        response=planner.plan_v2_turn(request)
        self.assertFalse(response.toolCalls)
        self.assertFalse(json.loads(response.updatedConversationSummary)['awaitingExplicitConfirmation'])

    def test_kitchen_replacement_uses_catalog_choice_and_explicit_summary(self):
        from tests.test_multilingual_understanding import room_request, task_operation, result, order, edit
        request=room_request('dos hamburguesas');request.conversation.summary='{}'
        request.availableOfferings=[offering([item(groups=[group()])])]
        request.activeOperations=[task_operation('ROOM_SERVICE_ORDER_CHANGE_DETAILS')]
        task=request.activeOperations[0].pendingConversationTasks[0]
        request.conversation.focusedConversationTaskId=task.conversationTaskId
        self.extract.side_effect=None;self.extract.return_value=result(order(edit('dos hamburguesas','hamburguesas',2,'dos')))
        self.match.side_effect=None;self.match.return_value=OpenAiJsonResult({'status':'MATCH','candidateId':'burger','candidateIds':['burger'],
                   'evidence':'hamburguesas','confidence':.99},OpenAiTokenUsage(),'test')
        response=planner.plan_v2_turn(request)
        request=follow_up(request,response,'Grande',response.messages[0].interaction.options[1].id)
        response=planner.plan_v2_turn(request)
        self.assertFalse(response.toolCalls)
        self.assertIn('250.00',response.messages[0].text)
        request=follow_up(request,response,'Confirmar',response.messages[0].interaction.options[0].id)
        response=planner.plan_v2_turn(request)
        self.assertEqual('COMPLETE_CONVERSATION_TASK',response.toolCalls[0].toolName.value)
        self.assertEqual(task.conversationTaskId,response.toolCalls[0].targetConversationTaskId)
        self.assertEqual('burger',response.toolCalls[0].arguments['result']['items'][0]['catalogSelection']['itemId'])

class CatalogExtraTest(unittest.TestCase):
    def test_unknown_extra_cannot_be_smuggled_in_as_free_text(self):
        _, pending = normalize_order(offering(), [row(notes=['con langosta'])], semantic=False)
        self.assertEqual('MODIFIER', pending['kind'])
        valid, pending = normalize_order(offering(), [row(notes=['sin cebolla', 'salsa aparte', 'bien cocida'])], semantic=False)
        self.assertIsNone(pending)
        self.assertEqual(['sin cebolla', 'salsa aparte', 'bien cocida'], valid[0]['modifications'])

    def test_multiselect_required_group_collects_each_explicit_choice(self):
        g=group();g['minimumSelections']=2;g['maximumSelections']=2
        off=offering([item(groups=[g])]);items,pending=normalize_order(off,[row()],semantic=False)
        for identity in ('small','large'):
            choice=next(c for c in pending['choices'] if c['id']==identity)
            items[0]=apply_choice(items[0],pending,choice)
            items,pending=normalize_order(off,items,semantic=False)
        self.assertIsNone(pending)
        self.assertEqual(['small','large'],items[0]['catalogSelection']['optionIds'])

class CatalogVariantCorrectionTest(unittest.TestCase):
    def test_explicit_variant_choice_removes_contradictory_standalone_notes_only(self):
        off=offering([item(groups=[group()])])
        items,pending=normalize_order(off,[row(notes=['Chico','Grande','sin cebolla','alergia a nueces'])],semantic=False)
        self.assertEqual('OPTION',pending['kind'])
        items[0]=apply_choice(items[0],pending,pending['choices'][1])
        items,pending=normalize_order(off,items,semantic=False)
        self.assertIsNone(pending)
        self.assertEqual(['Grande'],items[0]['catalogSelection']['optionNames'])
        self.assertEqual(['sin cebolla','alergia a nueces'],items[0]['modifications'])

class CatalogMalformedInputTest(unittest.TestCase):
    def test_malformed_model_arguments_fail_closed_without_an_unhandled_exception(self):
        for rows in ['not a list', ['bad row'], [row(notes='sin cebolla')], [{**row(), 'catalogSelection': 'bad'}],
                     [{**row(), 'catalogSelection': {'optionIds': [None]}}]]:
            with self.subTest(rows=rows), self.assertRaises(AgentModelError):
                normalize_order(offering(), rows, semantic=False)
