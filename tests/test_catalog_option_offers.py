"""Optional complements remain explicit, priced and scoped to the current draft."""
import json
import unittest
from copy import deepcopy
from unittest.mock import patch
from app.services.catalog_orders import normalize_order, pending_choice, apply_choice, clarification
from tests.test_catalog_orders import item, group, offering, row, catalog
from tests import test_catalog_orders as base
from tests.test_spa_turns import request_for, follow_up
from app.agents import v2_turn_planner as planner


def extras(maximum=5):
    return {'id': 'extras', 'name': 'Extras', 'required': False, 'minimumSelections': 0,
            'maximumSelections': maximum, 'offerToGuest': True, 'options': [
                {'id': 'egg', 'name': 'Huevo', 'priceAdjustment': 35, 'available': True},
                {'id': 'chicken', 'name': 'Pechuga de Pollo', 'priceAdjustment': 100, 'available': True},
                {'id': 'beef', 'name': 'Arrachera', 'priceAdjustment': 150, 'available': True}]}


def choose(rows, pending, identity):
    choice = next(c for c in pending['choices'] if c['id'] == identity)
    result = deepcopy(rows)
    result[pending.get('itemIndex', 0)] = apply_choice(result[pending.get('itemIndex', 0)], pending, choice)
    return result


class CatalogOptionOffersTest(unittest.TestCase):
    def setUp(self):
        self.model = patch('app.services.catalog_orders.call_openai_json_result', side_effect=AssertionError('No live model in deterministic tests'))
        self.model.start(); self.addCleanup(self.model.stop)

    def test_required_variant_precedes_optional_offer_and_decline_survives_salsa_edit(self):
        off = offering([item(groups=[extras(), group()])])
        rows, pending = normalize_order(off, [row(notes=['sin cebolla'])], semantic=False)
        self.assertEqual('size', pending['groupId'])
        rows, pending = normalize_order(off, choose(rows, pending, 'small'), semantic=False)
        self.assertEqual('extras', pending['groupId'])
        rows, pending = normalize_order(off, choose(rows, pending, '__skip__'), semantic=False)
        self.assertIsNone(pending)
        rows[0]['modifications'] = ['Grande', 'sin cebolla']
        rows, pending = normalize_order(off, rows, semantic=False)
        self.assertIsNone(pending)
        self.assertEqual(['Grande'], rows[0]['catalogSelection']['optionNames'])
        self.assertEqual('DECLINED', rows[0]['catalogSelection']['groupDecisions']['extras'])

    def test_multiple_extras_toggle_then_done_produces_exact_total(self):
        off = offering([item(groups=[extras()])])
        rows, pending = normalize_order(off, [row()], semantic=False)
        for identity in ['egg', 'chicken', 'egg']:
            rows, pending = normalize_order(off, choose(rows, pending, identity), semantic=False)
            self.assertIsNotNone(pending)
        rows, pending = normalize_order(off, choose(rows, pending, '__done__'), semantic=False)
        self.assertIsNone(pending)
        self.assertEqual(['Pechuga de Pollo'], rows[0]['catalogSelection']['optionNames'])
        self.assertEqual('400.00', rows[0]['catalogSelection']['lineTotal'])
        self.assertEqual((rows, None), normalize_order(off, rows, semantic=False))

    def test_disabled_or_empty_offers_do_not_block_and_single_optional_choice_is_not_automatic(self):
        for options in [[], [extras()['options'][0]]]:
            extra = extras(); extra['options'] = options
            off = offering([item(groups=[extra])])
            rows, pending = normalize_order(off, [row()], semantic=False)
            self.assertEqual(bool(options), bool(pending))
            extra['offerToGuest'] = False
            self.assertIsNone(normalize_order(offering([item(groups=[extra])]), [row()], semantic=False)[1])

    def test_skip_and_done_have_english_text_and_draft_scoped_buttons(self):
        off = offering([item(groups=[extras()])]); request = request_for('extras')
        request.guest.preferredLanguage = 'en'
        rows, pending = normalize_order(off, [row()], semantic=False)
        output = clarification(request, off, 'items', pending)
        self.assertIn('No extras', output['text']); self.assertIn('100.00', output['text'])
        msg = request.conversation.recentMessages[-1]; msg.text = 'no thanks'; msg.interactionReplyId = None
        self.assertEqual('SKIP', pending_choice(pending, msg)['action'])
        msg.interactionReplyId = 'catalog-choice:old:__skip__'
        self.assertIsNone(pending_choice(pending, msg))

    def test_direct_named_extras_are_not_reasked_and_allergy_never_selects_an_extra(self):
        off = offering([item(groups=[extras()])])
        rows, pending = normalize_order(off, [row(notes=['con pollo', 'sin cebolla'])], semantic=False)
        self.assertIsNone(pending)
        self.assertEqual(['Pechuga de Pollo'], rows[0]['catalogSelection']['optionNames'])
        rows, pending = normalize_order(off, [row(notes=['alergia a pollo'])], semantic=False)
        self.assertIsNotNone(pending)
        self.assertFalse(any(c.get('selected') for c in pending['choices']))

    def test_free_text_selects_multiple_available_options_without_inventing_ids(self):
        off = offering([item(groups=[extras()])]); request = request_for('con pollo y huevo')
        rows, pending = normalize_order(off, [row()], semantic=False)
        selected = pending_choice(pending, request.conversation.recentMessages[-1])
        self.assertEqual({'egg', 'chicken'}, set(selected['ids']))
        rows[0] = apply_choice(rows[0], pending, selected)
        rows, pending = normalize_order(off, rows, semantic=False)
        self.assertIsNone(pending)
        self.assertEqual('470.00', rows[0]['catalogSelection']['lineTotal'])

    def test_partial_match_does_not_discard_unknown_extras_or_preparation_notes(self):
        off = offering([item(groups=[extras()])])
        _, pending = normalize_order(off, [row()], semantic=False)
        for text in ['con pollo y langosta', 'pollo sin sal', 'pollo para otro platillo', '2 pollo']:
            request = request_for(text)
            self.assertIsNone(pending_choice(pending, request.conversation.recentMessages[-1]), text)
        self.assertEqual('MODIFIER', normalize_order(off, [row(notes=['con pollo y langosta'])], semantic=False)[1]['kind'])

    def test_declining_after_multiple_selection_does_not_leave_paid_notes_in_summary(self):
        off = offering([item(groups=[group(), extras()])])
        rows, pending = normalize_order(off, [row(notes=['con pollo y huevo', 'sin sal'])], semantic=False)
        rows, pending = normalize_order(off, choose(rows, pending, 'small'), semantic=False)
        self.assertIsNone(pending)
        # Reopening the group explicitly must remove only its selected-extra notes.
        rows[0]['catalogSelection']['groupDecisions']['extras'] = 'SELECTING'
        rows, pending = normalize_order(off, rows, semantic=False)
        rows, pending = normalize_order(off, choose(rows, pending, '__skip__'), semantic=False)
        self.assertIsNone(pending)
        self.assertEqual(['sin sal'], rows[0]['modifications'])

    def test_removed_selection_blocks_and_price_change_recomputes_without_reoffering(self):
        off = offering([item(groups=[extras()])])
        rows, pending = normalize_order(off, [row(notes=['con pollo'])], semantic=False)
        catalog(off)['options'][0]['optionGroups'][0]['options'][1]['priceAdjustment'] = 120
        changed, pending = normalize_order(off, rows, semantic=False)
        self.assertIsNone(pending); self.assertEqual('440.00', changed[0]['catalogSelection']['lineTotal'])
        catalog(off)['options'][0]['optionGroups'][0]['options'][1]['available'] = False
        self.assertIsNotNone(normalize_order(off, rows, semantic=False)[1])

    def test_large_group_pages_keep_skip_reachable_without_losing_selection(self):
        extra = extras(maximum=1); extra['options'] = [{'id': str(n), 'name': 'Extra ' + str(n), 'priceAdjustment': n, 'available': True} for n in range(18)]
        off = offering([item(groups=[extra])]); request = request_for('extras')
        rows, pending = normalize_order(off, [row()], semantic=False)
        for page in [1, 2]:
            msg = request.conversation.recentMessages[-1]; msg.interactionReplyId = 'catalog-choice:' + pending['token'] + ':__next__'
            rows[0] = apply_choice(rows[0], pending, pending_choice(pending, msg))
            rows, pending = normalize_order(off, rows, semantic=False)
            self.assertEqual(page, pending['page'])
            self.assertLessEqual(len(clarification(request, off, 'items', pending)['interaction']['options']), 10)
        self.assertIn('Sin extras', clarification(request, off, 'items', pending)['text'])

    def test_multiple_options_are_text_only_and_one_named_option_finishes_the_group(self):
        off = offering([item(groups=[extras()])])
        for locale in ['es', 'en']:
            request = request_for('Huevo'); request.guest.preferredLanguage = locale
            rows, pending = normalize_order(off, [row()], semantic=False)
            output = clarification(request, off, 'items', pending)
            self.assertIsNone(output['interaction'])
            self.assertIn('35.00', output['text'])
            choice = pending_choice(pending, request.conversation.recentMessages[-1])
            self.assertEqual({'action': 'SELECT_SET', 'ids': ['egg']}, choice)
            rows[0] = apply_choice(rows[0], pending, choice)
            rows, pending = normalize_order(off, rows, semantic=False)
            self.assertIsNone(pending)
            self.assertEqual(['Huevo'], rows[0]['catalogSelection']['optionNames'])

    def test_single_protein_rejects_combination_and_required_multiple_has_no_buttons(self):
        single = offering([item(groups=[extras(maximum=1)])])
        _, pending = normalize_order(single, [row()], semantic=False)
        request = request_for('pollo y huevo')
        self.assertIsNone(pending_choice(pending, request.conversation.recentMessages[-1]))
        self.assertIsNotNone(clarification(request, single, 'items', pending)['interaction'])
        self.assertIsNotNone(normalize_order(single, [row(notes=['con pollo y huevo'])], semantic=False)[1])
        required = extras(); required.update(required=True, minimumSelections=2)
        off = offering([item(groups=[required])])
        rows, pending = normalize_order(off, [row()], semantic=False)
        self.assertIsNone(clarification(request, off, 'items', pending)['interaction'])
        request.conversation.recentMessages[-1].text = 'Huevo'
        self.assertIsNone(pending_choice(pending, request.conversation.recentMessages[-1]))
        request.conversation.recentMessages[-1].text = 'pollo y huevo'
        rows[0] = apply_choice(rows[0], pending, pending_choice(pending, request.conversation.recentMessages[-1]))
        self.assertIsNone(normalize_order(off, rows, semantic=False)[1])

    def test_decline_belongs_to_item_and_never_skips_another_item(self):
        off = offering([item(groups=[extras()]), item('other', 'Otro', groups=[extras()])])
        rows, pending = normalize_order(off, [row()], semantic=False)
        rows, pending = normalize_order(off, choose(rows, pending, '__skip__'), semantic=False)
        rows[0]['name'] = 'Otro'
        self.assertEqual('extras', normalize_order(off, rows, semantic=False)[1]['groupId'])


class CatalogOfferConversationTest(unittest.TestCase):
    setUp = base.CatalogConversationTest.setUp
    capture = base.CatalogConversationTest.capture

    def test_natural_multiple_reply_keeps_english_summary_and_does_not_repeat_offer(self):
        off = offering([item(groups=[extras()])]); request, response = self.capture(off)
        self.assertIsNone(response.messages[0].interaction)
        request = follow_up(request, response, 'pollo y huevo')
        request.guest.preferredLanguage = 'en'
        with patch('app.agents.v2_turn_planner.classify_hotel_scope', side_effect=AssertionError('Catalog selections inherit English')):
            response = planner.plan_v2_turn(request)
        self.assertFalse(response.toolCalls)
        self.assertEqual('en', response.messages[0].language)
        self.assertIn('Order confirmation', response.messages[0].text)
        self.assertIn('470.00', response.messages[0].text)
        self.assertTrue(json.loads(response.updatedConversationSummary)['awaitingExplicitConfirmation'])

    def test_catalog_name_does_not_switch_english_confirmation_to_spanish(self):
        from app.services.conversation_language import resolve_language
        from tests.test_v2_scope_router import decision
        from tests.test_multilingual_foundation import multilingual_request
        off = offering([item(name='Chilaquiles Clásicos', groups=[extras(maximum=1)])])
        for text in ['Chilaquiles Clásicos', '2 Chilaquiles Clásicos', 'Huevo']:
            request = multilingual_request(text); request.availableOfferings = [off]
            request.guest.preferredLanguage = 'en'
            route = decision('SERVICE_REQUEST', text, offering='ROOM_SERVICE')
            route.detectedLanguage, route.languageConfidence = 'es', .99
            localized, change = resolve_language(request, request.conversation.recentMessages[0], route)
            self.assertIsNone(change)
            self.assertEqual('en', localized.guest.preferredLanguage)
            rows, pending = normalize_order(off, [dict(row(), name='Chilaquiles Clásicos')], semantic=False)
            rows = choose(rows, pending, 'egg')
            rows, pending = normalize_order(off, rows, semantic=False)
            message = planner._room_service_confirmation_message(localized, off, {'items': rows})
            self.assertTrue(message['text'].startswith('Order confirmation'))
            self.assertEqual('en', message['language'])
            self.assertEqual('Order confirmation', message['interaction']['title'])
            self.assertEqual(message['text'], message['interaction']['body'])
            self.assertEqual(['Confirm', 'Change', 'Cancel'], [o['label'] for o in message['interaction']['options']])
            route.requestedLanguage = 'es'
            self.assertEqual('es', resolve_language(request, request.conversation.recentMessages[0], route)[0].guest.preferredLanguage)

    def test_no_thanks_is_bound_to_extras_without_global_scope_or_order_cancellation(self):
        off = offering([item(groups=[extras()])]); request, response = self.capture(off)
        request = follow_up(request, response, 'no thanks')
        with patch('app.agents.v2_turn_planner.classify_hotel_scope', side_effect=AssertionError('A scoped skip must not go through global intent')):
            response = planner.plan_v2_turn(request)
        self.assertFalse(response.toolCalls)
        state = json.loads(response.updatedConversationSummary)
        self.assertEqual('ROOM_SERVICE', state['pendingOffering'])
        self.assertTrue(state['awaitingExplicitConfirmation'])
    def test_offer_skip_stale_button_and_confirm_create_only_current_order(self):
        from tests.conversation_regression.room_confirmation_support import current_room_button
        off = offering([item(groups=[extras(maximum=1)])]); request, response = self.capture(off)
        skip = next(o.id for o in response.messages[0].interaction.options if o.id.endswith(':__skip__'))
        request = follow_up(request, response, 'Sin extras', skip)
        response = planner.plan_v2_turn(request)
        self.assertFalse(response.toolCalls)
        self.assertIn('200.00', response.messages[0].text)
        request = follow_up(request, response, 'Sin extras', skip)
        response = planner.plan_v2_turn(request)
        self.assertFalse(response.toolCalls)
        request = follow_up(request, response, 'Confirmar', current_room_button(response))
        response = planner.plan_v2_turn(request)
        self.assertEqual('START_SERVICE', response.toolCalls[0].toolName.value)
        self.assertEqual([], response.toolCalls[0].arguments['input']['items'][0]['catalogSelection']['optionIds'])
