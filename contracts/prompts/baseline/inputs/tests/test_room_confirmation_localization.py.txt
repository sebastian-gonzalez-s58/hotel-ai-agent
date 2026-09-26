"""Exercise catalog confirmation through the real presentation-localization stage."""
import json
import time
import unittest
from unittest.mock import patch

from app.agents import v2_turn_planner as planner
from app.services import localized_content as localization
from app.services.catalog_orders import normalize_order
from app.services.conversation_language import greeting_language, resolve_language
from app.schemas.v2_turns import AgentMessage
from app.services.openai_client import OpenAiJsonResult
from app.services.telemetry_client import OpenAiTokenUsage
from tests.test_catalog_orders import offering, item, row
from tests.test_catalog_option_offers import extras
from tests.test_spa_turns import request_for, follow_up
from tests.conversation_regression.room_confirmation_support import current_room_button


def pending_extra(locale='en'):
    off = offering([item(name='Chilaquiles Clásicos', price=190, groups=[extras(maximum=1)])])
    rows, pending = normalize_order(off, [row('Chilaquiles Clásicos', 1, ['sin cebolla'])], semantic=False)
    request = request_for('Skirt steak 150')
    request.availableOfferings = [off]
    request.guest.preferredLanguage = locale
    request.trigger.eventPayload['languageContext'] = {'version': 1, 'explicit': False}
    request.conversation.summary = json.dumps({
        'pendingOffering': 'ROOM_SERVICE', 'phase': 'NEEDS_CATALOG_SELECTION',
        'capturedFields': {'deliveryLocation': 'ROOM', 'items': rows},
        'catalogPending': pending, 'awaitingExplicitConfirmation': False})
    request.conversation.recentMessages[-1].interactionReplyId = f"catalog-choice:{pending['token']}:beef"
    return request


def wrong_language_translation(prompt, **kwargs):
    # Valid placeholders, amounts and JSON shape, but wrong language: the production failure.
    texts = json.loads(prompt.split('\n', 1)[1])
    substitutions = {'Order confirmation': 'Confirmación del pedido', 'Items:': 'Artículos:',
        'Delivery location:': 'Lugar de entrega:',
        'Would you like to confirm, change, or cancel the order?': '¿Desea confirmar, cambiar o cancelar el pedido?',
        'Change': 'Cambiar'}
    result = {}
    for entry in texts:
        text = entry['text']
        for source, target in substitutions.items():
            text = text.replace(source, target)
        result[entry['id']] = text
    return OpenAiJsonResult({'texts': result}, OpenAiTokenUsage(), 'synthetic')


class RoomConfirmationLocalizationTest(unittest.TestCase):
    def setUp(self):
        localization._cache.clear()
        for path in ['app.agents.v2_turn_planner.classify_hotel_scope',
                     'app.agents.v2_turn_planner.call_openai_json_result',
                     'app.services.catalog_orders.call_openai_json_result',
                     'app.services.input_understanding.call_openai_json_result']:
            p = patch(path, side_effect=AssertionError('Unexpected interpretation call: ' + path))
            p.start(); self.addCleanup(p.stop)

    def test_english_catalog_confirmation_cannot_be_rewritten_in_spanish(self):
        for locale in ['en', 'en-US', 'en-GB']:
            with self.subTest(locale=locale), patch.object(localization, 'call_openai_json_result', side_effect=wrong_language_translation) as translate:
                response = planner.plan_v2_turn(pending_extra(locale))
                message = response.messages[0]
                self.assertTrue(message.text.startswith('Order confirmation'))
                self.assertIn('Delivery location: Room', message.text)
                self.assertIn('Chilaquiles Clásicos — Arrachera (sin cebolla) — MXN 340.00', message.text)
                self.assertIn('Total: MXN 340.00', message.text)
                self.assertEqual(locale, message.language)
                self.assertEqual(message.text, message.interaction.body)
                self.assertEqual('Order confirmation', message.interaction.title)
                self.assertEqual(['Confirm', 'Change', 'Cancel'], [o.label for o in message.interaction.options])
                self.assertFalse(response.toolCalls)
                self.assertIsNone(response.languageDecision)
                translate.assert_not_called()

    def test_spanish_summary_preserves_catalog_and_explicit_confirmation(self):
        request = pending_extra('es-MX')
        with patch.object(localization, 'call_openai_json_result', side_effect=AssertionError('Summary needs no translator')):
            response = planner.plan_v2_turn(request)
            self.assertTrue(response.messages[0].text.startswith('Confirmación de pedido'))
            self.assertEqual(['Confirmar', 'Cambiar', 'Cancelar'], [o.label for o in response.messages[0].interaction.options])
            request = follow_up(request, response, 'Confirmar', current_room_button(response))
            started = planner.plan_v2_turn(request)
        self.assertEqual('START_SERVICE', started.toolCalls[0].toolName.value)
        selected = started.toolCalls[0].arguments['input']['items'][0]
        self.assertEqual(['beef'], selected['catalogSelection']['optionIds'])
        self.assertEqual(['sin cebolla'], selected['modifications'])

    def test_other_locales_still_use_presentation_translation(self):
        with patch.object(localization, 'call_openai_json_result', side_effect=wrong_language_translation) as translate:
            response = planner.plan_v2_turn(pending_extra('fr'))
        translate.assert_called_once()
        self.assertTrue(translate.call_args.args[0].startswith('Translate these hotel interface messages to fr.'))
        self.assertEqual('fr', response.messages[0].language)

    def test_language_flag_alone_cannot_bypass_localization(self):
        request = pending_extra()
        with patch.object(localization, 'call_openai_json_result', side_effect=wrong_language_translation):
            response = planner.plan_v2_turn(request)
        captured = json.loads(response.updatedConversationSummary)['capturedFields']
        response.messages = [AgentMessage.model_validate(planner._room_service_confirmation_message(
            request, request.availableOfferings[0], captured))]
        self.assertTrue(planner._is_verified_room_confirmation(request, response))
        for target in ['text', 'body', 'title', 'label', 'id', 'purpose']:
            changed = response.model_copy(deep=True)
            message = changed.messages[0]
            if target == 'text': message.text = 'Confirma tu pedido'
            elif target == 'purpose': message.purpose = 'ANSWER'
            elif target in {'body', 'title'}: setattr(message.interaction, target, 'Confirmación del pedido')
            else: setattr(message.interaction.options[0], target, 'Cambiar')
            self.assertFalse(planner._is_verified_room_confirmation(request, changed), target)
        changed = response.model_copy(deep=True)
        changed.messages[0].text = 'Confirma tu pedido'
        with patch.object(localization, 'call_openai_json_result', side_effect=wrong_language_translation) as translate:
            localization.localize_response(request, changed, time.perf_counter())
        translate.assert_called()

    def test_custom_destination_uses_approved_label_and_preserves_source_value(self):
        request = pending_extra()
        off = request.availableOfferings[0]
        options = off.inputSchema['properties']['deliveryLocation']['x-chatbotinn-capture']['catalog']['options']
        options[0]['label'] = 'Terraza Azul'
        off.inputSchema['x-chatbotinn-localization'] = {'variants': {'en': {'Terraza Azul': 'Blue Terrace'}}}
        with patch.object(localization, 'call_openai_json_result', side_effect=AssertionError('No summary translation')):
            response = planner.plan_v2_turn(request)
        self.assertIn('Delivery location: Blue Terrace', response.messages[0].text)
        self.assertEqual('ROOM', json.loads(response.updatedConversationSummary)['capturedFields']['deliveryLocation'])
        off.inputSchema['x-chatbotinn-localization']['variants'] = {}
        with patch.object(localization, 'call_openai_json_result', side_effect=AssertionError('Retain canonical place name')):
            response = planner.plan_v2_turn(request)
        self.assertIn('Delivery location: Terraza Azul', response.messages[0].text)


class GreetingTypoLocaleTest(unittest.TestCase):
    def test_hellow_opens_english_menu_and_preserves_pending_order(self):
        request = pending_extra('es-MX')
        request.conversation.recentMessages[-1].text = 'Hellow!'
        request.conversation.recentMessages[-1].interactionReplyId = None
        before = json.loads(request.conversation.summary)['capturedFields']
        off = request.availableOfferings[0]
        off.inputSchema['x-chatbotinn-localization'] = {'variants': {'en': {off.name: 'Room service'}}}
        with patch.object(planner, 'classify_hotel_scope', side_effect=AssertionError('Greeting needs no scope model')), patch.object(localization, 'call_openai_json_result', side_effect=wrong_language_translation):
            response = planner.plan_v2_turn(request)
        self.assertEqual('en', response.languageDecision.locale)
        self.assertTrue(response.messages[0].text.startswith('Hello,'))
        self.assertEqual('en', response.messages[0].language)
        self.assertEqual(before, json.loads(response.updatedConversationSummary)['capturedFields'])

    def test_hellow_establishes_english_without_interpretation(self):
        request = request_for('Hellow!')
        request.guest.preferredLanguage = 'es-MX'
        request.availableOfferings = []
        request.trigger.eventPayload['languageContext'] = {'version': 1, 'explicit': False}
        for text in ['Hellow', 'HELLOW!', '  Hellow.  ']:
            request.conversation.recentMessages[-1].text = text
            localized, decision = resolve_language(request, request.conversation.recentMessages[-1])
            self.assertEqual('en', localized.guest.preferredLanguage)
            self.assertEqual('DETECTED', decision.source)

    def test_typo_does_not_override_explicit_spanish_or_match_a_request(self):
        request = request_for('Hellow')
        request.guest.preferredLanguage = 'es-MX'
        request.trigger.eventPayload['languageContext'] = {'version': 1, 'explicit': True}
        localized, decision = resolve_language(request, request.conversation.recentMessages[-1])
        self.assertEqual('es-MX', localized.guest.preferredLanguage)
        self.assertIsNone(decision)
        self.assertIsNone(greeting_language('Hellow, I need room service'))
