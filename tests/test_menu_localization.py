import json
import os
import unittest
from unittest.mock import patch

from app.agents.v2_turn_planner import plan_v2_turn
from app.agents.v2_scope_router import classify_hotel_scope
from app.agents.social_opening import classify_social_opening
from app.schemas.v2_turns import AgentTurnRequest, OfferingCapability
from app.services.localized_content import _cache, translate_batch
from app.services.openai_client import OpenAiJsonResult
from app.services.telemetry_client import OpenAiTokenUsage
from tests.test_multilingual_foundation import multilingual_request, model_result, response_for
from tests.test_v2_scope_router import decision, maintenance_offering


def menu_request(locale="en"):
    request = multilingual_request("Hello")
    request.guest.preferredLanguage = locale
    request.trigger.eventPayload["languageContext"]["explicit"] = True
    names = [("FAQ", "Preguntas frecuentes"), ("ROOM_SERVICE", "Servicio a la habitaci\u00f3n"),
             ("MAINTENANCE", "Mantenimiento"), ("SPA", "Reservaci\u00f3n de spa"),
             ("FRONT_DESK", "Recepci\u00f3n")]
    request.availableOfferings = [OfferingCapability.model_validate(
        dict(maintenance_offering(), offeringCode=code, name=name)) for code, name in names]
    return request


def translation_inputs(call):
    return json.loads(call.call_args.args[0].rsplit("\n", 1)[1])


class MenuLocalizationTest(unittest.TestCase):
    def setUp(self):
        _cache.clear()
        opening = patch("app.agents.v2_turn_planner.classify_social_opening", return_value=(False, OpenAiTokenUsage()))
        self.opening = opening.start()
        self.addCleanup(opening.stop)

    @patch("app.agents.v2_turn_planner.call_openai_json_result")
    @patch("app.services.localized_content.call_openai_json_result")
    @patch("app.agents.v2_turn_planner.classify_hotel_scope")
    def test_semantic_greeting_including_typo_always_uses_localized_menu(self, classify, translate, planner):
        for text in ["Hellow", "Hi there", "Hello again", "Good evening, how are you?"]:
            with self.subTest(text=text):
                request = menu_request()
                request.conversation.recentMessages[0].text = text
                classify.return_value = decision("SOCIAL", text, details=False), OpenAiTokenUsage()
                self.opening.return_value = True, OpenAiTokenUsage(input_tokens=5, output_tokens=2, total_tokens=7)
                translate.return_value = model_result(["Room service", "Maintenance", "Spa reservations", "Contact reception"])
                response = plan_v2_turn(request)
                self.assertEqual("LIST", response.messages[0].interaction.type)
                self.assertEqual(5, len(response.messages[0].interaction.options))
                self.assertEqual("Hotel questions", response.messages[0].interaction.options[0].label)
                self.assertEqual([], response.toolCalls)
                self.assertGreaterEqual(response.usage.totalTokens, 7)
                self.assertNotIn("LOCALIZATION_FALLBACK", response.warnings)
                planner.assert_not_called()

    @patch("app.agents.v2_turn_planner.call_openai_json_result")
    @patch("app.services.localized_content.call_openai_json_result")
    @patch("app.agents.v2_turn_planner.classify_hotel_scope")
    def test_semantic_greeting_keeps_pending_operations_and_independent_drafts(self, classify, translate, planner):
        from uuid import uuid4
        from app.schemas.v2_turns import OperationSnapshot
        from tests.test_v2_turn_planner import operation, conversation_task
        request = menu_request()
        request.conversation.recentMessages[0].text = "Hellow"
        request.conversation.summary = json.dumps({
            "pendingOffering": "ROOM_SERVICE", "capturedFields": {"deliveryLocation": "ROOM"},
            "spaDraft": {"serviceName": "Massage", "reservationDate": "2026-09-20"},
        })
        op_id, task_id = str(uuid4()), str(uuid4())
        op = operation(op_id)
        op["offeringCode"] = "MAINTENANCE"
        op["pendingConversationTasks"] = [conversation_task(task_id, op_id)]
        request.activeOperations = [OperationSnapshot.model_validate(op)]
        before = request.model_dump()
        classify.return_value = decision("SOCIAL", "Hellow", details=False), OpenAiTokenUsage()
        self.opening.return_value = True, OpenAiTokenUsage()
        translate.return_value = model_result(["Room service", "Maintenance", "Spa reservations", "Contact reception"])
        response = plan_v2_turn(request)
        self.assertEqual(5, len(response.messages[0].interaction.options))
        self.assertEqual(request.conversation.summary, response.updatedConversationSummary)
        self.assertEqual([], response.toolCalls)
        self.assertEqual([], response.messages[0].operationIds)
        self.assertEqual([], response.messages[0].conversationTaskIds)
        self.assertEqual(before, request.model_dump())
        planner.assert_not_called()

    @patch("app.agents.v2_turn_planner.call_openai_json_result")
    @patch("app.services.localized_content.call_openai_json_result")
    @patch("app.agents.v2_turn_planner.classify_hotel_scope")
    def test_thanks_and_farewell_are_not_forced_back_to_menu(self, classify, translate, planner):
        for text, answer in [("Thanks", "You're welcome."), ("Goodbye", "Goodbye.")]:
            with self.subTest(text=text):
                request = menu_request()
                request.conversation.recentMessages[0].text = text
                classify.return_value = decision("SOCIAL", text, details=False), OpenAiTokenUsage()
                planned = response_for(request, answer)
                planned.messages[0].interaction = None
                planned.messages[0].operationIds = []
                planned.messages[0].purpose = "ANSWER"
                planner.return_value = OpenAiJsonResult(planned.model_dump(mode="json"), OpenAiTokenUsage(), "social")
                translate.return_value = model_result([answer])
                response = plan_v2_turn(request)
                self.assertIsNone(response.messages[0].interaction)
                self.assertEqual([], response.toolCalls)

    @patch("app.agents.v2_turn_planner.call_openai_json_result")
    @patch("app.agents.v2_turn_planner.classify_hotel_scope")
    def test_greeting_with_concrete_maintenance_request_starts_service_not_menu(self, classify, planner):
        request = menu_request()
        text = "Hello, the shower is leaking"
        request.conversation.recentMessages[0].text = text
        classify.return_value = decision("SERVICE_REQUEST", text, "MAINTENANCE"), OpenAiTokenUsage()
        response = plan_v2_turn(request)
        self.assertEqual([], response.messages)
        self.assertEqual("START_SERVICE", response.toolCalls[0].toolName)
        self.assertEqual(text, response.toolCalls[0].arguments["input"]["issue"])
        planner.assert_not_called()
        self.opening.assert_not_called()

    @patch("app.services.localized_content.call_openai_json_result")
    def test_english_greeting_translates_all_five_options_with_stable_ids(self, call):
        call.return_value = model_result(["Room service", "Maintenance", "Spa reservations", "Contact reception"])
        request = menu_request()
        response = plan_v2_turn(request)
        message = response.messages[0]
        self.assertTrue(message.text.startswith("Hello"))
        self.assertEqual("en", message.language)
        self.assertEqual("Hotel services", message.interaction.title)
        self.assertEqual("View services", message.interaction.buttonText)
        self.assertEqual(["Hotel questions", "Room service", "Maintenance", "Spa reservations", "Contact reception"],
                         [option.label for option in message.interaction.options])
        self.assertEqual([f"offering:{o.offeringCode}" for o in request.availableOfferings],
                         [option.id for option in message.interaction.options])
        self.assertEqual([], response.toolCalls)
        self.assertNotIn("LOCALIZATION_FALLBACK", response.warnings)
        self.assertTrue(all(item["maxLength"] == 24 for item in translation_inputs(call)))
        fields = call.call_args.kwargs["response_schema"]["properties"]["texts"]["properties"]
        self.assertTrue(all(field["maxLength"] == 24 for field in fields.values()))
        self.assertTrue(call.call_args.kwargs["strict_schema"])
        self.opening.assert_not_called()

    @patch("app.services.localized_content.call_openai_json_result")
    def test_one_invalid_label_does_not_discard_other_translations_or_poison_cache(self, call):
        call.return_value = model_result(["A room service label that is too long", "Maintenance",
                                          "Spa reservations", "Contact reception"])
        response = plan_v2_turn(menu_request())
        self.assertIn("LOCALIZATION_FALLBACK", response.warnings)
        self.assertEqual("mul", response.messages[0].language)
        self.assertEqual("Hotel questions", response.messages[0].interaction.options[0].label)
        self.assertEqual("Maintenance", response.messages[0].interaction.options[2].label)
        self.assertEqual(3, len(_cache))
        call.return_value = model_result(["Room service"])
        response = plan_v2_turn(menu_request())
        self.assertEqual(1, len(translation_inputs(call)))
        self.assertEqual("Room service", response.messages[0].interaction.options[1].label)
        self.assertNotIn("LOCALIZATION_FALLBACK", response.warnings)

    @patch("app.services.localized_content.call_openai_json_result")
    def test_full_custom_offering_name_is_translated_before_channel_shortening(self, call):
        request = menu_request()
        request.availableOfferings = [request.availableOfferings[-1]]
        request.availableOfferings[0].name = "Asistencia personalizada de recepci\u00f3n"
        call.return_value = model_result(["Reception help"])
        response = plan_v2_turn(request)
        self.assertEqual("BUTTONS", response.messages[0].interaction.type)
        self.assertEqual("Reception help", response.messages[0].interaction.options[0].label)
        self.assertEqual(request.availableOfferings[0].name, translation_inputs(call)[0]["text"])
        self.assertEqual(20, translation_inputs(call)[0]["maxLength"])

    @patch("app.services.localized_content.call_openai_json_result")
    def test_approved_translation_matches_full_name_and_overrides_default_faq_label(self, call):
        request = menu_request()
        request.availableOfferings = request.availableOfferings[:1]
        offering = request.availableOfferings[0]
        offering.inputSchema["x-chatbotinn-localization"] = {
            "variants": {"en": {offering.name: "Hotel information"}}}
        response = plan_v2_turn(request)
        self.assertEqual("Hotel information", response.messages[0].interaction.options[0].label)
        call.assert_not_called()

    @patch("app.services.localized_content.call_openai_json_result")
    def test_spanish_menu_stays_spanish_without_extra_translation(self, call):
        request = menu_request("es-MX")
        response = plan_v2_turn(request)
        self.assertEqual([o.name for o in request.availableOfferings],
                         [o.label for o in response.messages[0].interaction.options])
        call.assert_not_called()

    @patch("app.services.localized_content.call_openai_json_result")
    def test_french_menu_localizes_labels_and_deduplicates_body_with_strictest_limit(self, call):
        request = menu_request("fr")
        call.return_value = model_result(["Bonjour [[P0]]. Comment pouvons-nous vous aider ?",
            "Services de l'h\u00f4tel", "Voir les services", "Questions fr\u00e9quentes",
            "Service en chambre", "Maintenance", "R\u00e9server au spa", "Contacter l'accueil"])
        response = plan_v2_turn(request)
        self.assertEqual("fr", response.messages[0].language)
        self.assertEqual(response.messages[0].text, response.messages[0].interaction.body)
        self.assertEqual(1024, translation_inputs(call)[0]["maxLength"])
        self.assertTrue(all(item["maxLength"] == 24 for item in translation_inputs(call)[3:]))
        self.assertNotIn("LOCALIZATION_FALLBACK", response.warnings)

    @patch("app.services.localized_content.call_openai_json_result")
    def test_cache_separates_long_body_translation_from_short_label(self, call):
        call.return_value = model_result(["Frequently Asked Questions"])
        translate_batch(["Preguntas frecuentes"], "en", max_lengths=[1024])
        call.return_value = model_result(["Hotel questions"])
        texts, _ = translate_batch(["Preguntas frecuentes"], "en", max_lengths=[24])
        self.assertEqual(["Hotel questions"], texts)
        self.assertEqual(2, call.call_count)

    @patch("app.services.localized_content.call_openai_json_result")
    def test_limits_count_restored_parameters_and_do_not_cache_invalid_values(self, call):
        call.return_value = model_result(["Visit [[P0]]"])
        with self.assertRaisesRegex(ValueError, "channel limits"):
            translate_batch(["Visitar https://hotel.example/menu"], "en", max_lengths=[20])
        self.assertEqual(0, len(_cache))

    @patch("app.services.localized_content.call_openai_json_result")
    def test_cache_accounts_for_different_protected_value_lengths(self, call):
        call.return_value = model_result(["Hi [[P0]]"])
        translate_batch(["Hola Ana"], "en", ["Ana"], max_lengths=[10])
        with self.assertRaisesRegex(ValueError, "channel limits"):
            translate_batch(["Hola Alexandra"], "en", ["Alexandra"], max_lengths=[10])
        self.assertEqual(2, call.call_count)

    @patch("app.services.localized_content.call_openai_json_result")
    def test_invalid_limits_fail_before_model_call(self, call):
        for limits in [[], [0], [True], [20001], [24, 24]]:
            with self.assertRaises(ValueError):
                translate_batch(["Menu"], "en", max_lengths=limits)
        call.assert_not_called()


@unittest.skipUnless(os.getenv("RUN_LIVE_MULTILINGUAL_EVALS") == "1", "Opt-in real model evaluation")
class LiveMenuLocalizationTest(unittest.TestCase):
    def test_typo_after_previous_welcome_still_returns_menu(self):
        from uuid import uuid4
        _cache.clear()
        request = menu_request()
        latest = request.conversation.recentMessages[0]
        latest.text = "Hellow"
        previous = latest.model_copy(update={"messageId": uuid4(), "direction": "OUTBOUND", "actor": "ASSISTANT",
            "text": "Hello. How can we help you today? Please choose an option from the menu."})
        request.conversation.recentMessages.insert(0, previous)
        request = AgentTurnRequest.model_validate_json(request.model_dump_json())
        response = plan_v2_turn(request)
        self.assertNotIn("LOCALIZATION_FALLBACK", response.warnings)
        self.assertEqual("en", response.messages[0].language)
        self.assertEqual(5, len(response.messages[0].interaction.options))
        self.assertEqual("Hotel questions", response.messages[0].interaction.options[0].label)
        self.assertEqual([], response.toolCalls)

    def test_scope_distinguishes_openings_from_specific_requests_and_social_closings(self):
        for text, expected, is_opening in [
            ("Hi there", "SOCIAL", True), ("Good evening, how are you?", "SOCIAL", True),
            ("Thanks", "SOCIAL", False), ("Goodbye", "SOCIAL", False),
            ("Hello, the shower is leaking", "SERVICE_REQUEST", False),
            ("Hi, when does the pool close?", "HOTEL_QUESTION", False),
        ]:
            with self.subTest(text=text):
                request = menu_request()
                latest = request.conversation.recentMessages[0]
                latest.text = text
                scope, _ = classify_hotel_scope(request, latest, {})
                self.assertEqual(expected, scope.kind)
                opening, _ = classify_social_opening(text)
                self.assertEqual(is_opening, opening)

    def test_initial_english_menu_from_spanish_offerings(self):
        _cache.clear()
        response = plan_v2_turn(menu_request())
        self.assertNotIn("LOCALIZATION_FALLBACK", response.warnings)
        self.assertEqual("en", response.messages[0].language)
        self.assertEqual("Hotel questions", response.messages[0].interaction.options[0].label)
        for option, offering in zip(response.messages[0].interaction.options, menu_request().availableOfferings):
            self.assertLessEqual(len(option.label), 24)
            self.assertNotEqual(offering.name, option.label)
            self.assertEqual(f"offering:{offering.offeringCode}", option.id)
        self.assertEqual([], response.toolCalls)

    def test_translation_respects_small_limits_for_custom_labels(self):
        _cache.clear()
        labels, _ = translate_batch(["Preguntas frecuentes", "Asistencia personalizada de recepci\u00f3n"],
                                    "en", max_lengths=[20, 20])
        self.assertTrue(all(len(label) <= 20 for label in labels))
        self.assertEqual(2, len(labels))
