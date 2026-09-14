import json
import time
import unittest
from string import Formatter
from unittest.mock import patch
from uuid import UUID, uuid4

from app.agents.v2_turn_planner import plan_v2_turn
from app.core.errors import AgentTimeoutError
from app.schemas.v2_turns import AgentTurnResponse
from app.services.conversation_language import resolve_language, normalize_locale
from app.services.localized_content import REGISTRY, _cache, template, translate_batch, localize_response
from app.services.openai_client import OpenAiJsonResult
from app.services.telemetry_client import OpenAiTokenUsage
from app.prompts.v2_turn import build_v2_turn_prompt
from tests.test_v2_scope_router import request_for, decision, maintenance_offering
from tests.test_v2_turn_endpoint import MESSAGE_ID


def multilingual_request(text="Hola"):
    request = request_for(text)
    request.trigger.eventPayload["languageContext"] = {"version": 1, "explicit": False}
    return request


def model_result(texts):
    return OpenAiJsonResult({"texts": texts}, OpenAiTokenUsage(input_tokens=4, output_tokens=3, total_tokens=7), "translation")


def response_for(request, text="Update for RS-123: https://hotel.example/menu"):
    return AgentTurnResponse(schemaVersion="2.0", agentTurnId=request.agentTurnId,
        disposition="RESPONSE_READY", messages=[{
            "messageDraftId": str(uuid4()), "purpose": "STATUS_UPDATE", "text": text,
            "language": "en", "operationIds": [str(uuid4())], "conversationTaskIds": [],
            "interaction": {"type": "BUTTONS", "title": "Order", "body": text,
                            "options": [{"id": "task:123:CONFIRM", "label": "Confirm"}]},
        }], toolCalls=[], usage={"model": "test", "inputTokens": 0, "cachedInputTokens": 0,
            "outputTokens": 0, "reasoningTokens": 0, "totalTokens": 0}, warnings=[])


class LanguagePolicyTest(unittest.TestCase):
    def test_planner_uses_runtime_locale_not_language_of_latest_reply(self):
        request = multilingual_request("sin queso")
        request.guest.preferredLanguage = "en"
        request.trigger.eventPayload["languageContext"]["explicit"] = True
        prompt = build_v2_turn_prompt(request)
        self.assertIn("guest.preferredLanguage as the effective output locale", prompt)
        self.assertNotIn("language consistent with the guest's latest message", prompt)
        self.assertIn('"preferredLanguage":"en"', prompt)
        self.assertIn("sin queso", prompt)

    @patch("app.services.localized_content.call_openai_json_result")
    @patch("app.agents.v2_turn_planner.classify_hotel_scope")
    def test_switch_to_english_preserves_draft_without_restarting_menu(self, classify, translate):
        request = multilingual_request("Please speak English")
        request.conversation.summary = '{"pendingOffering":"ROOM_SERVICE","capturedFields":{"deliveryLocation":"ROOM"}}'
        route = decision("SOCIAL", "Please speak English")
        route.requestedLanguage, route.languageConfidence, route.languageChangeOnly = "en", 1, True
        classify.return_value = route, OpenAiTokenUsage()
        response = plan_v2_turn(request)
        self.assertEqual(request.conversation.summary, response.updatedConversationSummary)
        self.assertEqual("en", response.languageDecision.locale)
        self.assertIn("previously recorded details have not changed", response.messages[0].text)
        self.assertNotIn("How else", response.messages[0].text)
        self.assertIsNone(response.messages[0].interaction)
        self.assertEqual([], response.toolCalls)
        translate.assert_not_called()

    def test_openapi_accepts_new_and_legacy_response_envelopes(self):
        from pathlib import Path
        from jsonschema import Draft202012Validator
        from app.schemas.v2_turns import LanguageDecision
        contract = json.loads(Path("contracts/v2/agent-runtime.openapi.json").read_text(encoding="utf-8"))
        validator = Draft202012Validator({"$ref": "#/components/schemas/AgentTurnResponse",
                                          "components": contract["components"]})
        response = response_for(multilingual_request())
        legacy = response.model_dump(mode="json")
        legacy.pop("languageDecision")
        validator.validate(legacy)
        response.languageDecision = LanguageDecision(locale="fr", source="EXPLICIT", confidence=1,
                                                     messageId=UUID(MESSAGE_ID))
        validator.validate(response.model_dump(mode="json"))

    def test_greetings_override_inherited_language_without_mutating_request(self):
        for greeting, locale in [("Hello", "en"), ("Bonjour", "fr"), ("你好", "zh"),
                                 ("こんにちは", "ja"), ("مرحبا", "ar"), ("Hallo", "de")]:
            request = multilingual_request(greeting)
            updated, language = resolve_language(request, request.conversation.recentMessages[0])
            self.assertEqual(locale, language.locale)
            self.assertEqual(UUID(MESSAGE_ID), language.messageId)
            self.assertEqual("es-MX", request.guest.preferredLanguage)
            self.assertEqual(request.conversation, updated.conversation)

    def test_button_ok_number_and_emoji_inherit(self):
        for text in ["OK", "1", "👍", "Spa"]:
            request = multilingual_request(text)
            request.guest.preferredLanguage = "fr-CA"
            route = decision("CONTEXT_REPLY", text)
            route.detectedLanguage, route.languageConfidence = "en", 0.99
            updated, language = resolve_language(request, request.conversation.recentMessages[0], route)
            self.assertIsNone(language)
            self.assertEqual("fr-CA", updated.guest.preferredLanguage)
        request.conversation.recentMessages[0].interactionReplyId = "offering:SPA"
        updated, language = resolve_language(request, request.conversation.recentMessages[0], route)
        self.assertIsNone(language)

    def test_explicit_preference_is_sticky_but_can_be_changed(self):
        request = multilingual_request("Please speak French")
        request.trigger.eventPayload["languageContext"]["explicit"] = True
        request.guest.preferredLanguage = "de"
        route = decision("SOCIAL", "Please speak French")
        route.detectedLanguage, route.languageConfidence = "en", 0.99
        self.assertIsNone(resolve_language(request, request.conversation.recentMessages[0], route)[1])
        route.requestedLanguage = "fr"
        updated, language = resolve_language(request, request.conversation.recentMessages[0], route)
        self.assertEqual("fr", updated.guest.preferredLanguage)
        self.assertEqual("EXPLICIT", language.source)

    def test_tool_results_and_low_confidence_never_detect_again(self):
        request = multilingual_request("The shower is broken")
        route = decision("SERVICE_REQUEST", request.conversation.recentMessages[0].text, "MAINTENANCE")
        route.detectedLanguage, route.languageConfidence = "en", 0.6
        self.assertIsNone(resolve_language(request, request.conversation.recentMessages[0], route)[1])
        request.trigger.type = "TOOL_RESULTS"
        route.languageConfidence = 1
        self.assertIsNone(resolve_language(request, request.conversation.recentMessages[0], route)[1])

    def test_locale_normalization_and_invalid_tags(self):
        self.assertEqual("zh-Hant-TW", normalize_locale("ZH-hant-tw"))
        for invalid in ["English", "en_US", "und", "en; DROP", "e", None]:
            self.assertIsNone(normalize_locale(invalid))

    @patch("app.agents.v2_turn_planner.classify_hotel_scope")
    def test_direct_maintenance_keeps_evidence_and_business_input(self, classify):
        request = multilingual_request("The shower is leaking")
        from app.schemas.v2_turns import OfferingCapability
        request.availableOfferings.append(OfferingCapability.model_validate(maintenance_offering()))
        route = decision("SERVICE_REQUEST", request.conversation.recentMessages[0].text, "MAINTENANCE")
        route.detectedLanguage, route.languageConfidence = "en", 0.99
        classify.return_value = route, OpenAiTokenUsage()
        result = plan_v2_turn(request)
        self.assertEqual("en", result.languageDecision.locale)
        self.assertEqual("START_SERVICE", result.toolCalls[0].toolName)
        self.assertEqual("The shower is leaking", result.toolCalls[0].arguments["input"]["issue"])
        self.assertEqual([UUID(MESSAGE_ID)], result.toolCalls[0].evidenceMessageIds)
        self.assertEqual([], result.messages)

    @patch("app.services.localized_content.call_openai_json_result", side_effect=AgentTimeoutError("timeout"))
    @patch("app.agents.v2_turn_planner.classify_hotel_scope")
    def test_language_only_request_keeps_pending_order_without_tools(self, classify, translate):
        request = multilingual_request("Please speak French")
        request.conversation.summary = '{"pendingOffering":"ROOM_SERVICE","capturedFields":{"items":[]}}'
        route = decision("SOCIAL", "Please speak French")
        route.requestedLanguage, route.languageConfidence, route.languageChangeOnly = "fr", 1, True
        classify.return_value = route, OpenAiTokenUsage()
        result = plan_v2_turn(request)
        self.assertEqual([], result.toolCalls)
        self.assertEqual(request.conversation.summary, result.updatedConversationSummary)
        self.assertEqual("EXPLICIT", result.languageDecision.source)
        self.assertIn("LOCALIZATION_FALLBACK", result.warnings)


class LocalizedContentTest(unittest.TestCase):
    def setUp(self):
        _cache.clear()

    @patch("app.services.localized_content.call_openai_json_result")
    def test_optional_operation_reference_cannot_crash_localization(self, call):
        call.return_value = model_result(["Bonjour [[P0]]"])
        texts, _ = translate_batch(["Hello Ana"], "fr", [None, "", "Ana"])
        self.assertEqual(["Bonjour Ana"], texts)

    def test_all_template_variants_use_identical_placeholders(self):
        for key, variants in REGISTRY["templates"].items():
            expected = {name for _, name, _, _ in Formatter().parse(variants["en"]) if name}
            for locale, text in variants.items():
                self.assertEqual(expected, {name for _, name, _, _ in Formatter().parse(text) if name}, (key, locale))
        with self.assertRaises(ValueError):
            template("service.started", "es", reference="M-1")

    @patch("app.services.localized_content.call_openai_json_result")
    def test_translation_cache_protects_parameters_and_invalidates_on_source_change(self, call):
        call.return_value = model_result(["Bonjour [[P0]], référence [[P1]]. Menu : [[P2]]"])
        first, usage = translate_batch(["Hello Ana, reference MA-123. Menu: https://hotel.test/menu"], "fr",
                                      ["Ana", "MA-123"])
        second, cached_usage = translate_batch(["Hello Bob, reference MA-456. Menu: https://other.test/menu"], "fr",
                                               ["Bob", "MA-456"])
        self.assertIn("Ana", first[0])
        self.assertIn("MA-456", second[0])
        self.assertIn("https://other.test/menu", second[0])
        self.assertIsNone(cached_usage)
        self.assertEqual(7, usage.total_tokens)
        call.assert_called_once()
        call.return_value = model_result(["Nouveau texte"])
        translate_batch(["New source"], "fr")
        self.assertEqual(2, call.call_count)

    @patch("app.services.localized_content.call_openai_json_result")
    def test_rejects_changed_links_numbers_and_duplicate_parameters(self, call):
        for invalid in ["Missing parameter", "[[P0]] [[P0]]", "[[P0]] 999", "[[P0]] https://evil.test"]:
            call.return_value = model_result([invalid])
            with self.assertRaises(ValueError):
                translate_batch(["Request 123"], "fr")
        self.assertEqual(0, len(_cache))

    @patch("app.services.localized_content.call_openai_json_result")
    def test_response_localization_never_changes_action_ids_or_operation_links(self, call):
        request = multilingual_request()
        request.guest.preferredLanguage = "fr"
        source = response_for(request)
        call.return_value = model_result(["Mise à jour RS-[[P0]] : [[P1]]", "Commande", "Confirmer"])
        result = localize_response(request, source, time.perf_counter())
        self.assertEqual("fr", result.messages[0].language)
        self.assertEqual("task:123:CONFIRM", result.messages[0].interaction.options[0].id)
        self.assertEqual(source.messages[0].operationIds, result.messages[0].operationIds)
        self.assertEqual(source.messages[0].messageDraftId, result.messages[0].messageDraftId)
        self.assertIn("RS-123", result.messages[0].text)
        self.assertIn("https://hotel.example/menu", result.messages[0].text)
        self.assertEqual("Confirm", source.messages[0].interaction.options[0].label)

    @patch("app.services.localized_content.call_openai_json_result", side_effect=AgentTimeoutError("timeout"))
    def test_translation_timeout_keeps_valid_response_without_retrying_tools(self, call):
        request = multilingual_request()
        request.guest.preferredLanguage = "fr"
        source = response_for(request)
        result = localize_response(request, source, time.perf_counter())
        self.assertEqual(source.messages[0].text, result.messages[0].text)
        self.assertEqual(source.toolCalls, result.toolCalls)
        self.assertIn("LOCALIZATION_FALLBACK", result.warnings)
        self.assertEqual("mul", result.messages[0].language)
        call.assert_called_once()
        self.assertLessEqual(call.call_args.kwargs["timeout_seconds"], 4)

    @patch("app.services.localized_content.call_openai_json_result")
    def test_spanish_has_no_added_model_call(self, call):
        request = multilingual_request()
        source = response_for(request, "Solicitud registrada")
        result = localize_response(request, source, time.perf_counter())
        self.assertEqual(source.messages[0].text, result.messages[0].text)
        self.assertEqual("Confirmar", result.messages[0].interaction.options[0].label)
        call.assert_not_called()

    @patch("app.services.localized_content.call_openai_json_result")
    def test_reviewed_english_templates_do_not_need_translation(self, call):
        request = multilingual_request()
        request.guest.preferredLanguage = "en"
        source = response_for(request, template("greeting.named", "en", name="Sebastian"))
        source.messages[0].interaction = None
        result = localize_response(request, source, time.perf_counter())
        self.assertEqual(source, result)
        call.assert_not_called()

    @patch("app.services.localized_content.call_openai_json_result")
    def test_approved_catalog_text_is_used_without_retranslation(self, call):
        request = multilingual_request()
        request.guest.preferredLanguage = "fr-CA"
        request.availableOfferings[0].inputSchema["x-chatbotinn-localization"] = {
            "version": 1, "variants": {"fr": {"Consulta el menú": "Consultez le menu"}}}
        source = response_for(request, "Consulta el menú\nhttps://hotel.example/menu")
        source.messages[0].interaction = None
        result = localize_response(request, source, time.perf_counter())
        self.assertEqual("Consultez le menu\nhttps://hotel.example/menu", result.messages[0].text)
        call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
