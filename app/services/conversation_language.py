"""Session language policy. Message text and business identifiers remain untouched."""
import re

from app.schemas.v2_turns import AgentTurnRequest, LanguageDecision


GREETINGS = {
    "hola": "es", "buenos días": "es", "buenos dias": "es", "buenas tardes": "es",
    "buenas noches": "es", "hello": "en", "hi": "en", "good morning": "en",
    "good afternoon": "en", "good evening": "en", "bonjour": "fr", "bonsoir": "fr",
    "guten tag": "de", "guten morgen": "de", "hallo": "de", "olá": "pt",
    "bom dia": "pt", "boa tarde": "pt", "buongiorno": "it", "buonasera": "it",
    "こんにちは": "ja", "你好": "zh", "您好": "zh", "안녕하세요": "ko",
    "مرحبا": "ar", "مرحباً": "ar", "привет": "ru", "здравствуйте": "ru",
}


def normalize_locale(value: str | None) -> str | None:
    if not isinstance(value, str) or len(value) > 35:
        return None
    if not re.fullmatch(r"[a-zA-Z]{2,3}(?:-[a-zA-Z0-9]{2,8})*", value) or value.lower() == "und":
        return None
    parts = value.split("-")
    return "-".join([parts[0].lower()] + [
        part.title() if len(part) == 4 else part.upper() if len(part) == 2 else part.lower()
        for part in parts[1:]
    ])


def language_enabled(request: AgentTurnRequest) -> bool:
    metadata = request.trigger.eventPayload.get("languageContext", {})
    return isinstance(metadata, dict) and metadata.get("version") == 1


def greeting_language(text: str) -> str | None:
    return GREETINGS.get(" ".join(text.casefold().split()).strip("¡!¿?., "))


def resolve_language(request: AgentTurnRequest, message, scope=None):
    effective = normalize_locale(request.guest.preferredLanguage) or normalize_locale(request.hotel.defaultLanguage) or "en"
    decision = None
    if (language_enabled(request) and request.trigger.type == "INBOUND_MESSAGE"
            and not request.previousToolResults and message is not None and not message.interactionReplyId):
        explicit = normalize_locale(getattr(scope, "requestedLanguage", None))
        detected = normalize_locale(getattr(scope, "detectedLanguage", None)) or greeting_language(message.text)
        confidence = getattr(scope, "languageConfidence", 0) if scope else 1.0
        locked = request.trigger.eventPayload.get("languageContext", {}).get("explicit", False)
        candidate = explicit or (detected if not locked else None)
        # Short, ambiguous replies (OK, digits, emoji, names) inherit the existing language.
        meaningful = explicit or greeting_language(message.text) or len(re.findall(r"[^\W\d_]", message.text)) >= 8
        if candidate and meaningful and confidence >= 0.85:
            if not explicit and candidate == effective.split("-")[0]:
                candidate = effective
            decision = LanguageDecision(locale=candidate, source="EXPLICIT" if explicit else "DETECTED",
                                        confidence=confidence, messageId=message.messageId)
            effective = candidate
    localized = request.model_copy(deep=True)
    localized.guest.preferredLanguage = effective
    return localized, decision
