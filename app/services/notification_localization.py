"""Render already-approved notifications. No planning, tools or process mutations."""
from app.core.latency import timed
import json
import re
from pathlib import Path
from string import Formatter

from app.core.errors import AgentModelError
from app.schemas.localization import LocalizationRequest, LocalizationResponse
from app.services.localized_content import _known_text, translate_batch


REGISTRY = json.loads(Path(__file__).with_name("process_message_templates.json").read_text(encoding="utf-8"))


def reviewed_notification(text, locale):
    language = locale.split("-")[0]
    if language == "es" and text == "No, sigue sin resolver":
        return "No, sigue pendiente"
    for variants in REGISTRY["templates"].values():
        target = variants.get(locale, variants.get(language))
        if target is None:
            continue
        for source in variants.values():
            parts = list(Formatter().parse(source))
            names = [name for _, name, _, _ in parts if name is not None]
            def placeholder(name):
                if name == "url":
                    return r"(https?://\S+)"
                if name == "reference":
                    return r"(REQ-[A-Z0-9-]+)"
                return r"(.+?)" if name else ""

            pattern = "".join(re.escape(literal) + placeholder(name) for literal, name, _, _ in parts)
            match = re.fullmatch(pattern, text, re.DOTALL)
            if match:
                return target.format(**dict(zip(names, match.groups())))
    return _known_text(text, locale, {})


@timed("agent.localize_notification")
def localize_notification(request: LocalizationRequest) -> LocalizationResponse:
    texts = [reviewed_notification(item.text, request.locale) for item in request.texts]
    pending = [index for index, text in enumerate(texts)
               if text is None or len(text) > request.texts[index].maxLength]
    if pending:
        translated, _ = translate_batch(
            [request.texts[index].text for index in pending], request.locale,
            request.protectedValues, namespace=f"outbound:{REGISTRY['version']}:{request.namespace}",
            timeout=4.0, max_lengths=[request.texts[index].maxLength for index in pending],
        )
        for index, text in zip(pending, translated):
            texts[index] = text
    if any(not text or len(text) > item.maxLength for text, item in zip(texts, request.texts)):
        raise AgentModelError("Localized notification exceeds channel limits")
    return LocalizationResponse(locale=request.locale, texts=texts)
