"""Resolve a pending catalog choice without changing the guest's original evidence."""
from app.core.latency import timed
from dataclasses import dataclass
import re
import unicodedata

from app.services.conversation_language import language_enabled


def ordered_capture_fields(offering):
    schema = offering.inputSchema
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    fields = []
    for position, code in enumerate(required):
        field = properties.get(code)
        if not isinstance(field, dict) or str(field.get("x-source") or "GUEST").upper() == "STAY":
            continue
        capture = field.get("x-chatbotinn-capture")
        if not isinstance(capture, dict):
            continue
        order = capture.get("displayOrder")
        fields.append((order if isinstance(order, int) else position, position, code, field))
    fields.sort(key=lambda item: (item[0], item[1]))
    return [(code, field) for _, _, code, field in fields]


def _normalized_label(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.split())


@dataclass
class PendingCatalogSelection:
    offering: object
    field_code: str
    field_schema: dict
    options: list
    current_value: str | None = None

    def context(self):
        context = {"offeringCode": self.offering.offeringCode, "fieldCode": self.field_code,
                "title": self.field_schema.get("title", self.field_code), "options": self.options}
        if self.current_value is not None:
            context["currentValue"] = self.current_value
        return context

    def reply_id(self, code):
        return f"field:{self.offering.offeringCode}:{self.field_code}:{code}"

    def _exact_matches(self, text):
        normalized = _normalized_label(text)
        matches = set()
        for option in self.options:
            labels = [option["label"], *option["translations"]]
            # Codes are opaque identities, but spaces/underscores are interchangeable in typed input.
            code_label = _normalized_label(option["code"].replace("_", " "))
            if normalized == code_label or normalized == _normalized_label(option["code"]) or any(
                    normalized == _normalized_label(label) for label in labels):
                matches.add(option["code"])
        return matches

    def exact_code(self, text):
        matches = self._exact_matches(text)
        return next(iter(matches)) if len(matches) == 1 else None

    def semantic_code(self, scope, text):
        if (scope.kind != "CONTEXT_REPLY" or scope.containsUnrelatedTopic or scope.separateRequest
                or scope.replyAction != "NONE" or scope.selectionConfidence < 0.9
                or scope.selectionEvidence != text or scope.offeringCode not in {None, self.offering.offeringCode}
                or len(self._exact_matches(text)) > 1):
            return None
        option = next((o for o in self.options if o["code"] == scope.selectionCode), None)
        if option is None:
            return None
        numbers = re.findall(r"\d+", unicodedata.normalize("NFKC", text))
        target_numbers = re.findall(r"\d+", option["code"] + " " + option["label"])
        if numbers and target_numbers and set(map(int, numbers)) != set(map(int, target_numbers)):
            return None
        return option["code"]


@timed("catalog.pending_selection")
def pending_catalog_selection(request, state):
    if (request.trigger.type != "INBOUND_MESSAGE" or request.previousToolResults
            or state.get("readyToStart")
            or state.get("phase") == "STARTING" or request.trigger.conversationTaskId):
        return None
    message = next((m for m in request.conversation.recentMessages if m.messageId == request.trigger.messageId), None)
    if message is not None and message.conversationTaskIds:
        return None
    offering = next((o for o in request.availableOfferings if o.offeringCode == state.get("pendingOffering")), None)
    if offering is None:
        return None
    captured = state.get("capturedFields")
    captured = captured if isinstance(captured, dict) else {}
    # BC-003: a saved location remains editable until the draft starts.
    # Keep this change bounded to room-service drafts; generic capture is unchanged.
    editable_location = (language_enabled(request) and offering.offeringCode == "ROOM_SERVICE"
                         and isinstance(captured.get("deliveryLocation"), str)
                         and bool(captured["deliveryLocation"]))
    if state.get("awaitingExplicitConfirmation") and not editable_location:
        return None
    localization = offering.inputSchema.get("x-chatbotinn-localization")
    variants = localization.get("variants", {}) if isinstance(localization, dict) else {}
    variants = variants if isinstance(variants, dict) else {}
    for code, field in ordered_capture_fields(offering):
        editing = editable_location and code == "deliveryLocation"
        if captured.get(code) not in (None, "", [], {}) and not editing:
            continue
        capture = field["x-chatbotinn-capture"]
        if str(capture.get("inputMode", "")).upper() != "SINGLE_SELECT":
            return None
        catalog = capture.get("catalog")
        raw = catalog.get("options", []) if isinstance(catalog, dict) else []
        options = []
        for option in raw if isinstance(raw, list) else []:
            if not isinstance(option, dict) or not all(isinstance(option.get(k), str) and option[k].strip()
                                                       for k in ("code", "label")):
                continue
            translations = [v[option["label"]] for v in variants.values()
                            if isinstance(v, dict) and isinstance(v.get(option["label"]), str)]
            options.append({"code": option["code"], "label": option["label"], "translations": translations})
        return PendingCatalogSelection(offering, code, field, options,
                                       captured[code] if editing else None) if options else None
    return None
