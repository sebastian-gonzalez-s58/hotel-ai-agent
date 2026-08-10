import logging
from typing import Any

from pydantic import ValidationError

from app.agents.tools import resolve_authorized_resources
from app.core.errors import AgentModelError
from app.prompts.capabilities import capability_prompt
from app.schemas.tasks import AgentTaskRequest, AgentTaskResponse, AgentTaskType
from app.services.openai_client import call_openai_json_result


logger = logging.getLogger("chatbotinn-agent.capabilities")
_VALID_STATUSES = {"COMPLETED", "NEEDS_CLARIFICATION", "NEEDS_HUMAN", "UNRESOLVED"}


def execute_agent_task(request: AgentTaskRequest) -> AgentTaskResponse:
    resources = resolve_authorized_resources(request)
    result = call_openai_json_result(
        capability_prompt(request, resources),
        purpose=request.taskType.value,
    )
    payload = dict(result.payload)
    payload["taskType"] = request.taskType.value
    payload["schemaVersion"] = request.schemaVersion
    payload["usage"] = result.usage.as_api_dict()
    _normalize_payload(payload)
    _apply_guardrails(payload, request, resources)

    try:
        return AgentTaskResponse.model_validate(payload)
    except ValidationError as exc:
        logger.warning(
            "Invalid normalized task response task_type=%s task_id=%s errors=%s",
            request.taskType.value,
            request.taskId,
            exc.errors(include_input=False),
        )
        raise AgentModelError("OpenAI returned an invalid agent task result") from exc


def _apply_guardrails(
    payload: dict[str, Any],
    request: AgentTaskRequest,
    resources: dict[str, Any],
) -> None:
    payload.setdefault("status", "COMPLETED")
    payload.setdefault("containsEmergency", False)
    payload.setdefault("extractedValues", [])
    payload.setdefault("catalogSelections", [])
    payload.setdefault("missingFieldCodes", [])
    payload.setdefault("complete", False)
    payload.setdefault("evidence", [])
    payload.setdefault("warnings", [])

    if request.taskType == AgentTaskType.CLASSIFY_OFFERING:
        _validate_selected_offering(payload, resources)
    if request.taskType == AgentTaskType.EXTRACT_REQUIREMENT_VALUES:
        _filter_unknown_required_fields(payload, request, resources)
    if request.taskType == AgentTaskType.MATCH_CATALOG_ITEMS:
        _filter_unknown_catalog_items(payload, resources)
    if (
        request.taskType == AgentTaskType.GENERATE_CLARIFICATION
        and request.taskConfig.get("offerActiveOfferings")
        and not request.offeringCode
    ):
        _apply_active_offerings_menu(payload, request, resources)


def _normalize_payload(payload: dict[str, Any]) -> None:
    status = str(payload.get("status") or "COMPLETED").strip().upper()
    payload["status"] = status if status in _VALID_STATUSES else "UNRESOLVED"
    payload["confidence"] = _normalized_confidence(payload.get("confidence"))
    payload["containsEmergency"] = _as_bool(payload.get("containsEmergency"))
    payload["complete"] = _as_bool(payload.get("complete"))
    payload["offeringCode"] = _optional_string(payload.get("offeringCode"))
    payload["decision"] = _optional_string(payload.get("decision"))
    payload["summary"] = _optional_string(payload.get("summary"))
    payload["message"] = _normalized_message(payload.get("message"))
    payload["extractedValues"] = _normalized_extracted_values(payload.get("extractedValues"))
    payload["catalogSelections"] = _normalized_catalog_selections(payload.get("catalogSelections"))
    payload["missingFieldCodes"] = _string_list(payload.get("missingFieldCodes"))
    payload["evidence"] = _normalized_evidence(payload.get("evidence"))
    payload["warnings"] = _string_list(payload.get("warnings"))


def _apply_active_offerings_menu(
    payload: dict[str, Any],
    request: AgentTaskRequest,
    resources: dict[str, Any],
) -> None:
    actions = []
    for offering in resources.get("allowedOfferings", []):
        code = _offering_code(offering)
        summary = offering.get("offering", offering) if isinstance(offering, dict) else {}
        name = summary.get("name") if isinstance(summary, dict) else None
        if code and name:
            actions.append({"id": str(code), "label": str(name)})

    if not actions:
        payload["status"] = "UNRESOLVED"
        payload["complete"] = False
        payload["warnings"].append("No active guest-visible offerings were available")
        return

    message = payload.get("message") or {}
    text = _optional_string(message.get("text")) or _menu_fallback_text(request)
    title = _optional_string(request.taskConfig.get("menuTitle")) or "Menu principal"
    button_text = _optional_string(request.taskConfig.get("menuButtonText")) or "Ver opciones"
    message["text"] = text
    message["interaction"] = {
        "type": "BUTTONS" if len(actions) <= 3 else "LIST",
        "title": title,
        "body": text,
        "buttonText": button_text,
        "actions": actions,
    }
    payload["message"] = message
    payload["status"] = "NEEDS_CLARIFICATION"
    payload["offeringCode"] = None
    payload["complete"] = False


def _menu_fallback_text(request: AgentTaskRequest) -> str:
    language = (request.context.language or "").casefold()
    latest_message = (request.latestMessage or "").casefold()
    looks_spanish = language.startswith("es") or any(
        word in latest_message for word in ("hola", "buenas", "ayuda", "servicio")
    )
    if looks_spanish:
        return "Hola, ¿en qué servicio del hotel puedo ayudarte hoy?"
    return "Hello, which hotel service can I help you with today?"


def _normalized_message(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        text = value.strip()
        return {"text": text, "interaction": None} if text else None
    if not isinstance(value, dict):
        return None
    text = _optional_string(value.get("text") or value.get("message"))
    if not text:
        return None
    return {"text": text, "interaction": _normalized_interaction(value.get("interaction"))}


def _normalized_interaction(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    raw_actions = value.get("actions") or value.get("options") or []
    actions = []
    if isinstance(raw_actions, list):
        for raw_action in raw_actions:
            if not isinstance(raw_action, dict):
                continue
            action_id = _optional_string(
                raw_action.get("id") or raw_action.get("value") or raw_action.get("code")
            )
            label = _optional_string(
                raw_action.get("label") or raw_action.get("title") or raw_action.get("name")
            )
            if action_id and label:
                actions.append({"id": action_id, "label": label})
    if not actions:
        return None
    return {
        "type": "BUTTONS" if len(actions) <= 3 else "LIST",
        "title": _optional_string(value.get("title")),
        "body": _optional_string(value.get("body")),
        "buttonText": _optional_string(value.get("buttonText")),
        "actions": actions,
    }


def _normalized_extracted_values(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        if not isinstance(item, dict):
            continue
        field_code = _optional_string(item.get("fieldCode"))
        if field_code:
            normalized.append(
                {
                    "fieldCode": field_code,
                    "value": item.get("value"),
                    "confidence": _normalized_confidence(item.get("confidence")),
                }
            )
    return normalized


def _normalized_catalog_selections(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        if not isinstance(item, dict):
            continue
        item_id = _optional_string(item.get("itemId"))
        if not item_id:
            continue
        quantity = item.get("quantity", 1)
        try:
            quantity = float(quantity)
        except (TypeError, ValueError):
            quantity = 1
        if quantity <= 0:
            quantity = 1
        normalized.append(
            {
                "catalogId": _optional_string(item.get("catalogId")),
                "catalogCode": _optional_string(item.get("catalogCode")),
                "itemId": item_id,
                "itemCode": _optional_string(item.get("itemCode")),
                "quantity": quantity,
                "optionIds": _string_list(item.get("optionIds")),
                "confidence": _normalized_confidence(item.get("confidence")),
            }
        )
    return normalized


def _normalized_evidence(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        if not isinstance(item, dict):
            continue
        evidence_type = _optional_string(item.get("type"))
        if evidence_type:
            normalized.append(
                {
                    "type": evidence_type,
                    "id": _optional_string(item.get("id")),
                    "code": _optional_string(item.get("code")),
                    "title": _optional_string(item.get("title")),
                }
            )
    return normalized


def _normalized_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "1", "yes", "si", "sí"}
    return bool(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [normalized for item in value if (normalized := _optional_string(item))]


def _validate_selected_offering(
    payload: dict[str, Any],
    resources: dict[str, Any],
) -> None:
    allowed_codes = {
        code
        for offering in resources.get("allowedOfferings", [])
        if (code := _offering_code(offering))
    }
    selected = payload.get("offeringCode")
    if selected and selected not in allowed_codes:
        payload["offeringCode"] = None
        payload["status"] = "UNRESOLVED"
        payload["complete"] = False
        payload["warnings"].append("Model selected an offering outside the allowed set")


def _filter_unknown_required_fields(
    payload: dict[str, Any],
    request: AgentTaskRequest,
    resources: dict[str, Any],
) -> None:
    offering = request.offering or resources.get("offering") or {}
    required_fields = offering.get("requiredFields", [])
    allowed_codes = {
        field.get("code")
        for field in required_fields
        if isinstance(field, dict) and field.get("code")
    }
    if not allowed_codes:
        return

    original = payload.get("extractedValues", [])
    payload["extractedValues"] = [
        value
        for value in original
        if isinstance(value, dict) and value.get("fieldCode") in allowed_codes
    ]
    if len(payload["extractedValues"]) != len(original):
        payload["warnings"].append("Unknown offering fields were discarded")


def _filter_unknown_catalog_items(
    payload: dict[str, Any],
    resources: dict[str, Any],
) -> None:
    allowed_ids = {
        str(item.get("item", {}).get("item", {}).get("id"))
        for item in resources.get("catalogMatches", [])
        if item.get("item", {}).get("item", {}).get("id")
    }
    original = payload.get("catalogSelections", [])
    payload["catalogSelections"] = [
        selection
        for selection in original
        if isinstance(selection, dict) and str(selection.get("itemId")) in allowed_ids
    ]
    if len(payload["catalogSelections"]) != len(original):
        payload["warnings"].append("Catalog selections outside authorized results were discarded")
        payload["complete"] = False


def _offering_code(offering: dict[str, Any]) -> str | None:
    summary = offering.get("offering") if isinstance(offering, dict) else None
    if isinstance(summary, dict):
        return summary.get("code")
    return offering.get("code") if isinstance(offering, dict) else None
