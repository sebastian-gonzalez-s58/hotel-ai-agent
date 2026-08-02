from typing import Any

from pydantic import ValidationError

from app.agents.tools import resolve_authorized_resources
from app.core.errors import AgentModelError
from app.prompts.capabilities import capability_prompt
from app.schemas.tasks import AgentTaskRequest, AgentTaskResponse, AgentTaskType
from app.services.openai_client import call_openai_json_result


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
    _apply_guardrails(payload, request, resources)

    try:
        return AgentTaskResponse.model_validate(payload)
    except ValidationError as exc:
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
