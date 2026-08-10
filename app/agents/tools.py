from typing import Any

from app.schemas.tasks import AgentTaskRequest, AgentToolName
from app.services.chatbotinn_client import (
    get_offering_catalogs,
    get_service_offering,
    list_service_offerings,
)


def resolve_authorized_resources(request: AgentTaskRequest) -> dict[str, Any]:
    allowed_tools = set(request.toolPolicy.allowedTools)
    offering = request.offering or {}
    allowed_offerings = list(request.allowedOfferings)

    if (
        not allowed_offerings
        and AgentToolName.LIST_AVAILABLE_OFFERINGS in allowed_tools
    ):
        allowed_offerings = list_service_offerings()

    if (
        not offering
        and request.offeringCode
        and AgentToolName.GET_OFFERING_DEFINITION in allowed_tools
    ):
        offering = get_service_offering(request.offeringCode)

    offering_code = request.offeringCode or _offering_code(offering)
    catalogs: list[dict[str, Any]] = []
    if (
        offering_code
        and AgentToolName.GET_OFFERING_CATALOGS in allowed_tools
    ):
        catalogs = _filter_catalogs(
            get_offering_catalogs(offering_code),
            request.toolPolicy.allowedCatalogIds,
            request.toolPolicy.allowedCatalogCodes,
        )

    query = (request.latestMessage or request.staffMessage or "").strip()
    catalog_matches: list[dict[str, Any]] = []
    if AgentToolName.SEARCH_CATALOG_ITEMS in allowed_tools:
        catalog_matches = _search_catalog_items(
            catalogs,
            query,
            request.toolPolicy.maxCatalogResults,
            faq_only=False,
        )

    knowledge_matches: list[dict[str, Any]] = []
    if AgentToolName.SEARCH_KNOWLEDGE in allowed_tools:
        knowledge_matches = _search_catalog_items(
            catalogs,
            query,
            request.toolPolicy.maxCatalogResults,
            faq_only=True,
        )

    return {
        "offering": offering or None,
        "allowedOfferings": allowed_offerings,
        "catalogMatches": catalog_matches,
        "knowledgeMatches": knowledge_matches,
    }


def _offering_code(offering: dict[str, Any]) -> str | None:
    summary = offering.get("offering") if isinstance(offering, dict) else None
    if isinstance(summary, dict):
        return summary.get("code")
    return offering.get("code") if isinstance(offering, dict) else None


def _filter_catalogs(
    catalogs: list[dict[str, Any]],
    allowed_ids: list[str],
    allowed_codes: list[str],
) -> list[dict[str, Any]]:
    id_filter = set(allowed_ids)
    code_filter = set(allowed_codes)
    if not id_filter and not code_filter:
        return catalogs

    filtered = []
    for payload in catalogs:
        catalog = payload.get("catalog", {}).get("catalog", {})
        if catalog.get("id") in id_filter or catalog.get("code") in code_filter:
            filtered.append(payload)
    return filtered


def _search_catalog_items(
    catalogs: list[dict[str, Any]],
    query: str,
    limit: int,
    *,
    faq_only: bool,
) -> list[dict[str, Any]]:
    terms = {part.casefold() for part in query.split() if len(part) >= 2}
    matches: list[tuple[int, dict[str, Any]]] = []

    for payload in catalogs:
        detail = payload.get("catalog", {})
        catalog_summary = detail.get("catalog", {})
        for item_detail in detail.get("items", []):
            if faq_only and not item_detail.get("faqConfiguration"):
                continue
            item = item_detail.get("item", {})
            searchable = " ".join(
                str(value or "")
                for value in (
                    item.get("code"),
                    item.get("name"),
                    item.get("description"),
                    item.get("tags"),
                    item_detail.get("faqConfiguration"),
                )
            ).casefold()
            score = sum(1 for term in terms if term in searchable)
            if terms and score == 0:
                continue
            matches.append(
                (
                    score,
                    {
                        "catalog": {
                            "id": catalog_summary.get("id"),
                            "code": catalog_summary.get("code"),
                            "name": catalog_summary.get("name"),
                        },
                        "item": item_detail,
                    },
                )
            )

    matches.sort(key=lambda entry: entry[0], reverse=True)
    return [entry[1] for entry in matches[:limit]]
