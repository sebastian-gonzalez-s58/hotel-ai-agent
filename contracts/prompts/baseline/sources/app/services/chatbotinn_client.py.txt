import logging
import time
from typing import Any
from urllib.parse import quote

import httpx

from app.core.config import settings


logger = logging.getLogger("chatbotinn-agent.backend")

_cache: dict[str, tuple[float, Any]] = {}


def list_service_offerings() -> list[dict[str, Any]]:
    payload = _cached_get("offerings", "/api/agent/service-offerings")
    return payload if isinstance(payload, list) else []


def get_service_offering(code: str) -> dict[str, Any]:
    payload = _cached_get(
        f"offering:{code}",
        f"/api/agent/service-offerings/{quote(code, safe='')}",
    )
    return payload if isinstance(payload, dict) else {}


def get_offering_catalogs(offering_code: str) -> list[dict[str, Any]]:
    payload = _cached_get(
        f"offering-catalogs:{offering_code}",
        f"/api/agent/catalogs/by-offering/{quote(offering_code, safe='')}",
    )
    return payload if isinstance(payload, list) else []


def get_catalog(code: str) -> dict[str, Any]:
    payload = _cached_get(
        f"catalog:{code}",
        f"/api/agent/catalogs/{quote(code, safe='')}",
    )
    return payload if isinstance(payload, dict) else {}


def post_ai_model_call(payload: dict[str, Any]) -> None:
    if not settings.is_chatbotinn_api_configured:
        return
    try:
        _request(
            "POST",
            "/api/agent/ai-model-calls",
            payload,
            timeout=settings.telemetry_timeout_seconds,
        )
    except httpx.HTTPError as exc:
        logger.warning("Could not record AI model call error=%s", exc)


def _cached_get(cache_key: str, path: str) -> Any:
    if not settings.is_chatbotinn_api_configured:
        logger.info("ChatbotInn API client is not configured")
        return {}

    now = time.monotonic()
    cached = _cache.get(cache_key)
    if cached and now - cached[0] < settings.knowledge_cache_ttl_seconds:
        return cached[1]

    try:
        payload = _request("GET", path)
        _cache[cache_key] = (now, payload)
        return payload
    except httpx.HTTPError as exc:
        logger.warning("Could not load backend path=%s error=%s", path, exc)
        return {}


def _request(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> Any:
    base_url = settings.chatbotinn_api_base_url.rstrip("/")
    headers = {
        "Authorization": f"Bearer {settings.chatbotinn_api_internal_token.strip()}",
        "Accept": "application/json",
    }
    if payload is not None:
        headers["Content-Type"] = "application/json"

    with httpx.Client(timeout=timeout or settings.chatbotinn_api_timeout_seconds) as client:
        response = client.request(
            method,
            f"{base_url}{path}",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        if response.status_code == 204 or not response.content:
            return None
        return response.json()
