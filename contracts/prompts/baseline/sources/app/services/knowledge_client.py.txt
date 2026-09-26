import logging
import time
from typing import Any

import httpx

from app.core.config import settings
from app.core.latency import headers as latency_headers, timed, span


logger = logging.getLogger("chatbotinn-agent.knowledge")

_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def get_faq_knowledge() -> dict[str, Any]:
    return _get_knowledge("faqs", "/api/agent/knowledge/faqs")


def get_menu_knowledge() -> dict[str, Any]:
    return _get_knowledge("menu", "/api/agent/knowledge/menu")


@timed("knowledge.cached_resource")
def _get_knowledge(cache_key: str, path: str) -> dict[str, Any]:
    if not settings.is_chatbotinn_api_configured:
        logger.info("ChatbotInn API knowledge client is not configured")
        return {}

    now = time.monotonic()
    cached = _cache.get(cache_key)
    with span("knowledge.cache_lookup", cache_hit=bool(cached and now - cached[0] < settings.knowledge_cache_ttl_seconds)):
        pass
    if cached and now - cached[0] < settings.knowledge_cache_ttl_seconds:
        return cached[1]

    try:
        payload = _request_knowledge(path)
        _cache[cache_key] = (now, payload)
        return payload
    except httpx.HTTPError as exc:
        logger.warning("Could not load knowledge path=%s error=%s", path, exc)
        return {}


@timed("http.backend_knowledge")
def _request_knowledge(path: str) -> dict[str, Any]:
    base_url = settings.chatbotinn_api_base_url.rstrip("/")
    url = f"{base_url}{path}"
    headers = {
        "Authorization": f"Bearer {settings.chatbotinn_api_internal_token.strip()}",
        "Accept": "application/json",
    }

    headers.update(latency_headers())
    with httpx.Client(timeout=settings.chatbotinn_api_timeout_seconds) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
