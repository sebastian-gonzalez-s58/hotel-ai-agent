import json
from typing import Any

from openai import OpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError

from app.core.config import settings
from app.core.errors import AgentDependencyError, AgentModelError, AgentTimeoutError


client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    global client
    if not settings.is_openai_configured:
        raise AgentDependencyError("OpenAI API key is not configured")

    if client is None:
        client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=1,
        )

    return client


def call_openai_json(prompt: str) -> dict[str, Any]:
    try:
        response = get_openai_client().responses.create(
            model=settings.openai_model,
            input=prompt,
            temperature=0,
            text={
                "format": {
                    "type": "json_object",
                }
            },
        )
    except APITimeoutError as exc:
        raise AgentTimeoutError("OpenAI request timed out") from exc
    except APIConnectionError as exc:
        raise AgentDependencyError("Could not connect to OpenAI") from exc
    except APIStatusError as exc:
        raise AgentDependencyError(f"OpenAI returned status {exc.status_code}") from exc

    try:
        return json.loads(response.output_text)
    except json.JSONDecodeError as exc:
        raise AgentModelError("OpenAI returned non-JSON output") from exc
