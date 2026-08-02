import json
from dataclasses import dataclass
import time
from typing import Any

from openai import OpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError

from app.core.config import settings
from app.core.errors import AgentDependencyError, AgentModelError, AgentTimeoutError
from app.core.agent_tracking import get_agent_tracking_context
from app.services.telemetry_client import (
    OpenAiTokenUsage,
    extract_openai_usage,
    record_model_call,
)


client: OpenAI | None = None


@dataclass(frozen=True)
class OpenAiJsonResult:
    payload: dict[str, Any]
    usage: OpenAiTokenUsage
    response_id: str | None


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


def call_openai_json_result(
    prompt: str,
    *,
    purpose: str | None = None,
) -> OpenAiJsonResult:
    started_at = time.perf_counter()
    tracking_context = get_agent_tracking_context()
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
        _record_failure(started_at, tracking_context, purpose, "OpenAI request timed out")
        raise AgentTimeoutError("OpenAI request timed out") from exc
    except APIConnectionError as exc:
        _record_failure(started_at, tracking_context, purpose, "Could not connect to OpenAI")
        raise AgentDependencyError("Could not connect to OpenAI") from exc
    except APIStatusError as exc:
        _record_failure(
            started_at,
            tracking_context,
            purpose,
            f"OpenAI returned status {exc.status_code}",
        )
        raise AgentDependencyError(f"OpenAI returned status {exc.status_code}") from exc

    usage = extract_openai_usage(response)
    response_id = getattr(response, "id", None)
    try:
        payload = json.loads(response.output_text)
    except json.JSONDecodeError as exc:
        record_model_call(
            context=tracking_context,
            status="FAILED",
            usage=usage,
            latency_ms=_elapsed_ms(started_at),
            response_id=response_id,
            error_message="OpenAI returned non-JSON output",
            purpose=purpose,
        )
        raise AgentModelError("OpenAI returned non-JSON output") from exc

    record_model_call(
        context=tracking_context,
        status="SUCCEEDED",
        usage=usage,
        latency_ms=_elapsed_ms(started_at),
        response_id=response_id,
        purpose=purpose,
    )
    return OpenAiJsonResult(payload=payload, usage=usage, response_id=response_id)


def call_openai_json(prompt: str) -> dict[str, Any]:
    return call_openai_json_result(prompt).payload


def _record_failure(started_at, context, purpose, message: str) -> None:
    record_model_call(
        context=context,
        status="FAILED",
        usage=OpenAiTokenUsage(),
        latency_ms=_elapsed_ms(started_at),
        error_message=message,
        purpose=purpose,
    )


def _elapsed_ms(started_at: float) -> int:
    return round((time.perf_counter() - started_at) * 1000)
