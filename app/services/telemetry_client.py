from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import logging
from typing import Any

from app.core.agent_tracking import AgentTrackingContext
from app.core.config import settings
from app.services.chatbotinn_client import post_ai_model_call


logger = logging.getLogger("chatbotinn-agent.telemetry")


@dataclass(frozen=True)
class OpenAiTokenUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    def as_api_dict(self) -> dict[str, int]:
        return {
            "inputTokens": self.input_tokens,
            "cachedInputTokens": self.cached_input_tokens,
            "outputTokens": self.output_tokens,
            "reasoningTokens": self.reasoning_tokens,
            "totalTokens": self.total_tokens,
        }


def extract_openai_usage(response: Any) -> OpenAiTokenUsage:
    usage = getattr(response, "usage", None)
    input_tokens = int(_read(usage, "input_tokens", 0) or 0)
    output_tokens = int(_read(usage, "output_tokens", 0) or 0)
    total_tokens = int(_read(usage, "total_tokens", input_tokens + output_tokens) or 0)
    input_details = _read(usage, "input_tokens_details", {})
    output_details = _read(usage, "output_tokens_details", {})
    return OpenAiTokenUsage(
        input_tokens=input_tokens,
        cached_input_tokens=int(_read(input_details, "cached_tokens", 0) or 0),
        output_tokens=output_tokens,
        reasoning_tokens=int(_read(output_details, "reasoning_tokens", 0) or 0),
        total_tokens=total_tokens or input_tokens + output_tokens,
    )


def record_model_call(
    *,
    context: AgentTrackingContext | None,
    status: str,
    usage: OpenAiTokenUsage,
    latency_ms: int,
    response_id: str | None = None,
    error_message: str | None = None,
    purpose: str | None = None,
) -> None:
    if context is None or not _has_business_context(context):
        return

    payload = {
        "conversationId": context.conversation_id,
        "operationId": context.operation_id,
        "messageId": context.message_id,
        "unmatchedMessageId": context.unmatched_message_id,
        "purpose": (purpose or context.purpose)[:80],
        "provider": "OPENAI",
        "model": settings.openai_model,
        "providerResponseId": response_id,
        "requestId": context.request_id,
        "processInstanceId": context.process_instance_id,
        "processActivityId": context.process_activity_id,
        "promptVersion": settings.agent_prompt_version,
        "status": status,
        **usage.as_api_dict(),
        "estimatedCost": _estimate_cost(usage),
        "costCurrency": "USD" if _has_cost_configuration() else None,
        "latencyMs": max(latency_ms, 0),
        "errorMessage": error_message[:1000] if error_message else None,
        "occurredAt": datetime.now(timezone.utc).isoformat(),
    }
    post_ai_model_call(payload)


def _has_business_context(context: AgentTrackingContext) -> bool:
    return any(
        (
            context.conversation_id,
            context.operation_id,
            context.message_id,
            context.unmatched_message_id,
        )
    )


def _has_cost_configuration() -> bool:
    return any(
        value is not None
        for value in (
            settings.openai_input_cost_per_million,
            settings.openai_cached_input_cost_per_million,
            settings.openai_output_cost_per_million,
        )
    )


def _estimate_cost(usage: OpenAiTokenUsage) -> str | None:
    if not _has_cost_configuration():
        return None

    input_rate = Decimal(str(settings.openai_input_cost_per_million or 0))
    cached_rate = Decimal(str(settings.openai_cached_input_cost_per_million or 0))
    output_rate = Decimal(str(settings.openai_output_cost_per_million or 0))
    uncached_input = max(usage.input_tokens - usage.cached_input_tokens, 0)
    cost = (
        Decimal(uncached_input) * input_rate
        + Decimal(usage.cached_input_tokens) * cached_rate
        + Decimal(usage.output_tokens) * output_rate
    ) / Decimal(1_000_000)
    return str(cost.quantize(Decimal("0.00000001")))


def _read(value: Any, name: str, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)
