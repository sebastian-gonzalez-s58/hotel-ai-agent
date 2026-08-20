import secrets
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings


bearer_scheme = HTTPBearer(auto_error=False)


def verify_internal_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> None:
    expected_token = settings.agent_internal_token
    if expected_token is None or not expected_token.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent internal token is not configured",
        )

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )

    if not secrets.compare_digest(credentials.credentials, expected_token.strip()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
        )


def verify_v2_request_headers(
    *,
    timestamp: str | None,
    idempotency_key: str | None,
    request_id: str | None,
    agent_turn_id: str,
) -> None:
    if not request_id or not request_id.strip():
        raise HTTPException(status_code=400, detail="X-Request-Id is required")
    if not idempotency_key or idempotency_key.strip() != agent_turn_id:
        raise HTTPException(status_code=400, detail="Idempotency-Key must equal agentTurnId")
    if not timestamp:
        raise HTTPException(status_code=400, detail="X-ChatbotInn-Timestamp is required")
    try:
        normalized = timestamp.strip().replace("Z", "+00:00")
        request_time = datetime.fromisoformat(normalized)
        if request_time.tzinfo is None:
            request_time = request_time.replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid X-ChatbotInn-Timestamp") from exc
    skew = abs((datetime.now(timezone.utc) - request_time.astimezone(timezone.utc)).total_seconds())
    if skew > settings.v2_request_timestamp_tolerance_seconds:
        raise HTTPException(status_code=401, detail="Request timestamp is outside the allowed window")
