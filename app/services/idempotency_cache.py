from dataclasses import dataclass
import hashlib
import json
from threading import Lock
import time
from typing import Any

from fastapi import HTTPException, status

from app.core.config import settings
from app.schemas.v2_turns import AgentTurnResponse


@dataclass(frozen=True)
class CachedTurn:
    fingerprint: str
    response: AgentTurnResponse
    expires_at: float


class V2TurnIdempotencyCache:
    def __init__(self) -> None:
        self._entries: dict[str, CachedTurn] = {}
        self._lock = Lock()

    def fingerprint(self, payload: Any) -> str:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(self, key: str, fingerprint: str) -> AgentTurnResponse | None:
        with self._lock:
            self._prune()
            cached = self._entries.get(key)
            if cached is None:
                return None
            if cached.fingerprint != fingerprint:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Idempotency key was reused with a different payload",
                )
            return cached.response

    def put(self, key: str, fingerprint: str, response: AgentTurnResponse) -> None:
        with self._lock:
            self._prune()
            if len(self._entries) >= settings.v2_idempotency_max_entries:
                oldest_key = min(self._entries, key=lambda item: self._entries[item].expires_at)
                self._entries.pop(oldest_key, None)
            self._entries[key] = CachedTurn(
                fingerprint=fingerprint,
                response=response,
                expires_at=time.monotonic() + settings.v2_idempotency_ttl_seconds,
            )

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def _prune(self) -> None:
        now = time.monotonic()
        expired = [key for key, value in self._entries.items() if value.expires_at <= now]
        for key in expired:
            self._entries.pop(key, None)


v2_turn_idempotency_cache = V2TurnIdempotencyCache()
