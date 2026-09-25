import logging
import time
import uuid

from fastapi import Request
from app.core.latency import identifier, span, trace


logger = logging.getLogger("chatbotinn-agent.requests")


async def request_context_middleware(request: Request, call_next):
    request_id = identifier(request.headers.get("X-Request-ID")) or str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.perf_counter()
    with trace(request.headers.get("X-Chat-Latency-Trace") or request_id,
               request.headers.get("X-Chat-Latency-Parent"),
               enabled=request.url.path.startswith(("/hotel/", "/internal/v2/"))), span("http.agent_request") as measurement:
        response = await call_next(request)
        measurement.attributes["http_status"] = response.status_code
        if response.status_code >= 400:
            measurement.outcome = "http_error"
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response
