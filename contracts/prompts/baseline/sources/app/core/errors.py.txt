import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


logger = logging.getLogger("chatbotinn-agent.errors")


class AgentDependencyError(RuntimeError):
    pass


class AgentModelError(RuntimeError):
    pass


class AgentTimeoutError(RuntimeError):
    pass


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "requestId": _request_id(request),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.info(
            "Request validation failed path=%s request_id=%s errors=%s",
            request.url.path,
            _request_id(request),
            exc.errors(),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Invalid request payload",
                "requestId": _request_id(request),
                "details": exc.errors(),
            },
        )

    @app.exception_handler(AgentDependencyError)
    async def dependency_exception_handler(request: Request, exc: AgentDependencyError) -> JSONResponse:
        logger.error(
            "Agent dependency error path=%s request_id=%s error=%s",
            request.url.path,
            _request_id(request),
            exc,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": str(exc),
                "requestId": _request_id(request),
            },
        )

    @app.exception_handler(AgentTimeoutError)
    async def timeout_exception_handler(request: Request, exc: AgentTimeoutError) -> JSONResponse:
        logger.warning(
            "Agent timeout path=%s request_id=%s error=%s",
            request.url.path,
            _request_id(request),
            exc,
        )
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={
                "error": str(exc),
                "requestId": _request_id(request),
            },
        )

    @app.exception_handler(AgentModelError)
    async def model_exception_handler(request: Request, exc: AgentModelError) -> JSONResponse:
        logger.error(
            "Agent model error path=%s request_id=%s error=%s",
            request.url.path,
            _request_id(request),
            exc,
        )
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={
                "error": "The language model returned an invalid response",
                "requestId": _request_id(request),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled agent error path=%s request_id=%s",
            request.url.path,
            _request_id(request),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Unexpected agent error",
                "requestId": _request_id(request),
            },
        )
