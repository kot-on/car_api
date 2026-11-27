"""Единый формат ошибок: {"error": {"code": 422, "message": "..."}}"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.logging_setup import log


def error_response(code: int, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content={"error": {"code": code, "message": message}},
        headers=headers,
    )


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_exc(_: Request, exc: StarletteHTTPException):
        return error_response(exc.status_code, str(exc.detail), getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def validation_exc(_: Request, exc: RequestValidationError):
        parts = []
        for err in exc.errors():
            loc = ".".join(str(x) for x in err["loc"] if x != "body")
            parts.append(f"{loc}: {err['msg']}")
        return error_response(422, "; ".join(parts) or "неверные данные запроса")

    @app.exception_handler(Exception)
    async def unhandled_exc(request: Request, exc: Exception):
        log.error("необработанная ошибка", exc_info=True,
                  extra={"event": "unhandled", "path": request.url.path})
        return error_response(500, "внутренняя ошибка сервера")
