from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging

from backend.core.exceptions import BackendError

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(BackendError)
    async def _backend_error_handler(request: Request, exc: BackendError) -> JSONResponse:
        logger.warning(
            "Handled backend exception",
            extra={
                "path": request.url.path,
                "method": request.method,
                "status_code": exc.status_code,
                "error_code": exc.error_code,
            },
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status": "error",
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                },
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled server exception",
            extra={
                "path": request.url.path,
                "method": request.method,
            },
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "code": "SERVER_ERROR",
            },
        )
