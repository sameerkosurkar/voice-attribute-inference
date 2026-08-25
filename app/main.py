"""Application entrypoint."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.errors import install_exception_handlers
from app.logging_setup import configure_logging
from app.observability import ERRORS
from app.routers import analyze, health, stream
from app.service import AnalyzerService

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    log.info(
        "starting",
        service=settings.service_name,
        backend=settings.backend,
        model=settings.age_gender_model,
    )

    service = AnalyzerService(settings)
    app.state.service = service
    started = time.perf_counter()
    await service.startup()
    log.info("startup_complete", startup_ms=round((time.perf_counter() - started) * 1000, 1))
    try:
        yield
    finally:
        await service.shutdown()
        log.info("shutdown_complete")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Voice Attribute Inference",
        version="1.0.0",
        description=(
            "Estimates a caller's gender and age bracket from a short audio "
            "sample, with calibrated confidences and an explicit audio-quality "
            "verdict. Audio is processed in memory and never persisted; see "
            "PRIVACY.md."
        ),
        lifespan=lifespan,
    )

    # Permissive CORS so the browser demo page can talk to the service. In a
    # real deployment this sits behind the edge and should be restricted --
    # called out here rather than left as an accidental default.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            ERRORS.labels(code="INTERNAL_ERROR").inc()
            raise

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.headers["x-request-id"] = request_id
        if request.url.path not in {"/metrics", "/health", "/ready"}:
            # Our own access log. Note what is absent: no query string, no
            # body, no filename -- only metadata. See PRIVACY.md.
            log.info(
                "request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=round(elapsed_ms, 2),
            )
            if response.status_code >= 400:
                ERRORS.labels(code=str(response.status_code)).inc()
        return response

    install_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(analyze.router)
    app.include_router(stream.router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        access_log=False,
        log_config=None,
    )
