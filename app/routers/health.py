"""Liveness, readiness, and metrics.

The distinction is load-bearing on Kubernetes: /health must answer while the
model is still loading (otherwise the orchestrator kills the pod mid-startup
and loops forever), while /ready must NOT pass until a warmup forward pass has
completed (otherwise traffic arrives before the model can serve it at target
latency, and the first callers of every new pod get a slow response).
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.observability import REGISTRY

router = APIRouter(tags=["ops"])


@router.get("/health", summary="Liveness: is the process up?")
async def health(request: Request) -> dict:
    service = request.app.state.service
    return {
        "status": "ok",
        "service": service.settings.service_name,
        "model_ready": service.ready,
    }


@router.get("/ready", summary="Readiness: can it serve at target latency?")
async def ready(request: Request, response: Response) -> dict:
    service = request.app.state.service
    if not service.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "loading", "model_ready": False}
    return {
        "status": "ready",
        "model_ready": True,
        "backend": service.backend.name if service.backend else None,
        "language_id": service.language is not None,
    }


@router.get("/metrics", summary="Prometheus exposition")
async def metrics() -> Response:
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
