"""
FastAPI application entrypoint.

Starts APScheduler (nightly retrain) inside the lifespan context manager.
All business routes are versioned under /v1/.
Health/readiness endpoints are unversioned.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, Request
from sqlalchemy import text

from rrp.api.routes import admin, corrections, forecast, metrics
from rrp.api.schemas import HealthResponse, ReadyResponse
from rrp.core.config import settings
from rrp.core.db import SessionLocal
from rrp.core.logging import configure_logging, get_logger

configure_logging(settings.env)
log = get_logger(__name__)

_scheduler = AsyncIOScheduler()


def _nightly_retrain_job() -> None:
    """Called by APScheduler — runs in a threadpool executor via asyncio."""
    from rrp.forecasting.trainer import run_nightly_retrain  # deferred import

    db = SessionLocal()
    try:
        run_nightly_retrain(db)
    finally:
        db.close()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    log.info("app.startup")
    _scheduler.add_job(
        _nightly_retrain_job,
        "cron",
        hour=settings.retrain_cron_hour,
        minute=settings.retrain_cron_minute,
        id="nightly_retrain",
        replace_existing=True,
    )
    _scheduler.start()
    log.info("scheduler.started", cron=f"{settings.retrain_cron_hour}:{settings.retrain_cron_minute:02d}")
    yield
    _scheduler.shutdown(wait=False)
    log.info("app.shutdown")


app = FastAPI(
    title="Restaurant RRP Forecaster",
    description=(
        "Self-learning forecaster for hourly covers, staff scheduling, and ingredient orders. "
        "Accepts manager corrections and converges toward accuracy via an online feedback loop."
    ),
    version="1.0.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Versioned routes
app.include_router(forecast.router, prefix="/v1")
app.include_router(corrections.router, prefix="/v1")
app.include_router(metrics.router, prefix="/v1")
app.include_router(admin.router, prefix="/v1")


@app.get("/healthz", response_model=HealthResponse, tags=["health"])
def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/readyz", response_model=ReadyResponse, tags=["health"])
def readyz() -> ReadyResponse:
    db_status = "ok"
    model_status = "ok"

    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
    except Exception:
        db_status = "unavailable"

    models_dir = settings.models_path
    model_a = models_dir / "covers_stage_a.lgb"
    if not model_a.exists():
        model_status = "no_model"

    overall = "ok" if db_status == "ok" else "degraded"
    return ReadyResponse(status=overall, db=db_status, model=model_status)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next: object) -> object:
    import uuid
    request_id = str(uuid.uuid4())[:8]
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
    response = await call_next(request)  # type: ignore[operator]
    return response
