"""
Convergence metrics computation.

For each day, scores predictions twice:
  - baseline_mape: LightGBM only (no online residual)
  - corrected_mape: LightGBM + EWMA+ridge online residual

Writes one row per surface to daily_metrics.
This is the undeniable proof that the feedback loop is doing work.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from rrp.core.logging import get_logger
from rrp.domain.models import DailyMetric, ModelRun, ServicePeriod, Surface
from rrp.forecasting.covers import get_forecaster
from rrp.forecasting.features import mape

log = get_logger(__name__)


def compute_and_store_daily_metrics(
    target_date: date,
    db: Session,
    model_version: str | None = None,
) -> dict[str, Any]:
    """
    Compute baseline vs corrected MAPE for covers on target_date.
    Upsert into daily_metrics.
    """
    rows = (
        db.query(ServicePeriod)
        .filter(
            ServicePeriod.date == target_date,
            ServicePeriod.covers_actual.isnot(None),
        )
        .all()
    )
    if not rows:
        return {}

    forecaster = get_forecaster()
    actuals = np.array([r.covers_actual for r in rows], dtype=float)
    hours = [r.hour for r in rows]

    baseline_pred = forecaster.predict_baseline_hourly(target_date, db)
    baseline_arr = np.array([baseline_pred[h] for h in hours], dtype=float)

    forecast_result = forecaster.forecast(target_date, db)
    corrected_hourly = forecast_result["hourly_covers"]
    corrected_arr = np.array([corrected_hourly[h] for h in hours], dtype=float)

    baseline_m = mape(actuals, baseline_arr)
    corrected_m = mape(actuals, corrected_arr)

    if model_version is None:
        last_run = (
            db.query(ModelRun)
            .filter(ModelRun.promoted == True)  # noqa: E712
            .order_by(ModelRun.created_at.desc())
            .first()
        )
        model_version = last_run.version if last_run else "baseline"

    n_corrections = db.execute(
        select(DailyMetric.n_corrections)
        .where(
            DailyMetric.date == target_date,
            DailyMetric.surface == Surface.covers,
        )
    ).scalar() or 0

    existing = (
        db.query(DailyMetric)
        .filter(DailyMetric.date == target_date, DailyMetric.surface == Surface.covers)
        .first()
    )
    if existing:
        existing.baseline_mape = baseline_m
        existing.corrected_mape = corrected_m
        existing.model_version = model_version
    else:
        db.add(DailyMetric(
            date=target_date,
            surface=Surface.covers,
            baseline_mape=baseline_m,
            corrected_mape=corrected_m,
            n_corrections=n_corrections,
            model_version=model_version,
        ))
    db.commit()

    log.info(
        "convergence.stored",
        date=str(target_date),
        baseline_mape=round(baseline_m, 4),
        corrected_mape=round(corrected_m, 4),
    )
    return {
        "date": str(target_date),
        "baseline_mape": baseline_m,
        "corrected_mape": corrected_m,
        "model_version": model_version,
    }


def get_convergence_series(surface: str, db: Session) -> list[dict[str, Any]]:
    """Return the full daily_metrics series for a surface, sorted by date."""
    try:
        surf = Surface(surface)
    except ValueError:
        surf = Surface.covers
    rows = (
        db.query(DailyMetric)
        .filter(DailyMetric.surface == surf)
        .order_by(DailyMetric.date.asc())
        .all()
    )
    return [
        {
            "date": str(r.date),
            "baseline_mape": r.baseline_mape,
            "corrected_mape": r.corrected_mape,
            "n_corrections": r.n_corrections,
            "model_version": r.model_version,
        }
        for r in rows
    ]
