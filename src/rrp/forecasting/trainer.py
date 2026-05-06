"""
Nightly LightGBM retrain with lightweight champion/challenger gate.

After training a new model, evaluate it on the last `champion_eval_days` days
of actuals. Promote only if the new model's MAPE is strictly lower than the
current champion's. Records the outcome in model_runs with promoted=True/False.

~15 lines for the gate itself (see _promote_if_better).
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from rrp.core.config import settings
from rrp.core.logging import get_logger
from rrp.domain.models import DailyMetric, ModelRun, ServicePeriod, Surface
from rrp.forecasting.covers import CoversForecaster, reload_forecaster, train_covers_models
from rrp.forecasting.features import mape

log = get_logger(__name__)


def _eval_mape_last_n_days(
    n: int,
    db: Session,
    forecaster: CoversForecaster,
) -> float:
    """Evaluate baseline MAPE on the last n days of actuals."""
    today = date.today()
    errors: list[float] = []
    for delta in range(1, n + 1):
        eval_date = today - timedelta(days=delta)
        rows = (
            db.query(ServicePeriod)
            .filter(
                ServicePeriod.date == eval_date,
                ServicePeriod.covers_actual.isnot(None),
            )
            .all()
        )
        if not rows:
            continue
        actuals = np.array([r.covers_actual for r in rows], dtype=float)
        baseline = np.array(forecaster.predict_baseline_hourly(eval_date, db), dtype=float)
        m = mape(actuals, baseline)
        if not np.isnan(m):
            errors.append(m)
    return float(np.mean(errors)) if errors else float("inf")


def _promote_if_better(
    new_metrics: dict[str, float],
    current_champion_mape: float,
    new_model_path: Path,
    staging_path: Path,
    db: Session,
    model_name: str,
    version: str,
) -> bool:
    """
    Lightweight champion/challenger gate (~15 lines).
    Promote new model only if its hold-out MAPE is strictly lower.
    Records outcome in model_runs.
    """
    new_mape = new_metrics.get("val_mape_stage_a", float("inf"))
    promoted = new_mape < current_champion_mape

    if promoted:
        # Atomically replace champion files
        for suffix in [".lgb"]:
            for stage in ["covers_stage_a", "covers_stage_b"]:
                src = staging_path / f"{stage}{suffix}"
                dst = new_model_path / f"{stage}{suffix}"
                if src.exists():
                    src.replace(dst)
        log.info("trainer.promoted", new_mape=round(new_mape, 4), prev_mape=round(current_champion_mape, 4))
    else:
        # Discard — clean up staging
        for stage in ["covers_stage_a", "covers_stage_b"]:
            p = staging_path / f"{stage}.lgb"
            if p.exists():
                p.unlink()
        log.info("trainer.not_promoted", new_mape=round(new_mape, 4), prev_mape=round(current_champion_mape, 4))

    run = ModelRun(
        model_name=model_name,
        version=version,
        metrics_json={**new_metrics, "champion_mape_at_eval": round(current_champion_mape, 4)},
        model_path=str(new_model_path),
        promoted=promoted,
    )
    db.add(run)
    db.commit()
    return promoted


def run_nightly_retrain(db: Session) -> dict[str, object]:
    """
    Full nightly retrain pipeline:
    1. Train new models to a staging directory.
    2. Evaluate current champion on last N days.
    3. Evaluate new models on last N days.
    4. Promote if new MAPE < champion MAPE.
    5. Reload forecaster singletons.
    """
    log.info("trainer.start")
    models_dir = settings.models_path
    staging_dir = models_dir / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    # Evaluate current champion before training
    current_forecaster = CoversForecaster()
    current_mape = _eval_mape_last_n_days(settings.champion_eval_days, db, current_forecaster)

    # Train new models into staging
    new_metrics = train_covers_models(db, staging_dir)
    if not new_metrics:
        log.warning("trainer.skip", reason="insufficient training data")
        return {"promoted": False, "reason": "insufficient data"}

    # Swap and evaluate using staged models
    staging_forecaster = CoversForecaster.__new__(CoversForecaster)
    CoversForecaster.__init__(staging_forecaster)
    # Temporarily point to staging files for evaluation
    import lightgbm as lgb  # local import to avoid module-level dep at trainer import time
    path_a = staging_dir / "covers_stage_a.lgb"
    path_b = staging_dir / "covers_stage_b.lgb"
    if path_a.exists():
        staging_forecaster._model_a = lgb.Booster(model_file=str(path_a))
    if path_b.exists():
        staging_forecaster._model_b = lgb.Booster(model_file=str(path_b))

    new_mape_eval = _eval_mape_last_n_days(settings.champion_eval_days, db, staging_forecaster)
    new_metrics["val_mape_stage_a"] = new_mape_eval

    promoted = _promote_if_better(
        new_metrics=new_metrics,
        current_champion_mape=current_mape,
        new_model_path=models_dir,
        staging_path=staging_dir,
        db=db,
        model_name="covers",
        version=version,
    )

    if promoted:
        reload_forecaster()

    # Drift detection: check if rolling 7-day MAPE > 1.5× trailing 28-day
    _check_drift(db)

    log.info("trainer.done", version=version, promoted=promoted)
    return {"promoted": promoted, "version": version, "new_mape": round(new_mape_eval, 4)}


def _check_drift(db: Session) -> None:
    """Log a warning if recent MAPE > drift threshold × baseline."""
    today = date.today()
    recent = (
        db.query(DailyMetric)
        .filter(
            DailyMetric.surface == Surface.covers,
            DailyMetric.date >= today - timedelta(days=7),
            DailyMetric.corrected_mape.isnot(None),
        )
        .all()
    )
    baseline_28 = (
        db.query(DailyMetric)
        .filter(
            DailyMetric.surface == Surface.covers,
            DailyMetric.date >= today - timedelta(days=35),
            DailyMetric.date < today - timedelta(days=7),
            DailyMetric.corrected_mape.isnot(None),
        )
        .all()
    )
    if not recent or not baseline_28:
        return
    recent_mape = float(np.mean([r.corrected_mape for r in recent]))
    base_mape = float(np.mean([r.corrected_mape for r in baseline_28]))
    if base_mape > 0 and recent_mape > settings.drift_mape_multiplier * base_mape:
        log.warning(
            "trainer.drift_detected",
            recent_mape=round(recent_mape, 4),
            baseline_mape=round(base_mape, 4),
            multiplier=recent_mape / base_mape,
        )
