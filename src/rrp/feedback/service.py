"""
Corrections feedback service.

Every POST /v1/corrections flows through here:
  1. Plausibility check — quarantine if |actual - predicted| / predicted > threshold
     unless reason_code == 'closure' (legitimate zero-out).
  2. Persist to corrections table.
  3. Fan-out to online learners (covers EWMA+ridge, staff ratios, inventory z).
  4. Append corrected record to training tables.
  5. Return updated metrics summary.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy.orm import Session

from rrp.core.config import settings
from rrp.core.logging import get_logger
from rrp.domain.models import (
    Correction,
    CorrectionScope,
    ReasonCode,
    ServicePeriod,
)
from rrp.forecasting.covers import get_forecaster
from rrp.forecasting.inventory import get_inventory_recommender
from rrp.forecasting.staff import get_staff_recommender

log = get_logger(__name__)


def _is_plausible(
    predicted: float,
    actual: float,
    reason_code: ReasonCode,
) -> bool:
    """Return True if the correction is plausible (should be learned from)."""
    if reason_code == ReasonCode.closure:
        return True
    if predicted <= 0:
        return True
    delta_ratio = abs(actual - predicted) / abs(predicted)
    return delta_ratio <= settings.quarantine_delta_ratio


def apply_correction(
    scope: CorrectionScope,
    scope_id: str | None,
    predicted: float,
    actual: float,
    reason_code: ReasonCode,
    note: str | None,
    correction_date: date,
    hour: int | None,
    db: Session,
) -> dict[str, Any]:
    """
    Persist and fan-out a single correction.
    Returns a dict with quarantined flag and updated metrics snapshot.
    """
    quarantined = not _is_plausible(predicted, actual, reason_code)

    if quarantined:
        log.warning(
            "feedback.quarantined",
            scope=scope,
            predicted=predicted,
            actual=actual,
            reason_code=reason_code,
        )

    correction = Correction(
        scope=scope,
        scope_id=scope_id,
        predicted=predicted,
        actual=actual,
        reason_code=reason_code,
        note=note,
        quarantined=quarantined,
    )
    db.add(correction)
    db.flush()  # populate correction.id before returning

    updated_metrics: dict[str, Any] = {
        "correction_id": correction.id,
        "quarantined": quarantined,
    }

    if not quarantined:
        if scope == CorrectionScope.covers and hour is not None:
            _handle_covers_correction(
                correction_date, hour, actual, predicted, db
            )
            updated_metrics["online_updated"] = True

        elif scope == CorrectionScope.staff and hour is not None and scope_id:
            _handle_staff_correction(
                correction_date, hour, actual, scope_id, db
            )

        elif scope == CorrectionScope.inventory and scope_id:
            _handle_inventory_correction(scope_id, actual, predicted, db)

        # Append corrected actuals to training tables
        _append_to_training(scope, correction_date, hour, actual, scope_id, db)

    db.commit()

    log.info(
        "feedback.applied",
        scope=scope,
        quarantined=quarantined,
        delta=round(actual - predicted, 2),
        reason=reason_code,
    )
    return updated_metrics


def _handle_covers_correction(
    correction_date: date,
    hour: int,
    actual: float,
    predicted: float,
    db: Session,
) -> None:
    forecaster = get_forecaster()
    forecaster.update_online(
        target_date=correction_date,
        hour=hour,
        actual=actual,
        predicted=predicted,
        db=db,
    )


def _handle_staff_correction(
    correction_date: date,
    hour: int,
    actual_hc: float,
    role_name: str,
    db: Session,
) -> None:
    # Get covers for that hour to update the ratio
    sp = (
        db.query(ServicePeriod)
        .filter(ServicePeriod.date == correction_date, ServicePeriod.hour == hour)
        .first()
    )
    covers = sp.covers_actual or sp.covers_predicted or 50.0 if sp else 50.0
    recommender = get_staff_recommender()
    recommender.update_ratio(
        role=role_name,
        weekday=correction_date.weekday(),
        hour=hour,
        actual_hc=actual_hc,
        covers=float(covers),
    )


def _handle_inventory_correction(
    ingredient_id_str: str,
    actual_used: float,
    predicted: float,
    db: Session,
) -> None:
    try:
        ingredient_id = int(ingredient_id_str)
    except ValueError:
        return
    recommender = get_inventory_recommender()
    stockout = actual_used > predicted * 1.5
    excess_waste = actual_used < predicted * 0.5
    recommender.update_z(ingredient_id, stockout=stockout, excess_waste=excess_waste)


def _append_to_training(
    scope: CorrectionScope,
    correction_date: date,
    hour: int | None,
    actual: float,
    scope_id: str | None,
    db: Session,
) -> None:
    """Write corrected actuals back to training tables for next nightly retrain."""
    if scope == CorrectionScope.covers and hour is not None:
        existing = (
            db.query(ServicePeriod)
            .filter(ServicePeriod.date == correction_date, ServicePeriod.hour == hour)
            .first()
        )
        if existing:
            existing.covers_actual = actual
        else:
            db.add(ServicePeriod(date=correction_date, hour=hour, covers_actual=actual))
