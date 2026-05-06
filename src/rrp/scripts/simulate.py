"""
Deterministic end-to-end simulation.

Runs the full feedback loop over N simulated days, writing daily_metrics rows
with both baseline_mape and corrected_mape.

Convergence is defined as a TRAILING-WINDOW comparison, not endpoint-vs-endpoint:

    converged := mean(corrected_mape[-W:]) <= mean(baseline_mape[-W:])
                 AND no_regression: max(corrected_mape[-W:])
                                    <= mean(baseline_mape[-W:]) * R

W = settings.convergence_window_days  (default 7)
R = settings.convergence_no_regression_factor  (default 1.10)

Multi-hour, multi-day signal density:
    Each simulated day, corrections are submitted for several "manager-noticed"
    hours — typically the lunch and dinner peaks. Without this, only one of
    the 168 (weekday, hour) cells gets touched per day and the test silently
    only exercises the Saturday-19:00 segment.

Usage:
    python -m rrp.scripts.simulate --days 60 --seed 42
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np

from rrp.core.config import settings
from rrp.core.db import SessionLocal
from rrp.core.logging import configure_logging, get_logger
from rrp.domain.models import (
    CorrectionScope,
    DailyMetric,
    ReasonCode,
    ServicePeriod,
    Surface,
)
from rrp.feedback.service import apply_correction
from rrp.forecasting.convergence import compute_and_store_daily_metrics
from rrp.forecasting.covers import get_forecaster, reload_forecaster

configure_logging()
log = get_logger(__name__)

# Manager corrections happen on hours where they actually look at the schedule.
# These are the typical peaks — covers a healthy mix of (weekday, hour) cells.
PEAK_HOURS: tuple[int, ...] = (12, 13, 18, 19, 20)

REASON_CODES: tuple[ReasonCode, ...] = (
    ReasonCode.rain,
    ReasonCode.event,
    ReasonCode.unknown,
    ReasonCode.holiday,
    ReasonCode.unknown,
    ReasonCode.unknown,
)


def _get_sim_start(db: object) -> date:
    from sqlalchemy.orm import Session

    assert isinstance(db, Session)
    rows = (
        db.query(ServicePeriod.date)
        .filter(ServicePeriod.covers_actual.isnot(None))
        .distinct()
        .order_by(ServicePeriod.date.asc())
        .all()
    )
    if not rows:
        return date.today() - timedelta(days=60)
    dates: list[date] = [r.date for r in rows]
    if len(dates) > 60:
        return dates[-60]
    return dates[0]


def _evaluate_convergence(
    baseline: Sequence[float],
    corrected: Sequence[float],
    window: int,
    no_regression_factor: float,
) -> dict[str, object]:
    """
    Trailing-window convergence test.

    Returns a structured verdict explaining why the run did or did not converge.
    """
    if len(baseline) < window or len(corrected) < window:
        return {
            "converged": False,
            "reason": "insufficient_data",
            "n_days": len(baseline),
            "window_days": window,
        }

    baseline_late = list(baseline[-window:])
    corrected_late = list(corrected[-window:])
    baseline_mean = statistics.fmean(baseline_late)
    corrected_mean = statistics.fmean(corrected_late)
    worst_corrected = max(corrected_late)

    improved = corrected_mean <= baseline_mean
    no_regression = worst_corrected <= baseline_mean * no_regression_factor

    if improved and no_regression:
        verdict = "converged_with_improvement"
    elif no_regression:
        verdict = "neutral_no_regression"
    else:
        verdict = "regressed"

    return {
        "converged": improved and no_regression,
        "reason": verdict,
        "n_days": len(baseline),
        "window_days": window,
        "baseline_late_mean": round(baseline_mean, 4),
        "corrected_late_mean": round(corrected_mean, 4),
        "corrected_late_worst": round(worst_corrected, 4),
        "improvement_pct": round(
            100 * (baseline_mean - corrected_mean) / max(baseline_mean, 1e-9), 1
        ),
        "no_regression_factor": no_regression_factor,
    }


def run_simulation(days: int = 60, seed: int = 42) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    db = SessionLocal()
    results: list[dict[str, object]] = []

    try:
        sim_start = _get_sim_start(db)
        log.info("simulate.start", days=days, seed=seed, from_date=str(sim_start))

        reload_forecaster()
        forecaster = get_forecaster()

        for day_idx in range(days):
            sim_date = sim_start + timedelta(days=day_idx)

            rows = (
                db.query(ServicePeriod)
                .filter(
                    ServicePeriod.date == sim_date,
                    ServicePeriod.covers_actual.isnot(None),
                )
                .all()
            )
            if not rows:
                log.debug("simulate.skip_day", date=str(sim_date), reason="no actuals")
                continue

            metrics = compute_and_store_daily_metrics(sim_date, db)
            if not metrics:
                continue

            # Multi-hour corrections — managers look at peaks, not just 19:00.
            # Pick 2-3 peak hours per day, randomised by seed.
            n_corrections_today = int(rng.integers(2, len(PEAK_HOURS) + 1))
            chosen_hours = rng.choice(
                np.array(PEAK_HOURS), size=n_corrections_today, replace=False
            ).tolist()

            baseline_hourly = forecaster.predict_baseline_hourly(sim_date, db)

            for hour in chosen_hours:
                row = next((r for r in rows if r.hour == hour), None)
                if row is None or row.covers_actual is None:
                    continue
                actual = float(row.covers_actual)
                predicted = float(baseline_hourly[int(hour)])
                # Manager noise: ±5% on the count they report.
                noisy_actual = actual * float(rng.uniform(0.95, 1.05))
                reason = REASON_CODES[(day_idx + int(hour)) % len(REASON_CODES)]

                apply_correction(
                    scope=CorrectionScope.covers,
                    scope_id=None,
                    predicted=predicted,
                    actual=noisy_actual,
                    reason_code=reason,
                    note=None,
                    correction_date=sim_date,
                    hour=int(hour),
                    db=db,
                )

            dm = (
                db.query(DailyMetric)
                .filter(
                    DailyMetric.date == sim_date,
                    DailyMetric.surface == Surface.covers,
                )
                .first()
            )
            if dm:
                dm.n_corrections = (dm.n_corrections or 0) + n_corrections_today
                db.commit()

            results.append(
                {
                    "day": day_idx + 1,
                    "date": str(sim_date),
                    "baseline_mape": metrics.get("baseline_mape"),
                    "corrected_mape": metrics.get("corrected_mape"),
                    "n_corrections": n_corrections_today,
                }
            )

            log.info(
                "simulate.day",
                day=day_idx + 1,
                date=str(sim_date),
                baseline=round(float(metrics.get("baseline_mape") or 0), 4),
                corrected=round(float(metrics.get("corrected_mape") or 0), 4),
                n_corrections=n_corrections_today,
            )

    finally:
        db.close()

    if results:
        baseline = [float(r["baseline_mape"] or 0) for r in results]  # type: ignore[arg-type]
        corrected = [float(r["corrected_mape"] or 0) for r in results]  # type: ignore[arg-type]
        verdict = _evaluate_convergence(
            baseline,
            corrected,
            window=settings.convergence_window_days,
            no_regression_factor=settings.convergence_no_regression_factor,
        )
        log.info("simulate.summary", **verdict)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic feedback loop simulation")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_simulation(days=args.days, seed=args.seed)


if __name__ == "__main__":
    main()
