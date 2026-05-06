from typing import Any

import numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from rrp.api.schemas import ConvergencePoint, ConvergenceResponse
from rrp.core.config import settings
from rrp.core.db import get_db
from rrp.forecasting.convergence import get_convergence_series
from rrp.forecasting.covers import load_feature_importance

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get(
    "/convergence",
    response_model=ConvergenceResponse,
    summary="Baseline vs corrected MAPE convergence series",
)
def get_convergence(
    surface: str = "covers",
    db: Session = Depends(get_db),
) -> ConvergenceResponse:
    series = get_convergence_series(surface, db)

    baseline_all = [p["baseline_mape"] for p in series if p["baseline_mape"] is not None]
    corrected_all = [p["corrected_mape"] for p in series if p["corrected_mape"] is not None]

    summary: dict[str, Any] = {
        "n_days": len(series),
        "avg_baseline_mape": round(float(np.mean(baseline_all)), 4) if baseline_all else None,
        "avg_corrected_mape": round(float(np.mean(corrected_all)), 4) if corrected_all else None,
    }

    # Trailing-window verdict — kept in lock-step with simulate.py's
    # `_evaluate_convergence`: a mean no-regression check AND a per-day
    # no-regression check (worst corrected[i] / baseline[i] over the window).
    window = settings.convergence_window_days
    factor = settings.convergence_no_regression_factor
    if len(baseline_all) >= window and len(corrected_all) >= window:
        baseline_late = baseline_all[-window:]
        corrected_late = corrected_all[-window:]
        baseline_late_mean = float(np.mean(baseline_late))
        corrected_late_mean = float(np.mean(corrected_late))
        worst_corrected = float(np.max(corrected_late))
        per_day_ratios = [
            c / b for b, c in zip(baseline_late, corrected_late, strict=True) if b > 0
        ]
        worst_per_day_ratio = max(per_day_ratios) if per_day_ratios else 1.0

        improved = corrected_late_mean <= baseline_late_mean * factor
        per_day_ok = worst_per_day_ratio <= factor
        converged = improved and per_day_ok

        summary["window_days"] = window
        summary["baseline_late_mean"] = round(baseline_late_mean, 4)
        summary["corrected_late_mean"] = round(corrected_late_mean, 4)
        summary["worst_corrected_late"] = round(worst_corrected, 4)
        summary["worst_per_day_ratio"] = round(worst_per_day_ratio, 3)
        summary["mape_improvement_pct"] = round(
            100 * (baseline_late_mean - corrected_late_mean) / max(baseline_late_mean, 1e-9),
            2,
        )
        summary["no_regression_factor"] = factor
        summary["converged"] = converged
        summary["no_regression"] = per_day_ok

    return ConvergenceResponse(
        surface=surface,
        series=[ConvergencePoint(**p) for p in series],
        summary=summary,
    )


@router.get(
    "/feature_importance",
    summary="LightGBM feature importance (gain + split) for the promoted models",
)
def get_feature_importance() -> dict[str, Any]:
    """
    Return the feature importance artefact dumped at training time.
    Useful as a basic explainability surface for the model card.
    """
    return load_feature_importance() or {
        "message": "feature_importance.json not present — train a baseline model first."
    }
