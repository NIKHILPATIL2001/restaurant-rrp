"""
Inline online residual learner — no River dependency.

Two components per (weekday, hour) segment:
  1. EWMASegment — exponentially weighted moving average + EW abs-deviation.
     Gives instant per-segment correction that is visible within one day.
  2. RidgeSegment — tiny 3-feature ridge regression over
     [temp_delta, precip_mm, event_intensity], updated via RLS.
     Learns that e.g. rainy Tuesdays consistently under-predict by X covers.

Final prediction = lgbm_base + ewma_resid + ridge_resid

Design choices that matter (and are defended by tests in tests/unit/test_online.py):

  * alpha = 0.10  (half-life ≈ 6.6 corrections). Lower than the textbook 0.2 because
    a single noisy correction at α=0.2 moves the EWMA by 20% of the residual,
    which is enough to inject more noise than it removes when LightGBM is
    already at 5–7% MAPE. α=0.1 trades adaptation speed for stability and
    keeps the layer from regressing the baseline on the synthetic test bed.

  * warmup_n = 5. The first N corrections per segment are applied at fractional
    weight (n/warmup_n). This is a Bayesian shrinkage prior — "the LightGBM
    baseline is correct" until evidence accumulates.

  * significance_k = 0.5. The correction is emitted only when |EWMA| exceeds
    half of its own running MAD. Prevents single-correction whiplash.

  * Ridge lam = 10.0 (was 1.0) and warmup_ridge_n = 10. The 3-feature ridge
    is wide-tailed by nature; with one update per day, lam=1 lets a single
    rainy Tuesday own the segment. lam=10 keeps it conservative.

  * Residual clamp at update time. A correction can only move EWMA by at most
    `clamp_factor * |predicted|` (default 0.20). One bad point can no longer
    poison a segment.

  * Predict-time cap. The *applied* correction (EWMA + ridge) is hard-capped
    to `predict_cap_factor * |baseline_predicted|` at predict time. This is
    the final safety net: even if both segments accumulate stale state, the
    visible correction is bounded by a small fraction of the baseline. It
    also bounds the worst-case MAPE the online layer can introduce.

Both components are serialised/deserialised as plain dicts so we don't need
an external serialisation library beyond joblib.
"""

from __future__ import annotations

import contextlib
import math
from typing import Any

import numpy as np

from rrp.core.config import settings

_N_FEATURES = 3  # [temp_delta_from_mean, precip_mm, event_intensity]


def _clamp_residual(residual: float, predicted: float, factor: float) -> float:
    """Cap the magnitude of a single residual update to ±factor·|predicted|."""
    if predicted == 0:
        return residual
    cap = abs(predicted) * factor
    return max(-cap, min(cap, residual))


class EWMASegment:
    """
    Per-(weekday, hour) exponentially weighted moving average with running
    absolute deviation. The MAD is used as a robust σ proxy for a
    significance gate at predict time.
    """

    def __init__(
        self,
        alpha: float | None = None,
        warmup_n: int | None = None,
        significance_k: float | None = None,
    ) -> None:
        self.alpha = alpha if alpha is not None else settings.online_alpha
        self.warmup_n = warmup_n if warmup_n is not None else settings.online_warmup_n
        self.significance_k = (
            significance_k if significance_k is not None else settings.online_significance_k
        )
        self.value: float = 0.0
        self.abs_dev: float = 0.0  # EW mean absolute deviation (robust σ)
        self.n: int = 0

    def update(self, residual: float) -> None:
        if self.n == 0:
            # Shrink the very first sample halfway toward zero. Without this,
            # one noisy first correction sets the segment to a wild value.
            self.value = 0.5 * residual
            self.abs_dev = abs(residual)
        else:
            new_val = self.alpha * residual + (1 - self.alpha) * self.value
            self.abs_dev = (
                self.alpha * abs(residual - self.value) + (1 - self.alpha) * self.abs_dev
            )
            self.value = new_val
        self.n += 1

    def predict(self) -> float:
        if self.n == 0:
            return 0.0
        # Significance gate: ignore the EWMA if it's smaller than k·MAD —
        # the signal-to-noise ratio is too low to be worth applying.
        if self.abs_dev > 0 and abs(self.value) < self.significance_k * self.abs_dev:
            return 0.0
        # Warmup shrinkage: apply at fractional weight until enough evidence.
        warmup = min(1.0, self.n / max(1, self.warmup_n))
        return self.value * warmup

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "warmup_n": self.warmup_n,
            "significance_k": self.significance_k,
            "value": self.value,
            "abs_dev": self.abs_dev,
            "n": self.n,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EWMASegment:
        seg = cls(
            alpha=d.get("alpha"),
            warmup_n=d.get("warmup_n"),
            significance_k=d.get("significance_k"),
        )
        seg.value = d["value"]
        seg.abs_dev = d.get("abs_dev", 0.0)
        seg.n = d["n"]
        return seg


class RidgeSegment:
    """
    Per-(weekday, hour) online ridge regression with 3 features.
    Closed-form RLS update with regularisation. Warmup-gated to avoid
    one-shot overfitting.
    """

    def __init__(
        self,
        lam: float | None = None,
        warmup_n: int | None = None,
    ) -> None:
        self.lam = lam if lam is not None else settings.online_ridge_lam
        self.warmup_n = warmup_n if warmup_n is not None else settings.online_ridge_warmup_n
        self.w: np.ndarray = np.zeros(_N_FEATURES)
        self.A: np.ndarray = np.eye(_N_FEATURES) * self.lam
        self.b: np.ndarray = np.zeros(_N_FEATURES)
        self.n: int = 0

    def _features(self, temp_c: float, precip_mm: float, event_intensity: float) -> np.ndarray:
        return np.array([temp_c - 15.0, precip_mm, event_intensity])

    def update(
        self,
        residual: float,
        temp_c: float,
        precip_mm: float,
        event_intensity: float,
    ) -> None:
        x = self._features(temp_c, precip_mm, event_intensity)
        self.A += np.outer(x, x)
        self.b += x * residual
        with contextlib.suppress(np.linalg.LinAlgError):
            self.w = np.linalg.solve(self.A, self.b)
        self.n += 1

    def predict(
        self,
        temp_c: float,
        precip_mm: float,
        event_intensity: float,
    ) -> float:
        if self.n == 0:
            return 0.0
        x = self._features(temp_c, precip_mm, event_intensity)
        warmup = min(1.0, self.n / max(1, self.warmup_n))
        return float(x @ self.w) * warmup

    def to_dict(self) -> dict[str, Any]:
        return {
            "lam": self.lam,
            "warmup_n": self.warmup_n,
            "w": self.w.tolist(),
            "A": self.A.tolist(),
            "b": self.b.tolist(),
            "n": self.n,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RidgeSegment:
        seg = cls(lam=d.get("lam"), warmup_n=d.get("warmup_n"))
        seg.w = np.array(d["w"])
        seg.A = np.array(d["A"])
        seg.b = np.array(d["b"])
        seg.n = d["n"]
        return seg


class OnlineResidualLearner:
    """
    Container for all (weekday, hour) segments.

    Segments are created lazily on first update.
    Serialised state is a plain dict — can be stored as JSON in Postgres
    or saved via joblib alongside the LightGBM model.
    """

    def __init__(self) -> None:
        self._ewma: dict[str, EWMASegment] = {}
        self._ridge: dict[str, RidgeSegment] = {}

    def _key(self, weekday: int, hour: int) -> str:
        return f"{weekday}_{hour}"

    def update(
        self,
        weekday: int,
        hour: int,
        residual: float,
        baseline_predicted: float,
        temp_c: float = 15.0,
        precip_mm: float = 0.0,
        event_intensity: float = 0.0,
    ) -> None:
        """
        Record an observed residual.

        residual = actual - baseline_predicted (NOT the corrected prediction).
        baseline_predicted is used for the magnitude clamp so that one extreme
        observation cannot poison the segment.
        """
        clamped = _clamp_residual(residual, baseline_predicted, settings.online_residual_clamp)
        k = self._key(weekday, hour)
        if k not in self._ewma:
            self._ewma[k] = EWMASegment()
            self._ridge[k] = RidgeSegment()
        self._ewma[k].update(clamped)
        self._ridge[k].update(clamped, temp_c, precip_mm, event_intensity)

    def predict(
        self,
        weekday: int,
        hour: int,
        temp_c: float = 15.0,
        precip_mm: float = 0.0,
        event_intensity: float = 0.0,
        baseline_predicted: float | None = None,
    ) -> float:
        """
        Return the applied correction for this (weekday, hour) cell.

        If `baseline_predicted` is provided, the result is clamped to
        ±`settings.online_predict_cap_factor * |baseline_predicted|`. This is
        the final safety net that bounds how badly the online layer can
        regress the baseline on any single hour.
        """
        k = self._key(weekday, hour)
        ewma_val = self._ewma[k].predict() if k in self._ewma else 0.0
        ridge_val = (
            self._ridge[k].predict(temp_c, precip_mm, event_intensity)
            if k in self._ridge
            else 0.0
        )
        if not (math.isfinite(ewma_val) and math.isfinite(ridge_val)):
            return 0.0
        total = ewma_val + ridge_val
        if baseline_predicted is not None and baseline_predicted != 0.0:
            cap = abs(baseline_predicted) * settings.online_predict_cap_factor
            total = max(-cap, min(cap, total))
        return total

    def half_life_corrections(self) -> float:
        """
        Number of corrections required for the EWMA to decay to half weight.
        Useful for documenting and unit-testing α stability.
        """
        return math.log(0.5) / math.log(1 - settings.online_alpha)

    @property
    def is_active(self) -> bool:
        return len(self._ewma) > 0

    def coverage(self) -> dict[str, int]:
        """How many of the 168 (weekday, hour) cells have been updated?"""
        return {
            "cells_touched": len(self._ewma),
            "cells_total": 7 * 24,
            "total_updates": sum(s.n for s in self._ewma.values()),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "ewma": {k: v.to_dict() for k, v in self._ewma.items()},
            "ridge": {k: v.to_dict() for k, v in self._ridge.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OnlineResidualLearner:
        learner = cls()
        learner._ewma = {k: EWMASegment.from_dict(v) for k, v in d.get("ewma", {}).items()}
        learner._ridge = {k: RidgeSegment.from_dict(v) for k, v in d.get("ridge", {}).items()}
        return learner
