"""
Unit tests for the inline EWMA + ridge online residual learner.

These tests pin down the design choices defended in `online.py`'s docstring:
the half-life of the EWMA, warmup shrinkage, the significance gate, and the
residual clamp. If any of these change, a test should fail loudly so the
behaviour is reviewed rather than silently regressing the baseline.
"""

import math

import numpy as np
import pytest

from rrp.core.config import settings
from rrp.forecasting.online import (
    EWMASegment,
    OnlineResidualLearner,
    RidgeSegment,
    _clamp_residual,
)


class TestEWMASegmentBasics:
    def test_first_update_shrinks_toward_zero(self) -> None:
        """A noisy first observation must not own the segment."""
        seg = EWMASegment(alpha=0.2, warmup_n=1, significance_k=0.0)
        seg.update(10.0)
        # First update is shrunk to 50% of the residual.
        assert seg.value == pytest.approx(5.0)
        assert seg.n == 1

    def test_decay_applied_on_subsequent_updates(self) -> None:
        seg = EWMASegment(alpha=0.5, warmup_n=1, significance_k=0.0)
        seg.update(10.0)  # value -> 5.0
        seg.update(0.0)   # 0.5*0 + 0.5*5 = 2.5
        assert seg.value == pytest.approx(2.5)

    def test_converges_to_constant(self) -> None:
        seg = EWMASegment(alpha=0.3, warmup_n=1, significance_k=0.0)
        for _ in range(200):
            seg.update(7.0)
        assert seg.value == pytest.approx(7.0, abs=0.01)

    def test_serialise_roundtrip(self) -> None:
        seg = EWMASegment(alpha=0.2)
        for v in [5.0, 10.0, 3.0]:
            seg.update(v)
        restored = EWMASegment.from_dict(seg.to_dict())
        assert restored.value == pytest.approx(seg.value)
        assert restored.abs_dev == pytest.approx(seg.abs_dev)
        assert restored.n == seg.n


class TestEWMAStability:
    """The properties that prevent the layer from regressing the baseline."""

    def test_warmup_shrinkage_blocks_premature_correction(self) -> None:
        """First few corrections must not be applied at full strength."""
        seg = EWMASegment(alpha=0.5, warmup_n=10, significance_k=0.0)
        seg.update(100.0)  # raw value 50.0 after first-update shrinkage
        assert abs(seg.predict()) < abs(seg.value), "warmup must apply <100% weight at n=1"

    def test_warmup_clears_after_n_updates(self) -> None:
        seg = EWMASegment(alpha=0.5, warmup_n=5, significance_k=0.0)
        for _ in range(5):
            seg.update(10.0)
        # n>=warmup_n -> full strength, gate is 0.0 so always emits
        assert seg.predict() == pytest.approx(seg.value, rel=1e-9)

    def test_significance_gate_filters_noise(self) -> None:
        """
        On pure zero-mean noise, the EWMA inevitably wanders some,
        but the significance gate must keep the *applied* correction
        far smaller than the noise σ averaged across seeds.
        """
        sigma = 10.0
        applied_corrections = []
        for seed in range(20):
            seg = EWMASegment(alpha=0.10, warmup_n=1, significance_k=0.5)
            rng = np.random.default_rng(seed)
            for _ in range(60):
                seg.update(float(rng.normal(0.0, sigma)))
            applied_corrections.append(abs(seg.predict()))
        mean_applied = sum(applied_corrections) / len(applied_corrections)
        # On pure noise, mean applied correction should be << σ.
        assert mean_applied < 0.4 * sigma, (
            f"Significance gate not suppressing noise: "
            f"mean |applied correction| = {mean_applied:.2f} (σ = {sigma})"
        )

    def test_significance_gate_passes_real_signal(self) -> None:
        seg = EWMASegment(alpha=0.2, warmup_n=1, significance_k=0.5)
        for _ in range(50):
            seg.update(20.0)
        # Constant residual: |value| ~ 20, MAD ~ 0  -> gate trivially open
        assert seg.predict() == pytest.approx(20.0, abs=1.0)

    def test_half_life_is_documented(self) -> None:
        """Pin the EWMA's adaptation speed so a future α tweak is forced
        through code review instead of silently breaking convergence."""
        learner = OnlineResidualLearner()
        # log(0.5)/log(1-α). At α=0.10 -> ~6.58 corrections.
        expected = math.log(0.5) / math.log(1 - settings.online_alpha)
        assert learner.half_life_corrections() == pytest.approx(expected, rel=1e-6)
        # Sanity range — too fast -> overshoots, too slow -> never adapts.
        assert 4.0 <= learner.half_life_corrections() <= 30.0


class TestRidgeSegment:
    def test_predict_zero_before_updates(self) -> None:
        seg = RidgeSegment()
        assert seg.predict(15.0, 0.0, 0.0) == pytest.approx(0.0)

    def test_learns_rain_residual(self) -> None:
        seg = RidgeSegment(lam=0.01, warmup_n=1)
        for _ in range(50):
            seg.update(residual=-20.0, temp_c=15.0, precip_mm=10.0, event_intensity=0.0)
        pred = seg.predict(temp_c=15.0, precip_mm=10.0, event_intensity=0.0)
        assert pred < -5.0, f"Expected negative rain adjustment, got {pred}"

    def test_warmup_gates_first_predictions(self) -> None:
        seg = RidgeSegment(lam=0.01, warmup_n=10)
        seg.update(residual=-20.0, temp_c=15.0, precip_mm=10.0, event_intensity=0.0)
        warm_pred = seg.predict(temp_c=15.0, precip_mm=10.0, event_intensity=0.0)
        for _ in range(20):
            seg.update(residual=-20.0, temp_c=15.0, precip_mm=10.0, event_intensity=0.0)
        full_pred = seg.predict(temp_c=15.0, precip_mm=10.0, event_intensity=0.0)
        assert abs(warm_pred) < abs(full_pred)

    def test_serialise_roundtrip(self) -> None:
        seg = RidgeSegment()
        seg.update(5.0, 20.0, 0.0, 0.5)
        restored = RidgeSegment.from_dict(seg.to_dict())
        assert np.allclose(restored.w, seg.w)


class TestResidualClamp:
    def test_clamp_caps_extreme_residuals(self) -> None:
        # Predicted=100, factor=0.5 -> max |residual| = 50
        assert _clamp_residual(200.0, 100.0, 0.5) == 50.0
        assert _clamp_residual(-200.0, 100.0, 0.5) == -50.0

    def test_clamp_passes_through_when_within_cap(self) -> None:
        assert _clamp_residual(20.0, 100.0, 0.5) == 20.0

    def test_clamp_handles_zero_predicted(self) -> None:
        assert _clamp_residual(20.0, 0.0, 0.5) == 20.0


class TestOnlineResidualLearner:
    def test_inactive_before_first_update(self) -> None:
        learner = OnlineResidualLearner()
        assert not learner.is_active

    def test_predicts_zero_for_unseen_segment(self) -> None:
        learner = OnlineResidualLearner()
        result = learner.predict(weekday=0, hour=19)
        assert result == pytest.approx(0.0)

    def test_learns_persistent_positive_residual(self) -> None:
        learner = OnlineResidualLearner()
        for _ in range(40):
            learner.update(weekday=0, hour=19, residual=30.0, baseline_predicted=100.0)
        pred = learner.predict(weekday=0, hour=19)
        # Clamped to ±0.5*100=50, learned value should be substantial.
        assert pred > 5.0

    def test_clamp_prevents_one_shot_poisoning(self) -> None:
        """A single absurd residual must not move the segment by more than the cap."""
        learner = OnlineResidualLearner()
        # Predicted 100, residual 1000 -> clamped to ±50
        learner.update(weekday=0, hour=19, residual=1000.0, baseline_predicted=100.0)
        # First-update shrinkage halves it to 25 max
        assert abs(learner._ewma["0_19"].value) <= 25.0 + 1e-9

    def test_segments_are_independent(self) -> None:
        learner = OnlineResidualLearner()
        for _ in range(10):
            learner.update(weekday=0, hour=12, residual=50.0, baseline_predicted=100.0)
        assert learner.predict(weekday=1, hour=12) == pytest.approx(0.0)

    def test_coverage_reports(self) -> None:
        learner = OnlineResidualLearner()
        for h in (12, 13, 18, 19, 20):
            learner.update(weekday=2, hour=h, residual=5.0, baseline_predicted=80.0)
        cov = learner.coverage()
        assert cov["cells_touched"] == 5
        assert cov["cells_total"] == 168
        assert cov["total_updates"] == 5

    def test_serialise_roundtrip(self) -> None:
        learner = OnlineResidualLearner()
        for h in [12, 19]:
            for _ in range(5):
                learner.update(weekday=2, hour=h, residual=float(h), baseline_predicted=100.0)
        restored = OnlineResidualLearner.from_dict(learner.to_dict())
        assert restored.predict(weekday=2, hour=12) == pytest.approx(
            learner.predict(weekday=2, hour=12), abs=1e-6
        )

    def test_finite_output_after_many_updates(self) -> None:
        learner = OnlineResidualLearner()
        rng = np.random.default_rng(42)
        for _ in range(200):
            learner.update(
                weekday=int(rng.integers(0, 7)),
                hour=int(rng.integers(0, 24)),
                residual=float(rng.normal(0, 50)),
                baseline_predicted=100.0,
                temp_c=float(rng.normal(15, 5)),
                precip_mm=float(rng.uniform(0, 20)),
                event_intensity=float(rng.uniform(0, 2)),
            )
        pred = learner.predict(
            weekday=0, hour=19, temp_c=15.0, precip_mm=5.0, event_intensity=0.5
        )
        assert math.isfinite(pred)


class TestNoRegressionInvariant:
    """
    The headline correctness property: when corrections are noise around the
    true value, the online layer must not push the prediction further from
    the baseline. This is the invariant that the original implementation
    silently violated on ~50% of days in the live simulation log.
    """

    def test_zero_mean_noise_does_not_corrupt_baseline(self) -> None:
        learner = OnlineResidualLearner()
        rng = np.random.default_rng(123)
        baseline = 100.0
        # 30 zero-mean noisy residuals
        for _ in range(30):
            residual = float(rng.normal(0.0, 5.0))
            learner.update(
                weekday=3, hour=19, residual=residual, baseline_predicted=baseline
            )
        # Significance gate + warmup shrinkage should keep the correction near 0.
        correction = learner.predict(weekday=3, hour=19)
        assert abs(correction) < 3.0, (
            f"Online layer drifted to {correction} on pure zero-mean noise — "
            f"this is exactly the regression that caused corrected_mape > baseline_mape."
        )
