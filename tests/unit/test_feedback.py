"""Unit tests for the feedback/plausibility service."""

from rrp.domain.models import ReasonCode
from rrp.feedback.service import _is_plausible


def test_plausible_small_delta() -> None:
    assert _is_plausible(predicted=100.0, actual=85.0, reason_code=ReasonCode.rain) is True


def test_implausible_large_delta() -> None:
    # delta_ratio = |700-100|/100 = 6.0 > 5.0 threshold → implausible
    assert _is_plausible(predicted=100.0, actual=700.0, reason_code=ReasonCode.unknown) is False


def test_closure_always_plausible() -> None:
    # Even a full zero-out (closure) should be accepted
    assert _is_plausible(predicted=200.0, actual=0.0, reason_code=ReasonCode.closure) is True


def test_zero_predicted_always_plausible() -> None:
    assert _is_plausible(predicted=0.0, actual=100.0, reason_code=ReasonCode.unknown) is True


def test_exactly_at_threshold_is_plausible() -> None:
    # delta_ratio = |600-100|/100 = 5.0 == threshold → plausible (<=)
    assert _is_plausible(predicted=100.0, actual=600.0, reason_code=ReasonCode.event) is True


def test_just_above_threshold_is_implausible() -> None:
    # delta_ratio = |601-100|/100 = 5.01 > 5.0 → implausible
    assert _is_plausible(predicted=100.0, actual=601.0, reason_code=ReasonCode.event) is False
