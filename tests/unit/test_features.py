"""Unit tests for feature engineering utilities."""


import numpy as np
import pytest

from rrp.forecasting.features import mape


def test_mape_basic() -> None:
    actual = np.array([100.0, 200.0, 300.0])
    predicted = np.array([110.0, 180.0, 300.0])
    result = mape(actual, predicted)
    assert 0.0 < result < 1.0


def test_mape_perfect_prediction() -> None:
    actual = np.array([100.0, 200.0])
    result = mape(actual, actual)
    assert result == pytest.approx(0.0)


def test_mape_skips_zeros() -> None:
    actual = np.array([0.0, 100.0])
    predicted = np.array([50.0, 110.0])
    result = mape(actual, predicted)
    assert result == pytest.approx(0.1)


def test_mape_all_zeros_returns_nan() -> None:
    actual = np.array([0.0, 0.0])
    predicted = np.array([10.0, 20.0])
    result = mape(actual, predicted)
    assert np.isnan(result)
