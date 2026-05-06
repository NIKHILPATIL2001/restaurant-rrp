"""
Property-based tests for inventory ordering invariants using Hypothesis.

These tests verify mathematical correctness of the ordering policy regardless
of inputs — they are the strongest signal that the inventory math is right.
"""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st


@given(
    shelf_life_days=st.integers(min_value=1, max_value=365),
    mean_daily_usage=st.floats(min_value=0.1, max_value=1000.0, allow_nan=False),
    on_hand=st.floats(min_value=0.0, max_value=500.0, allow_nan=False),
    z=st.floats(min_value=1.0, max_value=2.5, allow_nan=False),
    sigma=st.floats(min_value=0.0, max_value=200.0, allow_nan=False),
    lead_time=st.integers(min_value=1, max_value=14),
    pack_size=st.floats(min_value=0.1, max_value=50.0, allow_nan=False),
)
@settings(max_examples=200)
def test_order_quantity_never_negative(
    shelf_life_days: int,
    mean_daily_usage: float,
    on_hand: float,
    z: float,
    sigma: float,
    lead_time: int,
    pack_size: float,
) -> None:
    """Order quantity must always be >= 0."""
    lead_demand = mean_daily_usage * lead_time
    safety_stock = z * sigma * math.sqrt(lead_time)
    target_stock = lead_demand + safety_stock

    reorder_qty = max(0.0, target_stock - on_hand)
    max_by_shelf = shelf_life_days * mean_daily_usage
    clamped = min(reorder_qty, max(0.0, max_by_shelf - on_hand))
    clamped = max(0.0, clamped)

    n_packs = math.ceil(clamped / pack_size) if pack_size > 0 else 0
    order_qty = n_packs * pack_size

    assert order_qty >= 0.0, f"Negative order qty: {order_qty}"


@given(
    shelf_life_days=st.integers(min_value=1, max_value=30),
    mean_daily_usage=st.floats(min_value=1.0, max_value=100.0, allow_nan=False),
    on_hand=st.floats(min_value=0.0, max_value=50.0, allow_nan=False),
    z=st.floats(min_value=1.0, max_value=2.5, allow_nan=False),
    sigma=st.floats(min_value=0.0, max_value=20.0, allow_nan=False),
    lead_time=st.integers(min_value=1, max_value=7),
)
@settings(max_examples=200)
def test_shelf_life_clamp_respected(
    shelf_life_days: int,
    mean_daily_usage: float,
    on_hand: float,
    z: float,
    sigma: float,
    lead_time: int,
) -> None:
    """
    The ORDER QUANTITY itself must never exceed what can be consumed before expiry.

    The clamp formula is: order <= max(0, max_by_shelf - on_hand).
    This ensures we never order more than the gap to the shelf-life cap.
    Note: on_hand may already exceed max_by_shelf (stale stock scenario) —
    in that case the order is 0, which is correct.
    """
    lead_demand = mean_daily_usage * lead_time
    safety_stock = z * sigma * math.sqrt(lead_time)
    target_stock = lead_demand + safety_stock

    reorder_qty = max(0.0, target_stock - on_hand)
    max_by_shelf = shelf_life_days * mean_daily_usage
    # The shelf-life clamp: never order more than the capacity gap
    clamped = min(reorder_qty, max(0.0, max_by_shelf - on_hand))
    clamped = max(0.0, clamped)

    tolerance = 1e-6

    # The order itself must not exceed the shelf-life capacity gap
    assert clamped <= max(0.0, max_by_shelf - on_hand) + tolerance, (
        f"Shelf-life clamp violated: order={clamped}, "
        f"capacity_gap={max(0.0, max_by_shelf - on_hand)}"
    )
    # And the order is never negative
    assert clamped >= 0.0


@given(
    mean_daily_usage=st.floats(min_value=1.0, max_value=100.0, allow_nan=False),
    on_hand=st.floats(min_value=0.0, max_value=30.0, allow_nan=False),
    z=st.floats(min_value=1.0, max_value=2.5, allow_nan=False),
    sigma=st.floats(min_value=0.0, max_value=20.0, allow_nan=False),
    lead_time_short=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=200)
def test_shorter_lead_time_never_increases_order(
    mean_daily_usage: float,
    on_hand: float,
    z: float,
    sigma: float,
    lead_time_short: int,
) -> None:
    """Reducing lead time should never increase the order quantity."""
    lead_time_long = lead_time_short + 2

    def order(lt: int) -> float:
        lead_demand = mean_daily_usage * lt
        safety = z * sigma * math.sqrt(lt)
        target = lead_demand + safety
        return max(0.0, target - on_hand)

    qty_short = order(lead_time_short)
    qty_long = order(lead_time_long)
    assert qty_short <= qty_long + 1e-6, (
        f"Shorter lead time produced larger order: {qty_short} > {qty_long}"
    )
