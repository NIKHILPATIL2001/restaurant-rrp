"""Unit tests for inventory math: z update bounds and order policy."""


from rrp.forecasting.inventory import InventoryRecommender


def _make_recommender() -> InventoryRecommender:
    rec = InventoryRecommender()
    rec._z_per_ingredient = {}
    return rec


def test_z_increases_on_stockout() -> None:
    rec = _make_recommender()
    initial = rec._get_z(1)
    rec.update_z(1, stockout=True, excess_waste=False)
    assert rec._get_z(1) > initial


def test_z_decreases_on_excess_waste() -> None:
    rec = _make_recommender()
    rec._z_per_ingredient[1] = 2.0
    rec.update_z(1, stockout=False, excess_waste=True)
    assert rec._get_z(1) < 2.0


def test_z_bounded_above() -> None:
    rec = _make_recommender()
    rec._z_per_ingredient[1] = rec._z_max - 0.01
    rec.update_z(1, stockout=True, excess_waste=False)
    assert rec._get_z(1) <= rec._z_max


def test_z_bounded_below() -> None:
    rec = _make_recommender()
    rec._z_per_ingredient[1] = rec._z_min + 0.01
    rec.update_z(1, stockout=False, excess_waste=True)
    assert rec._get_z(1) >= rec._z_min


def test_simultaneous_stockout_and_waste_increments() -> None:
    """Stockout takes priority over waste (net effect should be non-negative)."""
    rec = _make_recommender()
    initial = rec._get_z(1)
    rec.update_z(1, stockout=True, excess_waste=True)
    assert rec._get_z(1) >= initial
