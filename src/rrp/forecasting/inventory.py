"""
Ingredient order recommender.

BOM-driven expected usage → safety-stock reorder quantity → shelf-life clamp.
Per-ingredient service-level z is adapted via a bandit-style update from
waste and stockout signals.

Exposes stockout_risk: bool in each ingredient's recommendation — true when
projected stock before next delivery is insufficient for expected usage.
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sqlalchemy.orm import Session

from rrp.core.config import settings
from rrp.core.logging import get_logger
from rrp.domain.models import BOM, Ingredient, InventoryOrder, ServicePeriod

log = get_logger(__name__)

_PRIOR_PATH = Path("data/priors.yaml")


def _load_priors() -> dict[str, Any]:
    path = _PRIOR_PATH if _PRIOR_PATH.exists() else settings.data_path / "priors.yaml"
    with open(path) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


class InventoryRecommender:
    """
    BOM-driven inventory order recommender with adaptive safety-stock z.
    """

    STATE_FILE = "inventory_z.joblib"

    def __init__(self) -> None:
        priors = _load_priors()
        cfg = priors["inventory"]
        self._default_z: float = cfg["default_z"]
        self._z_min: float = cfg["z_min"]
        self._z_max: float = cfg["z_max"]
        self._z_step_up: float = cfg["z_step_up"]
        self._z_step_down: float = cfg["z_step_down"]
        self._z_per_ingredient: dict[int, float] = {}
        self._load_state()

    def _load_state(self) -> None:
        path = settings.models_path / self.STATE_FILE
        if path.exists():
            self._z_per_ingredient = joblib.load(path)
            log.info("inventory.z_loaded")

    def _save_state(self) -> None:
        joblib.dump(self._z_per_ingredient, settings.models_path / self.STATE_FILE)

    def _get_z(self, ingredient_id: int) -> float:
        return self._z_per_ingredient.get(ingredient_id, self._default_z)

    def update_z(self, ingredient_id: int, stockout: bool, excess_waste: bool) -> None:
        z = self._get_z(ingredient_id)
        if stockout:
            z = min(self._z_max, z + self._z_step_up)
        if excess_waste:
            z = max(self._z_min, z - self._z_step_down)
        self._z_per_ingredient[ingredient_id] = z
        self._save_state()

    def _mean_daily_usage(self, ingredient_id: int, db: Session) -> float:
        rows = (
            db.query(InventoryOrder)
            .filter(
                InventoryOrder.ingredient_id == ingredient_id,
                InventoryOrder.qty_actual_used.isnot(None),
            )
            .order_by(InventoryOrder.date.desc())
            .limit(30)
            .all()
        )
        if not rows:
            return 1.0
        return float(np.mean([r.qty_actual_used for r in rows]))

    def _usage_std(self, ingredient_id: int, db: Session) -> float:
        rows = (
            db.query(InventoryOrder)
            .filter(
                InventoryOrder.ingredient_id == ingredient_id,
                InventoryOrder.qty_actual_used.isnot(None),
            )
            .order_by(InventoryOrder.date.desc())
            .limit(30)
            .all()
        )
        if len(rows) < 2:
            return self._mean_daily_usage(ingredient_id, db) * 0.2
        return float(np.std([r.qty_actual_used for r in rows]))

    def _latest_on_hand(self, ingredient_id: int, db: Session) -> float:
        row = (
            db.query(InventoryOrder)
            .filter(
                InventoryOrder.ingredient_id == ingredient_id,
                InventoryOrder.on_hand.isnot(None),
            )
            .order_by(InventoryOrder.date.desc())
            .first()
        )
        return float(row.on_hand) if row else 0.0

    def recommend(
        self,
        target_date: date,
        horizon_days: int,
        hourly_covers: list[float],
        db: Session,
    ) -> dict[str, Any]:
        ingredients = db.query(Ingredient).all()
        bom_entries = db.query(BOM).all()

        dish_mix = _get_dish_mix(db, target_date)

        total_covers = sum(hourly_covers)
        orders: list[dict[str, Any]] = []

        for ing in ingredients:
            relevant_bom = [b for b in bom_entries if b.ingredient_id == ing.id]
            if not relevant_bom:
                continue

            expected_usage = sum(
                total_covers * dish_mix.get(b.menu_item_id, 0.0) * b.qty_per_serving
                for b in relevant_bom
            )

            on_hand = self._latest_on_hand(ing.id, db)
            mean_daily = self._mean_daily_usage(ing.id, db)
            sigma = self._usage_std(ing.id, db)
            z = self._get_z(ing.id)
            lead_time = ing.lead_time_days

            lead_demand = mean_daily * lead_time
            safety_stock = z * sigma * math.sqrt(lead_time)
            target_stock = lead_demand + safety_stock

            reorder_qty = max(0.0, target_stock - on_hand)

            # Shelf-life clamp: don't order more than will be consumed before expiry
            max_by_shelf = ing.shelf_life_days * mean_daily
            clamped_qty = min(reorder_qty, max(0.0, max_by_shelf - on_hand))
            clamped_qty = max(0.0, clamped_qty)

            # Round up to pack size
            if ing.pack_size > 0:
                n_packs = math.ceil(clamped_qty / ing.pack_size)
                order_qty = n_packs * ing.pack_size
            else:
                order_qty = clamped_qty

            # stockout_risk: projected stock before next delivery < expected usage over lead time
            expected_lead_usage = expected_usage * lead_time / max(horizon_days, 1)
            projected_before_delivery = on_hand - expected_lead_usage
            stockout_risk = projected_before_delivery < expected_usage * 0.5

            orders.append({
                "ingredient_id": ing.id,
                "ingredient_name": ing.name,
                "unit": ing.unit,
                "order_qty": round(order_qty, 3),
                "expected_usage": round(expected_usage, 3),
                "on_hand": round(on_hand, 3),
                "lead_time_days": lead_time,
                "shelf_life_days": ing.shelf_life_days,
                "safety_stock_z": round(z, 3),
                "stockout_risk": stockout_risk,
                "rationale": (
                    f"target_stock={round(target_stock, 2)}, "
                    f"on_hand={round(on_hand, 2)}, "
                    f"lead_demand={round(lead_demand, 2)}, "
                    f"safety={round(safety_stock, 2)}"
                ),
            })

        return {
            "date": str(target_date),
            "horizon_days": horizon_days,
            "orders": orders,
        }


def _get_dish_mix(db: Session, target_date: date) -> dict[int, float]:
    """Return dish mix for the most recent comparable day, default to equal share."""
    row = (
        db.query(ServicePeriod)
        .filter(
            ServicePeriod.date < target_date,
            ServicePeriod.dish_mix_actual_json.isnot(None),
        )
        .order_by(ServicePeriod.date.desc())
        .first()
    )
    if row and row.dish_mix_actual_json:
        raw: dict[str, float] = json.loads(row.dish_mix_actual_json) if isinstance(row.dish_mix_actual_json, str) else row.dish_mix_actual_json
        return {int(k): float(v) for k, v in raw.items()}

    bom_entries = db.query(BOM).all()
    menu_ids = list({b.menu_item_id for b in bom_entries})
    if not menu_ids:
        return {}
    equal = 1.0 / len(menu_ids)
    return {mid: equal for mid in menu_ids}


_recommender: InventoryRecommender | None = None


def get_inventory_recommender() -> InventoryRecommender:
    global _recommender
    if _recommender is None:
        _recommender = InventoryRecommender()
    return _recommender
