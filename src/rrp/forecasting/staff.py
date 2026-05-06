"""
Staff recommender.

Inherits the covers forecast as the upstream signal.

Two outputs:
  1. Per-role headcount per hour, derived from a per-(weekday, daypart)
     covers_per_headcount ratio. The ratio is EWMA-updated from staffing
     corrections.
  2. Per-station minute-load per hour, derived from
        station_minutes = sum_over_dishes( covers * dish_mix[d] * minutes[d→station] )
     A station's peak load tells the kitchen how many bodies it actually
     needs at the line, independent of the front-of-house ratio.

A greedy smoother enforces a minimum-shift continuity constraint on the
headcount schedule (no 1-hour stubs).
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import yaml
from sqlalchemy.orm import Session

from rrp.core.config import settings
from rrp.core.logging import get_logger
from rrp.domain.models import BOM, MenuItem, Role, ServicePeriod

log = get_logger(__name__)

_PRIOR_PATH = Path("data/priors.yaml")

DAYPARTS = {
    "morning": range(6, 12),
    "lunch": range(12, 15),
    "afternoon": range(15, 17),
    "dinner": range(17, 22),
    "late": range(22, 24),
}

STATION_MINUTES_PER_COVER: dict[str, float] = {
    "grill": 3.5,
    "saute": 3.0,
    "salad": 1.5,
    "expo": 1.0,
    "bar": 2.0,
    "pastry": 1.0,
}

SHIFT_MINUTES = 480.0     # 8-hour shift in minutes
UTILIZATION_TARGET = 0.85
MAX_STAFF_PER_ROLE = 20
MIN_SHIFT_HOURS = 4


def _daypart(hour: int) -> str:
    for dp, hrs in DAYPARTS.items():
        if hour in hrs:
            return dp
    return "late"


def _load_priors() -> dict[str, Any]:
    path = _PRIOR_PATH if _PRIOR_PATH.exists() else settings.data_path / "priors.yaml"
    with open(path) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


class StaffRecommender:
    """
    Per-role headcount recommender inheriting from the covers forecast.
    Adaptive ratios are persisted to disk after each correction.
    """

    STATE_FILE = "staff_ratios.joblib"

    def __init__(self) -> None:
        priors = _load_priors()
        self._alpha: float = priors["staff"]["ewma_alpha"]
        self._default_ratios: dict[str, float] = priors["staff"]["covers_per_headcount"]
        # ratios[role][f"{weekday}_{daypart}"] = covers_per_headcount
        self._ratios: dict[str, dict[str, float]] = {}
        self._load_state()

    def _load_state(self) -> None:
        path = settings.models_path / self.STATE_FILE
        if path.exists():
            self._ratios = joblib.load(path)
            log.info("staff.ratios_loaded")

    def _save_state(self) -> None:
        joblib.dump(self._ratios, settings.models_path / self.STATE_FILE)

    def _get_ratio(self, role: str, weekday: int, daypart: str) -> float:
        key = f"{weekday}_{daypart}"
        return self._ratios.get(role, {}).get(key, self._default_ratios.get(role, 30.0))

    def update_ratio(self, role: str, weekday: int, hour: int, actual_hc: float, covers: float) -> None:
        if covers <= 0 or actual_hc <= 0:
            return
        dp = _daypart(hour)
        key = f"{weekday}_{dp}"
        observed_ratio = covers / actual_hc
        current = self._get_ratio(role, weekday, dp)
        updated = self._alpha * observed_ratio + (1 - self._alpha) * current
        if role not in self._ratios:
            self._ratios[role] = {}
        self._ratios[role][key] = updated
        self._save_state()

    def recommend(
        self,
        target_date: date,
        hourly_covers: list[float],
        db: Session,
    ) -> dict[str, Any]:
        """Return per-hour, per-role headcount + per-station minute load."""
        roles = db.query(Role).all()
        role_names = [r.name for r in roles] if roles else list(self._default_ratios.keys())
        weekday = target_date.weekday()

        schedule: dict[str, list[int]] = {r: [] for r in role_names}
        raw: dict[str, list[float]] = {r: [] for r in role_names}

        for h, covers in enumerate(hourly_covers):
            dp = _daypart(h)
            for role in role_names:
                ratio = self._get_ratio(role, weekday, dp)
                raw_hc = covers / ratio if ratio > 0 else 0.0
                raw[role].append(max(0.0, raw_hc))

        for role in role_names:
            smoothed = _smooth_schedule(raw[role], min_shift_hours=MIN_SHIFT_HOURS)
            schedule[role] = [min(MAX_STAFF_PER_ROLE, int(math.ceil(v))) for v in smoothed]

        station_load = _station_load(db, target_date, hourly_covers)

        return {
            "date": str(target_date),
            "schedule": schedule,
            "hourly_raw": {r: [round(v, 2) for v in raw[r]] for r in role_names},
            "station_load": station_load,
        }


def _station_load(
    db: Session,
    target_date: date,
    hourly_covers: list[float],
) -> list[dict[str, Any]]:
    """
    Compute per-station per-hour minute load:

        load[station][h] = sum_over_dishes( covers[h] * mix[d] * minutes[d->station] )

    Uses the most recent observed dish_mix from `service_periods`. Falls back
    to a uniform mix over the menu when no actuals exist (cold start).
    """
    menu_items = db.query(MenuItem).all()
    if not menu_items:
        return []

    item_to_station: dict[int, str] = {m.id: m.station for m in menu_items}
    n_items = len(menu_items)

    # Most recent recorded dish_mix as the prior for this date.
    recent = (
        db.query(ServicePeriod)
        .filter(ServicePeriod.dish_mix_actual_json.isnot(None))
        .order_by(ServicePeriod.date.desc(), ServicePeriod.hour.desc())
        .first()
    )
    mix: dict[int, float] = {m.id: 1.0 / n_items for m in menu_items}
    if recent and recent.dish_mix_actual_json:
        # SQLAlchemy JSON columns decode to Python dicts/lists automatically.
        raw = recent.dish_mix_actual_json
        try:
            if isinstance(raw, str):
                raw = json.loads(raw)
            if isinstance(raw, dict):
                mix = {
                    int(k): float(v)
                    for k, v in raw.items()
                    if int(k) in item_to_station
                }
        except (ValueError, TypeError):
            pass

    # Per-station, per-hour minutes
    station_hourly: dict[str, list[float]] = {
        s: [0.0] * 24 for s in STATION_MINUTES_PER_COVER
    }
    for h, covers in enumerate(hourly_covers):
        for item_id, share in mix.items():
            station = item_to_station.get(item_id)
            if station is None:
                continue
            station_hourly.setdefault(station, [0.0] * 24)
            station_hourly[station][h] += covers * share * STATION_MINUTES_PER_COVER.get(
                station, 1.5
            )

    out: list[dict[str, Any]] = []
    for station, hourly in station_hourly.items():
        peak_idx = max(range(24), key=lambda i: hourly[i])
        out.append({
            "station": station,
            "hourly_minutes": [round(v, 1) for v in hourly],
            "peak_hour": peak_idx,
            "peak_minutes": round(hourly[peak_idx], 1),
        })
    out.sort(key=lambda d: d["peak_minutes"], reverse=True)
    return out


# Verify BOM is reachable for type-checkers; the model is used implicitly via
# MenuItem.station + dish_mix and may be needed if the recommender is later
# extended to back-out station load from BOM.
_ = BOM


def _smooth_schedule(raw: list[float], min_shift_hours: int = 4) -> list[float]:
    """
    Simple greedy smoother: once a role is scheduled, keep them for at least
    min_shift_hours contiguous hours to avoid 1-hour stubs.
    """
    result = list(raw)
    n = len(result)
    i = 0
    while i < n:
        if result[i] > 0:
            end = i
            while end < n and result[end] > 0:
                end += 1
            span = end - i
            if span < min_shift_hours and span > 0:
                # extend or zero out
                extend_to = min(n, i + min_shift_hours)
                for j in range(i, extend_to):
                    result[j] = max(result[j], 1.0)
            i = end
        else:
            i += 1
    return result


_recommender: StaffRecommender | None = None


def get_staff_recommender() -> StaffRecommender:
    global _recommender
    if _recommender is None:
        _recommender = StaffRecommender()
    return _recommender
