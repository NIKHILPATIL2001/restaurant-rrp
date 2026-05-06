"""
Synthetic data generator for 18 months of restaurant history.

Produces deterministic output from a seed; covers are based on realistic
weekday/seasonality/weather/event patterns. Staff and inventory are derived
from covers via fixed ratios + noise.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MENU_ITEMS = [
    {"id": 1, "name": "Grilled Salmon", "station": "grill", "price": 28.0},
    {"id": 2, "name": "Caesar Salad", "station": "salad", "price": 14.0},
    {"id": 3, "name": "Ribeye Steak", "station": "grill", "price": 45.0},
    {"id": 4, "name": "Pasta Carbonara", "station": "saute", "price": 22.0},
    {"id": 5, "name": "Margherita Pizza", "station": "grill", "price": 18.0},
    {"id": 6, "name": "Fish Tacos", "station": "grill", "price": 20.0},
    {"id": 7, "name": "Veggie Burger", "station": "grill", "price": 16.0},
    {"id": 8, "name": "Chocolate Lava Cake", "station": "pastry", "price": 10.0},
]

INGREDIENTS = [
    {"id": 1, "name": "Salmon fillet", "unit": "kg", "shelf_life_days": 3, "lead_time_days": 1, "pack_size": 2.0},
    {"id": 2, "name": "Romaine lettuce", "unit": "head", "shelf_life_days": 5, "lead_time_days": 1, "pack_size": 12.0},
    {"id": 3, "name": "Ribeye", "unit": "kg", "shelf_life_days": 4, "lead_time_days": 2, "pack_size": 5.0},
    {"id": 4, "name": "Pasta", "unit": "kg", "shelf_life_days": 365, "lead_time_days": 2, "pack_size": 5.0},
    {"id": 5, "name": "Eggs", "unit": "dozen", "shelf_life_days": 21, "lead_time_days": 1, "pack_size": 12.0},
    {"id": 6, "name": "Pizza dough", "unit": "ball", "shelf_life_days": 2, "lead_time_days": 1, "pack_size": 10.0},
    {"id": 7, "name": "Mozzarella", "unit": "kg", "shelf_life_days": 7, "lead_time_days": 2, "pack_size": 2.0},
    {"id": 8, "name": "Beef patty", "unit": "each", "shelf_life_days": 3, "lead_time_days": 1, "pack_size": 20.0},
    {"id": 9, "name": "Flour", "unit": "kg", "shelf_life_days": 180, "lead_time_days": 3, "pack_size": 25.0},
    {"id": 10, "name": "Butter", "unit": "kg", "shelf_life_days": 30, "lead_time_days": 2, "pack_size": 2.0},
]

BOM_ENTRIES = [
    (1, 1, 0.25), (1, 10, 0.02),
    (2, 2, 1.0), (2, 5, 0.1),
    (3, 3, 0.35), (3, 10, 0.02),
    (4, 4, 0.15), (4, 5, 0.15), (4, 10, 0.03),
    (5, 6, 1.0), (5, 7, 0.12),
    (6, 1, 0.12), (6, 9, 0.05),
    (7, 8, 1.0), (7, 9, 0.05),
    (8, 9, 0.08), (8, 10, 0.05),
]

ROLES = ["server", "line_cook", "busser", "host", "dishwasher", "manager"]
STATIONS = ["grill", "saute", "salad", "expo", "bar", "pastry"]

ROLE_RATIOS: dict[str, float] = {
    "server": 1 / 20,
    "line_cook": 1 / 30,
    "busser": 1 / 40,
    "host": 1 / 80,
    "dishwasher": 1 / 60,
    "manager": 1 / 120,
}

WEEKDAY_FACTOR = [0.65, 0.70, 0.75, 0.85, 1.00, 1.30, 1.20]
MONTH_FACTOR = [0.80, 0.78, 0.85, 0.90, 1.00, 1.10, 1.15, 1.15, 1.05, 0.95, 0.90, 1.20]

HOUR_SHARE = [
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  # 00-05 closed
    0.01, 0.02, 0.03, 0.04, 0.05, 0.06,  # 06-11 breakfast  (sum 0.21)
    0.08, 0.10, 0.07, 0.05, 0.04, 0.03,  # 12-17 lunch      (sum 0.37)
    0.08, 0.10, 0.09, 0.08, 0.06, 0.01,  # 18-23 dinner     (sum 0.42)
]
assert abs(sum(HOUR_SHARE) - 1.0) < 1e-9


def _dish_mix(rng: np.random.Generator, n_items: int = len(MENU_ITEMS)) -> list[float]:
    alpha = np.ones(n_items) * 2.0
    mix = rng.dirichlet(alpha)
    return mix.tolist()


def generate(
    seed: int = 42,
    months: int = 18,
    base_daily_covers: float = 200.0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    start = date(2024, 11, 1)
    n_days = months * 30

    dates = [start + timedelta(days=i) for i in range(n_days)]

    service_rows: list[dict[str, Any]] = []
    weather_rows: list[dict[str, Any]] = []
    staffing_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    on_hand: dict[int, float] = {ing["id"]: 20.0 for ing in INGREDIENTS}

    event_dates = set(
        (start + timedelta(days=int(d))).isoformat()
        for d in rng.choice(n_days, size=n_days // 15, replace=False)
    )

    for d in dates:
        wday = d.weekday()
        month_idx = d.month - 1
        temp_c = float(
            15.0
            + 10.0 * np.sin((d.timetuple().tm_yday / 365) * 2 * np.pi)
            + rng.normal(0, 2)
        )
        precip_mm = max(0.0, float(rng.normal(0, 5)) if rng.random() < 0.3 else 0.0)
        rain_factor = max(0.6, 1.0 - precip_mm * 0.02)

        is_event = d.isoformat() in event_dates
        event_intensity = float(rng.uniform(0.5, 1.5)) if is_event else 0.0
        event_boost = 1.0 + 0.2 * event_intensity

        daily_covers = (
            base_daily_covers
            * WEEKDAY_FACTOR[wday]
            * MONTH_FACTOR[month_idx]
            * rain_factor
            * event_boost
            * float(rng.lognormal(0, 0.08))
        )

        if is_event:
            event_rows.append({
                "date": d.isoformat(),
                "type": rng.choice(["concert", "sports", "festival", "private_event"]),
                "intensity": round(event_intensity, 3),
            })

        dish_mix_day = _dish_mix(rng)

        for hour in range(24):
            share = HOUR_SHARE[hour]
            if share == 0.0:
                wx_temp = temp_c + float(rng.normal(0, 0.5))
                weather_rows.append({
                    "date": d.isoformat(), "hour": hour,
                    "temp_c": round(wx_temp, 1), "precip_mm": round(precip_mm, 2),
                    "wind_kph": round(max(0, float(rng.normal(15, 5))), 1),
                    "condition": "clear",
                })
                service_rows.append({
                    "date": d.isoformat(), "hour": hour,
                    "covers_actual": 0.0,
                    "dish_mix_actual_json": json.dumps({}),
                })
                continue

            hourly_covers = daily_covers * share * float(rng.lognormal(0, 0.05))
            condition = "rain" if precip_mm > 5 else "cloudy" if precip_mm > 0 else "clear"

            weather_rows.append({
                "date": d.isoformat(), "hour": hour,
                "temp_c": round(temp_c + float(rng.normal(0, 0.5)), 1),
                "precip_mm": round(precip_mm, 2),
                "wind_kph": round(max(0, float(rng.normal(15, 5))), 1),
                "condition": condition,
            })

            service_rows.append({
                "date": d.isoformat(),
                "hour": hour,
                "covers_actual": round(max(0.0, hourly_covers), 1),
                "dish_mix_actual_json": json.dumps(
                    {str(i + 1): round(v, 4) for i, v in enumerate(dish_mix_day)}
                ),
            })

            for role in ROLES:
                ratio = ROLE_RATIOS[role]
                hc = max(1.0, hourly_covers * ratio * float(rng.lognormal(0, 0.1)))
                staffing_rows.append({
                    "date": d.isoformat(), "hour": hour, "role": role,
                    "headcount_actual": round(hc, 1),
                })

        for ing in INGREDIENTS:
            iid = ing["id"]
            usage = 0.0
            for (mi_id, ing_id, qty) in BOM_ENTRIES:
                if ing_id == iid:
                    mi_idx = mi_id - 1
                    usage += daily_covers * dish_mix_day[mi_idx] * qty
            usage *= float(rng.lognormal(0, 0.1))
            wasted = usage * float(rng.uniform(0.0, 0.05))
            on_hand[iid] = max(0.0, on_hand[iid] - usage)
            order = max(0.0, usage * 1.2 - on_hand[iid])
            on_hand[iid] += order
            inventory_rows.append({
                "date": d.isoformat(), "ingredient_id": iid,
                "qty_actual_used": round(usage, 3),
                "qty_wasted": round(wasted, 3),
                "on_hand": round(on_hand[iid], 3),
                "qty_predicted": round(order, 3),
            })

    return {
        "menu_items": MENU_ITEMS,
        "ingredients": INGREDIENTS,
        "bom": [{"menu_item_id": m, "ingredient_id": i, "qty_per_serving": q} for m, i, q in BOM_ENTRIES],
        "roles": [{"id": i + 1, "name": r} for i, r in enumerate(ROLES)],
        "stations": [{"id": i + 1, "name": s} for i, s in enumerate(STATIONS)],
        "service_periods": service_rows,
        "weather": weather_rows,
        "staffing": staffing_rows,
        "inventory_orders": inventory_rows,
        "events": event_rows,
        "restaurant": {"id": 1, "name": "The Demo Kitchen", "timezone": "America/Chicago"},
    }


def to_dataframes(data: dict[str, Any]) -> dict[str, pd.DataFrame]:
    return {k: pd.DataFrame(v) for k, v in data.items() if isinstance(v, list)}


def save_parquet(data: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dfs = to_dataframes(data)
    for name, df in dfs.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
