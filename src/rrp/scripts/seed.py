"""
Idempotent seed script.  Generates 18 months of synthetic data and loads
it into Postgres if the database is empty (no service_period rows).
Also saves parquet files for offline training.
"""

from __future__ import annotations

import argparse

from sqlalchemy import text

from rrp.core.config import settings
from rrp.core.db import SessionLocal
from rrp.core.logging import configure_logging, get_logger
from rrp.data.generator import generate, save_parquet
from rrp.domain.models import (
    BOM,
    Event,
    Ingredient,
    InventoryOrder,
    MenuItem,
    Restaurant,
    Role,
    ServicePeriod,
    Staffing,
    Station,
    Weather,
)

configure_logging()
log = get_logger(__name__)


def seed(force: bool = False) -> None:
    db = SessionLocal()
    try:
        count = db.execute(text("SELECT COUNT(*) FROM service_periods")).scalar()
        if count and count > 0 and not force:
            log.info("seed.skip", reason="service_periods already populated", rows=count)
            return

        log.info("seed.start", months=18)
        data = generate(seed=42, months=18)

        save_parquet(data, settings.data_path)

        db.execute(text("TRUNCATE restaurants, menu_items, ingredients, bom, roles, stations, weather, events, service_periods, staffing, inventory_orders CASCADE"))

        rest = data["restaurant"]
        db.add(Restaurant(id=rest["id"], name=rest["name"], timezone=rest["timezone"]))

        for item in data["menu_items"]:
            db.add(MenuItem(**item))
        for ing in data["ingredients"]:
            db.add(Ingredient(**ing))
        for entry in data["bom"]:
            db.add(BOM(**entry))
        for role in data["roles"]:
            db.add(Role(**role))
        for stn in data["stations"]:
            db.add(Station(**stn))

        db.flush()

        role_id_map = {r.name: r.id for r in db.query(Role).all()}

        batch_size = 1000
        weather_batch = []
        service_batch = []
        staff_batch = []
        inv_batch = []
        event_batch = []

        for row in data["weather"]:
            weather_batch.append(Weather(**row))
            if len(weather_batch) >= batch_size:
                db.bulk_save_objects(weather_batch)
                weather_batch = []
        if weather_batch:
            db.bulk_save_objects(weather_batch)

        for row in data["service_periods"]:
            service_batch.append(ServicePeriod(
                date=row["date"],
                hour=row["hour"],
                covers_actual=row["covers_actual"],
                dish_mix_actual_json=row["dish_mix_actual_json"],
            ))
            if len(service_batch) >= batch_size:
                db.bulk_save_objects(service_batch)
                service_batch = []
        if service_batch:
            db.bulk_save_objects(service_batch)

        db.flush()
        {
            (str(sp.date), sp.hour): sp.id
            for sp in db.query(ServicePeriod).all()
        }

        for row in data["staffing"]:
            role_id = role_id_map.get(row["role"])
            if role_id is None:
                continue
            staff_batch.append(Staffing(
                date=row["date"],
                hour=row["hour"],
                role_id=role_id,
                headcount_actual=row["headcount_actual"],
            ))
            if len(staff_batch) >= batch_size:
                db.bulk_save_objects(staff_batch)
                staff_batch = []
        if staff_batch:
            db.bulk_save_objects(staff_batch)

        for row in data["inventory_orders"]:
            inv_batch.append(InventoryOrder(
                date=row["date"],
                ingredient_id=row["ingredient_id"],
                qty_actual_used=row["qty_actual_used"],
                qty_wasted=row["qty_wasted"],
                on_hand=row["on_hand"],
                qty_predicted=row["qty_predicted"],
            ))
            if len(inv_batch) >= batch_size:
                db.bulk_save_objects(inv_batch)
                inv_batch = []
        if inv_batch:
            db.bulk_save_objects(inv_batch)

        for row in data["events"]:
            event_batch.append(Event(
                date=row["date"],
                type=row["type"],
                intensity=row["intensity"],
            ))
        if event_batch:
            db.bulk_save_objects(event_batch)

        db.commit()
        log.info("seed.done", service_periods=len(data["service_periods"]))
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--if-empty", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    seed(force=args.force)


if __name__ == "__main__":
    main()
