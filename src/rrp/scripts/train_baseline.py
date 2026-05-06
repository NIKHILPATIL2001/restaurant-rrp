"""
Idempotent baseline training script.
Trains LightGBM Stage A + Stage B and saves to models/.
Skips if models already exist and --if-missing is passed.
"""

from __future__ import annotations

import argparse

from rrp.core.config import settings
from rrp.core.db import SessionLocal
from rrp.core.logging import configure_logging, get_logger
from rrp.domain.models import ModelRun
from rrp.forecasting.covers import train_covers_models

configure_logging()
log = get_logger(__name__)


def train_baseline(if_missing: bool = False) -> None:
    models_dir = settings.models_path
    if if_missing and (models_dir / "covers_stage_a.lgb").exists():
        log.info("train_baseline.skip", reason="model already exists")
        return

    db = SessionLocal()
    try:
        log.info("train_baseline.start")
        metrics = train_covers_models(db, models_dir)
        if not metrics:
            log.warning("train_baseline.skip", reason="insufficient data")
            return

        run = ModelRun(
            model_name="covers",
            version="baseline_v1",
            metrics_json=metrics,
            model_path=str(models_dir),
            promoted=True,
        )
        db.add(run)
        db.commit()
        log.info("train_baseline.done", metrics=metrics)
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--if-missing", action="store_true")
    args = parser.parse_args()
    train_baseline(if_missing=args.if_missing)


if __name__ == "__main__":
    main()
