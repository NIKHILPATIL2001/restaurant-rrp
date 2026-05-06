"""
Two-stage LightGBM covers forecaster with cold-start regime selector.

Stage A: daily total covers.
Stage B: intraday hour share (normalised to 1, multiplied by Stage A).
Online layer: EWMA + ridge residual applied on top.

Regime selector:
  prior  — < 7 distinct days with actuals
  sparse — 7–27 distinct days
  full   — >= 28 distinct days (both LightGBM stages active)
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sqlalchemy.orm import Session

from rrp.core.config import settings
from rrp.core.logging import get_logger
from rrp.domain.models import Regime, ServicePeriod
from rrp.forecasting.features import (
    FEATURE_COLS_DAILY,
    FEATURE_COLS_HOURLY,
    build_daily_features,
    build_hourly_features,
    build_training_dataframe,
    get_event_intensity_for_date,
    get_weather_for_date,
    mape,
)
from rrp.forecasting.online import OnlineResidualLearner

log = get_logger(__name__)

_PRIOR_PATH = Path("data/priors.yaml")


def _load_priors() -> dict[str, Any]:
    path = _PRIOR_PATH if _PRIOR_PATH.exists() else settings.data_path / "priors.yaml"
    with open(path) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def _get_regime(db: Session) -> Regime:
    distinct_dates: int = (
        db.query(ServicePeriod.date)
        .filter(ServicePeriod.covers_actual.isnot(None))
        .distinct()
        .count()
    )
    if distinct_dates < 7:
        return Regime.prior
    if distinct_dates < settings.sparse_threshold_days:
        return Regime.sparse
    return Regime.full


def _prior_forecast(
    target_date: date,
    regime: Regime,
    db: Session,
    priors: dict[str, Any],
) -> tuple[list[float], float]:
    """Return per-hour covers list + confidence band factor from priors."""
    covers_cfg = priors["covers"]
    wday = target_date.weekday()
    base = covers_cfg["base_daily"]
    wday_factor = covers_cfg["weekday_factor"][wday]

    if regime == Regime.sparse:
        rows = (
            db.query(ServicePeriod)
            .filter(ServicePeriod.covers_actual.isnot(None))
            .all()
        )
        if rows:
            daily_sums = {}
            for r in rows:
                key = str(r.date)
                daily_sums[key] = daily_sums.get(key, 0.0) + (r.covers_actual or 0.0)
            base = float(np.mean(list(daily_sums.values())))

    daily_total = base * wday_factor
    hour_shares = covers_cfg["hour_share"]
    hourly = [daily_total * hour_shares[h] for h in range(24)]
    band = covers_cfg[f"confidence_band_{regime.value}"]
    return hourly, band


class CoversForecaster:
    """
    Two-stage LightGBM covers forecaster.

    Loads trained models from disk; falls back to prior when not trained.
    """

    MODEL_A = "covers_stage_a"
    MODEL_B = "covers_stage_b"

    def __init__(self) -> None:
        self._model_a: lgb.Booster | None = None
        self._model_b: lgb.Booster | None = None
        self._online: OnlineResidualLearner = OnlineResidualLearner()
        self._priors: dict[str, Any] = _load_priors()
        self._load_models()

        # Historical min/max per (weekday, hour) for clipping
        self._clip_bounds: dict[str, tuple[float, float]] = {}

    def _load_models(self) -> None:
        path_a = settings.models_path / f"{self.MODEL_A}.lgb"
        path_b = settings.models_path / f"{self.MODEL_B}.lgb"
        if path_a.exists():
            self._model_a = lgb.Booster(model_file=str(path_a))
            log.info("covers.model_loaded", stage="A")
        if path_b.exists():
            self._model_b = lgb.Booster(model_file=str(path_b))
            log.info("covers.model_loaded", stage="B")

        online_path = settings.models_path / "online_residual.joblib"
        if online_path.exists():
            state = joblib.load(online_path)
            self._online = OnlineResidualLearner.from_dict(state)
            log.info("covers.online_loaded")

        clip_path = settings.models_path / "clip_bounds.joblib"
        if clip_path.exists():
            self._clip_bounds = joblib.load(clip_path)

    def reload(self) -> None:
        self._load_models()

    def _get_clip_bounds(self, weekday: int, hour: int) -> tuple[float, float]:
        key = f"{weekday}_{hour}"
        if key in self._clip_bounds:
            lo, hi = self._clip_bounds[key]
            return lo * settings.clip_lower_factor, hi * settings.clip_upper_factor
        return 0.0, 2000.0

    def forecast(
        self,
        target_date: date,
        db: Session,
    ) -> dict[str, Any]:
        regime = _get_regime(db)
        degraded_features: list[str] = []

        if regime in (Regime.prior, Regime.sparse) or self._model_a is None:
            hourly, band = _prior_forecast(target_date, regime, db, self._priors)
            hourly_with_online = self._apply_online(
                hourly, target_date, db, degraded_features
            )
            confidence_band = band
        else:
            daily_feat, deg = build_daily_features(db, target_date)
            degraded_features.extend(deg)
            feat_df_a = pd.DataFrame([daily_feat])[FEATURE_COLS_DAILY]
            daily_total = float(self._model_a.predict(feat_df_a)[0])
            daily_total = max(0.0, daily_total)

            hourly_shares: list[float] = []
            for h in range(24):
                h_feat, h_deg = build_hourly_features(db, target_date, h)
                if h_deg and "weather" not in degraded_features:
                    degraded_features.extend(h_deg)
                feat_df_b = pd.DataFrame([h_feat])[FEATURE_COLS_HOURLY]
                share = float(self._model_b.predict(feat_df_b)[0]) if self._model_b else 0.0
                hourly_shares.append(max(0.0, share))

            share_sum = sum(hourly_shares) or 1.0
            hourly = [daily_total * (s / share_sum) for s in hourly_shares]
            hourly_with_online = self._apply_online(
                hourly, target_date, db, degraded_features
            )
            confidence_band = self._priors["covers"]["confidence_band_full"]

        # Apply clipping
        clipped_hourly = []
        for h, val in enumerate(hourly_with_online):
            lo, hi = self._get_clip_bounds(target_date.weekday(), h)
            clipped = max(lo, min(hi, val))
            if clipped != val:
                log.warning(
                    "covers.clipped",
                    hour=h,
                    original=round(val, 2),
                    clipped=round(clipped, 2),
                )
            clipped_hourly.append(max(0.0, clipped))

        return {
            "regime": regime.value,
            "date": str(target_date),
            "hourly_covers": [round(v, 1) for v in clipped_hourly],
            "daily_total": round(sum(clipped_hourly), 1),
            "confidence_band": confidence_band,
            "degraded_features": degraded_features,
            "online_layer_active": self._online.is_active,
        }

    def _apply_online(
        self,
        base_hourly: list[float],
        target_date: date,
        db: Session,
        degraded_features: list[str],
    ) -> list[float]:
        weather, _ = get_weather_for_date(db, target_date)
        event_intensity, _ = get_event_intensity_for_date(db, target_date)
        weekday = target_date.weekday()
        result = []
        for h, base in enumerate(base_hourly):
            correction = self._online.predict(
                weekday=weekday,
                hour=h,
                temp_c=weather["temp_c"],
                precip_mm=weather["precip_mm"],
                event_intensity=event_intensity,
            )
            result.append(base + correction)
        return result

    def update_online(
        self,
        target_date: date,
        hour: int,
        actual: float,
        predicted: float,
        db: Session,
    ) -> None:
        """
        Update the online residual learner with a new observation.

        The residual is computed against the BASELINE LightGBM prediction
        (not the already-corrected prediction) so each observation produces
        an independent measurement of bias. The baseline value is also passed
        to the learner for the residual magnitude clamp.
        """
        baseline_hourly = self.predict_baseline_hourly(target_date, db)
        baseline_at_hour = baseline_hourly[hour] if 0 <= hour < len(baseline_hourly) else predicted
        residual = actual - baseline_at_hour

        weather, _ = get_weather_for_date(db, target_date)
        event_intensity, _ = get_event_intensity_for_date(db, target_date)
        self._online.update(
            weekday=target_date.weekday(),
            hour=hour,
            residual=residual,
            baseline_predicted=baseline_at_hour,
            temp_c=weather["temp_c"],
            precip_mm=weather["precip_mm"],
            event_intensity=event_intensity,
        )
        joblib.dump(
            self._online.to_dict(),
            settings.models_path / "online_residual.joblib",
        )

    def predict_baseline_hourly(
        self, target_date: date, db: Session
    ) -> list[float]:
        """LightGBM-only prediction without online residual — used for MAPE baseline."""
        regime = _get_regime(db)
        if regime in (Regime.prior, Regime.sparse) or self._model_a is None:
            hourly, _ = _prior_forecast(target_date, regime, db, self._priors)
            return hourly

        daily_feat, _ = build_daily_features(db, target_date)
        feat_df_a = pd.DataFrame([daily_feat])[FEATURE_COLS_DAILY]
        daily_total = float(max(0.0, self._model_a.predict(feat_df_a)[0]))

        hourly_shares = []
        for h in range(24):
            h_feat, _ = build_hourly_features(db, target_date, h)
            feat_df_b = pd.DataFrame([h_feat])[FEATURE_COLS_HOURLY]
            share = float(self._model_b.predict(feat_df_b)[0]) if self._model_b else 0.0
            hourly_shares.append(max(0.0, share))
        share_sum = sum(hourly_shares) or 1.0
        return [daily_total * (s / share_sum) for s in hourly_shares]


def train_covers_models(db: Session, models_dir: Path) -> dict[str, float]:
    """
    Train Stage A and Stage B LightGBM models on all available data.
    Returns eval metrics dict.
    """
    daily_df, hourly_df = build_training_dataframe(db)
    if daily_df.empty or len(daily_df) < 14:
        log.warning("covers.train_skip", reason="insufficient data")
        return {}

    models_dir.mkdir(parents=True, exist_ok=True)

    # Stage A
    X_a = daily_df[FEATURE_COLS_DAILY].values
    y_a = daily_df["covers_day"].values
    split_a = int(len(X_a) * 0.8)
    X_train_a, X_val_a = X_a[:split_a], X_a[split_a:]
    y_train_a, y_val_a = y_a[:split_a], y_a[split_a:]

    ds_train_a = lgb.Dataset(X_train_a, label=y_train_a, feature_name=FEATURE_COLS_DAILY)
    ds_val_a = lgb.Dataset(X_val_a, label=y_val_a, reference=ds_train_a)
    params_a = {
        "objective": "regression",
        "metric": "mape",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
        "seed": 42,
    }
    callbacks = [lgb.early_stopping(20, verbose=False), lgb.log_evaluation(period=-1)]
    model_a = lgb.train(
        params_a, ds_train_a,
        num_boost_round=300,
        valid_sets=[ds_val_a],
        callbacks=callbacks,
    )
    model_a.save_model(str(models_dir / "covers_stage_a.lgb"))

    val_mape_a = mape(y_val_a, model_a.predict(X_val_a))

    # Stage B
    X_b = hourly_df[FEATURE_COLS_HOURLY].values
    y_b = hourly_df["hour_share"].values
    split_b = int(len(X_b) * 0.8)
    X_train_b, X_val_b = X_b[:split_b], X_b[split_b:]
    y_train_b, y_val_b = y_b[:split_b], y_b[split_b:]

    ds_train_b = lgb.Dataset(X_train_b, label=y_train_b, feature_name=FEATURE_COLS_HOURLY)
    ds_val_b = lgb.Dataset(X_val_b, label=y_val_b, reference=ds_train_b)
    params_b = {
        "objective": "regression",
        "metric": "mape",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "verbose": -1,
        "seed": 42,
    }
    model_b = lgb.train(
        params_b, ds_train_b,
        num_boost_round=200,
        valid_sets=[ds_val_b],
        callbacks=callbacks,
    )
    model_b.save_model(str(models_dir / "covers_stage_b.lgb"))
    val_mape_b = mape(y_val_b, model_b.predict(X_val_b))

    # Compute clip bounds from training data
    clip_bounds: dict[str, tuple[float, float]] = {}
    for (wday, hour), grp in hourly_df.groupby(
        [hourly_df["date"].dt.dayofweek, "hour"]
    ):
        covers_series = grp["covers_actual"]
        clip_bounds[f"{int(wday)}_{int(hour)}"] = (
            float(covers_series.min()),
            float(covers_series.max()),
        )
    joblib.dump(clip_bounds, models_dir / "clip_bounds.joblib")

    # Feature importance — basic explainability artefact for the model card.
    importance: dict[str, dict[str, list[Any]]] = {
        "stage_a": {
            "features": list(FEATURE_COLS_DAILY),
            "gain": [int(v) for v in model_a.feature_importance(importance_type="gain")],
            "split": [int(v) for v in model_a.feature_importance(importance_type="split")],
        },
        "stage_b": {
            "features": list(FEATURE_COLS_HOURLY),
            "gain": [int(v) for v in model_b.feature_importance(importance_type="gain")],
            "split": [int(v) for v in model_b.feature_importance(importance_type="split")],
        },
    }
    importance_path = models_dir / "feature_importance.json"
    importance_path.write_text(json.dumps(importance, indent=2))

    log.info(
        "covers.trained",
        val_mape_a=round(val_mape_a, 4),
        val_mape_b=round(val_mape_b, 4),
        feature_importance_dumped=str(importance_path.name),
    )
    return {"val_mape_stage_a": val_mape_a, "val_mape_stage_b": val_mape_b}


def load_feature_importance(models_dir: Path | None = None) -> dict[str, Any]:
    """Read the feature importance artefact written by `train_covers_models`."""
    path = (models_dir or settings.models_path) / "feature_importance.json"
    if not path.exists():
        return {}
    parsed: dict[str, Any] = json.loads(path.read_text())
    return parsed


# Global singleton — loaded once at import time, reloaded after retrain
_forecaster: CoversForecaster | None = None


def get_forecaster() -> CoversForecaster:
    global _forecaster
    if _forecaster is None:
        _forecaster = CoversForecaster()
    return _forecaster


def reload_forecaster() -> None:
    global _forecaster
    _forecaster = CoversForecaster()
