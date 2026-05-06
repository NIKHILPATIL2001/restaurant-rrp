"""
Shared feature engineering for all forecasting models.

Degraded-feature fallbacks are implemented here: if weather or event data is
missing for a requested date, we substitute historical means and log a warning
via structlog. The caller receives a 'degraded_features' list indicating which
features were fallen back.
"""

from __future__ import annotations

from datetime import date, timedelta

import holidays
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from rrp.core.logging import get_logger
from rrp.domain.models import Event, ServicePeriod, Weather

log = get_logger(__name__)

FEATURE_COLS_DAILY = [
    "day_of_week", "month", "is_weekend", "is_holiday",
    "lag_1d", "lag_7d", "lag_28d",
    "rolling_mean_7d", "rolling_mean_28d",
    "weather_temp_c", "weather_precip_mm",
    "event_intensity",
]

FEATURE_COLS_HOURLY = [
    "day_of_week", "month", "hour", "is_weekend", "is_holiday",
    "weather_temp_c", "weather_precip_mm",
    "event_intensity",
]

_us_holidays = holidays.country_holidays("US")


def _is_holiday(d: date) -> int:
    return 1 if d in _us_holidays else 0


def get_weather_for_date(
    db: Session, target_date: date
) -> tuple[dict[str, float], list[str]]:
    """Return aggregated weather for a date, with fallback to historical mean."""
    degraded: list[str] = []
    rows = db.query(Weather).filter(Weather.date == target_date).all()
    if rows:
        temp = float(np.mean([r.temp_c for r in rows]))
        precip = float(np.sum([r.precip_mm for r in rows]))
        return {"temp_c": temp, "precip_mm": precip}, degraded

    # Fallback: historical mean for this month.
    # Logged at debug level only — the helper has no request-level context, and
    # repeated identical warnings (one per hour) flood the logs. The route layer
    # emits a single aggregated summary via the `degraded_features` field on
    # the response and a structured log line per request.
    degraded.append("weather")
    log.debug("feature.fallback", feature="weather", date=str(target_date))
    historical = (
        db.query(Weather)
        .filter(
            Weather.date >= target_date - timedelta(days=365),
            Weather.date < target_date,
        )
        .all()
    )
    if historical:
        same_month = [r for r in historical if r.date.month == target_date.month]
        pool = same_month if same_month else historical
        temp = float(np.mean([r.temp_c for r in pool]))
        precip = float(np.mean([r.precip_mm for r in pool]))
    else:
        temp, precip = 15.0, 0.0

    return {"temp_c": temp, "precip_mm": precip}, degraded


def get_event_intensity_for_date(
    db: Session, target_date: date
) -> tuple[float, list[str]]:
    """Return summed event intensity for a date, defaulting to 0."""
    degraded: list[str] = []
    rows = db.query(Event).filter(Event.date == target_date).all()
    if rows:
        return float(sum(r.intensity for r in rows)), degraded
    return 0.0, degraded


def _get_daily_covers_series(db: Session, before: date, lookback: int = 60) -> pd.Series:
    rows = (
        db.query(ServicePeriod)
        .filter(
            ServicePeriod.date >= before - timedelta(days=lookback),
            ServicePeriod.date < before,
            ServicePeriod.covers_actual.isnot(None),
        )
        .all()
    )
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame({"date": [r.date for r in rows], "covers": [r.covers_actual for r in rows]})
    daily = df.groupby("date")["covers"].sum()
    daily.index = pd.to_datetime(daily.index)
    return daily.sort_index()


def build_daily_features(
    db: Session,
    target_date: date,
) -> tuple[dict[str, float], list[str]]:
    """Build feature dict for Stage A (daily total) model."""
    weather, deg_weather = get_weather_for_date(db, target_date)
    event_intensity, _ = get_event_intensity_for_date(db, target_date)

    series = _get_daily_covers_series(db, target_date, lookback=60)

    def _lag(n: int) -> float:
        lag_date = pd.Timestamp(target_date - timedelta(days=n))
        return float(series.get(lag_date, series.mean() if len(series) > 0 else 200.0))

    def _rolling(n: int) -> float:
        recent = series.iloc[-n:] if len(series) >= n else series
        return float(recent.mean()) if len(recent) > 0 else 200.0

    features = {
        "day_of_week": float(target_date.weekday()),
        "month": float(target_date.month),
        "is_weekend": float(target_date.weekday() >= 5),
        "is_holiday": float(_is_holiday(target_date)),
        "lag_1d": _lag(1),
        "lag_7d": _lag(7),
        "lag_28d": _lag(28),
        "rolling_mean_7d": _rolling(7),
        "rolling_mean_28d": _rolling(28),
        "weather_temp_c": weather["temp_c"],
        "weather_precip_mm": weather["precip_mm"],
        "event_intensity": event_intensity,
    }
    return features, deg_weather


def build_hourly_features(
    db: Session,
    target_date: date,
    hour: int,
) -> tuple[dict[str, float], list[str]]:
    """Build feature dict for Stage B (intraday share) model."""
    weather, deg_weather = get_weather_for_date(db, target_date)
    event_intensity, _ = get_event_intensity_for_date(db, target_date)

    features = {
        "day_of_week": float(target_date.weekday()),
        "month": float(target_date.month),
        "hour": float(hour),
        "is_weekend": float(target_date.weekday() >= 5),
        "is_holiday": float(_is_holiday(target_date)),
        "weather_temp_c": weather["temp_c"],
        "weather_precip_mm": weather["precip_mm"],
        "event_intensity": event_intensity,
    }
    return features, deg_weather


def build_training_dataframe(db: Session) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build (daily_df, hourly_df) from all available service_periods with actuals.
    Used for LightGBM training.
    """
    rows = (
        db.query(ServicePeriod)
        .filter(ServicePeriod.covers_actual.isnot(None))
        .all()
    )
    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame({
        "date": pd.to_datetime([r.date for r in rows]),
        "hour": [r.hour for r in rows],
        "covers_actual": [r.covers_actual for r in rows],
    }).sort_values(["date", "hour"])

    weather_rows = db.query(Weather).all()
    wx_df = pd.DataFrame({
        "date": pd.to_datetime([r.date for r in weather_rows]),
        "hour": [r.hour for r in weather_rows],
        "temp_c": [r.temp_c for r in weather_rows],
        "precip_mm": [r.precip_mm for r in weather_rows],
    })
    wx_daily = wx_df.groupby("date").agg(
        weather_temp_c=("temp_c", "mean"),
        weather_precip_mm=("precip_mm", "sum"),
    ).reset_index()

    event_rows = db.query(Event).all()
    ev_df = pd.DataFrame({
        "date": pd.to_datetime([r.date for r in event_rows]),
        "intensity": [r.intensity for r in event_rows],
    }).groupby("date").agg(event_intensity=("intensity", "sum")).reset_index()

    daily = df.groupby("date")["covers_actual"].sum().reset_index()
    daily.columns = pd.Index(["date", "covers_day"])
    daily = daily.merge(wx_daily, on="date", how="left").merge(ev_df, on="date", how="left")
    daily["weather_temp_c"] = daily["weather_temp_c"].fillna(15.0)
    daily["weather_precip_mm"] = daily["weather_precip_mm"].fillna(0.0)
    daily["event_intensity"] = daily["event_intensity"].fillna(0.0)
    daily["day_of_week"] = daily["date"].dt.dayofweek.astype(float)
    daily["month"] = daily["date"].dt.month.astype(float)
    daily["is_weekend"] = (daily["date"].dt.dayofweek >= 5).astype(float)
    daily["is_holiday"] = daily["date"].apply(lambda d: float(d.date() in _us_holidays))

    daily = daily.sort_values("date").reset_index(drop=True)
    for lag in [1, 7, 28]:
        daily[f"lag_{lag}d"] = daily["covers_day"].shift(lag)
    daily["rolling_mean_7d"] = daily["covers_day"].shift(1).rolling(7, min_periods=1).mean()
    daily["rolling_mean_28d"] = daily["covers_day"].shift(1).rolling(28, min_periods=1).mean()
    daily = daily.dropna(subset=["lag_1d"])

    hourly = df.merge(
        daily[["date", "covers_day"]],
        on="date", how="inner",
    )
    hourly["hour_share"] = hourly["covers_actual"] / hourly["covers_day"].clip(lower=1.0)
    hourly = hourly.merge(wx_daily, on="date", how="left").merge(ev_df, on="date", how="left")
    hourly["weather_temp_c"] = hourly["weather_temp_c"].fillna(15.0)
    hourly["weather_precip_mm"] = hourly["weather_precip_mm"].fillna(0.0)
    hourly["event_intensity"] = hourly["event_intensity"].fillna(0.0)
    hourly["day_of_week"] = hourly["date"].dt.dayofweek.astype(float)
    hourly["month"] = hourly["date"].dt.month.astype(float)
    hourly["is_weekend"] = (hourly["date"].dt.dayofweek >= 5).astype(float)
    hourly["is_holiday"] = hourly["date"].apply(lambda d: float(d.date() in _us_holidays))
    hourly["hour"] = hourly["hour"].astype(float)

    return daily, hourly


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error, skipping zeros."""
    mask = actual > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])))
