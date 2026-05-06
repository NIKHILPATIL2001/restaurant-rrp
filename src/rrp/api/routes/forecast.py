from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from rrp.api.schemas import (
    CoversRequest,
    CoversResponse,
    HourlyCover,
    InventoryRequest,
    InventoryResponse,
    StaffRequest,
    StaffResponse,
)
from rrp.core.db import get_db
from rrp.core.logging import get_logger
from rrp.forecasting.covers import get_forecaster
from rrp.forecasting.inventory import get_inventory_recommender
from rrp.forecasting.staff import get_staff_recommender

log = get_logger(__name__)
router = APIRouter(prefix="/forecast", tags=["forecast"])


def _log_forecast(
    surface: str,
    target_date: str,
    regime: str,
    degraded: list[str],
    extra: dict[str, object] | None = None,
) -> None:
    """Single structured log line per forecast request."""
    payload: dict[str, object] = {
        "surface": surface,
        "date": target_date,
        "regime": regime,
        "degraded_features": degraded,
    }
    if extra:
        payload.update(extra)
    log.info("forecast.served", **payload)


@router.post("/covers", response_model=CoversResponse, summary="Forecast hourly covers")
def forecast_covers(
    req: CoversRequest,
    db: Session = Depends(get_db),
) -> CoversResponse:
    forecaster = get_forecaster()
    result = forecaster.forecast(req.date, db)

    hourly = result["hourly_covers"]
    band = result["confidence_band"]
    detail = [
        HourlyCover(
            hour=h,
            covers=v,
            lower=round(max(0.0, v * (1 - band)), 1),
            upper=round(v * (1 + band), 1),
        )
        for h, v in enumerate(hourly)
    ]
    _log_forecast(
        "covers", str(result["date"]), result["regime"], result["degraded_features"],
        extra={
            "daily_total": round(result["daily_total"], 1),
            "online_layer_active": result["online_layer_active"],
        },
    )
    return CoversResponse(
        date=result["date"],
        restaurant_id=req.restaurant_id,
        regime=result["regime"],
        daily_total=result["daily_total"],
        hourly_covers=hourly,
        hourly_detail=detail,
        confidence_band=band,
        degraded_features=result["degraded_features"],
        online_layer_active=result["online_layer_active"],
    )


@router.post("/staff", response_model=StaffResponse, summary="Recommend staff schedule")
def forecast_staff(
    req: StaffRequest,
    db: Session = Depends(get_db),
) -> StaffResponse:
    forecaster = get_forecaster()
    covers_result = forecaster.forecast(req.date, db)
    recommender = get_staff_recommender()
    result = recommender.recommend(req.date, covers_result["hourly_covers"], db)
    _log_forecast(
        "staff", str(result["date"]), covers_result["regime"], covers_result["degraded_features"],
        extra={
            "roles": list(result["schedule"].keys()),
            "stations": [s["station"] for s in result.get("station_load", [])],
        },
    )
    return StaffResponse(
        date=result["date"],
        schedule=result["schedule"],
        hourly_raw=result["hourly_raw"],
        station_load=result.get("station_load", []),
    )


@router.post("/inventory", response_model=InventoryResponse, summary="Recommend ingredient orders")
def forecast_inventory(
    req: InventoryRequest,
    db: Session = Depends(get_db),
) -> InventoryResponse:
    forecaster = get_forecaster()
    covers_result = forecaster.forecast(req.date, db)
    recommender = get_inventory_recommender()
    result = recommender.recommend(
        target_date=req.date,
        horizon_days=req.horizon_days,
        hourly_covers=covers_result["hourly_covers"],
        db=db,
    )
    stockout_count = sum(1 for o in result.get("orders", []) if o.get("stockout_risk"))
    _log_forecast(
        "inventory", str(result["date"]), covers_result["regime"], covers_result["degraded_features"],
        extra={
            "horizon_days": req.horizon_days,
            "n_orders": len(result.get("orders", [])),
            "stockout_risk_count": stockout_count,
        },
    )
    return InventoryResponse(**result)
