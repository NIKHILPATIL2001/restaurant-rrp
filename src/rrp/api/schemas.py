"""Pydantic v2 request/response schemas with OpenAPI examples."""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, Field

# ---------- Forecast ----------

class CoversRequest(BaseModel):
    date: dt.date = Field(..., examples=["2025-06-15"])
    restaurant_id: int = Field(1, examples=[1])

    model_config = {"json_schema_extra": {"examples": [{"date": "2025-06-15", "restaurant_id": 1}]}}


class HourlyCover(BaseModel):
    hour: int
    covers: float
    lower: float
    upper: float


class CoversResponse(BaseModel):
    date: str
    restaurant_id: int
    regime: str
    daily_total: float
    hourly_covers: list[float]
    hourly_detail: list[HourlyCover]
    confidence_band: float
    degraded_features: list[str]
    online_layer_active: bool


class StaffRequest(BaseModel):
    date: dt.date = Field(..., examples=["2025-06-15"])
    restaurant_id: int = Field(1, examples=[1])

    model_config = {"json_schema_extra": {"examples": [{"date": "2025-06-15", "restaurant_id": 1}]}}


class StationLoad(BaseModel):
    station: str
    hourly_minutes: list[float]
    peak_hour: int
    peak_minutes: float


class StaffResponse(BaseModel):
    date: str
    schedule: dict[str, list[int]]
    hourly_raw: dict[str, list[float]]
    station_load: list[StationLoad] = Field(default_factory=list)


class InventoryRequest(BaseModel):
    date: dt.date = Field(..., examples=["2025-06-15"])
    restaurant_id: int = Field(1, examples=[1])
    horizon_days: int = Field(3, ge=1, le=14, examples=[3])

    model_config = {"json_schema_extra": {"examples": [{"date": "2025-06-15", "restaurant_id": 1, "horizon_days": 3}]}}


class IngredientOrder(BaseModel):
    ingredient_id: int
    ingredient_name: str
    unit: str
    order_qty: float
    expected_usage: float
    on_hand: float
    lead_time_days: int
    shelf_life_days: int
    safety_stock_z: float
    stockout_risk: bool
    rationale: str


class InventoryResponse(BaseModel):
    date: str
    horizon_days: int
    orders: list[IngredientOrder]


# ---------- Corrections ----------

class CorrectionRequest(BaseModel):
    scope: str = Field(..., examples=["covers"], description="covers | staff | inventory")
    scope_id: str | None = Field(None, examples=["server"], description="role name (staff) or ingredient_id (inventory)")
    date: dt.date = Field(..., examples=["2025-06-14"])
    hour: int | None = Field(None, ge=0, le=23, examples=[19])
    predicted: float = Field(..., examples=[120.0])
    actual: float = Field(..., examples=[85.0])
    reason_code: str = Field("unknown", examples=["rain"])
    note: str | None = Field(None, examples=["Heavy rain all evening"])

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "scope": "covers", "date": "2025-06-14", "hour": 19,
                "predicted": 120.0, "actual": 85.0,
                "reason_code": "rain", "note": "Heavy rain all evening"
            }]
        }
    }


class CorrectionResponse(BaseModel):
    correction_id: int
    quarantined: bool
    online_updated: bool
    message: str


# ---------- Metrics ----------

class ConvergencePoint(BaseModel):
    date: str
    baseline_mape: float | None
    corrected_mape: float | None
    n_corrections: int
    model_version: str | None


class ConvergenceResponse(BaseModel):
    surface: str
    series: list[ConvergencePoint]
    summary: dict[str, Any]


# ---------- Admin ----------

class RetrainResponse(BaseModel):
    promoted: bool
    version: str | None
    new_mape: float | None
    message: str


# ---------- Health ----------

class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    db: str
    model: str
