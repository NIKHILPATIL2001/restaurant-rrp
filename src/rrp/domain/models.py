import enum
from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rrp.core.db import Base


class CorrectionScope(enum.StrEnum):
    covers = "covers"
    staff = "staff"
    inventory = "inventory"


class ReasonCode(enum.StrEnum):
    rain = "rain"
    event = "event"
    holiday = "holiday"
    closure = "closure"
    staff_shortage = "staff_shortage"
    pos_error = "pos_error"
    unknown = "unknown"


class Surface(enum.StrEnum):
    covers = "covers"
    staff = "staff"
    inventory = "inventory"


class Regime(enum.StrEnum):
    prior = "prior"
    sparse = "sparse"
    full = "full"


class Restaurant(Base):
    __tablename__ = "restaurants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    open_hours_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    timezone: Mapped[str] = mapped_column(String(50), default="UTC")


class MenuItem(Base):
    __tablename__ = "menu_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    station: Mapped[str] = mapped_column(String(100), nullable=False)
    price: Mapped[float] = mapped_column(Float, default=0.0)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class Ingredient(Base):
    __tablename__ = "ingredients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    unit: Mapped[str] = mapped_column(String(50), nullable=False)
    shelf_life_days: Mapped[int] = mapped_column(Integer, default=7)
    lead_time_days: Mapped[int] = mapped_column(Integer, default=2)
    pack_size: Mapped[float] = mapped_column(Float, default=1.0)


class BOM(Base):
    """Bill of Materials: ingredient quantities per menu item."""

    __tablename__ = "bom"
    __table_args__ = (UniqueConstraint("menu_item_id", "ingredient_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    menu_item_id: Mapped[int] = mapped_column(ForeignKey("menu_items.id"), nullable=False)
    ingredient_id: Mapped[int] = mapped_column(ForeignKey("ingredients.id"), nullable=False)
    qty_per_serving: Mapped[float] = mapped_column(Float, nullable=False)

    menu_item: Mapped["MenuItem"] = relationship("MenuItem")
    ingredient: Mapped["Ingredient"] = relationship("Ingredient")


class Role(Base):
    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)


class Station(Base):
    __tablename__ = "stations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)


class Weather(Base):
    __tablename__ = "weather"
    __table_args__ = (UniqueConstraint("date", "hour"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    hour: Mapped[int] = mapped_column(Integer, nullable=False)
    temp_c: Mapped[float] = mapped_column(Float, default=15.0)
    precip_mm: Mapped[float] = mapped_column(Float, default=0.0)
    wind_kph: Mapped[float] = mapped_column(Float, default=10.0)
    condition: Mapped[str] = mapped_column(String(50), default="clear")


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    type: Mapped[str] = mapped_column(String(100), nullable=False)
    intensity: Mapped[float] = mapped_column(Float, default=1.0)


class ServicePeriod(Base):
    __tablename__ = "service_periods"
    __table_args__ = (UniqueConstraint("date", "hour"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    hour: Mapped[int] = mapped_column(Integer, nullable=False)
    covers_predicted: Mapped[float | None] = mapped_column(Float, nullable=True)
    covers_actual: Mapped[float | None] = mapped_column(Float, nullable=True)
    dish_mix_actual_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class Staffing(Base):
    __tablename__ = "staffing"
    __table_args__ = (UniqueConstraint("date", "hour", "role_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    hour: Mapped[int] = mapped_column(Integer, nullable=False)
    role_id: Mapped[int] = mapped_column(ForeignKey("roles.id"), nullable=False)
    headcount_predicted: Mapped[float | None] = mapped_column(Float, nullable=True)
    headcount_actual: Mapped[float | None] = mapped_column(Float, nullable=True)

    role: Mapped["Role"] = relationship("Role")


class InventoryOrder(Base):
    __tablename__ = "inventory_orders"
    __table_args__ = (UniqueConstraint("date", "ingredient_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    ingredient_id: Mapped[int] = mapped_column(ForeignKey("ingredients.id"), nullable=False)
    qty_predicted: Mapped[float | None] = mapped_column(Float, nullable=True)
    qty_actual_used: Mapped[float | None] = mapped_column(Float, nullable=True)
    qty_wasted: Mapped[float | None] = mapped_column(Float, nullable=True)
    on_hand: Mapped[float | None] = mapped_column(Float, nullable=True)

    ingredient: Mapped["Ingredient"] = relationship("Ingredient")


class Correction(Base):
    __tablename__ = "corrections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scope: Mapped[CorrectionScope] = mapped_column(Enum(CorrectionScope), nullable=False)
    scope_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    predicted: Mapped[float] = mapped_column(Float, nullable=False)
    actual: Mapped[float] = mapped_column(Float, nullable=False)
    reason_code: Mapped[ReasonCode] = mapped_column(
        Enum(ReasonCode), default=ReasonCode.unknown, nullable=False
    )
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    quarantined: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    model_path: Mapped[str] = mapped_column(String(500), nullable=False)
    promoted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DailyMetric(Base):
    __tablename__ = "daily_metrics"
    __table_args__ = (UniqueConstraint("date", "surface"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    surface: Mapped[Surface] = mapped_column(Enum(Surface), nullable=False)
    baseline_mape: Mapped[float | None] = mapped_column(Float, nullable=True)
    corrected_mape: Mapped[float | None] = mapped_column(Float, nullable=True)
    n_corrections: Mapped[int] = mapped_column(Integer, default=0)
    model_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
