"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-05-01

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "restaurants",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("open_hours_json", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("timezone", sa.String(50), nullable=False, server_default="UTC"),
    )

    op.create_table(
        "menu_items",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("station", sa.String(100), nullable=False),
        sa.Column("price", sa.Float, server_default="0.0"),
        sa.Column("active", sa.Boolean, server_default="true"),
    )

    op.create_table(
        "ingredients",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("unit", sa.String(50), nullable=False),
        sa.Column("shelf_life_days", sa.Integer, server_default="7"),
        sa.Column("lead_time_days", sa.Integer, server_default="2"),
        sa.Column("pack_size", sa.Float, server_default="1.0"),
    )

    op.create_table(
        "bom",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("menu_item_id", sa.Integer, sa.ForeignKey("menu_items.id"), nullable=False),
        sa.Column("ingredient_id", sa.Integer, sa.ForeignKey("ingredients.id"), nullable=False),
        sa.Column("qty_per_serving", sa.Float, nullable=False),
        sa.UniqueConstraint("menu_item_id", "ingredient_id"),
    )

    op.create_table(
        "roles",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(100), nullable=False, unique=True),
    )

    op.create_table(
        "stations",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(100), nullable=False, unique=True),
    )

    op.create_table(
        "weather",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("hour", sa.Integer, nullable=False),
        sa.Column("temp_c", sa.Float, server_default="15.0"),
        sa.Column("precip_mm", sa.Float, server_default="0.0"),
        sa.Column("wind_kph", sa.Float, server_default="10.0"),
        sa.Column("condition", sa.String(50), server_default="clear"),
        sa.UniqueConstraint("date", "hour"),
    )

    op.create_table(
        "events",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("date", sa.Date, nullable=False, index=True),
        sa.Column("type", sa.String(100), nullable=False),
        sa.Column("intensity", sa.Float, server_default="1.0"),
    )

    op.create_table(
        "service_periods",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("date", sa.Date, nullable=False, index=True),
        sa.Column("hour", sa.Integer, nullable=False),
        sa.Column("covers_predicted", sa.Float, nullable=True),
        sa.Column("covers_actual", sa.Float, nullable=True),
        sa.Column("dish_mix_actual_json", sa.JSON, nullable=True),
        sa.UniqueConstraint("date", "hour"),
    )

    op.create_table(
        "staffing",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("date", sa.Date, nullable=False, index=True),
        sa.Column("hour", sa.Integer, nullable=False),
        sa.Column("role_id", sa.Integer, sa.ForeignKey("roles.id"), nullable=False),
        sa.Column("headcount_predicted", sa.Float, nullable=True),
        sa.Column("headcount_actual", sa.Float, nullable=True),
        sa.UniqueConstraint("date", "hour", "role_id"),
    )

    op.create_table(
        "inventory_orders",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("date", sa.Date, nullable=False, index=True),
        sa.Column("ingredient_id", sa.Integer, sa.ForeignKey("ingredients.id"), nullable=False),
        sa.Column("qty_predicted", sa.Float, nullable=True),
        sa.Column("qty_actual_used", sa.Float, nullable=True),
        sa.Column("qty_wasted", sa.Float, nullable=True),
        sa.Column("on_hand", sa.Float, nullable=True),
        sa.UniqueConstraint("date", "ingredient_id"),
    )

    op.create_table(
        "corrections",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "scope",
            sa.Enum("covers", "staff", "inventory", name="correctionscope"),
            nullable=False,
        ),
        sa.Column("scope_id", sa.String(200), nullable=True),
        sa.Column("predicted", sa.Float, nullable=False),
        sa.Column("actual", sa.Float, nullable=False),
        sa.Column(
            "reason_code",
            sa.Enum(
                "rain",
                "event",
                "holiday",
                "closure",
                "staff_shortage",
                "pos_error",
                "unknown",
                name="reasoncode",
            ),
            server_default="unknown",
            nullable=False,
        ),
        sa.Column("note", sa.Text, nullable=True),
        sa.Column("quarantined", sa.Boolean, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

    op.create_table(
        "model_runs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("metrics_json", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("model_path", sa.String(500), nullable=False),
        sa.Column("promoted", sa.Boolean, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

    op.create_table(
        "daily_metrics",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("date", sa.Date, nullable=False, index=True),
        sa.Column(
            "surface",
            sa.Enum("covers", "staff", "inventory", name="surface"),
            nullable=False,
        ),
        sa.Column("baseline_mape", sa.Float, nullable=True),
        sa.Column("corrected_mape", sa.Float, nullable=True),
        sa.Column("n_corrections", sa.Integer, server_default="0"),
        sa.Column("model_version", sa.String(50), nullable=True),
        sa.UniqueConstraint("date", "surface"),
    )


def downgrade() -> None:
    op.drop_table("daily_metrics")
    op.drop_table("model_runs")
    op.drop_table("corrections")
    op.drop_table("inventory_orders")
    op.drop_table("staffing")
    op.drop_table("service_periods")
    op.drop_table("events")
    op.drop_table("weather")
    op.drop_table("stations")
    op.drop_table("roles")
    op.drop_table("bom")
    op.drop_table("ingredients")
    op.drop_table("menu_items")
    op.drop_table("restaurants")
    op.execute("DROP TYPE IF EXISTS correctionscope")
    op.execute("DROP TYPE IF EXISTS reasoncode")
    op.execute("DROP TYPE IF EXISTS surface")
