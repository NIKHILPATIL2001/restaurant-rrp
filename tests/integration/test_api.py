"""
Integration tests for the FastAPI routes.

Uses TestClient with an in-memory/test DB.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from rrp.domain.models import Restaurant, Role, ServicePeriod


def _seed_basic(db: Session) -> None:
    db.add(Restaurant(id=1, name="Test Kitchen", timezone="UTC"))
    for i, role in enumerate(["server", "line_cook", "busser"], start=1):
        db.add(Role(id=i, name=role))
    d = date.today() - timedelta(days=3)
    for h in range(24):
        db.add(ServicePeriod(date=d, hour=h, covers_actual=float(10 + h)))
    db.flush()


class TestHealth:
    def test_healthz(self, client: TestClient) -> None:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_readyz(self, client: TestClient) -> None:
        r = client.get("/readyz")
        assert r.status_code == 200
        data = r.json()
        assert "db" in data
        assert "model" in data


class TestForecastCovers:
    def test_returns_24_hourly_covers(self, client: TestClient, db: Session) -> None:
        _seed_basic(db)
        tomorrow = str(date.today() + timedelta(days=1))
        r = client.post("/v1/forecast/covers", json={"date": tomorrow, "restaurant_id": 1})
        assert r.status_code == 200
        data = r.json()
        assert len(data["hourly_covers"]) == 24
        assert data["daily_total"] >= 0
        assert "regime" in data
        assert "degraded_features" in data

    def test_regime_is_valid(self, client: TestClient, db: Session) -> None:
        _seed_basic(db)
        tomorrow = str(date.today() + timedelta(days=1))
        r = client.post("/v1/forecast/covers", json={"date": tomorrow, "restaurant_id": 1})
        assert r.json()["regime"] in ("prior", "sparse", "full")

    def test_all_covers_non_negative(self, client: TestClient, db: Session) -> None:
        _seed_basic(db)
        tomorrow = str(date.today() + timedelta(days=1))
        r = client.post("/v1/forecast/covers", json={"date": tomorrow, "restaurant_id": 1})
        assert all(v >= 0 for v in r.json()["hourly_covers"])


class TestCorrectionEndpoint:
    def test_valid_correction_accepted(self, client: TestClient, db: Session) -> None:
        _seed_basic(db)
        yesterday = str(date.today() - timedelta(days=1))
        r = client.post("/v1/corrections", json={
            "scope": "covers",
            "date": yesterday,
            "hour": 19,
            "predicted": 100.0,
            "actual": 80.0,
            "reason_code": "rain",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["quarantined"] is False

    def test_implausible_correction_quarantined(self, client: TestClient, db: Session) -> None:
        _seed_basic(db)
        yesterday = str(date.today() - timedelta(days=1))
        r = client.post("/v1/corrections", json={
            "scope": "covers",
            "date": yesterday,
            "hour": 12,
            "predicted": 100.0,
            "actual": 800.0,
            "reason_code": "unknown",
        })
        assert r.status_code == 200
        assert r.json()["quarantined"] is True

    def test_invalid_scope_returns_422(self, client: TestClient) -> None:
        r = client.post("/v1/corrections", json={
            "scope": "invalid_scope",
            "date": "2025-01-01",
            "hour": 12,
            "predicted": 100.0,
            "actual": 90.0,
        })
        assert r.status_code == 422


class TestConvergenceEndpoint:
    def test_convergence_returns_series(self, client: TestClient) -> None:
        r = client.get("/v1/metrics/convergence?surface=covers")
        assert r.status_code == 200
        data = r.json()
        assert "series" in data
        assert "summary" in data
        assert isinstance(data["series"], list)

    def test_convergence_summary_has_n_days(self, client: TestClient) -> None:
        r = client.get("/v1/metrics/convergence?surface=covers")
        assert "n_days" in r.json()["summary"]


class TestColdStartRegime:
    def test_empty_db_returns_prior_regime(self, client: TestClient) -> None:
        """With no historical data, regime should be 'prior'."""
        tomorrow = str(date.today() + timedelta(days=1))
        r = client.post("/v1/forecast/covers", json={"date": tomorrow, "restaurant_id": 1})
        assert r.status_code == 200
        assert r.json()["regime"] == "prior"
        assert r.json()["confidence_band"] == pytest.approx(0.40)
