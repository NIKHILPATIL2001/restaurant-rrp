"""
Shared pytest fixtures for all test suites.

Uses a fresh SQLite in-memory DB per test (fast, no Postgres required for unit/integration tests).
The DATABASE_URL environment variable can override to use real Postgres in CI.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_rrp.db")
os.environ.setdefault("RRP_MODELS_DIR", tempfile.mkdtemp())
os.environ.setdefault("RRP_DATA_DIR", str(Path(__file__).parent.parent / "data"))

# Must import after env vars set
from rrp.api.main import app  # noqa: E402
from rrp.core.db import Base, get_db  # noqa: E402

TEST_DB_URL = os.environ.get("DATABASE_URL", "sqlite:///./test_rrp.db")

_engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False} if "sqlite" in TEST_DB_URL else {},
)
TestingSession = sessionmaker(bind=_engine, autocommit=False, autoflush=False)


@pytest.fixture(scope="session", autouse=True)
def create_tables() -> Generator[None, None, None]:
    import rrp.domain.models  # noqa: F401 — registers models
    Base.metadata.create_all(bind=_engine)
    yield
    Base.metadata.drop_all(bind=_engine)


@pytest.fixture(autouse=True)
def _clean_tables() -> Generator[None, None, None]:
    """Truncate all tables before each test for isolation."""
    yield
    session = TestingSession()
    try:
        # Delete in reverse FK order
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()
    finally:
        session.close()


@pytest.fixture
def db() -> Generator[Session, None, None]:
    session = TestingSession()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@pytest.fixture
def client(db: Session) -> Generator[TestClient, None, None]:
    def override_get_db() -> Generator[Session, None, None]:
        yield db

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
