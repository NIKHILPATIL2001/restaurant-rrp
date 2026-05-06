from collections.abc import Generator

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from rrp.core.config import settings

# Register adapters so psycopg2 can handle numpy scalar types directly.
# Without this, numpy 2.x's new repr (e.g. "np.float64(6.0)") leaks into SQL
# and produces InvalidSchemaName: schema "np" does not exist.
try:
    from psycopg2.extensions import AsIs, register_adapter

    def _adapt_np_float(val: np.floating) -> AsIs:
        return AsIs(repr(float(val)))

    def _adapt_np_int(val: np.integer) -> AsIs:
        return AsIs(repr(int(val)))

    # psycopg2's `register_adapter` is typed against `_ISQLQuoteProto`,
    # which our small AsIs-returning callables don't formally implement.
    # The runtime contract is correct; ignore the strict-typing arg-type.
    for _t_float in (np.float64, np.float32, np.float16):
        register_adapter(_t_float, _adapt_np_float)  # type: ignore[arg-type]
    for _t_int in (np.int64, np.int32, np.int16, np.int8):
        register_adapter(_t_int, _adapt_np_int)  # type: ignore[arg-type]
except ImportError:
    pass


class Base(DeclarativeBase):
    pass


_engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)


def get_engine() -> object:
    return _engine


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
