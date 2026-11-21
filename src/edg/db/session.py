"""SQLAlchemy engine, session factory, and database initialization."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import Connection, Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from edg.db.immutability import install_immutability_triggers
from edg.db.models import Base


def _ensure_sqlite_parent_dir(database_url: str) -> None:
    if database_url.startswith("sqlite:///") and not database_url.endswith(":memory:"):
        raw_path = database_url.removeprefix("sqlite:///")
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)


def create_db_engine(database_url: str, *, echo: bool = False) -> Engine:
    """Create a SQLAlchemy Engine with foreign-key enforcement on SQLite."""
    _ensure_sqlite_parent_dir(database_url)
    connect_args: dict[str, Any] = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    engine = create_engine(database_url, echo=echo, future=True, connect_args=connect_args)

    if database_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection: Any, _connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


_ACTIVE_SESSION: ContextVar[Session | None] = ContextVar("edg_active_session", default=None)


def get_active_session() -> Session | None:
    """Return the innermost session opened by `get_session` in the current context, if any.

    Side-channel writers (e.g. the LLM response cache) use it to write through the pipeline's own
    connection instead of a second connection, which on SQLite would block on the write lock the
    pipeline transaction already holds (`database is locked`).
    """
    return _ACTIVE_SESSION.get()


@contextmanager
def get_session(engine: Engine | Connection) -> Iterator[Session]:
    """Provide a transactional SQLAlchemy session scope."""
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = factory()
    token = _ACTIVE_SESSION.set(session)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        _ACTIVE_SESSION.reset(token)
        session.close()


def _find_migrations_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "migrations"


def init_db(database_url: str, *, use_alembic: bool = True, echo: bool = False) -> Engine:
    """Initialize the database schema via Alembic migrations (or direct metadata fallback)."""
    _ensure_sqlite_parent_dir(database_url)
    engine = create_db_engine(database_url, echo=echo)

    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    if use_alembic:
        migrations_dir = _find_migrations_dir()
        alembic_cfg = AlembicConfig()
        alembic_cfg.set_main_option("script_location", str(migrations_dir))
        alembic_cfg.set_main_option("sqlalchemy.url", database_url)
        command.upgrade(alembic_cfg, "head")
    else:
        Base.metadata.create_all(engine)
        with engine.begin() as conn:
            install_immutability_triggers(conn)

    return engine
