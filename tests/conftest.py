"""Shared pytest fixtures for EDG tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Engine

from edg.config import AppConfig, load_config
from edg.db.session import init_db


@pytest.fixture
def tmp_db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'test_edg.db'}"


@pytest.fixture
def small_config(tmp_db_url: str) -> AppConfig:
    return load_config(
        small=True,
        overrides={
            "database": {"url": tmp_db_url},
            "llm": {
                "provider": "fake",
                "retry_base_delay_seconds": 0.0,
                "retry_backoff_factor": 1.0,
            },
        },
    )


@pytest.fixture
def db_engine(tmp_db_url: str) -> Engine:
    engine = init_db(tmp_db_url, use_alembic=True)
    yield engine
    engine.dispose()
