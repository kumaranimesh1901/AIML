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


@pytest.fixture(scope="session")
def small_pipeline_db(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Build one fully populated --small pipeline DB (fake LLM, offline) shared across tests.

    Tests that need a realistic end-to-end DB (dashboard pages, cross-step invariants) use this
    instead of depending on a developer's local `data/edg.db`.
    """
    from edg.pipeline.orchestrator import run_full_pipeline

    root = tmp_path_factory.mktemp("small_pipeline")
    db_url = f"sqlite:///{root / 'edg_small.db'}"
    config = load_config(
        small=True,
        overrides={
            "database": {"url": db_url},
            "llm": {"provider": "fake", "retry_base_delay_seconds": 0.0, "retry_backoff_factor": 1.0},
        },
    )
    run_full_pipeline(
        config,
        release_id="v1.0",
        annotations_dir=root / "annotations",
        freeze_registry_path=root / "annotations" / "frozen_judges.json",
        summary_output_path=root / "results_summary.md",
    )
    return db_url
