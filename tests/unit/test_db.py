"""Unit tests for database schema initialization, CLI, immutability, and pipeline_runs tracking."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine, inspect
from typer.testing import CliRunner

from edg.cli import app
from edg.config import AppConfig
from edg.db.models import (
    Base,
    Customer,
    InteractionRaw,
    Order,
    PipelineRun,
    StateSnapshot,
)
from edg.db.repositories import PipelineRunRepository, track_run
from edg.db.session import create_db_engine, get_session

runner = CliRunner()

EXPECTED_TABLES = {
    "pipeline_runs",
    "policy_documents",
    "customers",
    "products",
    "orders",
    "order_items",
    "payments",
    "shipments",
    "refunds_returns",
    "state_snapshots",
    "interactions_raw",
    "pii_mappings",
    "interactions_clean",
    "embeddings",
    "clustering_runs",
    "clusters",
    "cluster_membership",
    "priority_scores",
    "judge_samples",
    "judgements",
    "labels",
    "human_annotations",
    "test_cases",
    "benchmark_releases",
    "release_items",
    "model_versions",
    "eval_results",
    "regression_reports",
    "llm_cache",
}


def test_cli_db_init_creates_all_tables(tmp_path: Path) -> None:
    db_url = f"sqlite:///{tmp_path / 'cli_init.db'}"
    result = runner.invoke(app, ["db", "init", "--small", "--db-url", db_url])
    assert result.exit_code == 0, result.output
    assert "Initialized database" in result.output

    engine = create_db_engine(db_url)
    try:
        tables = set(inspect(engine).get_table_names())
        assert EXPECTED_TABLES.issubset(tables)
        assert "alembic_version" in tables
    finally:
        engine.dispose()


def test_cli_db_status_and_config_commands(tmp_path: Path) -> None:
    db_url = f"sqlite:///{tmp_path / 'status.db'}"
    assert runner.invoke(app, ["db", "init", "--small", "--db-url", db_url]).exit_code == 0
    status_res = runner.invoke(app, ["db", "status", "--small", "--db-url", db_url])
    assert status_res.exit_code == 0
    assert "pipeline_runs" in status_res.output

    hash_res1 = runner.invoke(app, ["config", "hash", "--small"])
    hash_res2 = runner.invoke(app, ["config", "hash", "--small"])
    assert hash_res1.exit_code == 0
    assert hash_res1.output.strip() == hash_res2.output.strip()
    assert len(hash_res1.output.strip()) == 64


def test_every_table_has_created_at() -> None:
    for table_name, table in Base.metadata.tables.items():
        assert "created_at" in table.columns, f"Table {table_name} missing created_at"


def test_track_run_records_success_and_failure(db_engine: Engine, small_config: AppConfig) -> None:
    with get_session(db_engine) as session:
        with track_run(session, step="world_build", config=small_config) as (run, counts):
            counts["orders"] = 10
            run_id = run.run_id

    with get_session(db_engine) as session:
        repo = PipelineRunRepository(session)
        stored = repo.get(run_id)
        assert stored is not None
        assert stored.status == "completed"
        assert stored.step == "world_build"
        assert stored.config_hash == small_config.config_hash()
        assert stored.counts_json == {"orders": 10}
        assert stored.finished_at is not None
        assert stored.error is None

    failed_id: str | None = None
    with pytest.raises(RuntimeError, match="boom"):
        with get_session(db_engine) as session:
            with track_run(session, step="broken_step", config=small_config) as (run, _counts):
                failed_id = run.run_id
                raise RuntimeError("boom")

    assert failed_id is not None
    with get_session(db_engine) as session:
        stored_failed = session.get(PipelineRun, failed_id)
        assert stored_failed is not None
        assert stored_failed.status == "failed"
        assert stored_failed.error is not None and "boom" in stored_failed.error


def test_interactions_raw_is_immutable(db_engine: Engine, small_config: AppConfig) -> None:
    now = datetime.now(UTC)
    with get_session(db_engine) as session:
        with track_run(session, step="seed_raw", config=small_config) as (run, _counts):
            session.add(
                Customer(
                    customer_id="cust_1",
                    name_token="NAME_1",
                    email_token="EMAIL_1",
                    run_id=run.run_id,
                )
            )
            session.flush()
            session.add(
                Order(
                    order_id="ord_1",
                    customer_id="cust_1",
                    placed_at=now,
                    status="shipped",
                    total_amount=49.99,
                    currency="USD",
                    run_id=run.run_id,
                )
            )
            session.flush()
            session.add(
                StateSnapshot(
                    snapshot_id="snap_1",
                    order_id="ord_1",
                    policy_version="v1",
                    snapshot_json={"order_id": "ord_1"},
                    hard_state_tags=[],
                    run_id=run.run_id,
                )
            )
            session.flush()
            session.add(
                InteractionRaw(
                    interaction_id="int_1",
                    conversation_id="conv_1",
                    turn_index=0,
                    user_query="Where is my order?",
                    ai_response="Your order has shipped.",
                    model_version="v1",
                    timestamp=now,
                    source_tag="synthetic_natural",
                    snapshot_id="snap_1",
                    raw_json={"q": "Where is my order?"},
                    run_id=run.run_id,
                )
            )

    with pytest.raises(ValueError, match="immutable"):
        with get_session(db_engine) as session:
            row = session.get(InteractionRaw, "int_1")
            assert row is not None
            row.user_query = "Modified query"
            session.flush()

    with pytest.raises(ValueError, match="immutable"):
        with get_session(db_engine) as session:
            row = session.get(InteractionRaw, "int_1")
            assert row is not None
            session.delete(row)
            session.flush()
