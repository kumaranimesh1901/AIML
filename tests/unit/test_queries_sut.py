"""Unit and acceptance tests for Phase 2: Queries, binding, SUT assistant, tools, and failure injector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import Engine, select
from typer.testing import CliRunner

from edg.cli import app
from edg.config import AppConfig
from edg.db.models import InteractionRaw, ModelVersion, StateSnapshot
from edg.db.repositories import track_run
from edg.db.session import create_db_engine, get_session
from edg.llm.fake import FakeLLMClient
from edg.queries.binder import bind_queries_to_snapshots
from edg.queries.bitext_loader import load_bitext_queries
from edg.queries.hard_query_generator import generate_hard_queries
from edg.sut.assistant import CustomerSupportAssistant
from edg.sut.failure_injector import (
    CONTROLLED_SOURCE_TAG,
    query_release_eligible_interactions,
)
from edg.sut.runner import run_sut_pipeline
from edg.sut.tools import SnapshotAccessError, SnapshotToolBox
from edg.world.store_generator import generate_store_world
from edg.world.taxonomy import load_taxonomy

runner = CliRunner()


@pytest.fixture
def populated_world(db_engine: Engine, small_config: AppConfig) -> Engine:
    with get_session(db_engine) as session:
        with track_run(session, step="world_build", config=small_config) as (run, _counts):
            generate_store_world(
                session,
                small_config,
                run_id=run.run_id,
                prefer_hf_catalogue=False,
                use_sentence_transformers=False,
            )
    return db_engine


def test_toolbox_strictly_isolates_bound_snapshot(populated_world: Engine) -> None:
    with get_session(populated_world) as session:
        snaps = list(session.scalars(select(StateSnapshot).limit(2)).all())
        assert len(snaps) == 2
        snap_a, snap_b = snaps[0], snaps[1]

        tb_v1 = SnapshotToolBox(snap_a, session=session, retrieval_enabled=False)
        order_data = tb_v1.get_order(snap_a.order_id)
        assert order_data["order"]["order_id"] == snap_a.order_id

        # Accessing a different order must raise SnapshotAccessError
        with pytest.raises(SnapshotAccessError):
            tb_v1.get_order(snap_b.order_id)
        with pytest.raises(SnapshotAccessError):
            tb_v1.get_shipment(snap_b.order_id)
        with pytest.raises(SnapshotAccessError):
            tb_v1.get_payments(snap_b.order_id)
        with pytest.raises(SnapshotAccessError):
            tb_v1.get_returns(snap_b.order_id)

        # Dispatch returns error dict instead of leaking data
        err = tb_v1.dispatch("get_order", {"order_id": snap_b.order_id})
        assert "error" in err and "Access denied" in err["error"]

        # Policy retrieval disabled on V1
        pol_disabled = tb_v1.search_policy("30-day return window")
        assert pol_disabled["enabled"] is False
        assert pol_disabled["chunks"] == []

        # Policy retrieval enabled on V2
        tb_v2 = SnapshotToolBox(snap_a, session=session, retrieval_enabled=True)
        pol_enabled = tb_v2.search_policy("30-day return window", top_k=3)
        assert pol_enabled["enabled"] is True
        assert len(pol_enabled["chunks"]) == 3


def test_assistant_tool_calling_loop_with_scripted_llm(
    populated_world: Engine, small_config: AppConfig
) -> None:
    with get_session(populated_world) as session:
        snap = session.scalars(select(StateSnapshot).limit(1)).first()
        assert snap is not None
        tb = SnapshotToolBox(snap, session=session, retrieval_enabled=True)

        fake_llm = FakeLLMClient(
            responses=[
                json.dumps(
                    {
                        "action": "tool_call",
                        "tool": "get_order",
                        "arguments": {"order_id": snap.order_id},
                    }
                ),
                json.dumps(
                    {
                        "action": "tool_call",
                        "tool": "search_policy",
                        "arguments": {"query": "return window"},
                    }
                ),
                json.dumps(
                    {
                        "action": "final_answer",
                        "answer": f"Order {snap.order_id} has been checked against our return policy.",
                    }
                ),
            ],
            cache_enabled=False,
        )
        assistant = CustomerSupportAssistant(
            version_config=small_config.sut.versions["v2"],
            llm_client=fake_llm,
            max_tool_calls=4,
        )
        res = assistant.answer(
            query=f"Check order {snap.order_id}",
            order_id=snap.order_id,
            toolbox=tb,
        )
        assert snap.order_id in res.answer
        assert len(res.tool_calls) == 2
        assert res.tool_calls[0]["tool"] == "get_order"
        assert res.tool_calls[1]["tool"] == "search_policy"


def test_binder_and_hard_query_generator(populated_world: Engine) -> None:
    with get_session(populated_world) as session:
        raw_q = load_bitext_queries(sample_size=54, seed=42, prefer_hf=False)
        assert len(raw_q) == 54
        assert all(q.source_tag == "synthetic_natural" for q in raw_q)

        bound = bind_queries_to_snapshots(session, raw_q, seed=42)
        assert len(bound) == 54
        for bq in bound:
            assert bq.snapshot_id.startswith("SNAP-")
            assert "{{" not in bq.user_query
            assert bq.order_id in bq.user_query

        hard = generate_hard_queries(session, count=30, seed=42)
        assert len(hard) == 30
        for hq in hard:
            assert hq.source_tag == "synthetic_natural"
            assert hq.hard_state_tags
            assert hq.order_id in hq.user_query


def test_sut_pipeline_and_controlled_failure_tier(
    populated_world: Engine, small_config: AppConfig
) -> None:
    # Use a compact query count for fast unit testing
    cfg = small_config.model_copy(deep=True)
    cfg.queries.bitext_sample_size = 60
    cfg.queries.hard_query_count = 20

    with get_session(populated_world) as session:
        with track_run(session, step="sut_run_v1", config=cfg) as (run, counts):
            res = run_sut_pipeline(
                session,
                cfg,
                run_id=run.run_id,
                version_name="v1",
                prefer_hf_bitext=False,
                inject_failures=True,
            )
            counts.update(res)

    taxonomy = load_taxonomy()
    valid_failure_types = set(taxonomy.failure_types.keys())

    with get_session(populated_world) as session:
        versions = {mv.model_id: mv for mv in session.scalars(select(ModelVersion)).all()}
        assert "v1" in versions and "v2" in versions
        assert versions["v1"].retrieval_enabled is False
        assert versions["v2"].retrieval_enabled is True

        all_rows = list(session.scalars(select(InteractionRaw)).all())
        natural_rows = [r for r in all_rows if r.source_tag != CONTROLLED_SOURCE_TAG]
        controlled_rows = [r for r in all_rows if r.source_tag == CONTROLLED_SOURCE_TAG]

        assert len(natural_rows) == 80
        assert len(controlled_rows) == 8  # 10% of 80

        # Every interaction has a valid snapshot_id
        snap_ids = set(session.scalars(select(StateSnapshot.snapshot_id)).all())
        for row in all_rows:
            assert row.snapshot_id in snap_ids

        # Controlled tier has exactly one valid failure type per row
        for crow in controlled_rows:
            assert crow.injected_failure in valid_failure_types
            assert isinstance(crow.injected_failure, str)

        # Controlled tier is strictly excluded from release-eligible queries
        eligible = query_release_eligible_interactions(session)
        assert len(eligible) == 80
        assert all(r.source_tag != CONTROLLED_SOURCE_TAG for r in eligible)
        assert all(r.injected_failure is None for r in eligible)


def test_cli_sut_run_small(tmp_path: Path) -> None:
    db_url = f"sqlite:///{tmp_path / 'cli_sut.db'}"
    w_res = runner.invoke(
        app,
        ["world", "build", "--small", "--db-url", db_url, "--synthetic-catalogue", "--no-st"],
    )
    assert w_res.exit_code == 0, w_res.output

    s_res = runner.invoke(
        app,
        ["sut", "run", "--small", "--db-url", db_url, "--provider", "fake", "--no-hf"],
    )
    assert s_res.exit_code == 0, s_res.output
    data = json.loads(s_res.stdout)
    assert data["bitext_queries"] == 400
    assert data["hard_queries"] == 100
    assert data["natural_inserted"] == 500
    assert data["controlled_injected"] == 50

    engine = create_db_engine(db_url)
    try:
        with get_session(engine) as session:
            eligible = query_release_eligible_interactions(session)
            assert len(eligible) == 500
    finally:
        engine.dispose()
