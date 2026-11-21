"""Unit and acceptance tests for Phase 5: Priority signals, ranking, and weighted three-part judge sampling."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from sqlalchemy import Engine, select
from typer.testing import CliRunner

from edg.cleaning.pipeline import run_cleaning_pipeline
from edg.cli import app
from edg.clustering.pipeline import run_clustering_pipeline
from edg.config import AppConfig
from edg.db.models import JudgeSample, PriorityScore
from edg.db.repositories import track_run
from edg.db.session import get_session
from edg.embeddings.embedder import embed_interactions
from edg.llm.fake import FakeLLMClient
from edg.priority.sampler import run_priority_and_sampling
from edg.priority.signals import (
    compute_multi_intent_signal,
    compute_novelty_signal,
    compute_policy_boundary_signal,
    compute_response_drift_signal,
    compute_small_cluster_signal,
    compute_state_conflict_signal,
)
from edg.sut.runner import run_sut_pipeline
from edg.world.store_generator import generate_store_world

runner = CliRunner()


def test_individual_signals_are_bounded_0_to_1() -> None:
    assert compute_novelty_signal(-1, 0.2, 0.8) == 1.0
    assert compute_novelty_signal(0, 0.9, 0.8) == 1.0
    assert 0.0 <= compute_novelty_signal(0, 0.4, 0.8) < 1.0

    assert compute_small_cluster_signal(-1, 10, 1.0) == 1.0
    assert 0.0 <= compute_small_cluster_signal(0, 50, 1.0) <= 1.0

    assert compute_multi_intent_signal("Where is my order?") == 0.0
    assert compute_multi_intent_signal("Cancel my order and also refund my duplicate charge?") >= 0.75

    snap = {
        "order": {"status": "delivered", "total_amount": 205.0},
        "shipments": [{"actual_delivery": "2026-10-10T12:00:00+00:00", "customer_disputed": True}],
        "payments": [{"is_duplicate": True}],
        "refunds_returns": [{"requested_at": "2026-11-09T12:00:00+00:00"}],
    }
    assert compute_state_conflict_signal("I never received it!", snap, ["delivered_disputed"]) == 1.0
    assert compute_policy_boundary_signal(snap, ["boundary_return_window"]) == 1.0

    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert abs(compute_response_drift_signal(v1, v1) - 0.0) < 1e-6
    assert abs(compute_response_drift_signal(v1, v2) - 0.5) < 1e-6


def test_priority_scoring_and_true_uniform_random_control(
    db_engine: Engine, small_config: AppConfig
) -> None:
    cfg = small_config.model_copy(deep=True)
    cfg.world.n_orders = 80
    cfg.world.n_customers = 40
    cfg.queries.bitext_sample_size = 90
    cfg.queries.hard_query_count = 30
    cfg.priority.judge_sample_size = 80

    with get_session(db_engine) as session:
        with track_run(session, step="world", config=cfg) as (run, _c):
            generate_store_world(
                session, cfg, run_id=run.run_id, prefer_hf_catalogue=False, use_sentence_transformers=False
            )
        with track_run(session, step="sut", config=cfg) as (run, _c):
            run_sut_pipeline(session, cfg, run_id=run.run_id, prefer_hf_bitext=False, inject_failures=True)
        with track_run(session, step="clean", config=cfg) as (run, _c):
            run_cleaning_pipeline(session, cfg, run_id=run.run_id, use_spacy=False)
        with track_run(session, step="embed", config=cfg) as (run, _c):
            embed_interactions(session, cfg, run_id=run.run_id, use_sentence_transformers=False)
        with track_run(session, step="cluster", config=cfg) as (run, _c):
            run_clustering_pipeline(session, cfg, run_id=run.run_id, llm_client=FakeLLMClient())

    with get_session(db_engine) as session:
        with track_run(session, step="priority", config=cfg) as (run, counts):
            res = run_priority_and_sampling(session, cfg, run_id=run.run_id)
            counts.update(res)

    assert res["scored_count"] == 120
    assert res["sampled_total"] == 80
    assert res["stratum_counts"]["random_control"] == 16  # 20% of 80

    with get_session(db_engine) as session:
        scores = list(session.scalars(select(PriorityScore).order_by(PriorityScore.rank)).all())
        assert len(scores) == 120
        assert [s.rank for s in scores] == list(range(1, 121))
        for s in scores:
            for val in (
                s.novelty,
                s.small_cluster,
                s.multi_intent,
                s.state_conflict,
                s.policy_boundary,
                s.response_drift,
                s.total_score,
            ):
                assert 0.0 <= val <= 1.0

        # Non-increasing total_score by rank
        for prev, curr in zip(scores, scores[1:], strict=False):
            assert prev.total_score >= curr.total_score

        # Every sampled row has a valid stratum and positive sampling weight
        samples = list(session.scalars(select(JudgeSample)).all())
        assert len(samples) == 80
        valid_strata = {"random_control", "cluster", "priority"}
        for smp in samples:
            assert smp.stratum in valid_strata
            assert smp.sampling_weight >= 1.0

        # Verify random_control is an exact seeded uniform sample over the full population
        all_pop_ids = sorted(s.interaction_id for s in scores)
        expected_random_ids = set(random.Random(cfg.seed).sample(all_pop_ids, k=16))
        actual_random_ids = {s.interaction_id for s in samples if s.stratum == "random_control"}
        assert actual_random_ids == expected_random_ids


def test_cli_priority_run_small(tmp_path: Path) -> None:
    db_url = f"sqlite:///{tmp_path / 'cli_prio.db'}"
    assert (
        runner.invoke(
            app, ["world", "build", "--small", "--db-url", db_url, "--synthetic-catalogue", "--no-st"]
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(
            app, ["sut", "run", "--small", "--db-url", db_url, "--provider", "fake", "--no-hf"]
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(app, ["clean", "run", "--small", "--db-url", db_url, "--no-spacy"]).exit_code
        == 0
    )
    assert (
        runner.invoke(app, ["embed", "run", "--small", "--db-url", db_url, "--no-st"]).exit_code == 0
    )
    assert (
        runner.invoke(
            app, ["cluster", "run", "--small", "--db-url", db_url, "--provider", "fake"]
        ).exit_code
        == 0
    )

    p_res = runner.invoke(app, ["priority", "run", "--small", "--db-url", db_url])
    assert p_res.exit_code == 0, p_res.output
    data = json.loads(p_res.stdout)
    assert data["scored_count"] == 500
    assert data["sampled_total"] == 300
    assert data["stratum_counts"]["random_control"] == 60
