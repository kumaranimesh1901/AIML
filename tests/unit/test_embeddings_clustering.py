"""Unit and acceptance tests for Phase 4: Embeddings, caching, UMAP 10D/2D, HDBSCAN, stability, and intent mapping."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import Engine, select
from typer.testing import CliRunner

from edg.cleaning.pipeline import run_cleaning_pipeline
from edg.cli import app
from edg.clustering.hdbscan_cluster import fit_hdbscan, grid_search_hdbscan
from edg.clustering.intent_mapping import map_cluster_intent
from edg.clustering.pipeline import run_clustering_pipeline
from edg.clustering.stability import pairwise_non_noise_ari
from edg.config import AppConfig
from edg.db.models import Cluster, ClusteringRun, ClusterMembership, InteractionClean
from edg.db.repositories import track_run
from edg.db.session import get_session
from edg.embeddings.embedder import compare_embedding_models, embed_interactions
from edg.llm.fake import FakeLLMClient
from edg.sut.runner import run_sut_pipeline
from edg.world.store_generator import generate_store_world

runner = CliRunner()


def test_hdbscan_rejects_2d_projection() -> None:
    pts_2d = np.random.default_rng(42).normal(size=(50, 2)).astype(np.float32)
    with pytest.raises(ValueError, match="2D plot projection"):
        fit_hdbscan(pts_2d, min_cluster_size=5, min_samples=3)
    with pytest.raises(ValueError, match="2D plot projection"):
        grid_search_hdbscan(pts_2d, min_cluster_sizes=[5], min_samples_list=[3])


def test_hdbscan_finds_three_clusters_plus_noise_on_synthetic_blobs() -> None:
    rng = np.random.default_rng(42)
    centers = [
        np.array([0.0] * 10),
        np.array([15.0] * 10),
        np.array([-15.0] * 10),
    ]
    blobs = [rng.normal(loc=c, scale=0.4, size=(30, 10)) for c in centers]
    noise = rng.uniform(low=-40.0, high=40.0, size=(8, 10))
    X = np.vstack([*blobs, noise]).astype(np.float32)

    res = fit_hdbscan(X, min_cluster_size=10, min_samples=5)
    assert res.n_clusters == 3
    # At least some noise points detected as -1
    assert np.any(res.labels == -1)
    assert res.dbcv > 0.0
    assert res.glosh_scores.shape == (X.shape[0],)


def test_pairwise_non_noise_ari_and_intent_mapping() -> None:
    a = np.array([0, 0, 1, 1, -1, 2])
    b = np.array([1, 1, 0, 0, 0, -1])
    # Non-noise in both: indices 0,1,2,3 -> identical partition up to label permutation
    ari = pairwise_non_noise_ari(a, b)
    assert ari is not None and abs(ari - 1.0) < 1e-6

    pure = map_cluster_intent(
        ["track_order", "track_order", "track_order", "get_refund"],
        is_hard_query_flags=[False, False, False, False],
        purity_threshold=0.5,
    )
    assert pure.mapped_intent == "track_order"
    assert pure.purity == 0.75
    assert pure.is_new_intent is False

    mixed = map_cluster_intent(
        ["track_order", "get_refund", "cancel_order", "payment_issue"],
        is_hard_query_flags=[False, False, False, False],
        purity_threshold=0.5,
    )
    assert mixed.is_new_intent is True


def test_embedding_cache_100_percent_hit_and_clustering_pipeline(
    db_engine: Engine, small_config: AppConfig
) -> None:
    cfg = small_config.model_copy(deep=True)
    cfg.world.n_orders = 80
    cfg.world.n_customers = 40
    cfg.queries.bitext_sample_size = 90
    cfg.queries.hard_query_count = 30

    with get_session(db_engine) as session:
        with track_run(session, step="world", config=cfg) as (run, _c):
            generate_store_world(
                session, cfg, run_id=run.run_id, prefer_hf_catalogue=False, use_sentence_transformers=False
            )
        with track_run(session, step="sut", config=cfg) as (run, _c):
            run_sut_pipeline(session, cfg, run_id=run.run_id, prefer_hf_bitext=False, inject_failures=True)
        with track_run(session, step="clean", config=cfg) as (run, _c):
            run_cleaning_pipeline(session, cfg, run_id=run.run_id, use_spacy=False)

    # First embedding run
    with get_session(db_engine) as session:
        with track_run(session, step="embed_1", config=cfg) as (run, counts):
            r1 = embed_interactions(session, cfg, run_id=run.run_id, use_sentence_transformers=False)
            counts.update(r1)
    assert r1["rows_inserted"] > 0
    assert r1["cache_misses"] > 0

    # Second embedding run must hit the cache 100%
    with get_session(db_engine) as session:
        with track_run(session, step="embed_2", config=cfg) as (run, counts):
            r2 = embed_interactions(session, cfg, run_id=run.run_id, use_sentence_transformers=False)
            counts.update(r2)
    assert r2["rows_inserted"] == 0
    assert r2["cache_misses"] == 0
    assert r2["cache_hit_rate"] == 1.0

    # Model comparison report
    with get_session(db_engine) as session:
        cmp = compare_embedding_models(session, cfg, use_sentence_transformers=False)
    assert cmp["n_queries"] > 0
    assert cfg.embeddings.model_name in cmp["models"]
    assert cfg.embeddings.alternative_model_name in cmp["models"]

    # Clustering pipeline
    with get_session(db_engine) as session:
        with track_run(session, step="cluster", config=cfg) as (run, counts):
            cres = run_clustering_pipeline(
                session, cfg, run_id=run.run_id, llm_client=FakeLLMClient()
            )
            counts.update(cres)

    assert cres["representatives_clustered"] > 0
    assert cres["dbcv"] is not None
    assert cres["ari_vs_other_seeds"] is not None

    with get_session(db_engine) as session:
        crun = session.get(ClusteringRun, cres["clustering_run_id"])
        assert crun is not None
        assert crun.dbcv is not None
        assert crun.ari_vs_other_seeds is not None
        assert crun.umap_params["n_components"] == 10

        clusters = list(
            session.scalars(
                select(Cluster).where(Cluster.clustering_run_id == crun.clustering_run_id)
            ).all()
        )
        assert len(clusters) == crun.n_clusters
        for cl in clusters:
            assert cl.label and cl.description

        # Verify duplicates inherited their representative's cluster_id
        memberships = {
            m.interaction_id: m.cluster_id
            for m in session.scalars(
                select(ClusterMembership).where(
                    ClusterMembership.clustering_run_id == crun.clustering_run_id
                )
            ).all()
        }
        clean_rows = list(session.scalars(select(InteractionClean)).all())
        rep_by_group = {
            r.dup_group_id: r.interaction_id
            for r in clean_rows
            if r.is_representative and r.interaction_id in memberships
        }
        for r in clean_rows:
            if not r.is_representative and r.interaction_id in memberships and r.dup_group_id in rep_by_group:
                rep_id = rep_by_group[r.dup_group_id]
                assert memberships[r.interaction_id] == memberships[rep_id]


def test_cli_embed_and_cluster_commands(tmp_path: Path) -> None:
    db_url = f"sqlite:///{tmp_path / 'cli_ec.db'}"
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

    e1 = runner.invoke(app, ["embed", "run", "--small", "--db-url", db_url, "--no-st"])
    assert e1.exit_code == 0, e1.output
    e2 = runner.invoke(app, ["embed", "run", "--small", "--db-url", db_url, "--no-st"])
    assert e2.exit_code == 0, e2.output
    assert json.loads(e2.stdout)["cache_hit_rate"] == 1.0

    cmp = runner.invoke(app, ["embed", "compare", "--small", "--db-url", db_url, "--no-st"])
    assert cmp.exit_code == 0, cmp.output
    assert "recall_at_5" in cmp.stdout

    cl = runner.invoke(app, ["cluster", "run", "--small", "--db-url", db_url, "--provider", "fake"])
    assert cl.exit_code == 0, cl.output
    cl_data = json.loads(cl.stdout)
    assert cl_data["n_clusters"] >= 1
    assert cl_data["dbcv"] is not None
