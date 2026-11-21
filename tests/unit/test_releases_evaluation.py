"""Unit and acceptance tests for Phase 9: Releases, cluster splits, replay, grading, stats, and regressions."""

from __future__ import annotations

import pytest
from scipy.stats import binomtest
from sqlalchemy import Engine, select

from edg.benchmark.releases import ReleaseImmutabilityError, create_benchmark_release
from edg.cleaning.pipeline import run_cleaning_pipeline
from edg.clustering.pipeline import run_clustering_pipeline
from edg.config import AppConfig
from edg.db.models import BenchmarkRelease, RegressionReport, ReleaseItem, TestCase
from edg.db.repositories import track_run
from edg.db.session import get_session
from edg.embeddings.embedder import embed_interactions
from edg.evaluation.regression import compare_models_on_release, is_regression_flagged
from edg.evaluation.replay import replay_and_grade_release
from edg.evaluation.stats import (
    holm_bonferroni_adjust,
    mcnemar_exact_pvalue,
    paired_bootstrap_ci,
)
from edg.judge.llm_judge import run_judge_pipeline
from edg.labeling.rules import run_labeling_pipeline
from edg.llm.fake import FakeLLMClient
from edg.metrics.dataset_metrics import compute_dataset_metrics
from edg.priority.sampler import run_priority_and_sampling
from edg.sut.runner import run_sut_pipeline
from edg.testcases.generator import run_testcase_generation
from edg.world.store_generator import generate_store_world


def test_mcnemar_bootstrap_holm_and_flagging_match_scipy() -> None:
    """Phase 9 acceptance test: McNemar, bootstrap, Holm correction, and regression flagging."""
    # 50 paired outcomes: 30 both pass, 14 baseline pass / candidate fail (b=14),
    # 2 baseline fail / candidate pass (c=2), 4 both fail.
    base = [True] * 30 + [True] * 14 + [False] * 2 + [False] * 4
    cand = [True] * 30 + [False] * 14 + [True] * 2 + [False] * 4

    p_mcnemar = mcnemar_exact_pvalue(base, cand)
    expected_p = float(binomtest(14, 16, 0.5).pvalue)
    assert abs(p_mcnemar - expected_p) < 1e-12

    diff, lo, hi = paired_bootstrap_ci(
        [1.0 if x else 0.0 for x in base],
        [1.0 if x else 0.0 for x in cand],
        n_resamples=2000,
        alpha=0.05,
        seed=42,
    )
    # Exact diff = (32 - 44) / 50 = -0.24
    assert abs(diff - (-0.24)) < 1e-6
    assert lo < diff < hi < 0.0

    # Holm-Bonferroni correction with a None entry (insufficient category)
    adj = holm_bonferroni_adjust([0.001, 0.04, None, 0.02])
    assert adj[2] is None
    # Sorted valid p-values: 0.001 (x3=0.003), 0.02 (x2=0.04), 0.04 (x1=0.04)
    assert adj[0] == 0.003
    assert adj[3] == 0.04
    assert adj[1] == 0.04

    # Flagging requires p_adjusted < 0.05, CI excluding 0, AND |diff| > judge_flip_rate
    assert is_regression_flagged(
        metric="pass_rate",
        diff=-0.24,
        ci_low=-0.38,
        ci_high=-0.10,
        p_adjusted=0.003,
        alpha=0.05,
        judge_flip_rate=0.02,
    )
    # Not flagged if |diff| <= judge_flip_rate
    assert not is_regression_flagged(
        metric="pass_rate",
        diff=-0.04,
        ci_low=-0.07,
        ci_high=-0.01,
        p_adjusted=0.01,
        alpha=0.05,
        judge_flip_rate=0.05,
    )
    # Improvement (positive diff on pass_rate) is NOT flagged as regression
    assert not is_regression_flagged(
        metric="pass_rate",
        diff=0.20,
        ci_low=0.10,
        ci_high=0.30,
        p_adjusted=0.001,
        alpha=0.05,
        judge_flip_rate=0.01,
    )


def test_release_immutability_cluster_split_replay_and_comparison(
    db_engine: Engine, small_config: AppConfig
) -> None:
    cfg = small_config.model_copy(deep=True)
    cfg.world.n_orders = 80
    cfg.world.n_customers = 40
    cfg.queries.bitext_sample_size = 80
    cfg.queries.hard_query_count = 30
    cfg.priority.judge_sample_size = 70
    cfg.evaluation.bootstrap_resamples = 200

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
        with track_run(session, step="priority", config=cfg) as (run, _c):
            run_priority_and_sampling(session, cfg, run_id=run.run_id)
        with track_run(session, step="judge", config=cfg) as (run, _c):
            run_judge_pipeline(session, cfg, run_id=run.run_id, include_controlled=True)
        with track_run(session, step="label", config=cfg) as (run, _c):
            run_labeling_pipeline(session, cfg, run_id=run.run_id)
        with track_run(session, step="testcases", config=cfg) as (run, _c):
            run_testcase_generation(session, cfg, run_id=run.run_id, llm_client=FakeLLMClient())

    # Create immutable release v1.0
    with get_session(db_engine) as session:
        with track_run(session, step="release", config=cfg) as (run, counts):
            rel = create_benchmark_release(session, cfg, release_id="v1.0", run_id=run.run_id)
            counts.update(rel)
    assert rel["total_cases"] > 0

    # Recreating v1.0 raises ReleaseImmutabilityError
    with pytest.raises(ReleaseImmutabilityError):
        with get_session(db_engine) as session:
            create_benchmark_release(session, cfg, release_id="v1.0", run_id="run_dup")

    # Modifying BenchmarkRelease raises ReleaseImmutabilityError
    with pytest.raises(ReleaseImmutabilityError):
        with get_session(db_engine) as session:
            br = session.get(BenchmarkRelease, "v1.0")
            assert br is not None
            br.description = "tampered"
            session.flush()

    # Verify NO cluster spans dev and heldout
    with get_session(db_engine) as session:
        rows = session.execute(
            select(ReleaseItem.split, TestCase.cluster_id)
            .join(TestCase, TestCase.test_case_id == ReleaseItem.test_case_id)
            .where(ReleaseItem.release_id == "v1.0")
        ).all()
        dev_clusters = {cid for spl, cid in rows if spl == "dev"}
        heldout_clusters = {cid for spl, cid in rows if spl == "heldout"}
        assert dev_clusters.isdisjoint(heldout_clusters)

        ds_metrics = compute_dataset_metrics(session, "v1.0")
        assert ds_metrics["total_cases"] == rel["total_cases"]

    # Replay v1 and v2 on frozen release v1.0
    for model_name in ("v1", "v2"):
        with get_session(db_engine) as session:
            with track_run(session, step=f"replay_{model_name}", config=cfg) as (run, counts):
                rep = replay_and_grade_release(
                    session, cfg, release_id="v1.0", model_version_name=model_name, run_id=run.run_id
                )
                counts.update(rep)
        assert rep["evaluated_new"] == rel["total_cases"]

    # Compare v1 vs v2
    with get_session(db_engine) as session:
        with track_run(session, step="compare", config=cfg) as (run, _c):
            cmp = compare_models_on_release(
                session,
                cfg,
                release_id="v1.0",
                baseline_model_id="v1",
                candidate_model_id="v2",
                run_id=run.run_id,
            )

    assert cmp["release_id"] == "v1.0"
    assert cmp["n_shared_cases"] == rel["total_cases"]
    assert cmp["overall_pass_rate"] is not None
    assert cmp["mined_slice_pass_rate"] is not None
    assert cmp["random_control_slice_pass_rate"] is not None
    assert cmp["judge_flip_rate"]["flip_rate"] == 0.0

    # Verify categories with < 30 cases were marked insufficient (baseline_value is None, flagged is False)
    with get_session(db_engine) as session:
        reg_rows = list(session.scalars(select(RegressionReport)).all())
        assert len(reg_rows) > 0
        insufficient_rows = [r for r in reg_rows if r.baseline_value is None]
        assert len(insufficient_rows) > 0
        assert all(r.flagged is False and r.p_value is None for r in insufficient_rows)
