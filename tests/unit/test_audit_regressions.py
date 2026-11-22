"""Regression tests for bugs found in the full-pipeline verification audit.

Each test reproduces a defect against a real `--small` run-all database (fake LLM, offline)
or a freshly migrated database, and guards the root-cause fix.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine, func, select, text, update
from sqlalchemy.exc import DatabaseError

from edg.annotation.agreement import _pair_annotations_with_judgements
from edg.config import AppConfig, load_config
from edg.db.models import (
    BenchmarkRelease,
    Customer,
    EvalResult,
    HumanAnnotation,
    InteractionClean,
    InteractionRaw,
    Judgement,
    Label,
    Order,
    PipelineRun,
    RegressionReport,
    ReleaseItem,
    StateSnapshot,
)
from edg.db.repositories import track_run
from edg.db.session import create_db_engine, get_session
from edg.embeddings.embedder import compare_embedding_models
from edg.evaluation.regression import compare_models_on_release
from edg.judge.llm_judge import HybridJudge
from edg.labeling.rules import run_labeling_pipeline
from edg.llm.fake import FakeLLMClient
from edg.pipeline.report import write_results_summary_markdown
from edg.sut.failure_injector import CONTROLLED_SOURCE_TAG


def _cfg(db_url: str) -> AppConfig:
    return load_config(
        small=True,
        overrides={
            "database": {"url": db_url},
            "llm": {"provider": "fake", "retry_base_delay_seconds": 0.0, "retry_backoff_factor": 1.0},
        },
    )


def _replay_judgement_ids(session) -> set[str]:  # type: ignore[no-untyped-def]
    return set(
        session.scalars(select(EvalResult.judgement_id).where(EvalResult.judgement_id.is_not(None))).all()
    )


# ---------------------------------------------------------------------------
# Bug 1: replay judgements (graded on a NEW model response) were persisted under the
# source interaction_id and then silently reused as if they judged the ORIGINAL logged
# response: in judge_test calibration metrics, the judge cache, and labeling.
# ---------------------------------------------------------------------------


def test_calibration_pairs_only_source_judgements_after_replay(small_pipeline_db: str) -> None:
    engine = create_db_engine(small_pipeline_db)
    try:
        with get_session(engine) as session:
            replay_ids = _replay_judgement_ids(session)
            assert replay_ids, "fixture must contain replay judgements"

            # Interactions in judge_test that also have a replay judgement are the risky ones
            test_iids = set(
                session.scalars(
                    select(HumanAnnotation.interaction_id).where(HumanAnnotation.split == "judge_test")
                ).all()
            )
            replayed_iids = set(
                session.scalars(
                    select(Judgement.interaction_id).where(Judgement.judgement_id.in_(replay_ids))
                ).all()
            )
            overlap = test_iids & replayed_iids
            assert overlap, "fixture must exercise judge_test items that were also replayed"

            source_by_iid = {
                j.interaction_id: j
                for j in session.scalars(
                    select(Judgement).where(
                        Judgement.interaction_id.in_(overlap),
                        Judgement.judgement_id.notin_(replay_ids),
                    )
                ).all()
            }
            paired = {
                p["interaction_id"]: p
                for p in _pair_annotations_with_judgements(session, split="judge_test", judge_version="j1.5")
            }
            for iid in overlap:
                src = source_by_iid[iid]
                for dim in ("correctness", "relevance", "helpfulness", "completeness", "safety"):
                    assert paired[iid][f"judge_{dim}_verdict"] == getattr(src, f"{dim}_verdict")
                    assert paired[iid][f"judge_{dim}_score"] == getattr(src, f"{dim}_score")
                assert paired[iid]["judge_failure_types"] == list(src.failure_types or [])
    finally:
        engine.dispose()


def test_judge_cache_never_returns_replay_judgement(small_pipeline_db: str) -> None:
    engine = create_db_engine(small_pipeline_db)
    try:
        with get_session(engine) as session:
            replay_ids = _replay_judgement_ids(session)
            replayed_iids = sorted(
                set(
                    session.scalars(
                        select(Judgement.interaction_id).where(Judgement.judgement_id.in_(replay_ids))
                    ).all()
                )
            )
            judge = HybridJudge(config=_cfg(small_pipeline_db), llm_client=FakeLLMClient())
            for iid in replayed_iids[:25]:
                cached = judge._find_cached_judgement(session, iid)
                assert cached is not None
                assert cached.judgement_id not in replay_ids
            session.rollback()
    finally:
        engine.dispose()


def test_relabeling_after_replay_does_not_label_replay_judgements(small_pipeline_db: str) -> None:
    engine = create_db_engine(small_pipeline_db)
    config = _cfg(small_pipeline_db)
    try:
        with get_session(engine) as session:
            labels_before = session.scalar(select(func.count(Label.label_id)))
            run = PipelineRun(
                run_id="run_audit_relabel",
                step="label_run",
                config_hash=config.config_hash(),
                git_commit="test",
                started_at=datetime.now(UTC),
                status="running",
                counts_json={},
            )
            session.add(run)
            session.flush()
            stats = run_labeling_pipeline(session, config, run_id=run.run_id)
            assert stats["labeled_count"] == 0
            assert session.scalar(select(func.count(Label.label_id))) == labels_before
            session.rollback()
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Bug 2: synthetic_controlled rewrites share the exact user_query of their natural source,
# so cleaning deduped them into the SAME duplicate group. Because 'CTRL-' sorts before 'INT-',
# the controlled row became the representative; clustering excludes controlled rows, so the
# natural duplicates had no representative and were silently forced into noise (-1).
# ---------------------------------------------------------------------------


def test_controlled_tier_never_shares_duplicate_groups_with_natural_tier(small_pipeline_db: str) -> None:
    engine = create_db_engine(small_pipeline_db)
    try:
        with get_session(engine) as session:
            rows = session.execute(
                select(InteractionClean.dup_group_id, InteractionRaw.source_tag)
                .join(InteractionRaw, InteractionRaw.interaction_id == InteractionClean.interaction_id)
                .where(InteractionClean.valid.is_(True))
            ).all()
            tiers_by_group: dict[str, set[bool]] = {}
            for gid, tag in rows:
                tiers_by_group.setdefault(gid, set()).add(tag == CONTROLLED_SOURCE_TAG)
            mixed = [g for g, tiers in tiers_by_group.items() if len(tiers) > 1]
            assert mixed == []

            # Every natural duplicate group must have a natural representative
            natural_groups = {g for g, tiers in tiers_by_group.items() if tiers == {False}}
            rep_groups = set(
                session.scalars(
                    select(InteractionClean.dup_group_id)
                    .join(InteractionRaw, InteractionRaw.interaction_id == InteractionClean.interaction_id)
                    .where(
                        InteractionClean.is_representative.is_(True),
                        InteractionRaw.source_tag != CONTROLLED_SOURCE_TAG,
                    )
                ).all()
            )
            assert natural_groups <= rep_groups
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Bug 3: the embedding-model comparison included synthetic_controlled rows, whose queries are
# verbatim copies of natural queries, inflating within-intent recall@5.
# ---------------------------------------------------------------------------


def test_embedding_comparison_excludes_controlled_tier(small_pipeline_db: str) -> None:
    engine = create_db_engine(small_pipeline_db)
    try:
        with get_session(engine) as session:
            n_natural_reps = session.scalar(
                select(func.count(InteractionClean.interaction_id))
                .join(InteractionRaw, InteractionRaw.interaction_id == InteractionClean.interaction_id)
                .where(
                    InteractionClean.valid.is_(True),
                    InteractionClean.is_representative.is_(True),
                    InteractionRaw.intent_label.is_not(None),
                    InteractionRaw.source_tag != CONTROLLED_SOURCE_TAG,
                )
            )
            report = compare_embedding_models(
                session, _cfg(small_pipeline_db), use_sentence_transformers=False
            )
            assert report["n_queries"] == n_natural_reps
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Bug 4: results_summary.md aggregated RegressionReport rows from EVERY compare run, so
# re-running `edg eval compare` duplicated table rows and doubled the totals.
# ---------------------------------------------------------------------------


def test_summary_uses_only_latest_comparison_run(small_pipeline_db: str, tmp_path) -> None:  # type: ignore[no-untyped-def]
    engine = create_db_engine(small_pipeline_db)
    config = _cfg(small_pipeline_db)
    try:
        with get_session(engine) as session:
            with track_run(session, step="eval_compare_rerun", config=config) as (run, _counts):
                compare_models_on_release(
                    session,
                    config,
                    release_id="v1.0",
                    baseline_model_id="v1",
                    candidate_model_id="v2",
                    run_id=run.run_id,
                    measure_flip_rate=False,
                )
            per_run = session.scalar(
                select(func.count(RegressionReport.report_id)).where(RegressionReport.run_id == run.run_id)
            )
            out = write_results_summary_markdown(
                session, config, output_path=tmp_path / "summary.md"
            )
            body = out.read_text(encoding="utf-8")
            assert f"**Total Scope × Metric Comparisons**: {per_run}" in body
            assert body.count("| `overall` | `pass_rate` |") == 1
            session.rollback()
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# A-vs-A sanity: comparing a model against itself must yield zero diffs and zero flags.
# ---------------------------------------------------------------------------


def test_a_vs_a_comparison_yields_no_regressions(small_pipeline_db: str) -> None:
    engine = create_db_engine(small_pipeline_db)
    config = _cfg(small_pipeline_db)
    try:
        with get_session(engine) as session:
            with track_run(session, step="eval_compare_a_vs_a", config=config) as (run, _counts):
                res = compare_models_on_release(
                    session,
                    config,
                    release_id="v1.0",
                    baseline_model_id="v1",
                    candidate_model_id="v1",
                    run_id=run.run_id,
                    measure_flip_rate=False,
                )
            assert res["flagged_regressions_count"] == 0
            rows = session.scalars(
                select(RegressionReport).where(RegressionReport.run_id == run.run_id)
            ).all()
            assert rows
            for r in rows:
                if r.diff is not None:
                    assert r.diff == 0.0
                    assert r.p_value == 1.0
            session.rollback()
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Bug 5: immutability of interactions_raw / benchmark_releases / release_items was only an
# ORM event hook, so raw SQL and ORM bulk UPDATE/DELETE bypassed it entirely.
# ---------------------------------------------------------------------------


def _seed_raw_and_release(engine: Engine, config: AppConfig) -> None:
    now = datetime.now(UTC)
    with get_session(engine) as session:
        with track_run(session, step="seed", config=config) as (run, _counts):
            session.add(Customer(customer_id="c1", name_token="NAME_1", email_token="EMAIL_1", run_id=run.run_id))
            session.flush()
            session.add(
                Order(
                    order_id="o1",
                    customer_id="c1",
                    placed_at=now,
                    status="shipped",
                    total_amount=10.0,
                    currency="USD",
                    run_id=run.run_id,
                )
            )
            session.flush()
            session.add(
                StateSnapshot(
                    snapshot_id="s1",
                    order_id="o1",
                    policy_version="v1",
                    snapshot_json={"order_id": "o1"},
                    hard_state_tags=[],
                    run_id=run.run_id,
                )
            )
            session.flush()
            session.add(
                InteractionRaw(
                    interaction_id="i1",
                    conversation_id="conv1",
                    turn_index=0,
                    timestamp=now,
                    user_query="Where is my order?",
                    ai_response="It shipped.",
                    model_version="v1",
                    snapshot_id="s1",
                    source_tag="synthetic_natural",
                    raw_json={},
                    run_id=run.run_id,
                )
            )
            session.add(
                BenchmarkRelease(
                    release_id="r1",
                    frozen_at=now,
                    content_hash="h",
                    judge_version="j1.0",
                    description="d",
                    counts_json={},
                    run_id=run.run_id,
                )
            )


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE interactions_raw SET user_query = 'tampered'",
        "DELETE FROM interactions_raw",
        "UPDATE benchmark_releases SET content_hash = 'tampered'",
        "DELETE FROM benchmark_releases",
    ],
)
def test_immutability_enforced_at_database_level(db_engine: Engine, small_config: AppConfig, sql: str) -> None:
    _seed_raw_and_release(db_engine, small_config)
    with pytest.raises(DatabaseError, match="immutable"):
        with db_engine.begin() as conn:
            conn.execute(text(sql))


def test_orm_bulk_update_on_raw_is_blocked(db_engine: Engine, small_config: AppConfig) -> None:
    _seed_raw_and_release(db_engine, small_config)
    with pytest.raises(DatabaseError, match="immutable"):
        with get_session(db_engine) as session:
            session.execute(update(InteractionRaw).values(user_query="tampered"))
    with get_session(db_engine) as session:
        assert session.get(InteractionRaw, "i1").user_query == "Where is my order?"


def test_release_items_rows_cannot_be_modified(small_pipeline_db: str) -> None:
    engine = create_db_engine(small_pipeline_db)
    try:
        with pytest.raises(DatabaseError, match="immutable"):
            with engine.begin() as conn:
                conn.execute(text("UPDATE release_items SET split = 'dev'"))
        with pytest.raises(DatabaseError, match="immutable"):
            with engine.begin() as conn:
                conn.execute(text("DELETE FROM release_items"))
        with get_session(engine) as session:
            assert (session.scalar(select(func.count()).select_from(ReleaseItem)) or 0) > 0
    finally:
        engine.dispose()
