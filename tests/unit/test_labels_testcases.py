"""Unit and acceptance tests for Phase 8: Labelling, severity matrix, checklists, dedupe, and test cases."""

from __future__ import annotations

import json

from sqlalchemy import Engine, select

from edg.cleaning.pipeline import run_cleaning_pipeline
from edg.clustering.pipeline import run_clustering_pipeline
from edg.config import AppConfig
from edg.db.models import InteractionRaw, Label, StateSnapshot, TestCase
from edg.db.repositories import track_run
from edg.db.session import get_session
from edg.embeddings.embedder import embed_interactions
from edg.judge.llm_judge import run_judge_pipeline
from edg.labeling.review_queue import evaluate_review_need
from edg.labeling.rules import run_labeling_pipeline
from edg.labeling.severity import resolve_severity
from edg.llm.fake import FakeLLMClient
from edg.priority.sampler import run_priority_and_sampling
from edg.sut.failure_injector import CONTROLLED_SOURCE_TAG
from edg.sut.runner import run_sut_pipeline
from edg.testcases.checklist import (
    ExpectedChecklist,
    generate_and_validate_checklist,
    validate_checklist_against_snapshot,
)
from edg.testcases.dedupe import dedupe_test_cases_within_cluster
from edg.testcases.generator import run_testcase_generation
from edg.world.store_generator import generate_store_world
from edg.world.taxonomy import load_taxonomy


def test_severity_follows_taxonomy_matrix_exactly() -> None:
    """Phase 8 acceptance test: severity follows matrix exactly, highest severity wins."""
    tax = load_taxonomy()
    for failure_type, areas in tax.severity_matrix.items():
        for area, expected_sev in areas.items():
            assert resolve_severity(
                failure_types=[failure_type], impact_area=area, taxonomy=tax
            ) == expected_sev

    # Highest severity wins when multiple failure types are present
    assert (
        resolve_severity(
            failure_types=["incomplete_response", "factual_error"],
            impact_area="money",
            taxonomy=tax,
        )
        == "critical"
    )
    assert (
        resolve_severity(
            failure_types=["incomplete_response", "factual_error"],
            impact_area="delivery",
            taxonomy=tax,
        )
        == "high"
    )


def test_review_queue_triggers() -> None:
    # Critical severity triggers review
    need, reason = evaluate_review_need(
        failure_types=["factual_error"],
        severity="critical",
        confidence=0.95,
        low_confidence_threshold=0.70,
        claim_checks_json=[],
        raw_output="{}",
    )
    assert need is True and reason is not None and "critical_severity" in reason

    # Disagreement between contradicted claim check and raw LLM pass triggers review
    need2, reason2 = evaluate_review_need(
        failure_types=["factual_error"],
        severity="high",
        confidence=0.95,
        low_confidence_threshold=0.70,
        claim_checks_json=[{"status": "contradicted"}],
        raw_output=json.dumps({"correctness": {"verdict": "pass"}}),
    )
    assert need2 is True and reason2 is not None and "claim_verifier_disagreement" in reason2


def test_checklist_validation_filters_hallucinated_facts_and_matches_snapshot() -> None:
    """Phase 8 acceptance test: checklist facts match snapshot; wrong dates/amounts are filtered."""
    snapshot = {
        "order": {
            "order_id": "ORD-000042",
            "status": "shipped",
            "total_amount": 89.50,
            "placed_at": "2026-11-10T12:00:00+00:00",
        },
        "shipments": [
            {
                "shipment_id": "SHP-000042",
                "expected_delivery": "2026-11-22T12:00:00+00:00",
                "carrier_scan_status": "in_transit",
                "carrier_scan_at": "2026-11-12T12:00:00+00:00",
                "actual_delivery": None,
                "customer_disputed": False,
            }
        ],
        "payments": [{"amount": 89.50, "status": "captured", "is_duplicate": False}],
        "refunds_returns": [],
        "hard_state_tags": ["cancel_after_shipping"],
    }

    # Drafted checklist with a hallucinated date and wrong amount gets cleaned by code validator
    draft = ExpectedChecklist(
        must_do=[
            "must give expected delivery 2026-11-22",
            "must give wrong delivery 2099-01-01",  # invalid date
            "must quote wrong total $999.99",  # invalid amount
            "must state order ORD-000042 status is 'shipped'",
        ],
        must_not_do=["must not claim the order is cancelled"],
    )
    validated = validate_checklist_against_snapshot(draft, snapshot)
    assert "must give expected delivery 2026-11-22" in validated.must_do
    assert "must state order ORD-000042 status is 'shipped'" in validated.must_do
    assert all("2099-01-01" not in x for x in validated.must_do)
    assert all("$999.99" not in x for x in validated.must_do)

    gen = generate_and_validate_checklist(
        query="Cancel ORD-000042 please",
        snapshot_json=snapshot,
        intent="cancel_order",
    )
    assert any("2026-11-22" in x for x in gen.must_do)
    assert any("cannot be cancelled" in x for x in gen.must_do)


def test_duplicate_test_case_fixtures_are_merged() -> None:
    """Phase 8 acceptance test: duplicate fixtures within a cluster are merged."""
    chk = {"must_do": ["must state status is 'shipped'"], "must_not_do": ["must not lie"]}
    candidates = [
        {"test_case_id": "TC-1", "cluster_id": 3, "input_query": "Where is my order ORD-1?", "expected_checklist_json": chk},
        {"test_case_id": "TC-2", "cluster_id": 3, "input_query": "Where is my order ORD-1?", "expected_checklist_json": chk},
        # Different cluster -> NOT merged
        {"test_case_id": "TC-3", "cluster_id": 4, "input_query": "Where is my order ORD-1?", "expected_checklist_json": chk},
    ]
    kept, merged = dedupe_test_cases_within_cluster(candidates, cosine_threshold=0.92)
    assert merged == 1
    assert len(kept) == 2
    assert {c["test_case_id"] for c in kept} == {"TC-1", "TC-3"}


def test_end_to_end_labeling_and_testcase_generation_never_stores_bad_response(
    db_engine: Engine, small_config: AppConfig
) -> None:
    cfg = small_config.model_copy(deep=True)
    cfg.world.n_orders = 60
    cfg.world.n_customers = 30
    cfg.queries.bitext_sample_size = 60
    cfg.queries.hard_query_count = 20
    cfg.priority.judge_sample_size = 50

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

    with get_session(db_engine) as session:
        with track_run(session, step="label", config=cfg) as (run, counts):
            lbl_res = run_labeling_pipeline(session, cfg, run_id=run.run_id)
            counts.update(lbl_res)
    assert lbl_res["labeled_count"] > 0
    assert lbl_res["failures_count"] > 0

    with get_session(db_engine) as session:
        with track_run(session, step="testcases", config=cfg) as (run, counts):
            tc_res = run_testcase_generation(
                session, cfg, run_id=run.run_id, llm_client=FakeLLMClient(), generate_variants=True
            )
            counts.update(tc_res)
    assert tc_res["test_cases_created"] > 0

    with get_session(db_engine) as session:
        labels_by_iid = {lbl.interaction_id: lbl for lbl in session.scalars(select(Label)).all()}
        raw_by_iid = {r.interaction_id: r for r in session.scalars(select(InteractionRaw)).all()}
        snaps_by_id = {s.snapshot_id: s for s in session.scalars(select(StateSnapshot)).all()}
        tcs = list(session.scalars(select(TestCase)).all())
        assert len(tcs) > 0

        for tc in tcs:
            src = raw_by_iid[tc.source_interaction_id]
            # Controlled-tier rows must NEVER enter test_cases
            assert src.source_tag != CONTROLLED_SOURCE_TAG
            assert src.injected_failure is None
            # No test case stores the bad AI response as reference_answer or input_query
            if labels_by_iid[src.interaction_id].failure_detected:
                assert tc.reference_answer != src.ai_response
                assert tc.input_query != src.ai_response
            # Checklist must have valid must_do / must_not_do grounded in snapshot
            snap_order_id = snaps_by_id[tc.snapshot_id].order_id
            assert any(snap_order_id in m for m in tc.expected_checklist_json["must_do"])
            if tc.trust_tier == "variant":
                assert tc.parent_test_case_id is not None
                assert tc.status == "candidate"
