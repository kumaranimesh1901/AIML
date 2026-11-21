"""Unit and acceptance tests for Phase 6: Evidence-based hybrid LLM judge."""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import Engine, select
from typer.testing import CliRunner

from edg.cleaning.pipeline import run_cleaning_pipeline
from edg.clustering.pipeline import run_clustering_pipeline
from edg.config import AppConfig
from edg.db.models import Judgement
from edg.db.repositories import track_run
from edg.db.session import get_session
from edg.embeddings.embedder import embed_interactions
from edg.judge.claims import extract_claims_deterministic
from edg.judge.evidence import EvidenceBundle
from edg.judge.llm_judge import HybridJudge, run_judge_pipeline
from edg.judge.verifier import verify_claims
from edg.llm.fake import FakeLLMClient
from edg.priority.sampler import run_priority_and_sampling
from edg.sut.runner import run_sut_pipeline
from edg.world.store_generator import generate_store_world

runner = CliRunner()


def test_delivered_yesterday_against_null_actual_delivery_yields_contradicted_fail_factual_error(
    small_config: AppConfig,
) -> None:
    """Phase 6 acceptance test: 'delivered yesterday' vs actual_delivery=null."""
    snapshot = {
        "order": {
            "order_id": "ORD-000100",
            "status": "shipped",
            "total_amount": 75.00,
            "placed_at": "2026-11-10T10:00:00+00:00",
        },
        "shipments": [
            {
                "shipment_id": "SHP-000100",
                "carrier": "NovaExpress",
                "expected_delivery": "2026-11-18T12:00:00+00:00",
                "carrier_scan_status": "in_transit",
                "carrier_scan_at": "2026-11-12T12:00:00+00:00",
                "actual_delivery": None,
                "customer_disputed": False,
            }
        ],
        "payments": [{"amount": 75.00, "status": "captured", "is_duplicate": False}],
        "refunds_returns": [],
    }
    evidence = EvidenceBundle(
        snapshot_id="SNAP-000100",
        snapshot_json=snapshot,
        policy_chunks=[{"policy_id": "v1:09:chunk_0", "title": "Delivery", "body": "3-5 days"}],
        evidence_status="sufficient",
    )

    response_text = "Your order ORD-000100 was delivered yesterday."
    claims = extract_claims_deterministic(response_text)
    checks = verify_claims(claims, snapshot, evidence.policy_chunks)
    assert any(c.status == "contradicted" for c in checks)

    # Even if an LLM erroneously tried to return pass, code rules MUST force fail + factual_error
    optimistic_llm_json = json.dumps(
        {
            "correctness": {"verdict": "pass", "score": 5},
            "relevance": {"verdict": "pass", "score": 5},
            "helpfulness": {"verdict": "pass", "score": 5},
            "completeness": {"verdict": "pass", "score": 5},
            "safety": {"verdict": "pass", "score": 5},
            "failure_types": [],
            "explanation": "Looks fine to me.",
            "confidence": 0.9,
        }
    )
    judge = HybridJudge(
        config=small_config,
        llm_client=FakeLLMClient(responses=[optimistic_llm_json], cache_enabled=False),
    )
    jdg, cached = judge.judge_one(
        session=None,
        interaction_id="INT-FIX-1",
        query="Where is order ORD-000100?",
        response=response_text,
        evidence=evidence,
        run_id="run_test",
        persist=False,
    )
    assert cached is False
    assert jdg.correctness_verdict == "fail"
    assert jdg.correctness_score is not None and jdg.correctness_score <= 2
    assert "factual_error" in jdg.failure_types
    assert any(c["status"] == "contradicted" for c in jdg.claim_checks_json)


def test_missing_evidence_yields_not_assessable(small_config: AppConfig) -> None:
    """Phase 6 acceptance test: missing evidence forces correctness to not_assessable."""
    evidence = EvidenceBundle(
        snapshot_id=None,
        snapshot_json=None,
        policy_chunks=[],
        evidence_status="missing",
    )
    judge = HybridJudge(config=small_config, llm_client=FakeLLMClient(cache_enabled=False))
    jdg, _cached = judge.judge_one(
        session=None,
        interaction_id="INT-MISS-1",
        query="Where is my order?",
        response="Your order was delivered on 2026-11-14.",
        evidence=evidence,
        run_id="run_test",
        persist=False,
    )
    assert jdg.evidence_status == "missing"
    assert jdg.correctness_verdict == "not_assessable"
    assert jdg.correctness_score is None


def test_invalid_json_retried_then_marked_judge_error(small_config: AppConfig) -> None:
    """Phase 6 acceptance test: invalid JSON is retried max_retries times then marked judge_error."""
    cfg = small_config.model_copy(deep=True)
    cfg.judge.max_retries = 2

    failing_llm = FakeLLMClient(
        responses=["not valid json", "{broken", "still broken"],
        max_retries=2,
        retry_base_delay_seconds=0.0,
        cache_enabled=False,
    )
    evidence = EvidenceBundle(
        snapshot_id="SNAP-1",
        snapshot_json={"order": {"order_id": "ORD-1", "status": "shipped", "total_amount": 20.0}},
        policy_chunks=[{"policy_id": "p1", "title": "Policy", "body": "Rules"}],
        evidence_status="sufficient",
    )
    judge = HybridJudge(config=cfg, llm_client=failing_llm)
    jdg, _cached = judge.judge_one(
        session=None,
        interaction_id="INT-ERR-1",
        query="Where is ORD-1?",
        response="Your order ORD-1 is shipped.",
        evidence=evidence,
        run_id="run_test",
        persist=False,
    )
    assert failing_llm.call_count == 3  # 1 initial + 2 retries
    assert "judge_error" in jdg.failure_types
    assert jdg.correctness_verdict == "not_assessable"
    assert jdg.confidence == 0.0


def test_hallucinated_tracking_number_flagged_in_code(small_config: AppConfig) -> None:
    snapshot = {
        "order": {"order_id": "ORD-1", "status": "shipped", "total_amount": 40.0},
        "shipments": [
            {
                "shipment_id": "SHP-000001",
                "carrier": "NovaExpress",
                "expected_delivery": "2026-11-20T00:00:00+00:00",
                "carrier_scan_status": "in_transit",
                "carrier_scan_at": "2026-11-15T00:00:00+00:00",
                "actual_delivery": None,
                "customer_disputed": False,
            }
        ],
        "payments": [{"amount": 40.0, "status": "captured", "is_duplicate": False}],
        "refunds_returns": [],
    }
    evidence = EvidenceBundle(
        snapshot_id="SNAP-1",
        snapshot_json=snapshot,
        policy_chunks=[{"policy_id": "p1", "title": "Policy", "body": "Rules"}],
        evidence_status="sufficient",
    )
    judge = HybridJudge(config=small_config, llm_client=FakeLLMClient(cache_enabled=False))
    jdg, _cached = judge.judge_one(
        session=None,
        interaction_id="INT-HALL-1",
        query="Where is ORD-1?",
        response="Your order is shipped under tracking number TRK-999888777.",
        evidence=evidence,
        run_id="run_test",
        persist=False,
    )
    assert "hallucination" in jdg.failure_types
    assert any(c["status"] == "not_in_evidence" for c in jdg.claim_checks_json)


def test_judge_pipeline_caching_and_cli(db_engine: Engine, small_config: AppConfig, tmp_path: Path) -> None:
    cfg = small_config.model_copy(deep=True)
    cfg.world.n_orders = 60
    cfg.world.n_customers = 30
    cfg.queries.bitext_sample_size = 50
    cfg.queries.hard_query_count = 20
    cfg.priority.judge_sample_size = 40

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

    # First judge run
    with get_session(db_engine) as session:
        with track_run(session, step="judge_1", config=cfg) as (run, counts):
            r1 = run_judge_pipeline(session, cfg, run_id=run.run_id, include_controlled=True)
            counts.update(r1)
    assert r1["judged_new"] == r1["total_targets"]
    assert r1["cache_hits"] == 0
    assert r1["failures_detected"] > 0

    # Second judge run hits DB cache 100%
    with get_session(db_engine) as session:
        with track_run(session, step="judge_2", config=cfg) as (run, counts):
            r2 = run_judge_pipeline(session, cfg, run_id=run.run_id, include_controlled=True)
            counts.update(r2)
    assert r2["judged_new"] == 0
    assert r2["cache_hits"] == r1["total_targets"]

    with get_session(db_engine) as session:
        jdgs = list(session.scalars(select(Judgement)).all())
        assert len(jdgs) == r1["total_targets"]
        assert all(j.judge_model == cfg.judge.model_name for j in jdgs)
        assert all(j.judge_version == cfg.judge.judge_version for j in jdgs)
