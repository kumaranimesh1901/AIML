"""Test case candidate generation from failed interactions, high-priority passes, and random_control."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from edg.config import AppConfig
from edg.db.models import (
    ClusteringRun,
    ClusterMembership,
    InteractionClean,
    InteractionRaw,
    Judgement,
    JudgeSample,
    Label,
    PriorityScore,
    StateSnapshot,
    TestCase,
)
from edg.llm.client import BaseLLMClient
from edg.logging import get_logger
from edg.sut.failure_injector import CONTROLLED_SOURCE_TAG
from edg.testcases.checklist import generate_and_validate_checklist
from edg.testcases.dedupe import dedupe_test_cases_within_cluster
from edg.testcases.variants import generate_critical_variants

logger = get_logger(__name__)


def run_testcase_generation(
    session: Session,
    config: AppConfig,
    *,
    run_id: str,
    llm_client: BaseLLMClient | None = None,
    generate_variants: bool = True,
    max_variants: int = 10,
) -> dict[str, Any]:
    """Generate replayable test cases from natural-tier labelled interactions.

    Never stores the bad AI response as the expected output.
    Strictly excludes synthetic_controlled rows.
    """
    latest_crun = session.scalars(
        select(ClusteringRun).order_by(ClusteringRun.created_at.desc()).limit(1)
    ).first()
    c_run_id = latest_crun.clustering_run_id if latest_crun is not None else None

    rows = session.execute(
        select(
            Label,
            Judgement,
            InteractionRaw,
            InteractionClean,
            StateSnapshot,
            JudgeSample.stratum,
            PriorityScore.rank,
        )
        .join(Judgement, Judgement.judgement_id == Label.judgement_id)
        .join(InteractionRaw, InteractionRaw.interaction_id == Label.interaction_id)
        .join(InteractionClean, InteractionClean.interaction_id == Label.interaction_id)
        .join(StateSnapshot, StateSnapshot.snapshot_id == InteractionRaw.snapshot_id)
        .outerjoin(JudgeSample, JudgeSample.interaction_id == Label.interaction_id)
        .outerjoin(PriorityScore, PriorityScore.interaction_id == Label.interaction_id)
        .where(
            InteractionRaw.source_tag != CONTROLLED_SOURCE_TAG,
            InteractionRaw.injected_failure.is_(None),
            InteractionClean.valid.is_(True),
        )
        .order_by(Label.interaction_id)
    ).all()

    cluster_by_iid: dict[str, int] = {}
    if c_run_id:
        for iid, cid in session.execute(
            select(ClusterMembership.interaction_id, ClusterMembership.cluster_id).where(
                ClusterMembership.clustering_run_id == c_run_id
            )
        ).all():
            cluster_by_iid[iid] = int(cid)

    existing_source_ids = set(session.scalars(select(TestCase.source_interaction_id)).all())

    # Candidate selection: all failures, all random_control, and high-priority passes (top 25% rank or priority stratum)
    all_ranks = [int(r.rank) for r in rows if r.rank is not None]
    high_priority_rank_cutoff = max(1, len(all_ranks) // 4) if all_ranks else 50

    staged: list[dict[str, Any]] = []
    for lbl, jdg, raw, clean, snap, stratum, rank in rows:
        if raw.interaction_id in existing_source_ids:
            continue

        is_failure = bool(lbl.failure_detected)
        is_random_control = stratum == "random_control"
        is_high_priority_pass = (not is_failure) and (
            stratum == "priority" or (rank is not None and int(rank) <= high_priority_rank_cutoff)
        )

        if not (is_failure or is_random_control or is_high_priority_pass):
            continue

        checklist = generate_and_validate_checklist(
            query=clean.clean_query,
            snapshot_json=snap.snapshot_json,
            intent=raw.intent_label,
            llm_client=llm_client,
        )

        # Trust tier: silver+approved if high confidence & no review needed, else silver+candidate
        if (
            float(jdg.confidence) >= config.testcases.silver_confidence_threshold
            and not lbl.needs_human_review
        ):
            trust_tier = "silver"
            status = "approved"
        else:
            trust_tier = "silver"
            status = "candidate"

        tc_id = f"TC-{uuid.uuid5(uuid.NAMESPACE_DNS, f'tc:{raw.interaction_id}').hex[:12]}"
        staged.append(
            {
                "test_case_id": tc_id,
                "source_interaction_id": raw.interaction_id,
                "input_query": clean.clean_query,  # Original input query, NEVER the bad response
                "snapshot_id": raw.snapshot_id,
                "policy_version": snap.policy_version,
                "expected_checklist_json": checklist.model_dump(),
                "reference_answer": raw.reference_answer,  # Only original reference_answer, NEVER ai_response
                "original_failure_types": list(lbl.failure_types or []),
                "severity": lbl.severity,
                "cluster_id": cluster_by_iid.get(raw.interaction_id),
                "intent": raw.intent_label,
                "trust_tier": trust_tier,
                "status": status,
                "provenance_json": {
                    "source_interaction_id": raw.interaction_id,
                    "judgement_id": jdg.judgement_id,
                    "judge_version": jdg.judge_version,
                    "stratum": stratum or "unstratified",
                    "selection_reason": (
                        "failure"
                        if is_failure
                        else ("random_control" if is_random_control else "high_priority_pass")
                    ),
                    "needs_human_review": lbl.needs_human_review,
                },
            }
        )

    kept, dupes_merged = dedupe_test_cases_within_cluster(
        staged, cosine_threshold=config.testcases.dedupe_cosine_threshold
    )

    approved_count = 0
    candidate_count = 0
    for item in kept:
        if item["status"] == "approved":
            approved_count += 1
        else:
            candidate_count += 1
        session.add(
            TestCase(
                test_case_id=item["test_case_id"],
                source_interaction_id=item["source_interaction_id"],
                input_query=item["input_query"],
                snapshot_id=item["snapshot_id"],
                policy_version=item["policy_version"],
                expected_checklist_json=item["expected_checklist_json"],
                reference_answer=item["reference_answer"],
                original_failure_types=item["original_failure_types"],
                severity=item["severity"],
                cluster_id=item["cluster_id"],
                intent=item["intent"],
                trust_tier=item["trust_tier"],
                status=item["status"],
                parent_test_case_id=None,
                provenance_json=item["provenance_json"],
                run_id=run_id,
            )
        )
    session.flush()

    variants_created = (
        generate_critical_variants(
            session, run_id=run_id, max_variants=max_variants, llm_client=llm_client
        )
        if generate_variants
        else 0
    )

    stats = {
        "candidates_considered": len(staged),
        "duplicates_merged": dupes_merged,
        "test_cases_created": len(kept),
        "approved_silver": approved_count,
        "pending_candidate": candidate_count,
        "variants_created": variants_created,
    }
    logger.info("testcase_generation_completed", extra=stats)
    return stats
