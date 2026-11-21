"""Labelling rules turning judgements into failure labels, severities, and review queue entries."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from edg.config import AppConfig
from edg.db.models import InteractionRaw, Judgement, Label, StateSnapshot
from edg.db.repositories import source_judgements_only
from edg.judge.schema import DIMENSIONS
from edg.labeling.review_queue import evaluate_review_need
from edg.labeling.severity import resolve_severity_for_interaction
from edg.logging import get_logger
from edg.world.taxonomy import TaxonomyConfig, load_taxonomy

logger = get_logger(__name__)


def label_judgement(
    judgement: Judgement,
    *,
    intent_label: str | None,
    hard_state_tags: list[str] | None,
    config: AppConfig,
    taxonomy: TaxonomyConfig,
) -> dict[str, Any]:
    """Derive failure_detected, failure_types, severity, and review-queue status from a Judgement."""
    dim_verdicts = [getattr(judgement, f"{dim}_verdict") for dim in DIMENSIONS]
    failure_types = list(judgement.failure_types or [])
    any_dim_fail = any(v == "fail" for v in dim_verdicts)
    failure_detected = any_dim_fail or len(failure_types) > 0

    if failure_detected:
        _area, severity = resolve_severity_for_interaction(
            failure_types=failure_types,
            intent=intent_label,
            hard_state_tags=hard_state_tags,
            taxonomy=taxonomy,
        )
    else:
        severity = "low"

    needs_review, review_reason = evaluate_review_need(
        failure_types=failure_types,
        severity=severity,
        confidence=float(judgement.confidence),
        low_confidence_threshold=config.judge.low_confidence_threshold,
        claim_checks_json=list(judgement.claim_checks_json or []),
        raw_output=judgement.raw_output or "",
    )

    return {
        "failure_detected": failure_detected,
        "failure_types": failure_types,
        "severity": severity,
        "needs_human_review": needs_review,
        "review_reason": review_reason,
    }


def run_labeling_pipeline(
    session: Session,
    config: AppConfig,
    *,
    run_id: str,
) -> dict[str, Any]:
    """Create Label rows for all Judgement rows not yet labelled."""
    taxonomy = load_taxonomy()
    already_labeled_jids = select(Label.judgement_id)

    rows = session.execute(
        select(
            Judgement,
            InteractionRaw.intent_label,
            StateSnapshot.hard_state_tags,
        )
        .join(InteractionRaw, InteractionRaw.interaction_id == Judgement.interaction_id)
        .join(StateSnapshot, StateSnapshot.snapshot_id == InteractionRaw.snapshot_id)
        .where(
            Judgement.judgement_id.notin_(already_labeled_jids),
            source_judgements_only(),
        )
        .order_by(Judgement.judgement_id)
    ).all()

    labeled_count = 0
    failures_count = 0
    review_count = 0
    severity_counts: dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}

    for jdg, intent, hard_tags in rows:
        info = label_judgement(
            jdg,
            intent_label=intent,
            hard_state_tags=list(hard_tags or []),
            config=config,
            taxonomy=taxonomy,
        )
        session.add(
            Label(
                interaction_id=jdg.interaction_id,
                judgement_id=jdg.judgement_id,
                failure_detected=info["failure_detected"],
                failure_types=info["failure_types"],
                severity=info["severity"],
                needs_human_review=info["needs_human_review"],
                review_reason=info["review_reason"],
                run_id=run_id,
            )
        )
        labeled_count += 1
        if info["failure_detected"]:
            failures_count += 1
        if info["needs_human_review"]:
            review_count += 1
        severity_counts[info["severity"]] = severity_counts.get(info["severity"], 0) + 1

    session.flush()
    stats = {
        "labeled_count": labeled_count,
        "failures_count": failures_count,
        "needs_human_review_count": review_count,
        "severity_counts": severity_counts,
    }
    logger.info("labeling_pipeline_completed", extra=stats)
    return stats
