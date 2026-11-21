"""Lightweight Streamlit annotation helper (embedded into dashboard/pages/5_Judge.py in Phase 10)."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.orm import Session

from edg.db.models import HumanAnnotation


def save_single_annotation(
    session: Session,
    *,
    interaction_id: str,
    annotator_id: str,
    split: str,
    verdicts: dict[str, str],
    scores: dict[str, int | None],
    failure_types: list[str],
    evidence_status: str = "sufficient",
    notes: str | None = None,
) -> HumanAnnotation:
    """Persist one human annotation submitted from the Streamlit annotation page."""
    ann_id = f"ANN-{uuid.uuid5(uuid.NAMESPACE_DNS, f'{interaction_id}:{annotator_id}:{split}').hex[:12]}"
    existing = session.get(HumanAnnotation, ann_id)
    if existing is not None:
        existing.evidence_status = evidence_status
        existing.correctness_verdict = verdicts["correctness"]
        existing.correctness_score = scores.get("correctness")
        existing.relevance_verdict = verdicts["relevance"]
        existing.relevance_score = scores.get("relevance")
        existing.helpfulness_verdict = verdicts["helpfulness"]
        existing.helpfulness_score = scores.get("helpfulness")
        existing.completeness_verdict = verdicts["completeness"]
        existing.completeness_score = scores.get("completeness")
        existing.safety_verdict = verdicts["safety"]
        existing.safety_score = scores.get("safety")
        existing.failure_types = list(failure_types)
        existing.notes = notes
        session.flush()
        return existing

    ann = HumanAnnotation(
        annotation_id=ann_id,
        interaction_id=interaction_id,
        annotator_id=annotator_id,
        split=split,
        evidence_status=evidence_status,
        correctness_verdict=verdicts["correctness"],
        correctness_score=scores.get("correctness"),
        relevance_verdict=verdicts["relevance"],
        relevance_score=scores.get("relevance"),
        helpfulness_verdict=verdicts["helpfulness"],
        helpfulness_score=scores.get("helpfulness"),
        completeness_verdict=verdicts["completeness"],
        completeness_score=scores.get("completeness"),
        safety_verdict=verdicts["safety"],
        safety_score=scores.get("safety"),
        failure_types=list(failure_types),
        notes=notes,
    )
    session.add(ann)
    session.flush()
    return ann


def get_annotation_progress(session: Session, split: str) -> dict[str, Any]:
    """Return count of completed human annotations for a split."""
    from sqlalchemy import func, select

    total = session.scalar(
        select(func.count(HumanAnnotation.annotation_id)).where(HumanAnnotation.split == split)
    )
    return {"split": split, "annotated_count": int(total or 0)}
