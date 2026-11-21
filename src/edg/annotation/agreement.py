"""Split-guarded calibration evaluation, human-human ceiling, and judge version freezing."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from edg.db.models import HumanAnnotation, InteractionRaw, Judgement
from edg.judge.schema import DIMENSIONS
from edg.logging import get_logger
from edg.metrics.judge_metrics import (
    compute_controlled_tier_recall,
    compute_judge_agreement_report,
    compute_judge_reliability,
    compute_ordinal_agreement,
)
from edg.sut.failure_injector import CONTROLLED_SOURCE_TAG
from edg.world.taxonomy import load_taxonomy

logger = get_logger(__name__)

DEFAULT_FREEZE_REGISTRY = Path("data/annotations/frozen_judges.json")


class SplitLeakageError(PermissionError):
    """Raised when a prompt-tuning command attempts to read the held-out judge_test split."""


class JudgeFreezeError(RuntimeError):
    """Raised when judge freeze immutability or prerequisite checks fail."""


def _load_freeze_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_judge_version(
    *,
    judge_version: str,
    judge_model: str,
    prompt_hash: str,
    registry_path: str | Path = DEFAULT_FREEZE_REGISTRY,
) -> dict[str, Any]:
    """Freeze a judge version immutably before running on judge_test or at scale."""
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    registry = _load_freeze_registry(path)

    if judge_version in registry:
        existing = registry[judge_version]
        if existing["prompt_hash"] != prompt_hash or existing["judge_model"] != judge_model:
            raise JudgeFreezeError(
                f"Judge version '{judge_version}' is already frozen with prompt_hash="
                f"{existing['prompt_hash']} and cannot be modified."
            )
        return existing

    entry = {
        "judge_version": judge_version,
        "judge_model": judge_model,
        "prompt_hash": prompt_hash,
        "frozen_at": datetime.now(UTC).isoformat(),
    }
    registry[judge_version] = entry
    path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("judge_version_frozen", extra=entry)
    return entry


def is_judge_frozen(
    judge_version: str,
    *,
    registry_path: str | Path = DEFAULT_FREEZE_REGISTRY,
) -> bool:
    registry = _load_freeze_registry(Path(registry_path))
    return judge_version in registry


def _pair_annotations_with_judgements(
    session: Session,
    *,
    split: str,
    judge_version: str | None = None,
) -> list[dict[str, Any]]:
    anns = list(
        session.scalars(
            select(HumanAnnotation)
            .where(HumanAnnotation.split == split)
            .order_by(HumanAnnotation.interaction_id, HumanAnnotation.annotator_id)
        ).all()
    )
    if not anns:
        return []

    # Use first annotator per interaction_id for primary human-vs-judge agreement
    primary_anns: dict[str, HumanAnnotation] = {}
    for ann in anns:
        primary_anns.setdefault(ann.interaction_id, ann)

    j_stmt = select(Judgement).where(Judgement.interaction_id.in_(list(primary_anns.keys())))
    if judge_version:
        j_stmt = j_stmt.where(Judgement.judge_version == judge_version)
    j_stmt = j_stmt.order_by(Judgement.created_at.desc())

    judgements_by_id: dict[str, Judgement] = {}
    for j in session.scalars(j_stmt).all():
        judgements_by_id.setdefault(j.interaction_id, j)

    paired: list[dict[str, Any]] = []
    for iid, ann in sorted(primary_anns.items()):
        jdg = judgements_by_id.get(iid)
        if jdg is None:
            continue
        rec: dict[str, Any] = {
            "interaction_id": iid,
            "human_failure_types": list(ann.failure_types or []),
            "judge_failure_types": list(jdg.failure_types or []),
        }
        for dim in DIMENSIONS:
            rec[f"human_{dim}_verdict"] = getattr(ann, f"{dim}_verdict")
            rec[f"human_{dim}_score"] = getattr(ann, f"{dim}_score")
            rec[f"judge_{dim}_verdict"] = getattr(jdg, f"{dim}_verdict")
            rec[f"judge_{dim}_score"] = getattr(jdg, f"{dim}_score")
        paired.append(rec)
    return paired


def compute_human_human_ceiling(
    session: Session,
    *,
    split: str | None = None,
) -> dict[str, Any]:
    """Compute inter-annotator quadratic weighted kappa on double-annotated items."""
    stmt = select(HumanAnnotation)
    if split:
        stmt = stmt.where(HumanAnnotation.split == split)
    anns = list(session.scalars(stmt.order_by(HumanAnnotation.interaction_id, HumanAnnotation.annotator_id)).all())

    by_iid: dict[str, list[HumanAnnotation]] = defaultdict(list)
    for a in anns:
        by_iid[a.interaction_id].append(a)

    double_items = [pair[:2] for pair in by_iid.values() if len(pair) >= 2]
    per_dim: dict[str, Any] = {}
    for dim in DIMENSIONS:
        a1 = [getattr(pair[0], f"{dim}_score") for pair in double_items]
        a2 = [getattr(pair[1], f"{dim}_score") for pair in double_items]
        per_dim[dim] = compute_ordinal_agreement(a1, a2)

    return {"n_double_annotated": len(double_items), "by_dimension": per_dim}


def _evaluate_split(
    session: Session,
    *,
    split: str,
    judge_version: str | None = None,
) -> dict[str, Any]:
    taxonomy = load_taxonomy()
    paired = _pair_annotations_with_judgements(session, split=split, judge_version=judge_version)
    agreement = compute_judge_agreement_report(
        paired, failure_type_list=sorted(taxonomy.failure_types.keys())
    )
    ceiling = compute_human_human_ceiling(session, split=split)

    # Controlled-tier recall
    ctrl_rows = session.execute(
        select(InteractionRaw.injected_failure, Judgement.failure_types)
        .join(Judgement, Judgement.interaction_id == InteractionRaw.interaction_id)
        .where(
            InteractionRaw.source_tag == CONTROLLED_SOURCE_TAG,
            InteractionRaw.injected_failure.is_not(None),
        )
    ).all()
    controlled_recall = compute_controlled_tier_recall(
        [(str(r[0]), list(r[1] or [])) for r in ctrl_rows]
    )

    # Judge reliability
    j_rows = session.execute(
        select(Judgement, InteractionRaw.ai_response).join(
            InteractionRaw, InteractionRaw.interaction_id == Judgement.interaction_id
        )
    ).all()
    reliability_input = [
        {
            "correctness_verdict": j.correctness_verdict,
            "failure_types": j.failure_types,
            "correctness_score": j.correctness_score,
            "relevance_score": j.relevance_score,
            "helpfulness_score": j.helpfulness_score,
            "completeness_score": j.completeness_score,
            "safety_score": j.safety_score,
            "response_length": len(resp or ""),
        }
        for j, resp in j_rows
    ]
    reliability = compute_judge_reliability(reliability_input)

    return {
        "split": split,
        "judge_version": judge_version,
        "agreement": agreement,
        "human_human_ceiling": ceiling,
        "controlled_tier_recall": controlled_recall,
        "reliability": reliability,
    }


def evaluate_prompt_tuning(
    session: Session,
    *,
    split: str = "judge_dev",
    judge_version: str | None = None,
) -> dict[str, Any]:
    """Evaluate judge prompt tuning strictly on judge_dev; raises SplitLeakageError on judge_test."""
    if split != "judge_dev":
        raise SplitLeakageError(
            f"Prompt tuning may ONLY evaluate on 'judge_dev'; attempted split='{split}' is forbidden."
        )
    return _evaluate_split(session, split="judge_dev", judge_version=judge_version)


def evaluate_frozen_judge(
    session: Session,
    *,
    judge_version: str,
    registry_path: str | Path = DEFAULT_FREEZE_REGISTRY,
) -> dict[str, Any]:
    """Evaluate a frozen judge version on the held-out judge_test split."""
    if not is_judge_frozen(judge_version, registry_path=registry_path):
        raise JudgeFreezeError(
            f"Judge version '{judge_version}' must be frozen before evaluating on 'judge_test'."
        )
    return _evaluate_split(session, split="judge_test", judge_version=judge_version)
