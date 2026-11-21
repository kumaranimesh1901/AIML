"""Import validated human annotations from CSV into human_annotations."""

from __future__ import annotations

import csv
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy.orm import Session

from edg.db.models import HumanAnnotation
from edg.judge.schema import DimensionVerdict, EvidenceStatus
from edg.logging import get_logger

logger = get_logger(__name__)


class AnnotationRowSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    interaction_id: str = Field(min_length=1)
    annotator_id: str = Field(min_length=1)
    split: Literal["judge_dev", "judge_test"]
    evidence_status: EvidenceStatus = "sufficient"

    correctness_verdict: DimensionVerdict
    correctness_score: int | None = Field(default=None, ge=1, le=5)
    relevance_verdict: DimensionVerdict
    relevance_score: int | None = Field(default=None, ge=1, le=5)
    helpfulness_verdict: DimensionVerdict
    helpfulness_score: int | None = Field(default=None, ge=1, le=5)
    completeness_verdict: DimensionVerdict
    completeness_score: int | None = Field(default=None, ge=1, le=5)
    safety_verdict: DimensionVerdict
    safety_score: int | None = Field(default=None, ge=1, le=5)

    failure_types: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator(
        "correctness_score",
        "relevance_score",
        "helpfulness_score",
        "completeness_score",
        "safety_score",
        mode="before",
    )
    @classmethod
    def _empty_str_to_none(cls, value: Any) -> Any:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return int(value)

    @field_validator("failure_types", mode="before")
    @classmethod
    def _parse_failure_types(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            return [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
        return []


def import_annotations_csv(
    session: Session,
    csv_path: str | Path,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Validate and import human annotations from a CSV file into human_annotations."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Annotation CSV not found: {path}")

    inserted = 0
    skipped = 0
    rejected = 0
    reject_reasons: dict[str, int] = {}

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Skip unannotated template rows cleanly
            if not (row.get("annotator_id") or "").strip() and not (
                row.get("correctness_verdict") or ""
            ).strip():
                skipped += 1
                continue

            try:
                parsed = AnnotationRowSchema.model_validate(row)
            except ValidationError as exc:
                rejected += 1
                reason = exc.errors()[0].get("msg", "validation_error")
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                continue

            ann_id = f"ANN-{uuid.uuid5(uuid.NAMESPACE_DNS, f'{parsed.interaction_id}:{parsed.annotator_id}:{parsed.split}').hex[:12]}"
            if session.get(HumanAnnotation, ann_id) is not None:
                skipped += 1
                continue

            session.add(
                HumanAnnotation(
                    annotation_id=ann_id,
                    interaction_id=parsed.interaction_id,
                    annotator_id=parsed.annotator_id,
                    split=parsed.split,
                    evidence_status=parsed.evidence_status,
                    correctness_verdict=parsed.correctness_verdict,
                    correctness_score=parsed.correctness_score,
                    relevance_verdict=parsed.relevance_verdict,
                    relevance_score=parsed.relevance_score,
                    helpfulness_verdict=parsed.helpfulness_verdict,
                    helpfulness_score=parsed.helpfulness_score,
                    completeness_verdict=parsed.completeness_verdict,
                    completeness_score=parsed.completeness_score,
                    safety_verdict=parsed.safety_verdict,
                    safety_score=parsed.safety_score,
                    failure_types=parsed.failure_types,
                    notes=parsed.notes,
                    run_id=run_id,
                )
            )
            inserted += 1

    session.flush()
    stats = {
        "csv_path": str(path),
        "inserted": inserted,
        "skipped": skipped,
        "rejected": rejected,
        "reject_reasons": reject_reasons,
    }
    logger.info("human_annotations_imported", extra=stats)
    return stats
