"""Insert-only ingestion of raw interaction records and validation helpers."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError
from sqlalchemy.orm import Session

from edg.db.models import InteractionRaw
from edg.ingestion.schema import RawInteractionSchema
from edg.logging import get_logger

logger = get_logger(__name__)


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()))
        msg = err.get("msg", "invalid")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts)[:250]


def validate_raw_row(row: InteractionRaw) -> tuple[bool, str | None]:
    """Validate an existing InteractionRaw ORM row against RawInteractionSchema."""
    try:
        RawInteractionSchema.model_validate(
            {
                "interaction_id": row.interaction_id,
                "conversation_id": row.conversation_id,
                "turn_index": row.turn_index,
                "user_query": row.user_query,
                "ai_response": row.ai_response,
                "model_version": row.model_version,
                "timestamp": row.timestamp,
                "source_tag": row.source_tag,
                "snapshot_id": row.snapshot_id,
                "intent_label": row.intent_label,
                "reference_answer": row.reference_answer,
                "injected_failure": row.injected_failure,
            }
        )
        return True, None
    except ValidationError as exc:
        return False, _format_validation_error(exc)


def ingest_raw_records(
    session: Session,
    records: list[dict[str, Any]],
    *,
    run_id: str,
) -> dict[str, Any]:
    """Validate and insert raw interaction dicts into interactions_raw (insert-only)."""
    inserted = 0
    skipped_existing = 0
    rejected: list[dict[str, str]] = []
    reject_reasons: dict[str, int] = {}

    for idx, rec in enumerate(records):
        try:
            validated = RawInteractionSchema.model_validate(rec)
        except ValidationError as exc:
            reason = _format_validation_error(exc)
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
            rejected.append({"index": str(idx), "reason": reason})
            continue

        if session.get(InteractionRaw, validated.interaction_id) is not None:
            skipped_existing += 1
            continue

        session.add(
            InteractionRaw(
                interaction_id=validated.interaction_id,
                conversation_id=validated.conversation_id,
                turn_index=validated.turn_index,
                user_query=validated.user_query,
                ai_response=validated.ai_response,
                model_version=validated.model_version,
                timestamp=validated.timestamp,
                source_tag=validated.source_tag,
                intent_label=validated.intent_label,
                snapshot_id=validated.snapshot_id,
                reference_answer=validated.reference_answer,
                injected_failure=validated.injected_failure,
                raw_json=dict(rec),
                run_id=run_id,
            )
        )
        inserted += 1

    session.flush()
    stats = {
        "records_in": len(records),
        "inserted": inserted,
        "skipped_existing": skipped_existing,
        "rejected_count": len(rejected),
        "reject_reasons": reject_reasons,
    }
    logger.info("raw_records_ingested", extra=stats)
    return stats
