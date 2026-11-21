"""Incremental ingestion-validation, text cleaning, PII pseudonymisation, and deduplication pipeline."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from edg.cleaning.cleaner import clean_text
from edg.cleaning.dedupe import assign_duplicate_groups
from edg.cleaning.pii import PIIPseudonymizer, persist_pii_mappings
from edg.config import AppConfig
from edg.db.models import InteractionClean, InteractionRaw
from edg.ingestion.ingest import validate_raw_row
from edg.logging import get_logger

logger = get_logger(__name__)


def run_cleaning_pipeline(
    session: Session,
    config: AppConfig,
    *,
    run_id: str,
    use_spacy: bool = True,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Incrementally clean, pseudonymise, and deduplicate only new interactions_raw rows.

    Never modifies interactions_raw.
    """
    start = time.perf_counter()

    already_processed_ids = select(InteractionClean.interaction_id)
    new_rows = list(
        session.scalars(
            select(InteractionRaw)
            .where(InteractionRaw.interaction_id.notin_(already_processed_ids))
            .order_by(InteractionRaw.interaction_id)
        ).all()
    )

    if not new_rows:
        counts = {
            "records_in": 0,
            "records_out": 0,
            "valid_count": 0,
            "rejected_count": 0,
            "reject_reasons": {},
            "pii_replacements": 0,
            "dup_groups": 0,
            "dup_share": 0.0,
            "throughput_per_min": 0.0,
        }
        logger.info("cleaning_pipeline_no_new_rows", extra=counts)
        return counts

    pseudonymizer = PIIPseudonymizer(use_spacy=use_spacy)

    valid_staging: list[tuple[InteractionRaw, str, str, str | None]] = []
    reject_reasons: dict[str, int] = {}
    rejected_count = 0
    pii_replacements = 0

    for row in new_rows:
        is_valid, reject_reason = validate_raw_row(row)
        if not is_valid:
            rejected_count += 1
            reason_key = reject_reason or "invalid_schema"
            reject_reasons[reason_key] = reject_reasons.get(reason_key, 0) + 1
            session.add(
                InteractionClean(
                    interaction_id=row.interaction_id,
                    clean_query="",
                    clean_response="",
                    pii_map_ref=None,
                    dup_group_id=None,
                    dup_count=0,
                    is_representative=False,
                    valid=False,
                    reject_reason=reason_key,
                    run_id=run_id,
                )
            )
            continue

        c_query = clean_text(row.user_query)
        c_response = clean_text(row.ai_response)

        p_query, q_recs = pseudonymizer.pseudonymize(c_query)
        p_response, r_recs = pseudonymizer.pseudonymize(c_response)
        all_pii = [*q_recs, *r_recs]

        pii_ref: str | None = None
        if all_pii:
            pii_ref = f"PII-{row.interaction_id}"
            pii_replacements += persist_pii_mappings(
                session, pii_map_ref=pii_ref, records=all_pii, run_id=run_id
            )

        valid_staging.append((row, p_query, p_response, pii_ref))

    # Dedupe valid new rows within intent
    dedupe_input = [
        (row.interaction_id, row.intent_label or "unknown", p_query)
        for row, p_query, _p_resp, _ref in valid_staging
    ]
    dedupe_map = assign_duplicate_groups(
        dedupe_input,
        cosine_threshold=config.cleaning.near_dup_cosine_threshold,
        embed_fn=embed_fn,
    )

    dup_group_ids: set[str] = set()
    non_rep_count = 0
    for row, p_query, p_response, pii_ref in valid_staging:
        gid, dup_count, is_rep = dedupe_map[row.interaction_id]
        dup_group_ids.add(gid)
        if not is_rep:
            non_rep_count += 1
        session.add(
            InteractionClean(
                interaction_id=row.interaction_id,
                clean_query=p_query,
                clean_response=p_response,
                pii_map_ref=pii_ref,
                dup_group_id=gid,
                dup_count=dup_count,
                is_representative=is_rep,
                valid=True,
                reject_reason=None,
                run_id=run_id,
            )
        )

    session.flush()
    elapsed_min = max((time.perf_counter() - start) / 60.0, 1e-6)
    valid_count = len(valid_staging)
    dup_share = round(non_rep_count / valid_count, 4) if valid_count > 0 else 0.0

    counts = {
        "records_in": len(new_rows),
        "records_out": valid_count + rejected_count,
        "valid_count": valid_count,
        "rejected_count": rejected_count,
        "reject_reasons": reject_reasons,
        "pii_replacements": pii_replacements,
        "dup_groups": len(dup_group_ids),
        "dup_share": dup_share,
        "throughput_per_min": round(len(new_rows) / elapsed_min, 1),
    }
    logger.info("cleaning_pipeline_completed", extra=counts)
    return counts
