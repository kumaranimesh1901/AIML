"""Evidence assembly for judging: bound snapshot, top-k retrieved policies, and reference answer."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from edg.db.models import StateSnapshot
from edg.judge.retriever import retrieve_policy_chunks
from edg.judge.schema import EvidenceStatus


class EvidenceBundle(BaseModel):
    snapshot_id: str | None = None
    snapshot_json: dict[str, Any] | None = None
    policy_chunks: list[dict[str, Any]] = Field(default_factory=list)
    reference_answer: str | None = None
    evidence_status: EvidenceStatus = "missing"


def determine_evidence_status(
    snapshot_json: dict[str, Any] | None,
    policy_chunks: list[dict[str, Any]],
) -> EvidenceStatus:
    """Classify evidence completeness: sufficient, partial, or missing."""
    has_snapshot = bool(snapshot_json and isinstance(snapshot_json, dict) and snapshot_json.get("order"))
    has_policies = len(policy_chunks) > 0
    if has_snapshot and has_policies:
        return "sufficient"
    if has_snapshot:
        return "partial"
    return "missing"


def assemble_evidence(
    session: Session | None,
    *,
    snapshot_id: str | None,
    query: str,
    reference_answer: str | None = None,
    policy_version: str = "v1",
    top_k_policies: int = 3,
    snapshot_override: dict[str, Any] | None = None,
    policy_chunks_override: list[dict[str, Any]] | None = None,
    preloaded_policies: list[tuple[str, str, str, np.ndarray]] | None = None,
) -> EvidenceBundle:
    """Assemble snapshot JSON, top-k policy chunks, and reference answer into an EvidenceBundle."""
    snap_json = snapshot_override
    resolved_policy_version = policy_version

    if snap_json is None and session is not None and snapshot_id:
        snap_row = session.get(StateSnapshot, snapshot_id)
        if snap_row is not None:
            snap_json = snap_row.snapshot_json
            resolved_policy_version = snap_row.policy_version

    if policy_chunks_override is not None:
        policies = policy_chunks_override
    elif session is not None or preloaded_policies is not None:
        policies = retrieve_policy_chunks(
            session,  # type: ignore[arg-type]
            query=query,
            policy_version=resolved_policy_version,
            top_k=top_k_policies,
            preloaded_docs=preloaded_policies,
        )
    else:
        policies = []

    status = determine_evidence_status(snap_json, policies)
    return EvidenceBundle(
        snapshot_id=snapshot_id,
        snapshot_json=snap_json,
        policy_chunks=policies,
        reference_answer=reference_answer,
        evidence_status=status,
    )
