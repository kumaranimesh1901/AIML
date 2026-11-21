"""Judge prompt construction, rubric loading, and deterministic prompt hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from edg.config import _find_project_root
from edg.judge.evidence import EvidenceBundle
from edg.judge.schema import ClaimCheckResult


def load_judge_rubric(path: str | Path | None = None) -> dict[str, Any]:
    """Load configs/judge_rubric.yaml."""
    resolved = Path(path) if path is not None else _find_project_root() / "configs" / "judge_rubric.yaml"
    if not resolved.exists():
        raise FileNotFoundError(f"Judge rubric file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_judge_system_prompt(rubric: dict[str, Any]) -> str:
    """Build the system prompt embedding rubric dimensions and strict evidence rules."""
    dims_text = []
    for dim_name, dim_def in (rubric.get("dimensions") or {}).items():
        dims_text.append(f"- {dim_name}: {dim_def.get('description', '')}")
    return (
        "You are an impartial, evidence-based evaluation judge for NovaMart customer-support replies.\n"
        "NEVER guess truth. Base correctness strictly on the provided state snapshot, policy chunks, "
        "and deterministic claim checks.\n"
        "If evidence_status is 'missing', correctness MUST be 'not_assessable' with score null.\n"
        "Dimensions to score (verdict in [pass, partial, fail, not_assessable] and score 1-5):\n"
        + "\n".join(dims_text)
        + "\nReturn ONLY valid JSON matching LLMJudgeOutput."
    )


def compute_prompt_hash(system_prompt: str, rubric: dict[str, Any]) -> str:
    payload = json.dumps(
        {"system_prompt": system_prompt, "rubric": rubric},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_judge_user_prompt(
    *,
    query: str,
    response: str,
    evidence: EvidenceBundle,
    claim_checks: list[ClaimCheckResult],
) -> str:
    """Format the per-interaction user prompt for the judge LLM."""
    parts = [
        f"Customer Query:\n{query}",
        f"Assistant Response:\n{response}",
        f"Evidence Status: {evidence.evidence_status}",
        f"State Snapshot JSON:\n{json.dumps(evidence.snapshot_json, sort_keys=True) if evidence.snapshot_json else 'NONE'}",
        f"Retrieved Policy Chunks:\n{json.dumps(evidence.policy_chunks, sort_keys=True)}",
        f"Reference Answer:\n{evidence.reference_answer or 'NONE'}",
        "Deterministic Claim Checks:\n"
        + json.dumps([c.model_dump() for c in claim_checks], sort_keys=True),
    ]
    return "\n\n".join(parts)
