"""Human review queue routing rules."""

from __future__ import annotations

import json
from typing import Any


def _claim_verifier_disagreed(
    *,
    claim_checks_json: list[dict[str, Any]],
    raw_output: str,
) -> bool:
    """Detect disagreement between deterministic claim checks and the raw LLM verdict."""
    has_contradicted = any(c.get("status") == "contradicted" for c in (claim_checks_json or []))
    if not raw_output or raw_output.startswith("judge_error:"):
        return False
    try:
        parsed = json.loads(raw_output)
    except Exception:
        return False

    raw_corr = (parsed.get("correctness") or {}).get("verdict")
    if has_contradicted and raw_corr == "pass":
        return True
    return False


def evaluate_review_need(
    *,
    failure_types: list[str],
    severity: str,
    confidence: float,
    low_confidence_threshold: float,
    claim_checks_json: list[dict[str, Any]],
    raw_output: str,
) -> tuple[bool, str | None]:
    """Determine whether an interaction requires human review and why."""
    reasons: list[str] = []

    if "judge_error" in (failure_types or []):
        reasons.append("judge_error")
    if confidence < low_confidence_threshold:
        reasons.append("low_confidence")
    if severity == "critical":
        reasons.append("critical_severity")
    if _claim_verifier_disagreed(claim_checks_json=claim_checks_json, raw_output=raw_output):
        reasons.append("claim_verifier_disagreement")

    if reasons:
        return True, "; ".join(reasons)
    return False, None
