"""Optional paraphrase/condition variants of critical test cases (require human approval)."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from edg.db.models import TestCase
from edg.llm.client import BaseLLMClient, LLMError
from edg.logging import get_logger

logger = get_logger(__name__)


def _paraphrase_fallback(query: str) -> str:
    return f"Could you please double-check this for me? {query}"


def generate_critical_variants(
    session: Session,
    *,
    run_id: str,
    max_variants: int = 10,
    llm_client: BaseLLMClient | None = None,
) -> int:
    """Create variant test cases for critical parent test cases (trust_tier='variant', status='candidate')."""
    critical_parents = list(
        session.scalars(
            select(TestCase)
            .where(
                TestCase.severity == "critical",
                TestCase.trust_tier != "variant",
                TestCase.parent_test_case_id.is_(None),
            )
            .order_by(TestCase.test_case_id)
            .limit(max_variants)
        ).all()
    )
    if not critical_parents:
        return 0

    use_live_llm = llm_client is not None and llm_client.provider != "fake"
    created = 0

    for parent in critical_parents:
        variant_id = f"TCV-{uuid.uuid5(uuid.NAMESPACE_DNS, f'variant:{parent.test_case_id}').hex[:12]}"
        if session.get(TestCase, variant_id) is not None:
            continue

        new_query: str | None = None
        if use_live_llm and llm_client is not None:
            prompt = (
                "Paraphrase this e-commerce support query while keeping the exact same order ID "
                f"and customer intent:\n{parent.input_query}\nReturn ONLY the paraphrased query."
            )
            try:
                resp = llm_client.generate(prompt=prompt, temperature=0.5)
                if resp.text.strip():
                    new_query = resp.text.strip()
            except LLMError as exc:
                logger.warning("variant_llm_fallback", extra={"parent_id": parent.test_case_id, "error": str(exc)})

        if not new_query:
            new_query = _paraphrase_fallback(parent.input_query)

        prov: dict[str, Any] = dict(parent.provenance_json or {})
        prov["variant_of"] = parent.test_case_id
        prov["source_tag"] = "synthetic_variant"

        session.add(
            TestCase(
                test_case_id=variant_id,
                source_interaction_id=parent.source_interaction_id,
                input_query=new_query,
                snapshot_id=parent.snapshot_id,
                policy_version=parent.policy_version,
                expected_checklist_json=parent.expected_checklist_json,
                reference_answer=parent.reference_answer,
                original_failure_types=list(parent.original_failure_types or []),
                severity=parent.severity,
                cluster_id=parent.cluster_id,
                intent=parent.intent,
                trust_tier="variant",
                status="candidate",  # Variants always require human approval
                parent_test_case_id=parent.test_case_id,
                provenance_json=prov,
                run_id=run_id,
            )
        )
        created += 1

    session.flush()
    return created
