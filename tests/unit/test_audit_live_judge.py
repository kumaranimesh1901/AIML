"""Regression tests for live-LLM judge defects seen in a real Ollama run (llama3.1:8b).

Symptoms: 84% of interactions flagged as failures while only 10% of injected controlled-tier
failures were caught. Root causes covered here:
  * the judge prompt never listed the taxonomy failure-type names or decision rules, so the LLM
    invented names (e.g. "incorrect_information") that never match taxonomy-based recall,
    severity, or labeling;
  * the claim verifier compared LLM-extracted status values by exact string, so "In transit"
    contradicted "in_transit", and "not delivered" was treated as a claim of delivery.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from edg.config import load_config
from edg.judge.evidence import EvidenceBundle
from edg.judge.llm_judge import HybridJudge
from edg.judge.prompts import build_judge_system_prompt, load_judge_rubric
from edg.judge.schema import ExtractedClaim
from edg.judge.verifier import verify_claims
from edg.llm.client import BaseLLMClient, LLMRequest
from edg.world.taxonomy import load_taxonomy

_TAXONOMY = load_taxonomy()


def test_judge_system_prompt_lists_every_taxonomy_failure_type_with_decision_rule() -> None:
    prompt = build_judge_system_prompt(load_judge_rubric())
    for name, ft in _TAXONOMY.failure_types.items():
        assert name in prompt
        assert ft.decision_rule in prompt


class _RecordingClient(BaseLLMClient):
    provider = "stub_http"

    def __init__(self, replies: list[str]) -> None:
        super().__init__(default_model="llama3.1:8b", max_retries=0, retry_base_delay_seconds=0.0)
        self.replies = replies
        self.requests: list[LLMRequest] = []

    def _send(self, request: LLMRequest) -> tuple[str, dict[str, Any]]:
        self.requests.append(request)
        return self.replies.pop(0), {}


def _judge_json(failure_types: list[str]) -> str:
    import json

    return json.dumps(
        {
            **{d: {"verdict": "fail", "score": 2} for d in ("correctness", "relevance", "helpfulness", "completeness", "safety")},
            "failure_types": failure_types,
            "explanation": "The answer is wrong.",
            "confidence": 0.9,
        }
    )


def _evidence() -> EvidenceBundle:
    return EvidenceBundle(
        snapshot_id="s1",
        snapshot_json={"order": {"order_id": "ORD-1", "status": "shipped", "total_amount": 10.0}},
        policy_chunks=[],
        reference_answer=None,
        evidence_status="sufficient",
    )


def test_judge_schema_restricts_failure_types_to_taxonomy() -> None:
    cfg = load_config(small=True, overrides={"llm": {"provider": "fake"}})
    client = _RecordingClient(
        [
            '{"claims": []}',
            _judge_json(["incorrect_information"]),  # invented name -> must be rejected & retried
            _judge_json(["factual_error"]),
        ]
    )
    judge = HybridJudge(config=cfg, llm_client=client)
    cfg.judge.max_retries = 1
    jdg, _ = judge.judge_one(
        session=None, interaction_id="i1", query="Where is my order?", response="It was cancelled.",
        evidence=_evidence(), run_id="r1",
    )
    assert jdg.failure_types == ["factual_error"]
    schema = client.requests[1].json_schema
    assert schema is not None
    enum = schema["properties"]["failure_types"]["items"]["enum"]
    assert sorted(enum) == sorted(_TAXONOMY.failure_types)


def test_judge_output_model_rejects_unknown_failure_type() -> None:
    cfg = load_config(small=True, overrides={"llm": {"provider": "fake"}})
    judge = HybridJudge(config=cfg, llm_client=_RecordingClient([]))
    with pytest.raises(ValidationError):
        judge.output_schema.model_validate_json(_judge_json(["incorrect_information"]))


@pytest.mark.parametrize(
    ("claimed", "text", "expected"),
    [
        ("In transit", "Your package is in transit.", "supported"),
        ("Out for delivery", "It is out for delivery today.", "supported"),
        ("not delivered", "Your order has not been delivered yet.", "supported"),
        ("delivered", "Your order was delivered.", "contradicted"),
        ("cancelled", "Your order is cancelled.", "contradicted"),
    ],
)
def test_status_claims_are_normalised_and_negation_aware(claimed: str, text: str, expected: str) -> None:
    snapshot = {
        "order": {"order_id": "ORD-1", "status": "shipped", "total_amount": 10.0},
        "shipments": [
            {"shipment_id": "SHP-1", "carrier_scan_status": "in_transit", "actual_delivery": None},
            {"shipment_id": "SHP-2", "carrier_scan_status": "out_for_delivery", "actual_delivery": None},
        ],
    }
    res = verify_claims([ExtractedClaim(claim_text=text, claim_type="status", claimed_value=claimed)], snapshot)
    assert res[0].status == expected, res[0].evidence_detail
