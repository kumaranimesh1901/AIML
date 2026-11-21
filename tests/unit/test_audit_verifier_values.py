"""Regression tests for the second real-Ollama small run (judge j1.1, real Bitext + Amazon data).

Symptoms: 338/350 judged interactions flagged as failures (random_control 98%), controlled-tier
recall 18%, correctness scored on only 12/60 calibration items and completeness on 1/60.
Root causes covered here:
  * the claim verifier compared LLM-extracted dates by exact ISO string, so a correct
    "2026-11-25T14:00:00Z" or "November 25, 2026" was "not in evidence" and, being concrete,
    auto-labelled hallucination;
  * any amount absent from the snapshot was "contradicted" (factual_error + correctness fail),
    even a policy figure such as a $5.99 shipping fee, and the 30 in "30-day window";
  * any status value outside the snapshot ("being processed", "will be delivered on ...") was
    "contradicted", and payment statuses ("captured") were never considered;
  * the judge schema let every dimension score default to null, so the constrained Ollama
    output dropped scores while still giving a verdict.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from edg.judge.schema import DimensionAssessment, ExtractedClaim, LLMJudgeOutput
from edg.judge.verifier import verify_claims

_SNAPSHOT = {
    "order": {
        "order_id": "ORD-1",
        "status": "shipped",
        "total_amount": 89.99,
        "placed_at": "2026-11-10T09:30:00Z",
    },
    "items": [{"unit_price": 89.99}],
    "payments": [{"amount": 89.99, "status": "captured", "charged_at": "2026-11-10T09:31:00Z"}],
    "shipments": [
        {
            "shipment_id": "SHP-1",
            "carrier_scan_status": "in_transit",
            "expected_delivery": "2026-11-25T18:00:00Z",
            "actual_delivery": None,
        }
    ],
    "refunds_returns": [],
}

_POLICIES = [
    {"policy_id": "POL-1", "title": "Shipping and returns", "body": "Standard shipping costs $5.99 and is free on orders over $50. "
                   "Items can be returned within a 30-day return window."}
]


def _check(claim_type: str, value: str, text: str) -> str:
    claim = ExtractedClaim(claim_text=text, claim_type=claim_type, claimed_value=value)  # type: ignore[arg-type]
    res = verify_claims([claim], _SNAPSHOT, _POLICIES)
    return res[0].status


@pytest.mark.parametrize(
    "value",
    [
        "2026-11-25",
        "2026-11-25T18:00:00Z",
        "November 25, 2026",
        "Nov 25",
        "25 November 2026",
        "11/25/2026",
    ],
)
def test_dates_in_any_common_format_match_the_snapshot(value: str) -> None:
    assert _check("date", value, f"It should arrive by {value}.") == "supported"


def test_date_absent_from_snapshot_stays_not_in_evidence() -> None:
    assert _check("date", "December 24, 2026", "It will arrive on December 24, 2026.") == "not_in_evidence"


@pytest.mark.parametrize(
    ("value", "text", "expected"),
    [
        ("5.99", "Standard shipping costs $5.99.", "supported"),
        ("30", "You have a 30-day return window.", "supported"),
        ("12.50", "There is a $12.50 restocking fee.", "not_in_evidence"),
        ("89.99", "Your order total was $89.99.", "supported"),
        ("239.99", "Your order total was $239.99.", "contradicted"),
        ("120.00", "You were charged $120.00.", "contradicted"),
        ("40.00", "We will refund $40.00.", "not_in_evidence"),
    ],
)
def test_amounts_only_contradict_when_they_claim_a_recorded_amount(value: str, text: str, expected: str) -> None:
    assert _check("amount", value, text) == expected


@pytest.mark.parametrize(
    ("value", "text", "expected"),
    [
        ("being processed", "Your order is being processed.", "not_in_evidence"),
        ("will be delivered", "Your order will be delivered on November 25.", "not_in_evidence"),
        ("captured", "Your payment has been captured.", "supported"),
        ("shipped", "Your order has shipped.", "supported"),
        ("cancelled", "Your order is cancelled.", "contradicted"),
        ("delivered", "Your order was delivered.", "contradicted"),
    ],
)
def test_status_claims_only_contradict_on_a_known_conflicting_status(value: str, text: str, expected: str) -> None:
    assert _check("status", value, text) == expected


def test_deterministic_factual_error_template_is_still_contradicted() -> None:
    from edg.judge.claims import extract_claims_deterministic

    text = "Order ORD-1 currently has status 'cancelled' with a total of $239.99, and it was delivered yesterday."
    checks = verify_claims(extract_claims_deterministic(text), _SNAPSHOT, _POLICIES)
    amount = [c for c in checks if c.claim_type == "amount"]
    assert amount and amount[0].status == "contradicted"


def test_dimension_score_is_required_unless_not_assessable() -> None:
    with pytest.raises(ValidationError):
        DimensionAssessment.model_validate({"verdict": "pass", "score": None})
    with pytest.raises(ValidationError):
        DimensionAssessment.model_validate({"verdict": "fail"})
    assert DimensionAssessment.model_validate({"verdict": "not_assessable", "score": None}).score is None
    # A score given alongside not_assessable is dropped rather than kept
    assert DimensionAssessment.model_validate({"verdict": "not_assessable", "score": 3}).score is None


def test_judge_json_schema_requires_dimension_scores() -> None:
    schema = LLMJudgeOutput.model_json_schema()
    dim = schema["$defs"]["DimensionAssessment"]
    assert set(dim["required"]) == {"verdict", "score"}


# ---------------------------------------------------------------------------
# Third real run (j1.2): llama3.1:8b kept sending {"verdict": "fail", "score": null} even after
# corrective retries, so judgements ended as judge_error. The schema sent to Ollama allowed null,
# and constrained decoding let the model pick it. The LLM-facing schema now only allows an
# integer score; for not_assessable the code drops whatever score was sent.
# ---------------------------------------------------------------------------


def _judge_schema() -> dict:
    from edg.config import load_config
    from edg.judge.llm_judge import HybridJudge
    from edg.llm.fake import FakeLLMClient

    cfg = load_config(small=True, overrides={"llm": {"provider": "fake"}})
    return HybridJudge(config=cfg, llm_client=FakeLLMClient()).output_schema  # type: ignore[return-value]


def test_llm_facing_judge_schema_forbids_null_scores() -> None:
    model = _judge_schema()
    schema = model.model_json_schema()  # type: ignore[attr-defined]
    dims = [d for name, d in schema["$defs"].items() if "Dimension" in name]
    assert dims
    for dim in dims:
        score = dim["properties"]["score"]
        assert score.get("type") == "integer", score
        assert "anyOf" not in score
        assert "score" in dim["required"]


def test_llm_facing_judge_schema_drops_score_for_not_assessable() -> None:
    model = _judge_schema()
    payload = {
        "correctness": {"verdict": "not_assessable", "score": 3},
        "relevance": {"verdict": "fail", "score": 1},
        "helpfulness": {"verdict": "partial", "score": 3},
        "completeness": {"verdict": "fail", "score": 2},
        "safety": {"verdict": "pass", "score": 5},
        "failure_types": [],
        "explanation": "Off-topic reply.",
        "confidence": 0.8,
    }
    out = model.model_validate(payload)  # type: ignore[attr-defined]
    assert out.correctness.score is None
    assert out.relevance.score == 1
    with pytest.raises(ValidationError):
        model.model_validate({**payload, "relevance": {"verdict": "fail", "score": None}})  # type: ignore[attr-defined]
