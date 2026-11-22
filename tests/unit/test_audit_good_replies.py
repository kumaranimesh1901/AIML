"""Offline replay of the judge's deterministic half on realistic, CORRECT support replies.

The LLM judge is not available offline, so this covers everything else the judge pipeline adds:
claim verification and the code rules (Rule 2 factual_error, Rule 3 hallucination). Claims are
written the way llama3.1:8b extracts them from real replies (order ids inside the claim text,
real shipment ids, natural-language dates, policy figures, relative times). A correct reply
must never get a failure label from the code; a reply with an invented value still must.
"""

from __future__ import annotations

import pytest

from edg.judge.llm_judge import enforce_code_rules
from edg.judge.schema import ExtractedClaim, LLMJudgeOutput
from edg.judge.verifier import verify_claims

SNAPSHOT = {
    "reference_date": "2026-11-20T12:00:00Z",
    "customer": {"customer_id": "CUST-000042", "name_token": "[NAME_1]"},
    "order": {
        "order_id": "ORD-000123",
        "customer_id": "CUST-000042",
        "status": "shipped",
        "total_amount": 89.99,
        "placed_at": "2026-11-10T09:30:00Z",
    },
    "items": [{"product_id": "PROD-0007", "title": "Anker PowerCore 10000", "unit_price": 89.99}],
    "payments": [{"payment_id": "PAY-000123", "amount": 89.99, "status": "captured", "charged_at": "2026-11-10T09:31:00Z"}],
    "shipments": [
        {
            "shipment_id": "SHP-000088",
            "carrier": "UPS",
            "expected_delivery": "2026-11-25T18:00:00Z",
            "carrier_scan_status": "in_transit",
            "carrier_scan_at": "2026-11-19T08:00:00Z",
            "actual_delivery": None,
            "customer_disputed": False,
        }
    ],
    "refunds_returns": [],
}
POLICIES = [
    {"policy_id": "POL-01", "title": "Returns window", "body": "Items can be returned within 30 days of delivery."},
    {"policy_id": "POL-02", "title": "Refund timelines", "body": "Card refunds post in 5 to 7 business days after issuance."},
]


def C(text: str, ctype: str, value: str | None) -> ExtractedClaim:  # noqa: N802
    return ExtractedClaim(claim_text=text, claim_type=ctype, claimed_value=value)  # type: ignore[arg-type]


GOOD_REPLY_CLAIMS = [
    C("I checked your order ORD-000123", "action_taken", "checked order ORD-000123"),
    C("Your order ORD-000123 has shipped", "status", "shipped"),
    C("It is in transit with UPS", "status", "in transit"),
    C("Shipment SHP-000088 is on its way", "action_taken", "SHP-000088"),
    C("Your tracking number is SHP-000088", "action_taken", "SHP-000088"),
    C("It should arrive by November 25", "date", "November 25"),
    C("expected delivery 2026-11-25", "date", "2026-11-25"),
    C("You were charged $89.99", "amount", "$89.99"),
    C("Your payment for ORD-000123 was captured", "status", "captured"),
    C("You can return it within 30 days of delivery", "policy", "30 days"),
    C("Refunds post in 5 to 7 business days", "amount", "5 to 7"),
    C("It will arrive in a few days", "date", "in a few days"),
    C("Your order is on its way", "status", "on its way"),
    C("Your order was placed on November 10, 2026", "date", "November 10, 2026"),
    C("I have escalated this to our team", "action_taken", "escalated"),
    C("Your order ORD-000123 is not delivered yet", "status", "not delivered"),
]


def _clean() -> LLMJudgeOutput:
    return LLMJudgeOutput.model_validate(
        {
            **{d: {"verdict": "pass", "score": 5} for d in ("correctness", "relevance", "helpfulness", "completeness", "safety")},
            "failure_types": [],
            "explanation": "Accurate and helpful.",
            "confidence": 0.9,
        }
    )


@pytest.mark.parametrize("claim", GOOD_REPLY_CLAIMS, ids=[c.claim_text for c in GOOD_REPLY_CLAIMS])
def test_a_correct_claim_never_gets_a_failure_from_the_code(claim: ExtractedClaim) -> None:
    checks = verify_claims([claim], SNAPSHOT, POLICIES)
    out = enforce_code_rules(_clean(), evidence_status="sufficient", claim_checks=checks)
    assert out.failure_types == [], (checks[0].status, checks[0].evidence_detail)
    assert out.correctness.verdict == "pass"


BAD_REPLY_CLAIMS = [
    (C("Your tracking number is TRK-999888777", "action_taken", "TRK-999888777"), "hallucination"),
    (C("Your tracking number is SHP-999999", "action_taken", "SHP-999999"), "hallucination"),
    (C("It will arrive on December 24, 2026", "date", "December 24, 2026"), "hallucination"),
    (C("Your order total was $139.99", "amount", "$139.99"), "factual_error"),
    (C("Your order has been delivered", "status", "delivered"), "factual_error"),
    (C("Your order ORD-000123 was cancelled", "status", "cancelled"), "factual_error"),
    (C("We issued a $25 loyalty voucher", "action_taken", "loyalty_voucher"), "hallucination"),
]


@pytest.mark.parametrize(("claim", "expected"), BAD_REPLY_CLAIMS, ids=[c.claim_text for c, _ in BAD_REPLY_CLAIMS])
def test_a_wrong_or_invented_value_is_still_caught(claim: ExtractedClaim, expected: str) -> None:
    checks = verify_claims([claim], SNAPSHOT, POLICIES)
    out = enforce_code_rules(_clean(), evidence_status="sufficient", claim_checks=checks)
    assert expected in out.failure_types, (checks[0].status, checks[0].evidence_detail)
