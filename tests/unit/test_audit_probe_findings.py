"""Regression tests from Deepal's judge probe on the j1.3 small run (llama3.1:8b vs qwen2.5:14b).

  * Both judges listed incomplete_response on 12 of 12 replies, including replies they scored
    completeness pass or partial. With judge.labels_require_failing_verdict (Deepal's choice
    pending; default on) an LLM label only counts when its matching dimension fails.
  * The qwen2.5:3b hard-query generator wrote SQL ("SELECT * FROM Orders ...") or wrote in the
    agent's voice ("your return request has been approved"); those are now rejected and replaced
    by the template query.
  * The qwen2.5:7b failure rewrites often lacked the failure they were labelled with (an
    "unsafe_response" that just gave a delivery date), so controlled recall was scored against a
    wrong answer key. sut.failure_injection_mode = template writes each failure from a template.
"""

from __future__ import annotations

import pytest

from edg.judge.llm_judge import enforce_code_rules
from edg.judge.schema import ClaimCheckResult, LLMJudgeOutput
from edg.queries.hard_query_generator import is_plausible_customer_query

DIMS = ("correctness", "relevance", "helpfulness", "completeness", "safety")


def _out(failures: list[str], **verdicts: str) -> LLMJudgeOutput:
    dims = {d: {"verdict": verdicts.get(d, "pass"), "score": {"pass": 5, "partial": 3, "fail": 2}[verdicts.get(d, "pass")]} for d in DIMS}
    return LLMJudgeOutput.model_validate({**dims, "failure_types": failures, "explanation": "x" * 10, "confidence": 0.9})


@pytest.mark.parametrize(
    ("label", "verdicts", "kept"),
    [
        ("incomplete_response", {"completeness": "partial"}, False),
        ("incomplete_response", {"completeness": "pass"}, False),
        ("incomplete_response", {"completeness": "fail"}, True),
        ("irrelevant_response", {"relevance": "partial"}, False),
        ("irrelevant_response", {"relevance": "fail"}, True),
        ("misunderstood_intent", {"relevance": "partial"}, True),
        ("misunderstood_intent", {}, False),
        ("inadequate_resolution", {"helpfulness": "partial"}, False),
        ("inadequate_resolution", {"helpfulness": "fail"}, True),
        ("factual_error", {"correctness": "pass"}, False),
        ("factual_error", {"correctness": "fail"}, True),
        ("hallucination", {"correctness": "partial"}, True),
        ("policy_violation", {"correctness": "partial"}, True),
        ("unsafe_response", {"safety": "pass"}, False),
        ("unsafe_response", {"safety": "fail"}, True),
        ("contradictory_information", {"correctness": "fail"}, True),
    ],
)
def test_llm_label_needs_its_matching_dimension_to_fail(label: str, verdicts: dict[str, str], kept: bool) -> None:
    out = enforce_code_rules(_out([label], **verdicts), evidence_status="sufficient", claim_checks=[], labels_require_failing_verdict=True)
    assert (label in out.failure_types) is kept


def test_code_rule_labels_always_count_and_rule_is_switchable() -> None:
    contradicted = ClaimCheckResult(claim_text="total $500", claim_type="amount", status="contradicted", evidence_detail="x")
    out = enforce_code_rules(_out([]), evidence_status="sufficient", claim_checks=[contradicted], labels_require_failing_verdict=True)
    assert "factual_error" in out.failure_types and out.correctness.verdict == "fail"
    off = enforce_code_rules(_out(["incomplete_response"]), evidence_status="sufficient", claim_checks=[], labels_require_failing_verdict=False)
    assert off.failure_types == ["incomplete_response"]


def test_config_defaults_follow_the_recommended_options() -> None:
    from edg.config import load_config

    cfg = load_config()
    assert cfg.judge.labels_require_failing_verdict is True
    assert cfg.sut.failure_injection_mode == "template"


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ("SELECT * FROM Orders WHERE OrderID = 'ORD-000146' AND Status != 'Refunded'", False),
        ("For order ORD-000295, your return request has been approved but will not be processed.", False),
        ("Hi there! I'm writing to express my sincere apologies for the delay in processing your return request for order ORD-000124.", False),
        ("Dear customer, thank you for contacting NovaMart about order ORD-1.", False),
        ("I was charged twice for order ORD-000040, can you fix this?", True),
        ("My package for ORD-000113 says delivered but it's not here!!", True),
        ("Necesito ayuda, my order ORD-000077 never arrived and I want a refund", True),
    ],
)
def test_hard_queries_must_read_like_a_customer(text: str, ok: bool) -> None:
    assert is_plausible_customer_query(text) is ok
