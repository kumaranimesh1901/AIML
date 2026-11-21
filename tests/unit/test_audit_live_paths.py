"""Regression tests for code paths that only run with a real LLM (found in a pre-run review).

  * SUT: a model that keeps calling tools past the budget had its raw tool-call JSON logged as
    the customer reply;
  * checklist grading matched dates, amounts and status codes only in the exact snapshot
    spelling, so "November 25, 2026" or "out for delivery" failed a correct reply;
  * Bitext placeholders such as {{Customer Support Phone Number}} or {{Website URL}} were filled
    with the customer token or the order id, and {{Refund Amount}} with the order total even
    when no refund exists, so the reference answer shown to the judge contained wrong facts;
  * a judge version already frozen with another prompt hash only failed at calibration, after
    the whole SUT and judge stages had run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from edg.config import AppConfig
from edg.db.models import StateSnapshot
from edg.db.session import get_session
from edg.evaluation.grade import evaluate_checklist
from edg.llm.fake import FakeLLMClient
from edg.queries.binder import _fill_placeholders
from edg.sut.assistant import CustomerSupportAssistant
from edg.sut.tools import SnapshotToolBox
from edg.world.store_generator import generate_store_world


@pytest.fixture
def populated_world(db_engine: Any, small_config: AppConfig) -> Any:
    from edg.db.repositories import track_run

    with get_session(db_engine) as session:
        with track_run(session, step="world_build", config=small_config) as (run, _counts):
            generate_store_world(session, small_config, run_id=run.run_id)
    return db_engine


def test_sut_never_logs_tool_call_json_as_reply(populated_world: Any, small_config: AppConfig) -> None:
    with get_session(populated_world) as session:
        snap = session.scalars(select(StateSnapshot).limit(1)).first()
        assert snap is not None
        tool_call = json.dumps({"action": "tool_call", "tool": "get_order", "arguments": {"order_id": snap.order_id}})
        llm = FakeLLMClient(responses=[tool_call] * 6, cache_enabled=False)
        assistant = CustomerSupportAssistant(
            version_config=small_config.sut.versions["v1"], llm_client=llm, max_tool_calls=2
        )
        res = assistant.answer(
            query="Where is my order?",
            order_id=snap.order_id,
            toolbox=SnapshotToolBox(snap, session=session),
        )
        assert '"action"' not in res.answer
        # The last request may only produce a final answer
        last_schema = llm.requests[-1].json_schema
        assert last_schema is not None
        assert last_schema["properties"]["action"] == {"const": "final_answer"}


_CHECKLIST = {
    "must_do": [
        "must state order ORD-1 status is 'out_for_delivery'",
        "must state package has not been delivered and expected delivery is 2026-11-25",
        "must reference accurate order total $89.99",
    ],
    "must_not_do": [],
}


@pytest.mark.parametrize(
    "reply",
    [
        "Order ORD-1 is out_for_delivery, expected 2026-11-25, total $89.99.",
        "Your order ORD-1 is out for delivery and should arrive on November 25, 2026. Your total was $89.99.",
        "Order ORD-1 is out-for-delivery; it hasn't arrived yet and is due Nov 25. The order total is 89.99 USD.",
    ],
)
def test_checklist_accepts_any_spelling_of_the_right_fact(reply: str) -> None:
    res = evaluate_checklist(reply, _CHECKLIST)
    assert res["satisfaction_rate"] == 1.0, res


def test_checklist_still_rejects_wrong_facts() -> None:
    reply = "Your order ORD-1 was delivered on November 20, 2026. The total was $99.99."
    res = evaluate_checklist(reply, _CHECKLIST)
    assert res["satisfied"] == 0, res


_SNAPSHOT: dict[str, Any] = {
    "order": {"order_id": "ORD-000123", "total_amount": 89.99, "placed_at": "2026-11-10T09:30:00Z"},
    "customer": {"name_token": "[NAME_1]", "email_token": "[EMAIL_1]"},
    "items": [{"title": "USB-C Cable"}],
    "shipments": [{"shipment_id": "SHP-9", "expected_delivery": "2026-11-25T18:00:00Z"}],
    "refunds_returns": [],
}


@pytest.mark.parametrize(
    ("placeholder", "must_not_contain"),
    [
        ("{{Customer Support Phone Number}}", "[NAME_1]"),
        ("{{Customer Support Hours}}", "[NAME_1]"),
        ("{{Website URL}}", "ORD-000123"),
        ("{{Online Company Portal Info}}", "ORD-000123"),
        ("{{Online Order Interaction}}", "ORD-000123"),
        ("{{Refund Amount}}", "$89.99"),
    ],
)
def test_non_order_placeholders_are_not_filled_with_order_facts(placeholder: str, must_not_contain: str) -> None:
    filled = _fill_placeholders(f"Please use {placeholder} for help.", _SNAPSHOT)
    head = filled.split(" (Order ")[0]
    assert must_not_contain not in head, filled
    assert "{{" not in filled


def test_order_placeholders_still_get_snapshot_values() -> None:
    filled = _fill_placeholders("Order {{Order Number}} for {{Client First Name}} ships {{Delivery Date}}.", _SNAPSHOT)
    assert filled == "Order ORD-000123 for [NAME_1] ships 2026-11-25."


def test_run_all_refuses_a_conflicting_frozen_judge_before_doing_any_work(tmp_path: Path, small_config: AppConfig) -> None:
    from edg.annotation.agreement import JudgeFreezeError, freeze_judge_version
    from edg.pipeline.orchestrator import run_full_pipeline

    registry = tmp_path / "frozen_judges.json"
    freeze_judge_version(
        judge_version=small_config.judge.judge_version,
        judge_model=small_config.judge.model_name,
        prompt_hash="0" * 64,
        registry_path=registry,
    )
    cfg = small_config.model_copy(deep=True)
    cfg.database.url = f"sqlite:///{tmp_path / 'edg.db'}"
    cfg.llm.provider = "fake"
    with pytest.raises(JudgeFreezeError, match="already frozen"):
        run_full_pipeline(
            cfg,
            annotations_dir=tmp_path / "ann",
            freeze_registry_path=registry,
            summary_output_path=None,
        )
    db_file = tmp_path / "edg.db"
    if db_file.exists():
        from edg.db.session import create_db_engine

        with get_session(create_db_engine(cfg.database.url)) as s:
            assert s.scalar(select(StateSnapshot).limit(1)) is None


def test_replay_judge_error_is_excluded_from_model_comparison() -> None:
    """A judge_error on a replayed answer says nothing about the model, so it must not count as a failure."""
    from types import SimpleNamespace

    from edg.evaluation.regression import _extract_metric_flags
    from edg.world.taxonomy import load_taxonomy

    jdg = SimpleNamespace(
        failure_types=["judge_error"],
        **{f"{d}_verdict": "not_assessable" for d in ("correctness", "relevance", "helpfulness", "completeness", "safety")},
    )
    er = SimpleNamespace(passed=False)
    tc = SimpleNamespace(intent="track_order")
    flags = _extract_metric_flags(er, jdg, tc, [], load_taxonomy())  # type: ignore[arg-type]
    assert flags and all(v is None for v in flags.values()), flags
