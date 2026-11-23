"""Customer feedback, escalation, executive correction and re-evaluation (edg.support)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import create_engine, inspect, select

from edg.benchmark.splits import assign_slice
from edg.config import load_config
from edg.db.models import ChatTurn, InteractionRaw, StateSnapshot, SupportTicket
from edg.db.models import TestCase as TestCaseRow
from edg.db.session import _find_migrations_dir, create_db_engine, get_session
from edg.llm.client import BaseLLMClient, LLMRequest
from edg.support import (
    CUSTOMER_ESCALATION_TAG,
    SupportError,
    ai_escalation_reason,
    evaluate_turn,
    record_feedback,
    record_turn,
    reevaluate_ticket,
    resolve_ticket,
    support_metrics,
)
from edg.sut.assistant import AssistantResult

DIMS = ("correctness", "relevance", "helpfulness", "completeness", "safety")


class _Stub(BaseLLMClient):
    """A non-fake client replaying scripted replies (stands in for Ollama)."""

    provider = "stub_http"

    def __init__(self, replies: list[str]) -> None:
        super().__init__(default_model="stub", max_retries=0, retry_base_delay_seconds=0.0)
        self.replies = replies
        self.requests: list[LLMRequest] = []

    def _send(self, request: LLMRequest) -> tuple[str, dict[str, Any]]:
        self.requests.append(request)
        return self.replies.pop(0), {}


def _judge(verdict: str = "pass", failure_types: list[str] | None = None) -> _Stub:
    out = {
        **{d: {"verdict": verdict, "score": 5 if verdict == "pass" else 1} for d in DIMS},
        "failure_types": failure_types or [],
        "explanation": "stub verdict",
        "confidence": 0.9,
    }
    return _Stub(['{"claims": []}', json.dumps(out)])


@pytest.fixture(scope="module")
def world_db(tmp_path_factory: pytest.TempPathFactory) -> str:
    db_url = f"sqlite:///{tmp_path_factory.mktemp('support') / 'edg_support.db'}"
    edg = [sys.executable, "-m", "edg.cli"]
    env = {**os.environ, "EDG_LLM_PROVIDER": "fake"}
    subprocess.run([*edg, "db", "init", "--db-url", db_url], check=True, capture_output=True, env=env)
    subprocess.run(
        [*edg, "world", "build", "--small", "--synthetic-catalogue", "--no-st", "--db-url", db_url],
        check=True, capture_output=True, env=env,
    )
    return db_url


@pytest.fixture
def engine(world_db: str):  # noqa: ANN201
    eng = create_db_engine(world_db)
    yield eng
    eng.dispose()


@pytest.fixture
def config(world_db: str):  # noqa: ANN201
    return load_config(small=True, overrides={"database": {"url": world_db}, "llm": {"provider": "ollama"}})


def _snapshot(engine, *, tagged: bool) -> StateSnapshot:  # noqa: ANN001
    with get_session(engine) as s:
        for snap in s.scalars(select(StateSnapshot).order_by(StateSnapshot.order_id)):
            if bool(snap.hard_state_tags) == tagged:
                return snap
    raise AssertionError("no snapshot")


def _turn(engine, snapshot: StateSnapshot, conv: str = "CONV-test") -> str:  # noqa: ANN001
    with get_session(engine) as s:
        t = record_turn(
            s,
            conversation_id=conv,
            turn_index=0,
            snapshot=snapshot,
            version_name="v2",
            base_model="qwen2.5:7b",
            message="Where is my order?",
            history=[],
            result=AssistantResult(answer="It has shipped.", model_version="v2", latency_ms=10.0),
        )
        return t.turn_id


def _ticket(engine, turn_id: str) -> SupportTicket | None:  # noqa: ANN001
    with get_session(engine) as s:
        return s.scalars(select(SupportTicket).where(SupportTicket.turn_id == turn_id)).first()


def test_migration_0003_adds_support_tables_to_an_older_database(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'old.db'}"
    cfg = AlembicConfig()
    cfg.set_main_option("script_location", str(_find_migrations_dir()))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "0002_immutability_triggers")
    eng = create_engine(url)
    with eng.begin() as conn:  # an older DB built by 0001 already has every table; drop the new ones
        conn.exec_driver_sql("DROP TABLE support_tickets")
        conn.exec_driver_sql("DROP TABLE chat_turns")
    command.upgrade(cfg, "head")
    assert {"chat_turns", "support_tickets"} <= set(inspect(eng).get_table_names())


def test_satisfied_customer_does_not_escalate_or_mark_correct(engine, config) -> None:  # noqa: ANN001
    tid = _turn(engine, _snapshot(engine, tagged=False))
    with get_session(engine) as s:
        assert record_feedback(s, tid, satisfied=True) is None
    with get_session(engine) as s:
        turn = s.get(ChatTurn, tid)
        assert turn.customer_feedback == "satisfied"
        assert turn.ai_failure_detected is None and turn.eval_status == "pending"


def test_dissatisfied_escalates_even_when_judge_passes(engine, config) -> None:  # noqa: ANN001
    tid = _turn(engine, _snapshot(engine, tagged=False))
    turn, ticket = evaluate_turn(engine, config, tid, policies=[], llm_client=_judge("pass"))
    assert turn.eval_status == "done" and turn.ai_failure_detected is False and ticket is None
    with get_session(engine) as s:
        ticket = record_feedback(s, tid, satisfied=False, comment="Wrong delivery date")
    assert ticket.trigger == "customer" and ticket.status == "pending"
    assert "Wrong delivery date" in ticket.escalation_reason


def test_serious_ai_error_escalates_without_complaint_and_merges_triggers(engine, config) -> None:  # noqa: ANN001
    tid = _turn(engine, _snapshot(engine, tagged=True))
    turn, ticket = evaluate_turn(engine, config, tid, policies=[], llm_client=_judge("fail", ["unsafe_response"]))
    assert turn.ai_severity == "critical"
    assert ticket is not None and ticket.trigger == "ai_judge" and ticket.priority == "urgent"
    with get_session(engine) as s:
        same = record_feedback(s, tid, satisfied=True)  # happy customer does not close it
    assert same.ticket_id == ticket.ticket_id and same.status == "pending"
    with get_session(engine) as s:
        both = record_feedback(s, tid, satisfied=False)
    assert both.trigger == "customer_and_ai_judge"


def test_ai_escalation_rules() -> None:
    assert ai_escalation_reason(failure_detected=True, severity="low", failure_types=["incomplete_response"], claim_checks=[]) is None
    assert ai_escalation_reason(failure_detected=False, severity=None, failure_types=[], claim_checks=[]) is None
    assert "high" in ai_escalation_reason(failure_detected=True, severity="high", failure_types=["factual_error"], claim_checks=[])
    assert "contradict" in ai_escalation_reason(
        failure_detected=False, severity=None, failure_types=[], claim_checks=[{"status": "contradicted"}]
    )


def test_resolve_and_reevaluate_promotes_a_candidate_test_case(engine, config) -> None:  # noqa: ANN001
    snap = _snapshot(engine, tagged=True)
    tid = _turn(engine, snap)
    with get_session(engine) as s:
        ticket = record_feedback(s, tid, satisfied=False, comment="not helpful")
        with pytest.raises(SupportError):
            resolve_ticket(s, ticket.ticket_id, executive_name="Asha", corrected_answer="  ")
        resolve_ticket(s, ticket.ticket_id, executive_name="Asha", corrected_answer="Correct answer.", notes="n")
        ticket_id = ticket.ticket_id
    assert _ticket(engine, tid).reeval_status == "queued"

    checklist = _Stub([json.dumps({"must_do": ["state the order status"], "must_not_do": ["promise a refund"]})])
    done = reevaluate_ticket(engine, config, ticket_id, judge_client=_judge("pass"), helper_client=checklist)
    assert done.reeval_status == "validated" and done.promoted_test_case_id
    with get_session(engine) as s:
        raw = s.get(InteractionRaw, done.promoted_interaction_id)
        tc = s.get(TestCaseRow, done.promoted_test_case_id)
        assert raw.source_tag == CUSTOMER_ESCALATION_TAG and raw.reference_answer == "Correct answer."
        assert raw.ai_response == "It has shipped."
        assert tc.status == "candidate" and tc.trust_tier == "gold"
        assert tc.reference_answer == "Correct answer." and tc.expected_checklist_json["must_do"]
        assert assign_slice(tc) == "mined"


def test_reevaluation_that_still_fails_is_not_promoted(engine, config) -> None:  # noqa: ANN001
    tid = _turn(engine, _snapshot(engine, tagged=False))
    with get_session(engine) as s:
        ticket = record_feedback(s, tid, satisfied=False)
        resolve_ticket(s, ticket.ticket_id, executive_name="Asha", corrected_answer="Still wrong.")
        ticket_id = ticket.ticket_id
    done = reevaluate_ticket(engine, config, ticket_id, judge_client=_judge("fail", ["factual_error"]))
    assert done.reeval_status == "failed_validation" and done.promoted_test_case_id is None


def test_live_paths_refuse_fake_provider(engine, world_db: str) -> None:  # noqa: ANN001
    from edg.sut.chat import ChatUnavailableError

    fake_cfg = load_config(small=True, overrides={"database": {"url": world_db}, "llm": {"provider": "fake"}})
    tid = _turn(engine, _snapshot(engine, tagged=False))
    with pytest.raises(ChatUnavailableError):
        evaluate_turn(engine, fake_cfg, tid, policies=[])


def test_support_metrics_keep_signals_separate(engine) -> None:  # noqa: ANN001
    with get_session(engine) as s:
        m = support_metrics(s)
    assert m["total_interactions"] >= 5
    assert m["satisfied"] + m["dissatisfied"] + m["no_feedback"] == m["total_interactions"]
    assert m["customer_escalations"] >= 3 and m["ai_escalations"] >= 1
    assert m["tickets_pending"] + m["tickets_resolved"] == m["tickets_total"]
    assert m["reeval_validated"] >= 1
