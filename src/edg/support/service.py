"""Customer feedback, escalation to support executives, and re-evaluation of human corrections.

Two evaluation signals are kept apart on every chat turn:

* AI evaluation: the frozen hybrid judge plus deterministic claim checks (technical correctness).
* Customer feedback: satisfied or dissatisfied (customer experience).

Either signal alone can escalate a turn to a support ticket: a dissatisfied customer escalates
even when the judge passed the answer, and a serious judge or verification finding escalates
even when the customer said nothing or was satisfied. Positive feedback never marks an answer as
correct and never closes a ticket.

LLM calls (judge, checklist drafting) run outside any database transaction, so a slow model never
holds a SQLite lock. Functions that call a model therefore take an Engine, not a Session.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from edg.config import AppConfig
from edg.db.models import ChatTurn, InteractionRaw, Judgement, StateSnapshot, SupportTicket, TestCase
from edg.db.repositories import track_run
from edg.db.session import get_session
from edg.labeling.rules import label_judgement
from edg.llm.client import BaseLLMClient
from edg.logging import get_logger
from edg.sut.assistant import AssistantResult
from edg.sut.chat import build_live_client, grade_reply, load_policies
from edg.testcases.checklist import generate_and_validate_checklist
from edg.world.taxonomy import load_taxonomy

logger = get_logger(__name__)

CUSTOMER_ESCALATION_TAG = "customer_escalation"
SERIOUS_SEVERITIES = frozenset({"high", "critical"})
_PRIORITY_BY_SEVERITY = {"critical": "urgent", "high": "high"}


class SupportError(ValueError):
    """Raised for an invalid support action (unknown ticket, empty correction, ...)."""


def _now() -> datetime:
    return datetime.now(UTC)


def new_conversation_id() -> str:
    return f"CONV-{uuid.uuid4().hex[:12]}"


# ---- Recording chat turns -------------------------------------------------------------------------


def record_turn(
    session: Session,
    *,
    conversation_id: str,
    turn_index: int,
    snapshot: StateSnapshot,
    version_name: str,
    base_model: str,
    message: str,
    history: list[tuple[str, str]] | None,
    result: AssistantResult,
) -> ChatTurn:
    """Store one customer message and the chatbot's reply."""
    turn = ChatTurn(
        turn_id=f"CHAT-{uuid.uuid4().hex[:12]}",
        conversation_id=conversation_id,
        turn_index=turn_index,
        order_id=snapshot.order_id,
        snapshot_id=snapshot.snapshot_id,
        model_version=version_name,
        base_model=base_model,
        user_query=message,
        ai_response=result.answer,
        history_json=[list(h) for h in (history or [])],
        tool_calls_json=list(result.tool_calls),
        latency_ms=round(float(result.latency_ms), 2),
        error=result.error,
        eval_status="pending",
        ai_failure_types=[],
        ai_verdicts_json={},
        ai_claim_checks_json=[],
        created_at=_now(),
    )
    session.add(turn)
    session.flush()
    return turn


# ---- Escalation -----------------------------------------------------------------------------------


def ai_escalation_reason(
    *,
    failure_detected: bool,
    severity: str | None,
    failure_types: list[str],
    claim_checks: list[dict[str, Any]],
) -> str | None:
    """Why the AI evaluation alone should escalate this turn, or None if it shouldn't."""
    reasons: list[str] = []
    if failure_detected and severity in SERIOUS_SEVERITIES:
        names = ", ".join(t.replace("_", " ") for t in failure_types) or "a failing dimension"
        reasons.append(f"AI judge found a {severity}-severity problem ({names})")
    contradicted = [c for c in claim_checks if c.get("status") == "contradicted"]
    if contradicted:
        reasons.append(
            f"verification found {len(contradicted)} claim(s) that contradict the order record"
        )
    return "; ".join(reasons) or None


def _escalate(session: Session, turn: ChatTurn, *, source: str, reason: str) -> SupportTicket:
    """Open a ticket for this turn, or add a second trigger to its existing ticket."""
    ticket = session.scalars(select(SupportTicket).where(SupportTicket.turn_id == turn.turn_id)).first()
    priority = _PRIORITY_BY_SEVERITY.get(turn.ai_severity or "", "normal") if source == "ai_judge" else "normal"
    if ticket is None:
        ticket = SupportTicket(
            ticket_id=f"TKT-{uuid.uuid4().hex[:8].upper()}",
            turn_id=turn.turn_id,
            conversation_id=turn.conversation_id,
            trigger=source,
            priority=priority,
            status="pending",
            escalation_reason=reason,
            delivered_to_customer=False,
            reeval_status="not_started",
            reeval_json={},
            created_at=_now(),
            updated_at=_now(),
        )
        session.add(ticket)
    else:
        if source not in ticket.trigger:
            ticket.trigger = "customer_and_ai_judge"
        if reason not in ticket.escalation_reason:
            ticket.escalation_reason = f"{ticket.escalation_reason}; {reason}".strip("; ")
        order = ["normal", "high", "urgent"]
        if order.index(priority) > order.index(ticket.priority if ticket.priority in order else "normal"):
            ticket.priority = priority
        ticket.updated_at = _now()
    session.flush()
    logger.info("support_escalation", extra={"ticket_id": ticket.ticket_id, "source": source})
    return ticket


def apply_ai_evaluation(
    session: Session,
    turn_id: str,
    judgement: Judgement,
    *,
    config: AppConfig,
    hard_state_tags: list[str] | None,
) -> tuple[ChatTurn, SupportTicket | None]:
    """Store the judge's verdict on a turn and escalate it if the finding is serious."""
    turn = session.get(ChatTurn, turn_id)
    if turn is None:
        raise SupportError(f"Unknown chat turn {turn_id}")
    turn.judge_model = judgement.judge_model
    turn.judge_version = judgement.judge_version
    turn.evaluated_at = _now()
    turn.ai_explanation = judgement.explanation
    turn.ai_claim_checks_json = list(judgement.claim_checks_json or [])
    if "judge_error" in (judgement.failure_types or []):
        turn.eval_status = "error"
        session.flush()
        return turn, None

    labels = label_judgement(
        judgement,
        intent_label=None,
        hard_state_tags=hard_state_tags,
        config=config,
        taxonomy=load_taxonomy(),
    )
    turn.eval_status = "done"
    turn.ai_failure_detected = bool(labels["failure_detected"])
    turn.ai_severity = labels["severity"] if labels["failure_detected"] else None
    turn.ai_failure_types = list(labels["failure_types"])
    turn.ai_confidence = float(judgement.confidence)
    turn.ai_verdicts_json = {
        d: {"verdict": getattr(judgement, f"{d}_verdict"), "score": getattr(judgement, f"{d}_score")}
        for d in ("correctness", "relevance", "helpfulness", "completeness", "safety")
    }
    session.flush()

    reason = ai_escalation_reason(
        failure_detected=turn.ai_failure_detected,
        severity=turn.ai_severity,
        failure_types=turn.ai_failure_types,
        claim_checks=turn.ai_claim_checks_json,
    )
    ticket = _escalate(session, turn, source="ai_judge", reason=reason) if reason else None
    return turn, ticket


def evaluate_turn(
    engine: Engine,
    config: AppConfig,
    turn_id: str,
    *,
    policies: list[Any] | None = None,
    llm_client: BaseLLMClient | None = None,
) -> tuple[ChatTurn, SupportTicket | None]:
    """Grade a stored turn with the frozen judge (outside any transaction), then store the result."""
    with get_session(engine) as session:
        turn = session.get(ChatTurn, turn_id)
        if turn is None:
            raise SupportError(f"Unknown chat turn {turn_id}")
        snapshot = session.get(StateSnapshot, turn.snapshot_id)
        if policies is None:
            policies = load_policies(session, snapshot.policy_version)
        query, reply = turn.user_query, turn.ai_response
    judgement = grade_reply(
        config=config, snapshot=snapshot, message=query, reply=reply, policies=policies, llm_client=llm_client
    )
    with get_session(engine) as session:
        return apply_ai_evaluation(
            session, turn_id, judgement, config=config, hard_state_tags=list(snapshot.hard_state_tags or [])
        )


def record_feedback(
    session: Session, turn_id: str, *, satisfied: bool, comment: str | None = None
) -> SupportTicket | None:
    """Store the customer's answer to "Are you satisfied?". Dissatisfied always escalates.

    Satisfied feedback changes nothing about the AI evaluation and never closes a ticket."""
    turn = session.get(ChatTurn, turn_id)
    if turn is None:
        raise SupportError(f"Unknown chat turn {turn_id}")
    turn.customer_feedback = "satisfied" if satisfied else "dissatisfied"
    turn.feedback_comment = (comment or "").strip() or None
    turn.feedback_at = _now()
    session.flush()
    if satisfied:
        return session.scalars(select(SupportTicket).where(SupportTicket.turn_id == turn_id)).first()
    reason = "customer was not satisfied" + (f": \"{turn.feedback_comment}\"" if turn.feedback_comment else "")
    return _escalate(session, turn, source="customer", reason=reason)


# ---- Support executive ----------------------------------------------------------------------------


def resolve_ticket(
    session: Session,
    ticket_id: str,
    *,
    executive_name: str,
    corrected_answer: str,
    notes: str | None = None,
) -> SupportTicket:
    """Send the executive's corrected answer to the customer and queue it for re-evaluation."""
    ticket = session.get(SupportTicket, ticket_id)
    if ticket is None:
        raise SupportError(f"Unknown ticket {ticket_id}")
    if not corrected_answer.strip():
        raise SupportError("The corrected answer is empty.")
    if not executive_name.strip():
        raise SupportError("Enter the support executive's name.")
    ticket.executive_name = executive_name.strip()
    ticket.corrected_answer = corrected_answer.strip()
    ticket.resolution_notes = (notes or "").strip() or None
    ticket.status = "resolved"
    ticket.delivered_to_customer = True
    ticket.resolved_at = _now()
    ticket.reeval_status = "queued"
    ticket.reeval_json = {}
    ticket.updated_at = _now()
    session.flush()
    return ticket


def _promote(session: Session, config: AppConfig, ticket: SupportTicket, turn: ChatTurn, snapshot: StateSnapshot,
             checklist: dict[str, Any]) -> tuple[str, str]:
    """Log the escalated conversation into interactions_raw and add a candidate test case whose
    reference answer is the human correction. It enters a benchmark only after approval."""
    interaction_id = f"ESC-{ticket.ticket_id.removeprefix('TKT-')}"
    test_case_id = f"TC-ESC-{ticket.ticket_id.removeprefix('TKT-')}"
    with track_run(session, "support_promote", config) as (run, counts):
        if session.get(InteractionRaw, interaction_id) is None:
            session.add(
                InteractionRaw(
                    interaction_id=interaction_id,
                    conversation_id=turn.conversation_id,
                    turn_index=turn.turn_index,
                    user_query=turn.user_query,
                    ai_response=turn.ai_response,
                    model_version=turn.model_version,
                    timestamp=turn.created_at,
                    source_tag=CUSTOMER_ESCALATION_TAG,
                    intent_label=None,
                    snapshot_id=turn.snapshot_id,
                    reference_answer=ticket.corrected_answer,
                    injected_failure=None,
                    raw_json={
                        "ticket_id": ticket.ticket_id,
                        "chat_turn_id": turn.turn_id,
                        "trigger": ticket.trigger,
                        "customer_feedback": turn.customer_feedback,
                        "feedback_comment": turn.feedback_comment,
                        "ai_failure_types": turn.ai_failure_types,
                        "history": turn.history_json,
                        "tool_calls": turn.tool_calls_json,
                    },
                    run_id=run.run_id,
                )
            )
            session.flush()
        if session.get(TestCase, test_case_id) is None:
            session.add(
                TestCase(
                    test_case_id=test_case_id,
                    source_interaction_id=interaction_id,
                    input_query=turn.user_query,
                    snapshot_id=turn.snapshot_id,
                    policy_version=snapshot.policy_version,
                    expected_checklist_json=checklist,
                    reference_answer=ticket.corrected_answer,  # the human correction, never the AI reply
                    original_failure_types=list(turn.ai_failure_types or []),
                    severity=turn.ai_severity or ("high" if ticket.priority == "urgent" else "medium"),
                    cluster_id=None,
                    intent=None,
                    trust_tier="gold",
                    status="candidate",  # needs approval on the Test Dataset page before a release
                    parent_test_case_id=None,
                    provenance_json={
                        "stratum": CUSTOMER_ESCALATION_TAG,
                        "selection_reason": f"support_ticket:{ticket.trigger}",
                        "ticket_id": ticket.ticket_id,
                        "executive": ticket.executive_name,
                        "customer_feedback": turn.customer_feedback,
                        "reeval_judge_version": (ticket.reeval_json or {}).get("judge_version"),
                    },
                    run_id=run.run_id,
                )
            )
        counts.update({"ticket_id": ticket.ticket_id, "interaction_id": interaction_id, "test_case_id": test_case_id})
    return interaction_id, test_case_id


def reevaluate_ticket(
    engine: Engine,
    config: AppConfig,
    ticket_id: str,
    *,
    judge_client: BaseLLMClient | None = None,
    helper_client: BaseLLMClient | None = None,
) -> SupportTicket:
    """Judge the human-corrected answer with the frozen judge. If it passes, it is validated and
    becomes a candidate test case for future benchmark releases."""
    with get_session(engine) as session:
        ticket = session.get(SupportTicket, ticket_id)
        if ticket is None:
            raise SupportError(f"Unknown ticket {ticket_id}")
        if ticket.status != "resolved" or not ticket.corrected_answer:
            raise SupportError("Only resolved tickets with a corrected answer can be re-evaluated.")
        turn = session.get(ChatTurn, ticket.turn_id)
        snapshot = session.get(StateSnapshot, turn.snapshot_id)
        policies = load_policies(session, snapshot.policy_version)
        correction = ticket.corrected_answer

    judgement = grade_reply(
        config=config, snapshot=snapshot, message=turn.user_query, reply=correction, policies=policies,
        llm_client=judge_client,
    )
    errored = "judge_error" in (judgement.failure_types or [])
    labels = (
        None
        if errored
        else label_judgement(
            judgement, intent_label=None, hard_state_tags=list(snapshot.hard_state_tags or []),
            config=config, taxonomy=load_taxonomy(),
        )
    )
    validated = labels is not None and not labels["failure_detected"]
    checklist: dict[str, Any] | None = None
    if validated:
        helper = helper_client or build_live_client(config, provider=config.llm.provider, model=config.llm.helper_model)
        chunks = [{"policy_id": p[0], "title": p[1], "body": p[2]} for p in policies[:5]]
        checklist = generate_and_validate_checklist(
            query=turn.user_query, snapshot_json=snapshot.snapshot_json, policy_chunks=chunks, llm_client=helper
        ).model_dump()

    with get_session(engine) as session:
        ticket = session.get(SupportTicket, ticket_id)
        turn = session.get(ChatTurn, ticket.turn_id)
        ticket.reeval_at = _now()
        ticket.reeval_json = {
            "judge_model": judgement.judge_model,
            "judge_version": judgement.judge_version,
            "failure_detected": None if labels is None else labels["failure_detected"],
            "failure_types": [] if labels is None else labels["failure_types"],
            "severity": None if labels is None else labels["severity"],
            "verdicts": {
                d: getattr(judgement, f"{d}_verdict")
                for d in ("correctness", "relevance", "helpfulness", "completeness", "safety")
            },
            "explanation": judgement.explanation,
            "confidence": judgement.confidence,
        }
        ticket.reeval_status = "error" if errored else ("validated" if validated else "failed_validation")
        ticket.updated_at = _now()
        session.flush()
        if validated and checklist is not None:
            iid, tcid = _promote(session, config, ticket, turn, session.get(StateSnapshot, turn.snapshot_id), checklist)
            ticket.promoted_interaction_id, ticket.promoted_test_case_id = iid, tcid
        session.flush()
        return ticket


# ---- Analytics ------------------------------------------------------------------------------------


def support_metrics(session: Session) -> dict[str, Any]:
    """Counts behind the support dashboard. Satisfaction and AI correctness stay separate."""

    def n(stmt: Any) -> int:
        return int(session.scalar(stmt) or 0)

    total = n(select(func.count(ChatTurn.turn_id)))
    satisfied = n(select(func.count()).where(ChatTurn.customer_feedback == "satisfied"))
    dissatisfied = n(select(func.count()).where(ChatTurn.customer_feedback == "dissatisfied"))
    evaluated = n(select(func.count()).where(ChatTurn.eval_status == "done"))
    ai_failures = n(select(func.count()).where(ChatTurn.ai_failure_detected.is_(True)))
    tickets = session.execute(
        select(SupportTicket.trigger, SupportTicket.status, SupportTicket.reeval_status, func.count())
        .group_by(SupportTicket.trigger, SupportTicket.status, SupportTicket.reeval_status)
    ).all()

    def t(pred: Any) -> int:
        return sum(c for trig, st, re, c in tickets if pred(trig, st, re))

    matrix = session.execute(
        select(ChatTurn.customer_feedback, ChatTurn.ai_failure_detected, func.count()).group_by(
            ChatTurn.customer_feedback, ChatTurn.ai_failure_detected
        )
    ).all()
    return {
        "total_interactions": total,
        "satisfied": satisfied,
        "dissatisfied": dissatisfied,
        "no_feedback": total - satisfied - dissatisfied,
        "csat": satisfied / (satisfied + dissatisfied) if (satisfied + dissatisfied) else None,
        "ai_evaluated": evaluated,
        "ai_failures": ai_failures,
        "customer_escalations": t(lambda trig, st, re: "customer" in trig),
        "ai_escalations": t(lambda trig, st, re: "ai_judge" in trig),
        "tickets_total": t(lambda *_: True),
        "tickets_pending": t(lambda trig, st, re: st == "pending"),
        "tickets_resolved": t(lambda trig, st, re: st == "resolved"),
        "sent_for_reevaluation": t(lambda trig, st, re: re != "not_started"),
        "reeval_queued": t(lambda trig, st, re: re == "queued"),
        "reeval_validated": t(lambda trig, st, re: re == "validated"),
        "reeval_failed": t(lambda trig, st, re: re in ("failed_validation", "error")),
        "feedback_vs_ai": [
            {"feedback": fb or "none", "ai_failure": ai, "count": c} for fb, ai, c in matrix
        ],
    }


def ensure_support_tables(engine: Engine) -> None:
    """Create chat_turns / support_tickets if this database predates them (same as migration 0003),
    so the dashboard works on a database built before the feedback feature without a manual step."""
    from edg.db.models import Base

    tables = [Base.metadata.tables[name] for name in ("chat_turns", "support_tickets")]
    Base.metadata.create_all(bind=engine, tables=tables, checkfirst=True)
