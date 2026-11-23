"""Customer feedback, support escalation, human corrections and their re-evaluation."""

from edg.support.service import (
    CUSTOMER_ESCALATION_TAG,
    SupportError,
    ai_escalation_reason,
    apply_ai_evaluation,
    evaluate_turn,
    record_feedback,
    record_turn,
    reevaluate_ticket,
    resolve_ticket,
    support_metrics,
)

__all__ = [
    "CUSTOMER_ESCALATION_TAG",
    "SupportError",
    "ai_escalation_reason",
    "apply_ai_evaluation",
    "evaluate_turn",
    "record_feedback",
    "record_turn",
    "reevaluate_ticket",
    "resolve_ticket",
    "support_metrics",
]
