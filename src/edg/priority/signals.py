"""Six normalized [0, 1] priority signals for edge-case and anomaly ranking."""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any

import numpy as np

from edg.world.hard_states import (
    BOUNDARY_RETURN_WINDOW,
    CANCEL_AFTER_SHIPPING,
    DELIVERED_DISPUTED,
    DUPLICATE_CHARGE,
    PARTIAL_SHIPMENT,
    PAST_RETURN_WINDOW,
    WRONG_ITEM_DELIVERED,
)

_INTENT_CUE_PATTERNS: dict[str, re.Pattern[str]] = {
    "refund": re.compile(r"\b(refund|money back|reimburse|reembolso)\b", re.IGNORECASE),
    "cancel": re.compile(r"\b(cancel|cancela|stop order|undo)\b", re.IGNORECASE),
    "track": re.compile(r"\b(track|where is|delivered|package|shipment|arrive)\b", re.IGNORECASE),
    "address": re.compile(r"\b(address|shipping address|redirect)\b", re.IGNORECASE),
    "billing": re.compile(r"\b(charged twice|duplicate|invoice|billing|payment)\b", re.IGNORECASE),
    "return": re.compile(r"\b(return|send back|devolver|wrong item|replacement)\b", re.IGNORECASE),
}
_MULTI_JOIN_RE = re.compile(r"\b(and also|also|plus|as well as|in addition)\b", re.IGNORECASE)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def compute_novelty_signal(cluster_id: int, glosh_score: float, p90_glosh: float) -> float:
    """1.0 if noise (-1) or GLOSH >= 90th percentile, else scaled in [0, 0.99]."""
    if cluster_id == -1:
        return 1.0
    if p90_glosh > 0 and glosh_score >= p90_glosh:
        return 1.0
    if p90_glosh <= 0:
        return _clip01(glosh_score)
    return _clip01(0.99 * (glosh_score / p90_glosh))


def compute_small_cluster_signal(cluster_id: int, cluster_size: int, max_inv_log: float) -> float:
    """Inverse of log(1 + cluster_size) normalized to [0, 1]; noise gets 1.0."""
    if cluster_id == -1 or cluster_size <= 1:
        return 1.0
    raw = 1.0 / math.log1p(cluster_size)
    if max_inv_log <= 0:
        return _clip01(raw)
    return _clip01(raw / max_inv_log)


def compute_multi_intent_signal(query: str) -> float:
    """Heuristic count of distinct request cues in the query, scaled to [0, 1]."""
    matched_groups = sum(1 for pat in _INTENT_CUE_PATTERNS.values() if pat.search(query))
    has_join = bool(_MULTI_JOIN_RE.search(query))
    q_marks = query.count("?")

    score = 0.0
    if matched_groups >= 3:
        score = 1.0
    elif matched_groups == 2:
        score = 0.75 if (has_join or q_marks >= 2) else 0.55
    elif has_join or q_marks >= 2:
        score = 0.40
    return _clip01(score)


def compute_state_conflict_signal(
    query: str,
    snapshot_json: dict[str, Any],
    hard_state_tags: list[str],
) -> float:
    """Detect disagreement between customer claim and bound state snapshot, scaled to [0, 1]."""
    q_low = query.lower()
    tags = set(hard_state_tags or [])
    order = snapshot_json.get("order") or {}
    shipments = snapshot_json.get("shipments") or []
    payments = snapshot_json.get("payments") or []

    score = 0.0
    if tags & {DELIVERED_DISPUTED, CANCEL_AFTER_SHIPPING, DUPLICATE_CHARGE, WRONG_ITEM_DELIVERED}:
        score = max(score, 0.85)
    if PARTIAL_SHIPMENT in tags:
        score = max(score, 0.70)

    # Explicit query-vs-snapshot contradiction checks
    has_delivered = any(s.get("actual_delivery") for s in shipments)
    if has_delivered and any(
        phrase in q_low for phrase in ("never received", "didn't get", "nothing arrived", "not delivered", "missing")
    ):
        score = 1.0

    if order.get("status") in {"shipped", "out_for_delivery"} and any(
        phrase in q_low for phrase in ("cancel", "cancela", "stop order")
    ):
        score = max(score, 0.95)

    if any(p.get("is_duplicate") for p in payments) and any(
        phrase in q_low for phrase in ("twice", "duplicate", "double", "deux")
    ):
        score = 1.0

    return _clip01(score)


def compute_policy_boundary_signal(
    snapshot_json: dict[str, Any],
    hard_state_tags: list[str],
) -> float:
    """Detect proximity (within 3 days or 20% of dollar threshold) to a store policy limit."""
    tags = set(hard_state_tags or [])
    score = 0.0

    if BOUNDARY_RETURN_WINDOW in tags:
        return 1.0
    if PAST_RETURN_WINDOW in tags:
        score = max(score, 0.75)

    shipments = snapshot_json.get("shipments") or []
    returns = snapshot_json.get("refunds_returns") or []
    order = snapshot_json.get("order") or {}

    delivered_dts = [
        datetime.fromisoformat(s["actual_delivery"])
        for s in shipments
        if s.get("actual_delivery")
    ]
    if delivered_dts and returns:
        earliest_del = min(delivered_dts)
        for ret in returns:
            if ret.get("requested_at"):
                req_dt = datetime.fromisoformat(ret["requested_at"])
                delta_days = (req_dt - earliest_del).days
                dist_to_30 = abs(delta_days - 30)
                if dist_to_30 <= 3:
                    score = max(score, 1.0 - (dist_to_30 / 10.0))

    # Disputed delivery near the $200 escalation boundary
    total = float(order.get("total_amount") or 0.0)
    if any(s.get("customer_disputed") for s in shipments) and 160.0 <= total <= 240.0:
        score = max(score, 0.85)

    return _clip01(score)


def compute_response_drift_signal(
    response_vec: np.ndarray,
    cluster_mean_response_vec: np.ndarray | None,
) -> float:
    """Cosine distance of a response embedding from its cluster's mean response, scaled to [0, 1]."""
    if cluster_mean_response_vec is None:
        return 0.5
    r_norm = float(np.linalg.norm(response_vec))
    m_norm = float(np.linalg.norm(cluster_mean_response_vec))
    if r_norm == 0.0 or m_norm == 0.0:
        return 0.5
    cos_sim = float(np.dot(response_vec / r_norm, cluster_mean_response_vec / m_norm))
    # Cosine distance in [0, 2] mapped cleanly to [0, 1]
    return _clip01((1.0 - cos_sim) / 2.0)
