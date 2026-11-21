"""Checklist generation (must_do / must_not_do) with deterministic snapshot validation."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from edg.llm.client import BaseLLMClient, LLMError
from edg.logging import get_logger
from edg.world.hard_states import (
    BOUNDARY_RETURN_WINDOW,
    CANCEL_AFTER_SHIPPING,
    DELIVERED_DISPUTED,
    DUPLICATE_CHARGE,
    PARTIAL_SHIPMENT,
    PAST_RETURN_WINDOW,
    REFUND_APPROVED_NOT_ISSUED,
    WRONG_ITEM_DELIVERED,
)

logger = get_logger(__name__)

_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_AMOUNT_RE = re.compile(r"\$(\d+(?:\.\d{1,2})?)")
_ORDER_RE = re.compile(r"\b(ORD-\d+)\b")


class ExpectedChecklist(BaseModel):
    must_do: list[str] = Field(default_factory=list)
    must_not_do: list[str] = Field(default_factory=list)


def _deterministic_checklist(
    *,
    snapshot_json: dict[str, Any],
    intent: str | None,
) -> ExpectedChecklist:
    order = snapshot_json["order"]
    order_id = order["order_id"]
    status = order["status"]
    total = float(order["total_amount"])
    shipments = snapshot_json.get("shipments") or []
    payments = snapshot_json.get("payments") or []
    returns = snapshot_json.get("refunds_returns") or []
    tags = set(snapshot_json.get("hard_state_tags") or [])

    must_do: list[str] = [
        f"must state order {order_id} status is '{status}'",
    ]
    must_not_do: list[str] = [
        "must not invent tracking numbers, dates, or dollar amounts absent from the snapshot",
    ]

    if shipments:
        s0 = shipments[0]
        if s0.get("actual_delivery"):
            must_do.append(f"must state package was delivered on {s0['actual_delivery'][:10]}")
        else:
            must_do.append(
                f"must state package has not been delivered and expected delivery is {s0['expected_delivery'][:10]}"
            )

    if DELIVERED_DISPUTED in tags:
        must_do.append("must explain the 3 to 5 business day carrier investigation process")
        must_not_do.append("must not promise an immediate refund before the carrier investigation completes")

    if DUPLICATE_CHARGE in tags:
        dup_pays = [p for p in payments if p.get("is_duplicate")]
        amt = dup_pays[0]["amount"] if dup_pays else total
        must_do.append(f"must confirm duplicate captured charge of ${amt:.2f} will be refunded in 3 to 5 business days")

    if PAST_RETURN_WINDOW in tags:
        must_do.append("must explain the return is past the 30-day return window and ineligible for a standard refund")
        must_not_do.append("must not approve a standard refund past the 30-day return window")

    if BOUNDARY_RETURN_WINDOW in tags:
        must_do.append("must confirm the return requested on day 29 or 30 is eligible under the 30-day policy")

    if PARTIAL_SHIPMENT in tags:
        must_do.append("must report that the order shipped in separate packages and give status for each shipment")

    if CANCEL_AFTER_SHIPPING in tags:
        must_do.append("must state the order cannot be cancelled in transit after shipping")
        must_not_do.append("must not claim that an already-shipped order has been cancelled")

    if REFUND_APPROVED_NOT_ISSUED in tags and returns:
        must_do.append(
            f"must state refund of ${returns[0]['refund_amount']:.2f} is approved and in the 2-business-day issuance queue"
        )

    if WRONG_ITEM_DELIVERED in tags:
        must_do.append("must offer a free prepaid return shipping label and full refund or free replacement")

    if intent in {"check_invoice", "get_invoice", "payment_issue"}:
        must_do.append(f"must reference accurate order total ${total:.2f}")

    return ExpectedChecklist(must_do=must_do, must_not_do=must_not_do)


def validate_checklist_against_snapshot(
    checklist: ExpectedChecklist,
    snapshot_json: dict[str, Any],
) -> ExpectedChecklist:
    """Validate that every factual order ID, date, and amount in must_do matches the snapshot."""
    order = snapshot_json.get("order") or {}
    valid_order_id = str(order.get("order_id", ""))

    valid_dates: set[str] = set()
    if order.get("placed_at"):
        valid_dates.add(str(order["placed_at"])[:10])
    for s in snapshot_json.get("shipments") or []:
        for key in ("expected_delivery", "carrier_scan_at", "actual_delivery"):
            if s.get(key):
                valid_dates.add(str(s[key])[:10])
    for p in snapshot_json.get("payments") or []:
        if p.get("charged_at"):
            valid_dates.add(str(p["charged_at"])[:10])
    for r in snapshot_json.get("refunds_returns") or []:
        for key in ("requested_at", "refund_issued_at"):
            if r.get(key):
                valid_dates.add(str(r[key])[:10])

    valid_amounts: list[float] = []
    if order.get("total_amount") is not None:
        valid_amounts.append(float(order["total_amount"]))
    for p in snapshot_json.get("payments") or []:
        if p.get("amount") is not None:
            valid_amounts.append(float(p["amount"]))
    for r in snapshot_json.get("refunds_returns") or []:
        if r.get("refund_amount") is not None:
            valid_amounts.append(float(r["refund_amount"]))
    valid_amounts.extend([50.0, 200.0])  # Policy thresholds

    validated_must_do: list[str] = []
    for item in checklist.must_do:
        bad = False
        for m in _ORDER_RE.finditer(item):
            if valid_order_id and m.group(1) != valid_order_id:
                bad = True
        for m in _DATE_RE.finditer(item):
            if m.group(1) not in valid_dates:
                bad = True
        for m in _AMOUNT_RE.finditer(item):
            val = float(m.group(1))
            if not any(abs(val - a) <= 0.01 for a in valid_amounts):
                bad = True
        if not bad and item.strip():
            validated_must_do.append(item.strip())

    if not validated_must_do and valid_order_id:
        validated_must_do.append(f"must state order {valid_order_id} status is '{order.get('status')}'")

    validated_must_not_do = [i.strip() for i in checklist.must_not_do if i.strip()]
    return ExpectedChecklist(must_do=validated_must_do, must_not_do=validated_must_not_do)


def generate_and_validate_checklist(
    *,
    query: str,
    snapshot_json: dict[str, Any],
    policy_chunks: list[dict[str, Any]] | None = None,
    intent: str | None = None,
    llm_client: BaseLLMClient | None = None,
) -> ExpectedChecklist:
    """Draft must_do / must_not_do checklist and validate all facts against the snapshot."""
    det = _deterministic_checklist(snapshot_json=snapshot_json, intent=intent)
    use_live_llm = llm_client is not None and llm_client.provider != "fake"

    if use_live_llm and llm_client is not None:
        prompt = (
            "Draft a concise evaluation checklist (`must_do` and `must_not_do` lists of 2-4 items each) "
            "for a customer-support assistant answering this query.\n"
            f"Query: {query}\nIntent: {intent}\n"
            f"Snapshot JSON: {json.dumps(snapshot_json, sort_keys=True)}\n"
            f"Policies: {json.dumps(policy_chunks or [], sort_keys=True)}\n"
            'Return ONLY valid JSON: {"must_do": [...], "must_not_do": [...]}'
        )
        try:
            parsed, _resp = llm_client.generate_json(
                prompt=prompt,
                schema_model=ExpectedChecklist,
                temperature=0.0,
            )
            return validate_checklist_against_snapshot(parsed, snapshot_json)
        except LLMError as exc:
            logger.warning("checklist_llm_fallback", extra={"error": str(exc)})

    return validate_checklist_against_snapshot(det, snapshot_json)
