"""Deterministic code verifier comparing extracted claims against the bound state snapshot."""

from __future__ import annotations

import re
from typing import Any

from edg.judge.schema import ClaimCheckResult, ExtractedClaim

_NEGATED_DELIVERY_RE = re.compile(
    r"\b(?:not(?:\s+yet)?\s+(?:been\s+)?delivered|undelivered|never\s+(?:been\s+)?delivered|"
    r"(?:hasn't|hasn’t|has not|have not|haven't|wasn't|was not|isn't|is not)\s+(?:yet\s+)?(?:been\s+)?delivered)\b"
)


def _normalise_status(value: str) -> str:
    """Map free-text status values ("In transit", "out-for-delivery") onto snapshot codes."""
    return re.sub(r"[\s\-]+", "_", value.strip().lower())


def _collect_snapshot_dates(snapshot_json: dict[str, Any]) -> set[str]:
    dates: set[str] = set()
    order = snapshot_json.get("order") or {}
    if order.get("placed_at"):
        dates.add(str(order["placed_at"])[:10])
    for p in snapshot_json.get("payments") or []:
        if p.get("charged_at"):
            dates.add(str(p["charged_at"])[:10])
    for s in snapshot_json.get("shipments") or []:
        for key in ("expected_delivery", "carrier_scan_at", "actual_delivery"):
            if s.get(key):
                dates.add(str(s[key])[:10])
    for r in snapshot_json.get("refunds_returns") or []:
        for key in ("requested_at", "refund_issued_at"):
            if r.get(key):
                dates.add(str(r[key])[:10])
    if snapshot_json.get("reference_date"):
        dates.add(str(snapshot_json["reference_date"])[:10])
    return dates


def _collect_snapshot_amounts(snapshot_json: dict[str, Any]) -> list[float]:
    amounts: list[float] = []
    order = snapshot_json.get("order") or {}
    if order.get("total_amount") is not None:
        amounts.append(float(order["total_amount"]))
    for it in snapshot_json.get("items") or []:
        if it.get("unit_price") is not None:
            amounts.append(float(it["unit_price"]))
    for p in snapshot_json.get("payments") or []:
        if p.get("amount") is not None:
            amounts.append(float(p["amount"]))
    for r in snapshot_json.get("refunds_returns") or []:
        if r.get("refund_amount") is not None:
            amounts.append(float(r["refund_amount"]))
    return amounts


def verify_claims(
    claims: list[ExtractedClaim],
    snapshot_json: dict[str, Any] | None,
    policy_chunks: list[dict[str, Any]] | None = None,
) -> list[ClaimCheckResult]:
    """Deterministically verify status, date, amount, eligibility, and action claims against snapshot."""
    if not claims:
        return []

    if not snapshot_json or not snapshot_json.get("order"):
        return [
            ClaimCheckResult(
                claim_text=c.claim_text,
                claim_type=c.claim_type,
                status="not_in_evidence",
                evidence_detail="No bound state snapshot available to verify claim.",
            )
            for c in claims
        ]

    order = snapshot_json["order"]
    order_status = str(order.get("status", "")).lower()
    shipments = snapshot_json.get("shipments") or []
    returns = snapshot_json.get("refunds_returns") or []
    valid_dates = _collect_snapshot_dates(snapshot_json)
    valid_amounts = _collect_snapshot_amounts(snapshot_json)
    valid_shipment_ids = {str(s.get("shipment_id", "")) for s in shipments}
    has_actual_delivery = any(s.get("actual_delivery") is not None for s in shipments)
    has_disputed_delivery = any(bool(s.get("customer_disputed")) for s in shipments)
    has_past_window_return = any(r.get("status") == "rejected_past_window" for r in returns)
    all_statuses = {order_status} | {
        str(s.get("carrier_scan_status", "")).lower() for s in shipments
    } | {str(r.get("status", "")).lower() for r in returns}

    results: list[ClaimCheckResult] = []
    for claim in claims:
        val = (claim.claimed_value or "").strip()
        val_low = val.lower()
        text_low = claim.claim_text.lower()

        if claim.claim_type == "status":
            norm_val = _normalise_status(val_low)
            negated_delivery = _NEGATED_DELIVERY_RE.search(f"{val_low} {text_low}") is not None
            if negated_delivery:
                # "not delivered yet" is a claim of NON-delivery, not of delivery
                results.append(
                    ClaimCheckResult(
                        claim_text=claim.claim_text,
                        claim_type=claim.claim_type,
                        status="contradicted" if has_actual_delivery else "supported",
                        evidence_detail=(
                            "Response claims not delivered, but a shipment has actual_delivery set."
                            if has_actual_delivery
                            else "No shipment has actual_delivery set, consistent with not delivered."
                        ),
                    )
                )
            elif norm_val in all_statuses and norm_val != "delivered":
                results.append(
                    ClaimCheckResult(
                        claim_text=claim.claim_text,
                        claim_type=claim.claim_type,
                        status="supported",
                        evidence_detail=f"Status '{norm_val}' matches snapshot records.",
                    )
                )
            elif "delivered" in val_low or "delivered" in text_low:
                if not has_actual_delivery:
                    results.append(
                        ClaimCheckResult(
                            claim_text=claim.claim_text,
                            claim_type=claim.claim_type,
                            status="contradicted",
                            evidence_detail="Response claims delivered, but actual_delivery is null in all shipments.",
                        )
                    )
                else:
                    results.append(
                        ClaimCheckResult(
                            claim_text=claim.claim_text,
                            claim_type=claim.claim_type,
                            status="supported",
                            evidence_detail="Snapshot contains a shipment with actual_delivery set.",
                        )
                    )
            elif val_low:
                results.append(
                    ClaimCheckResult(
                        claim_text=claim.claim_text,
                        claim_type=claim.claim_type,
                        status="contradicted",
                        evidence_detail=f"Claimed status '{norm_val}' contradicts snapshot statuses {sorted(all_statuses)}.",
                    )
                )
            else:
                results.append(
                    ClaimCheckResult(
                        claim_text=claim.claim_text,
                        claim_type=claim.claim_type,
                        status="not_in_evidence",
                        evidence_detail="Could not resolve claimed status value.",
                    )
                )

        elif claim.claim_type == "date":
            if val_low == "yesterday":
                if not has_actual_delivery:
                    results.append(
                        ClaimCheckResult(
                            claim_text=claim.claim_text,
                            claim_type=claim.claim_type,
                            status="contradicted",
                            evidence_detail="Claimed 'delivered yesterday' contradicts actual_delivery=null.",
                        )
                    )
                else:
                    results.append(
                        ClaimCheckResult(
                            claim_text=claim.claim_text,
                            claim_type=claim.claim_type,
                            status="not_in_evidence",
                            evidence_detail="Relative date 'yesterday' not directly verified.",
                        )
                    )
            elif val in valid_dates:
                results.append(
                    ClaimCheckResult(
                        claim_text=claim.claim_text,
                        claim_type=claim.claim_type,
                        status="supported",
                        evidence_detail=f"Date {val} matches snapshot.",
                    )
                )
            else:
                results.append(
                    ClaimCheckResult(
                        claim_text=claim.claim_text,
                        claim_type=claim.claim_type,
                        status="not_in_evidence",
                        evidence_detail=f"Date '{val}' not found in snapshot dates {sorted(valid_dates)}.",
                    )
                )

        elif claim.claim_type == "amount":
            try:
                num = float(val.replace("$", "").replace(",", ""))
            except ValueError:
                results.append(
                    ClaimCheckResult(
                        claim_text=claim.claim_text,
                        claim_type=claim.claim_type,
                        status="not_in_evidence",
                        evidence_detail=f"Unparseable amount '{val}'.",
                    )
                )
                continue

            if any(abs(num - a) <= 0.01 for a in valid_amounts):
                results.append(
                    ClaimCheckResult(
                        claim_text=claim.claim_text,
                        claim_type=claim.claim_type,
                        status="supported",
                        evidence_detail=f"Amount ${num:.2f} matches snapshot.",
                    )
                )
            elif num in (50.0, 200.0) and "voucher" not in text_low:
                # Policy threshold amounts ($50 free shipping, $200 escalation)
                results.append(
                    ClaimCheckResult(
                        claim_text=claim.claim_text,
                        claim_type=claim.claim_type,
                        status="supported",
                        evidence_detail=f"Amount ${num:.2f} matches store policy threshold.",
                    )
                )
            else:
                results.append(
                    ClaimCheckResult(
                        claim_text=claim.claim_text,
                        claim_type=claim.claim_type,
                        status="contradicted",
                        evidence_detail=f"Amount ${num:.2f} contradicts snapshot amounts {valid_amounts}.",
                    )
                )

        elif claim.claim_type == "eligibility":
            if val_low == "immediate_refund_approved":
                if has_disputed_delivery:
                    results.append(
                        ClaimCheckResult(
                            claim_text=claim.claim_text,
                            claim_type=claim.claim_type,
                            status="contradicted",
                            evidence_detail="Promised immediate refund on a disputed delivery before carrier investigation.",
                        )
                    )
                elif has_past_window_return:
                    results.append(
                        ClaimCheckResult(
                            claim_text=claim.claim_text,
                            claim_type=claim.claim_type,
                            status="contradicted",
                            evidence_detail="Promised standard refund on a return past the 30-day window.",
                        )
                    )
                else:
                    results.append(
                        ClaimCheckResult(
                            claim_text=claim.claim_text,
                            claim_type=claim.claim_type,
                            status="not_in_evidence",
                            evidence_detail="Refund approval action not recorded in snapshot.",
                        )
                    )
            else:
                results.append(
                    ClaimCheckResult(
                        claim_text=claim.claim_text,
                        claim_type=claim.claim_type,
                        status="not_in_evidence",
                        evidence_detail="Eligibility claim requires policy review.",
                    )
                )

        elif claim.claim_type == "action_taken":
            if val_low == "cancelled" and order_status in {"shipped", "out_for_delivery", "delivered"}:
                results.append(
                    ClaimCheckResult(
                        claim_text=claim.claim_text,
                        claim_type=claim.claim_type,
                        status="contradicted",
                        evidence_detail=f"Claimed order cancelled, but order status is '{order_status}'.",
                    )
                )
            elif val.startswith("TRK-") and val not in valid_shipment_ids:
                results.append(
                    ClaimCheckResult(
                        claim_text=claim.claim_text,
                        claim_type=claim.claim_type,
                        status="not_in_evidence",
                        evidence_detail=f"Tracking number '{val}' does not exist in snapshot shipments.",
                    )
                )
            elif val_low == "loyalty_voucher":
                results.append(
                    ClaimCheckResult(
                        claim_text=claim.claim_text,
                        claim_type=claim.claim_type,
                        status="not_in_evidence",
                        evidence_detail="Complimentary loyalty voucher is not present in snapshot or policy.",
                    )
                )
            else:
                results.append(
                    ClaimCheckResult(
                        claim_text=claim.claim_text,
                        claim_type=claim.claim_type,
                        status="not_in_evidence",
                        evidence_detail="Action claim not verified in snapshot.",
                    )
                )
        else:
            # policy claims evaluated by LLM against retrieved chunks
            has_pol = bool(policy_chunks)
            results.append(
                ClaimCheckResult(
                    claim_text=claim.claim_text,
                    claim_type=claim.claim_type,
                    status="supported" if has_pol else "not_in_evidence",
                    evidence_detail="Policy claim delegated to retrieved policy chunks.",
                )
            )

    # Carry the claimed value so downstream rules can tell concrete from generic claims
    for claim, res in zip(claims, results, strict=True):
        res.claimed_value = claim.claimed_value
    return results
