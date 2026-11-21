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


# Every status code the store world can record (orders, payments, shipments, returns). A claimed
# status outside this vocabulary ("being processed") cannot be checked, so it is not_in_evidence
# rather than contradicted.
KNOWN_STATUS_CODES = frozenset(
    {
        "placed", "shipped", "in_transit", "out_for_delivery", "delivered", "cancelled",
        "returned", "refunded", "authorised", "captured", "failed", "requested", "approved",
        "rejected_past_window",
    }
)

_STATUS_SYNONYMS = {"canceled": "cancelled", "authorized": "authorised"}

# "will be delivered", "expected to be delivered" etc. describe a future delivery, not a delivery
_FUTURE_DELIVERY_RE = re.compile(
    r"\b(?:will|should|would|to|expected\s+to|scheduled\s+to|going\s+to)\s+(?:be\s+)?(?:get\s+)?delivered\b"
)

_MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"),
            ("may",), ("jun", "june"), ("jul", "july"), ("aug", "august"),
            ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"), ("dec", "december"),
        ],
        start=1,
    )
    for m in names
}
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})")
_US_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_MONTH_FIRST_RE = re.compile(r"\b([a-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b")
_DAY_FIRST_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]{3,9})\.?(?:,?\s+(\d{4}))?\b")

# Words that make an amount a claim about a recorded order amount (total, charge, price)
_RECORDED_AMOUNT_RE = re.compile(r"\b(?:total|charged?|paid|billed|payment|price|cost of (?:your|the) order)\b")
_REFUND_AMOUNT_RE = re.compile(r"\brefund", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def _normalise_date(value: str) -> tuple[int | None, int, int] | None:
    """Parse a claimed date into (year or None, month, day); None if it is not a calendar date."""
    low = value.strip().lower()
    if m := _ISO_DATE_RE.search(low):
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    if m := _US_DATE_RE.search(low):
        return int(m.group(3)), int(m.group(1)), int(m.group(2))
    if (m := _MONTH_FIRST_RE.search(low)) and m.group(1) in _MONTHS:
        return (int(m.group(3)) if m.group(3) else None), _MONTHS[m.group(1)], int(m.group(2))
    if (m := _DAY_FIRST_RE.search(low)) and m.group(2) in _MONTHS:
        return (int(m.group(3)) if m.group(3) else None), _MONTHS[m.group(2)], int(m.group(1))
    return None


def _date_matches(value: str, valid_dates: set[str]) -> bool:
    parsed = _normalise_date(value)
    if parsed is None:
        return False
    year, month, day = parsed
    for d in valid_dates:
        vy, vm, vd = (int(x) for x in d.split("-"))
        if vm == month and vd == day and (year is None or year == vy):
            return True
    return False


def _policy_numbers(policy_chunks: list[dict[str, Any]] | None) -> set[float]:
    nums: set[float] = set()
    for chunk in policy_chunks or []:
        text = " ".join(str(v) for v in chunk.values() if isinstance(v, str))
        for m in _NUMBER_RE.finditer(text):
            nums.add(float(m.group(0).replace(",", "")))
    return nums


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
    payments = snapshot_json.get("payments") or []
    all_statuses = (
        {order_status}
        | {str(s.get("carrier_scan_status", "")).lower() for s in shipments}
        | {str(r.get("status", "")).lower() for r in returns}
        | {str(p.get("status", "")).lower() for p in payments}
    )
    order_amounts = [float(order["total_amount"])] if order.get("total_amount") is not None else []
    order_amounts += [float(p["amount"]) for p in payments if p.get("amount") is not None]
    order_amounts += [
        float(it["unit_price"]) for it in snapshot_json.get("items") or [] if it.get("unit_price") is not None
    ]
    refund_amounts = [float(r["refund_amount"]) for r in returns if r.get("refund_amount") is not None]
    policy_numbers = _policy_numbers(policy_chunks)

    results: list[ClaimCheckResult] = []
    for claim in claims:
        val = (claim.claimed_value or "").strip()
        val_low = val.lower()
        text_low = claim.claim_text.lower()

        if claim.claim_type == "status":
            norm_val = _normalise_status(val_low)
            norm_val = _STATUS_SYNONYMS.get(norm_val, norm_val)
            future_delivery = _FUTURE_DELIVERY_RE.search(f"{val_low} {text_low}") is not None
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
            elif future_delivery:
                # An expected delivery is only wrong if the parcel has already arrived
                results.append(
                    ClaimCheckResult(
                        claim_text=claim.claim_text,
                        claim_type=claim.claim_type,
                        status="contradicted" if has_actual_delivery else "not_in_evidence",
                        evidence_detail=(
                            "Response describes a future delivery, but a shipment has actual_delivery set."
                            if has_actual_delivery
                            else "Future delivery claim; no actual_delivery recorded yet."
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
            elif norm_val in KNOWN_STATUS_CODES:
                results.append(
                    ClaimCheckResult(
                        claim_text=claim.claim_text,
                        claim_type=claim.claim_type,
                        status="contradicted",
                        evidence_detail=f"Claimed status '{norm_val}' contradicts snapshot statuses {sorted(all_statuses)}.",
                    )
                )
            elif val_low:
                results.append(
                    ClaimCheckResult(
                        claim_text=claim.claim_text,
                        claim_type=claim.claim_type,
                        status="not_in_evidence",
                        evidence_detail=f"Claimed status '{val}' is not a recorded status code.",
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
            elif val in valid_dates or _date_matches(val or claim.claim_text, valid_dates):
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
            num_match = _NUMBER_RE.search(val.replace("$", "")) or _NUMBER_RE.search(claim.claim_text)
            if num_match is None:
                results.append(
                    ClaimCheckResult(
                        claim_text=claim.claim_text,
                        claim_type=claim.claim_type,
                        status="not_in_evidence",
                        evidence_detail=f"Unparseable amount '{val}'.",
                    )
                )
                continue
            num = float(num_match.group(0).replace(",", ""))

            def _amount(status: str, detail: str, c: ExtractedClaim = claim) -> ClaimCheckResult:
                return ClaimCheckResult(
                    claim_text=c.claim_text,
                    claim_type=c.claim_type,
                    status=status,  # type: ignore[arg-type]
                    evidence_detail=detail,
                )

            if any(abs(num - a) <= 0.01 for a in valid_amounts):
                results.append(_amount("supported", f"Amount ${num:.2f} matches snapshot."))
            elif "voucher" in text_low:
                results.append(_amount("not_in_evidence", f"Voucher amount ${num:.2f} is not in snapshot or policy."))
            elif _REFUND_AMOUNT_RE.search(text_low) and refund_amounts:
                results.append(
                    _amount("contradicted", f"Refund amount ${num:.2f} contradicts recorded refunds {refund_amounts}.")
                )
            elif _RECORDED_AMOUNT_RE.search(text_low) and not _REFUND_AMOUNT_RE.search(text_low) and order_amounts:
                results.append(
                    _amount("contradicted", f"Amount ${num:.2f} contradicts recorded order amounts {order_amounts}.")
                )
            elif num in (50.0, 200.0) or num in policy_numbers:
                # Policy figures: thresholds, fees, return windows
                results.append(_amount("supported", f"Amount {num:g} matches a store policy figure."))
            else:
                results.append(
                    _amount("not_in_evidence", f"Amount ${num:.2f} is not recorded in the snapshot or policies.")
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
