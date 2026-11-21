"""Atomic factual claim extraction from assistant responses (LLM + deterministic extractor)."""

from __future__ import annotations

import re

from edg.judge.schema import ClaimExtractionOutput, ExtractedClaim
from edg.llm.client import BaseLLMClient, LLMError
from edg.logging import get_logger

logger = get_logger(__name__)

_STATUS_QUOTED_RE = re.compile(
    r"status\s+['\"]([a-z_]+)['\"]",
    re.IGNORECASE,
)
_DELIVERED_RE = re.compile(
    r"\b(?:was\s+delivered|delivered\s+yesterday|delivered\s+on\s+\d{4}-\d{2}-\d{2}|has\s+been\s+delivered)\b",
    re.IGNORECASE,
)
_CANCELLED_RE = re.compile(
    r"\b(?:has\s+been\s+(?:permanently\s+)?cancelled|submitted\s+a\s+cancellation)\b",
    re.IGNORECASE,
)
_DATE_ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_AMOUNT_RE = re.compile(r"\$(\d+(?:\.\d{1,2})?)")
_FAKE_TRACKING_RE = re.compile(r"\b(TRK-[A-Z0-9-]+)\b")
_IMMEDIATE_REFUND_RE = re.compile(
    r"\b(?:approve(?:d)?\s+an?\s+immediate\s+(?:full\s+)?refund|process\s+a\s+standard\s+refund)\b",
    re.IGNORECASE,
)
_VOUCHER_RE = re.compile(r"\b(complimentary\s+\$\d+\s+loyalty\s+voucher)\b", re.IGNORECASE)


def extract_claims_deterministic(response_text: str) -> list[ExtractedClaim]:
    """Deterministically extract atomic status, date, amount, eligibility, and action claims."""
    claims: list[ExtractedClaim] = []
    seen: set[tuple[str, str]] = set()

    def _add(c_type: str, text: str, val: str | None = None) -> None:
        key = (c_type, text.lower())
        if key not in seen:
            seen.add(key)
            claims.append(
                ExtractedClaim(
                    claim_text=text,
                    claim_type=c_type,  # type: ignore[arg-type]
                    claimed_value=val,
                )
            )

    for m in _STATUS_QUOTED_RE.finditer(response_text):
        _add("status", f"order status is '{m.group(1).lower()}'", m.group(1).lower())

    for m in _DELIVERED_RE.finditer(response_text):
        val = "delivered_yesterday" if "yesterday" in m.group(0).lower() else "delivered"
        _add("status", m.group(0), val)

    for m in _CANCELLED_RE.finditer(response_text):
        _add("action_taken", m.group(0), "cancelled")

    for m in _DATE_ISO_RE.finditer(response_text):
        _add("date", f"date mentioned: {m.group(1)}", m.group(1))

    if "yesterday" in response_text.lower() and not any(c.claim_type == "date" for c in claims):
        _add("date", "delivered yesterday", "yesterday")

    for m in _AMOUNT_RE.finditer(response_text):
        _add("amount", f"amount ${m.group(1)}", m.group(1))

    for m in _FAKE_TRACKING_RE.finditer(response_text):
        _add("action_taken", f"tracking number {m.group(1)}", m.group(1))

    for m in _VOUCHER_RE.finditer(response_text):
        _add("action_taken", m.group(1), "loyalty_voucher")

    for m in _IMMEDIATE_REFUND_RE.finditer(response_text):
        _add("eligibility", m.group(0), "immediate_refund_approved")

    return claims


def extract_claims(
    response_text: str,
    *,
    llm_client: BaseLLMClient | None = None,
    model: str | None = None,
) -> list[ExtractedClaim]:
    """Extract atomic factual claims using LLM when live, augmented by deterministic extraction."""
    det_claims = extract_claims_deterministic(response_text)
    use_live_llm = llm_client is not None and llm_client.provider != "fake"
    if not use_live_llm or llm_client is None:
        return det_claims

    prompt = (
        "Extract all atomic factual claims from the assistant response below.\n"
        "Allowed claim_type values: status, date, amount, eligibility, policy, action_taken.\n"
        f"Assistant response:\n{response_text}\n\n"
        'Return JSON: {"claims": [{"claim_text": "...", "claim_type": "...", "claimed_value": "..."}]}'
    )
    try:
        parsed, _resp = llm_client.generate_json(
            prompt=prompt,
            schema_model=ClaimExtractionOutput,
            model=model,
            temperature=0.0,
        )
        # Merge deterministic claims with LLM claims, deduplicating by (type, text)
        merged = list(det_claims)
        seen = {(c.claim_type, c.claim_text.lower()) for c in merged}
        for c in parsed.claims:
            key = (c.claim_type, c.claim_text.lower())
            if key not in seen:
                seen.add(key)
                merged.append(c)
        return merged
    except LLMError as exc:
        logger.warning("claim_extraction_llm_fallback", extra={"error": str(exc)})
        return det_claims
