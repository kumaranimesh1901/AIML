"""Consistent PII detection and pseudonymisation (Regex + optional spaCy NER)."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from sqlalchemy.orm import Session

from edg.db.models import PIIMapping
from edg.logging import get_logger

logger = get_logger(__name__)

# Patterns for PII detection
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}(?!\w)"
)
_CARD_RE = re.compile(r"(?<!\d)(?:\d{4}[-\s]?){3}\d{4}(?!\d)")
_ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+[A-Z][A-Za-z0-9.'-]+(?:\s+[A-Z][A-Za-z0-9.'-]+){0,3}\s+"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way)\b\.?",
    re.IGNORECASE,
)
# Contextual person-name regex fallback when spaCy en_core_web_sm is unavailable
_NAME_CONTEXT_RE = re.compile(
    r"(\b(?:[Mm]y name is|[Tt]his is|[Rr]egards,|[Ss]incerely,|[Tt]hanks,|[Ff]rom)\s+)"
    r"([A-Z][a-z]{1,20}\s+[A-Z][a-z]{1,20})\b"
)

# Pseudonymous IDs that must NEVER be masked (order IDs stay joinable)
_PROTECTED_ID_RE = re.compile(
    r"\b(?:ORD|CUST|SHP|PAY|RET|SNAP|PROD|QRY|HQRY|INT|CTRL)-\w+\b|\b(?:NAME|EMAIL)_\d+\b"
)


# Store, carrier and brand-like words spaCy may tag as PERSON in support replies
_STORE_WORDS = frozenset({"novamart", "ups", "fedex", "dhl", "usps", "customer", "dear customer"})


def _make_token(entity_type: str, value: str) -> tuple[str, str]:
    normalized = value.strip().lower()
    digest = hashlib.sha256(f"{entity_type}:{normalized}".encode()).hexdigest()
    token = f"<{entity_type}_{digest[:8].upper()}>"
    return token, digest


class PIIPseudonymizer:
    """Detects emails, phones, cards, addresses, and person names with consistent tokens."""

    def __init__(self, *, use_spacy: bool = True) -> None:
        self._nlp: Any = None
        if use_spacy:
            try:
                import spacy

                self._nlp = spacy.load("en_core_web_sm")
            except Exception:
                self._nlp = None

    def pseudonymize(
        self, text: str, *, protected_phrases: list[str] | None = None
    ) -> tuple[str, list[dict[str, str]]]:
        """Replace PII spans with deterministic tokens while preserving store/order IDs.

        `protected_phrases` (e.g. the bound order's product titles) are never masked as person
        names: spaCy tags parts of titles such as "MX Master 3S Wireless Mouse" as PERSON.
        """
        protected_low = [p.lower() for p in (protected_phrases or []) if p]
        records: list[dict[str, str]] = []
        result = text

        def _sub_pattern(pattern: re.Pattern[str], entity_type: str, current: str) -> str:
            def _repl(match: re.Match[str]) -> str:
                raw_val = match.group(0)
                if _PROTECTED_ID_RE.search(raw_val):
                    return raw_val
                token, orig_hash = _make_token(entity_type, raw_val)
                records.append(
                    {"entity_type": entity_type, "original_hash": orig_hash, "token": token}
                )
                return token

            return pattern.sub(_repl, current)

        result = _sub_pattern(_EMAIL_RE, "EMAIL", result)
        result = _sub_pattern(_CARD_RE, "CARD", result)
        result = _sub_pattern(_PHONE_RE, "PHONE", result)
        result = _sub_pattern(_ADDRESS_RE, "ADDRESS", result)

        if self._nlp is not None:
            doc = self._nlp(result)
            # Replace PERSON entities from right to left to preserve offsets
            spans = [ent for ent in doc.ents if ent.label_ == "PERSON"]
            for ent in sorted(spans, key=lambda e: e.start_char, reverse=True):
                raw_val = ent.text.strip()
                if not raw_val or raw_val.startswith("<") or _PROTECTED_ID_RE.search(raw_val):
                    continue
                if raw_val.lower() in _STORE_WORDS or any(raw_val.lower() in p for p in protected_low):
                    continue
                token, orig_hash = _make_token("PERSON", raw_val)
                records.append(
                    {"entity_type": "PERSON", "original_hash": orig_hash, "token": token}
                )
                result = result[: ent.start_char] + token + result[ent.end_char :]
        else:
            def _repl_name(match: re.Match[str]) -> str:
                prefix = match.group(1)
                raw_val = match.group(2)
                if _PROTECTED_ID_RE.search(raw_val):
                    return match.group(0)
                token, orig_hash = _make_token("PERSON", raw_val)
                records.append(
                    {"entity_type": "PERSON", "original_hash": orig_hash, "token": token}
                )
                return f"{prefix}{token}"

            result = _NAME_CONTEXT_RE.sub(_repl_name, result)

        return result, records


def persist_pii_mappings(
    session: Session,
    *,
    pii_map_ref: str,
    records: list[dict[str, str]],
    run_id: str,
) -> int:
    """Persist PII token mappings into pii_mappings table."""
    for rec in records:
        session.add(
            PIIMapping(
                pii_map_ref=pii_map_ref,
                entity_type=rec["entity_type"],
                original_hash=rec["original_hash"],
                token=rec["token"],
                run_id=run_id,
            )
        )
    return len(records)
