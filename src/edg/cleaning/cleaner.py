"""Text normalization (HTML stripping, Unicode NFKC, whitespace) preserving case and stopwords."""

from __future__ import annotations

import html
import re
import unicodedata
from datetime import UTC, datetime

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Clean text for downstream embeddings without lowercasing, stemming, or removing stopwords."""
    unescaped = html.unescape(text)
    no_html = _HTML_TAG_RE.sub(" ", unescaped)
    normalized = unicodedata.normalize("NFKC", no_html)
    collapsed = _WHITESPACE_RE.sub(" ", normalized).strip()
    return collapsed


def normalize_timestamp_utc(ts: datetime | str) -> str:
    """Standardize a datetime or ISO string to UTC ISO-8601 format."""
    if isinstance(ts, str):
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    else:
        dt = ts
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()
