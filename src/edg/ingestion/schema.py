"""Strict Pydantic v2 schema for validating raw interaction logs without silent repair."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ValidSourceTag = Literal[
    "public_bitext",
    "synthetic_natural",
    "synthetic_controlled",
    "synthetic_variant",
]


class RawInteractionSchema(BaseModel):
    """Strict validation schema for production-style interaction logs."""

    model_config = ConfigDict(extra="ignore")

    interaction_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    turn_index: int = Field(ge=0)
    user_query: str = Field(min_length=2)
    ai_response: str = Field(min_length=2)
    model_version: str = Field(min_length=1)
    timestamp: datetime
    source_tag: ValidSourceTag
    snapshot_id: str = Field(min_length=1)
    intent_label: str | None = None
    reference_answer: str | None = None
    injected_failure: str | None = None

    @field_validator("user_query", "ai_response")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        if not value or not value.strip() or len(value.strip()) < 2:
            raise ValueError("must contain at least 2 non-whitespace characters")
        return value

    @field_validator("timestamp", mode="before")
    @classmethod
    def _parse_timestamp_utc(cls, value: datetime | str) -> datetime:
        if isinstance(value, str):
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        elif isinstance(value, datetime):
            dt = value
        else:
            raise ValueError("timestamp must be a datetime or ISO-8601 string")
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
