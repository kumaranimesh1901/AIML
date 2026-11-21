"""Pydantic v2 schemas for the hybrid evidence-based LLM judge."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EvidenceStatus = Literal["sufficient", "partial", "missing"]
DimensionVerdict = Literal["pass", "partial", "fail", "not_assessable"]
ClaimType = Literal["status", "date", "amount", "eligibility", "policy", "action_taken"]
ClaimCheckStatus = Literal["supported", "contradicted", "not_in_evidence"]

DIMENSIONS: tuple[str, ...] = (
    "correctness",
    "relevance",
    "helpfulness",
    "completeness",
    "safety",
)


class ExtractedClaim(BaseModel):
    model_config = ConfigDict(extra="ignore")

    claim_text: str = Field(min_length=1)
    claim_type: ClaimType
    claimed_value: str | None = None


class ClaimExtractionOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    claims: list[ExtractedClaim] = Field(default_factory=list)


class ClaimCheckResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    claim_text: str
    claim_type: ClaimType
    status: ClaimCheckStatus
    evidence_detail: str


class DimensionAssessment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    verdict: DimensionVerdict
    score: int | None = Field(default=None, ge=1, le=5)


class LLMJudgeOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    correctness: DimensionAssessment
    relevance: DimensionAssessment
    helpfulness: DimensionAssessment
    completeness: DimensionAssessment
    safety: DimensionAssessment
    failure_types: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=5)
    confidence: float = Field(ge=0.0, le=1.0)
