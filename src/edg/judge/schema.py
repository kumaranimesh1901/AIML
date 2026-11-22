"""Pydantic v2 schemas for the hybrid evidence-based LLM judge."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    claimed_value: str | None = None
    # Concrete values (dates, amounts, record ids) in the claim that the snapshot and policies do
    # not contain. None = not computed (older judgements); [] = every concrete value checks out.
    unverified_values: list[str] | None = None


class DimensionAssessment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    verdict: DimensionVerdict
    # Required (no default) so schema-constrained providers always emit it; null only for
    # not_assessable. A defaulted score let the live judge drop scores on most items.
    score: int | None = Field(ge=1, le=5)

    @model_validator(mode="after")
    def _score_matches_verdict(self) -> DimensionAssessment:
        if self.verdict == "not_assessable":
            self.score = None
        elif self.score is None:
            raise ValueError(f"verdict '{self.verdict}' needs an integer score from 1 to 5")
        return self


class LLMDimensionAssessment(DimensionAssessment):
    """Dimension shape sent to the judge LLM: the score is always an integer.

    Allowing null in the provider schema let constrained decoding pick it for fail/partial
    verdicts; for not_assessable the inherited validator drops the score.
    """

    score: int = Field(ge=1, le=5)  # type: ignore[assignment]


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
