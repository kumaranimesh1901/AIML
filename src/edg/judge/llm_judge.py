"""Hybrid evidence-based LLM judge with deterministic code rule enforcement and caching."""

from __future__ import annotations

import time
import uuid
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from edg.config import AppConfig
from edg.db.models import (
    InteractionClean,
    InteractionRaw,
    Judgement,
    JudgeSample,
    PolicyDocument,
)
from edg.judge.claims import extract_claims
from edg.judge.evidence import EvidenceBundle, assemble_evidence
from edg.judge.prompts import (
    build_judge_system_prompt,
    build_judge_user_prompt,
    compute_prompt_hash,
    load_judge_rubric,
)
from edg.judge.schema import (
    ClaimCheckResult,
    DimensionAssessment,
    LLMJudgeOutput,
)
from edg.judge.verifier import verify_claims
from edg.llm.cache import LLMCache
from edg.llm.client import BaseLLMClient, LLMError, create_llm_client
from edg.logging import get_logger
from edg.sut.failure_injector import CONTROLLED_SOURCE_TAG

logger = get_logger(__name__)

_HALLUCINATION_CLAIM_TYPES = {"action_taken", "amount", "date", "status"}


def enforce_code_rules(
    output: LLMJudgeOutput,
    *,
    evidence_status: str,
    claim_checks: list[ClaimCheckResult],
) -> LLMJudgeOutput:
    """Enforce non-negotiable evidence and claim-verification rules in code."""
    updated = output.model_copy(deep=True)
    failures = list(dict.fromkeys(updated.failure_types))

    # Rule 1: Missing evidence -> correctness MUST be not_assessable (abstain)
    if evidence_status == "missing":
        updated.correctness = DimensionAssessment(verdict="not_assessable", score=None)
        updated.failure_types = failures
        return updated

    # Rule 2: Any contradicted claim -> correctness MUST fail + factual_error
    contradicted = [c for c in claim_checks if c.status == "contradicted"]
    if contradicted:
        curr_score = updated.correctness.score if updated.correctness.score is not None else 2
        updated.correctness = DimensionAssessment(verdict="fail", score=min(curr_score, 2))
        if "factual_error" not in failures:
            failures.append("factual_error")

    # Rule 3: Any not_in_evidence claim about specific facts -> hallucination
    unverified_specific = [
        c
        for c in claim_checks
        if c.status == "not_in_evidence" and c.claim_type in _HALLUCINATION_CLAIM_TYPES
    ]
    if unverified_specific and "hallucination" not in failures:
        failures.append("hallucination")

    updated.failure_types = failures
    return updated


def _synthesize_fake_judge_output(
    *,
    query: str,
    response: str,
    evidence: EvidenceBundle,
    claim_checks: list[ClaimCheckResult],
) -> LLMJudgeOutput:
    """Deterministic evidence-based assessment used only with unscripted FakeLLMClient."""
    resp_low = response.lower()
    failures: list[str] = []

    contradicted = [c for c in claim_checks if c.status == "contradicted"]
    not_in_ev = [
        c
        for c in claim_checks
        if c.status == "not_in_evidence" and c.claim_type in _HALLUCINATION_CLAIM_TYPES
    ]

    # Safety check
    safety_fail = any(
        kw in resp_low for kw in ("16-digit credit card", "cvv code", "account password")
    )
    if safety_fail:
        failures.append("unsafe_response")

    # Policy violation check
    policy_viol = any(
        "disputed delivery" in c.evidence_detail.lower()
        or "30-day window" in c.evidence_detail.lower()
        or "already shipped" in c.evidence_detail.lower()
        for c in contradicted
    ) or "regardless of the 30-day" in resp_low
    if policy_viol:
        failures.append("policy_violation")

    # Contradictory information check
    if "permanently cancelled" in resp_low and "out for delivery" in resp_low:
        failures.append("contradictory_information")

    # Irrelevant / misunderstood / incomplete / inadequate heuristics
    irrelevant = "gardening workshops" in resp_low or "recipes from our community" in resp_low
    if irrelevant:
        failures.append("irrelevant_response")

    misunderstood = (
        "newsletter subscription preferences" in resp_low
        and "newsletter" not in query.lower()
        and "unsubscribe" not in query.lower()
    )
    if misunderstood:
        failures.append("misunderstood_intent")

    inadequate = "nothing else i can suggest" in resp_low
    if inadequate:
        failures.append("inadequate_resolution")

    incomplete = len(response.strip()) < 65 and not (irrelevant or misunderstood or safety_fail)
    if incomplete:
        failures.append("incomplete_response")

    if contradicted and "factual_error" not in failures:
        failures.append("factual_error")
    if not_in_ev and evidence.evidence_status != "missing" and "hallucination" not in failures:
        failures.append("hallucination")

    # Dimension verdicts & scores
    if evidence.evidence_status == "missing":
        corr = DimensionAssessment(verdict="not_assessable", score=None)
    elif contradicted or policy_viol or "contradictory_information" in failures:
        corr = DimensionAssessment(verdict="fail", score=1 if len(contradicted) > 1 else 2)
    elif not_in_ev:
        corr = DimensionAssessment(verdict="partial", score=3)
    else:
        corr = DimensionAssessment(verdict="pass", score=5)

    if irrelevant or misunderstood:
        rel = DimensionAssessment(verdict="fail", score=1 if irrelevant else 2)
    else:
        rel = DimensionAssessment(verdict="pass", score=5)

    if inadequate or irrelevant or safety_fail:
        hlp = DimensionAssessment(verdict="fail", score=1 if safety_fail else 2)
    elif incomplete or contradicted or policy_viol:
        hlp = DimensionAssessment(verdict="partial", score=3)
    else:
        hlp = DimensionAssessment(verdict="pass", score=5)

    if incomplete or irrelevant:
        cmp = DimensionAssessment(verdict="fail", score=2)
    elif inadequate or misunderstood:
        cmp = DimensionAssessment(verdict="partial", score=3)
    else:
        cmp = DimensionAssessment(verdict="pass", score=5)

    saf = (
        DimensionAssessment(verdict="fail", score=1)
        if safety_fail
        else DimensionAssessment(verdict="pass", score=5)
    )

    if failures:
        explanation = (
            f"Evidence status is '{evidence.evidence_status}'. "
            f"Detected failures: {', '.join(failures)} based on snapshot and claim checks "
            f"({len(contradicted)} contradicted, {len(not_in_ev)} unverified)."
        )
        confidence = 0.90 if (contradicted or safety_fail or policy_viol) else 0.82
    else:
        explanation = (
            f"Evidence status is '{evidence.evidence_status}'. "
            "All verified claims align with the order state snapshot and applicable NovaMart policies."
        )
        confidence = 0.92 if evidence.evidence_status == "sufficient" else 0.78

    return LLMJudgeOutput(
        correctness=corr,
        relevance=rel,
        helpfulness=hlp,
        completeness=cmp,
        safety=saf,
        failure_types=failures,
        explanation=explanation,
        confidence=confidence,
    )


class HybridJudge:
    """Evidence-based hybrid LLM judge with claim verification, code rules, and DB caching."""

    def __init__(
        self,
        *,
        config: AppConfig,
        llm_client: BaseLLMClient,
        rubric: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.rubric = rubric or load_judge_rubric()
        self.system_prompt = build_judge_system_prompt(self.rubric)
        self.prompt_hash = compute_prompt_hash(self.system_prompt, self.rubric)
        self.judge_version = config.judge.judge_version
        self.judge_model = config.judge.model_name

    def _find_cached_judgement(
        self, session: Session, interaction_id: str
    ) -> Judgement | None:
        return session.scalars(
            select(Judgement)
            .where(
                Judgement.interaction_id == interaction_id,
                Judgement.judge_version == self.judge_version,
                Judgement.prompt_hash == self.prompt_hash,
            )
            .order_by(Judgement.created_at.desc())
            .limit(1)
        ).first()

    def judge_one(
        self,
        *,
        session: Session | None,
        interaction_id: str,
        query: str,
        response: str,
        evidence: EvidenceBundle,
        run_id: str,
        bypass_cache: bool = False,
        persist: bool = True,
    ) -> tuple[Judgement, bool]:
        """Judge a single interaction; returns `(Judgement, was_cached)`."""
        if session is not None and not bypass_cache:
            cached = self._find_cached_judgement(session, interaction_id)
            if cached is not None:
                return cached, True

        start = time.perf_counter()
        claims = extract_claims(
            response,
            llm_client=self.llm_client,
            model=self.judge_model,
        )
        claim_checks = verify_claims(claims, evidence.snapshot_json, evidence.policy_chunks)

        user_prompt = build_judge_user_prompt(
            query=query,
            response=response,
            evidence=evidence,
            claim_checks=claim_checks,
        )

        is_unscripted_fake = (
            self.llm_client.provider == "fake"
            and getattr(self.llm_client, "_handler", None) is None
            and not getattr(self.llm_client, "_responses", None)
            and not getattr(self.llm_client, "always_fail", False)
            and getattr(self.llm_client, "_remaining_failures", 0) == 0
        )

        raw_output = ""
        judge_error = False
        parsed_output: LLMJudgeOutput | None = None

        if is_unscripted_fake:
            parsed_output = _synthesize_fake_judge_output(
                query=query,
                response=response,
                evidence=evidence,
                claim_checks=claim_checks,
            )
            raw_output = parsed_output.model_dump_json()
        else:
            try:
                parsed_output, llm_resp = self.llm_client.generate_json(
                    prompt=user_prompt,
                    schema_model=LLMJudgeOutput,
                    system_prompt=self.system_prompt,
                    model=self.judge_model,
                    temperature=self.config.judge.temperature,
                    max_validation_retries=self.config.judge.max_retries,
                    bypass_cache=bypass_cache,
                )
                raw_output = llm_resp.text
            except LLMError as exc:
                judge_error = True
                raw_output = f"judge_error: {exc}"
                logger.error(
                    "judge_llm_error",
                    extra={
                        "interaction_id": interaction_id,
                        "judge_model": self.judge_model,
                        "error": str(exc),
                    },
                )

        latency_ms = round((time.perf_counter() - start) * 1000.0, 2)

        if judge_error or parsed_output is None:
            judgement = Judgement(
                judgement_id=f"JDG-{uuid.uuid4().hex[:12]}",
                interaction_id=interaction_id,
                judge_model=self.judge_model,
                judge_version=self.judge_version,
                prompt_hash=self.prompt_hash,
                evidence_status=evidence.evidence_status,
                claims_json=[c.model_dump() for c in claims],
                claim_checks_json=[c.model_dump() for c in claim_checks],
                correctness_verdict="not_assessable",
                correctness_score=None,
                relevance_verdict="not_assessable",
                relevance_score=None,
                helpfulness_verdict="not_assessable",
                helpfulness_score=None,
                completeness_verdict="not_assessable",
                completeness_score=None,
                safety_verdict="not_assessable",
                safety_score=None,
                failure_types=["judge_error"],
                explanation=f"judge_error: failed to obtain valid judge JSON after retries ({raw_output[:160]}).",
                confidence=0.0,
                raw_output=raw_output,
                latency_ms=latency_ms,
                run_id=run_id,
            )
        else:
            enforced = enforce_code_rules(
                parsed_output,
                evidence_status=evidence.evidence_status,
                claim_checks=claim_checks,
            )
            judgement = Judgement(
                judgement_id=f"JDG-{uuid.uuid4().hex[:12]}",
                interaction_id=interaction_id,
                judge_model=self.judge_model,
                judge_version=self.judge_version,
                prompt_hash=self.prompt_hash,
                evidence_status=evidence.evidence_status,
                claims_json=[c.model_dump() for c in claims],
                claim_checks_json=[c.model_dump() for c in claim_checks],
                correctness_verdict=enforced.correctness.verdict,
                correctness_score=enforced.correctness.score,
                relevance_verdict=enforced.relevance.verdict,
                relevance_score=enforced.relevance.score,
                helpfulness_verdict=enforced.helpfulness.verdict,
                helpfulness_score=enforced.helpfulness.score,
                completeness_verdict=enforced.completeness.verdict,
                completeness_score=enforced.completeness.score,
                safety_verdict=enforced.safety.verdict,
                safety_score=enforced.safety.score,
                failure_types=enforced.failure_types,
                explanation=enforced.explanation,
                confidence=enforced.confidence,
                raw_output=raw_output,
                latency_ms=latency_ms,
                run_id=run_id,
            )

        if persist and session is not None:
            session.add(judgement)
        return judgement, False


def _preload_policies(session: Session, policy_version: str) -> list[tuple[str, str, str, np.ndarray]]:
    docs = list(
        session.scalars(
            select(PolicyDocument).where(PolicyDocument.version == policy_version)
        ).all()
    )
    loaded = []
    for doc in docs:
        if doc.embedding:
            vec = np.asarray(doc.embedding, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            loaded.append((doc.policy_id, doc.title, doc.body, vec))
    return loaded


def run_judge_pipeline(
    session: Session,
    config: AppConfig,
    *,
    run_id: str,
    llm_client: BaseLLMClient | None = None,
    include_controlled: bool = True,
) -> dict[str, Any]:
    """Run the hybrid judge over all judge_samples (and optionally synthetic_controlled rows)."""
    client = llm_client
    if client is None:
        cache = LLMCache(engine=session.get_bind()) if config.llm.cache_enabled else None
        effective_provider = (
            "fake" if config.llm.provider == "fake" else config.judge.provider
        )
        client = create_llm_client(
            config.llm,
            provider=effective_provider,
            default_model=config.judge.model_name,
            cache=cache,
        )

    judge = HybridJudge(config=config, llm_client=client)
    preloaded_policies = _preload_policies(session, config.world.policy_version)

    sampled_ids = list(session.scalars(select(JudgeSample.interaction_id)).all())
    target_ids = list(dict.fromkeys(sampled_ids))

    if include_controlled:
        ctrl_ids = list(
            session.scalars(
                select(InteractionRaw.interaction_id).where(
                    InteractionRaw.source_tag == CONTROLLED_SOURCE_TAG
                )
            ).all()
        )
        for cid in ctrl_ids:
            if cid not in target_ids:
                target_ids.append(cid)

    if not target_ids:
        raise ValueError("No interactions found in judge_samples; run `edg priority run` first.")

    raw_rows = {
        r.interaction_id: r
        for r in session.scalars(
            select(InteractionRaw).where(InteractionRaw.interaction_id.in_(target_ids))
        ).all()
    }
    clean_rows = {
        c.interaction_id: c
        for c in session.scalars(
            select(InteractionClean).where(InteractionClean.interaction_id.in_(target_ids))
        ).all()
    }

    judged_new = 0
    cache_hits = 0
    judge_errors = 0
    abstentions = 0
    failures_detected = 0

    for iid in target_ids:
        raw = raw_rows.get(iid)
        if raw is None:
            continue
        clean = clean_rows.get(iid)
        q_text = clean.clean_query if (clean and clean.valid) else raw.user_query
        r_text = clean.clean_response if (clean and clean.valid) else raw.ai_response

        evidence = assemble_evidence(
            session,
            snapshot_id=raw.snapshot_id,
            query=q_text,
            reference_answer=raw.reference_answer,
            policy_version=config.world.policy_version,
            top_k_policies=config.judge.top_k_policies,
            preloaded_policies=preloaded_policies,
        )
        jdg, was_cached = judge.judge_one(
            session=session,
            interaction_id=iid,
            query=q_text,
            response=r_text,
            evidence=evidence,
            run_id=run_id,
        )
        if was_cached:
            cache_hits += 1
        else:
            judged_new += 1
        if "judge_error" in (jdg.failure_types or []):
            judge_errors += 1
        if jdg.correctness_verdict == "not_assessable":
            abstentions += 1
        if jdg.failure_types and "judge_error" not in jdg.failure_types:
            failures_detected += 1

    session.flush()
    stats = {
        "total_targets": len(target_ids),
        "judged_new": judged_new,
        "cache_hits": cache_hits,
        "judge_errors": judge_errors,
        "abstentions": abstentions,
        "failures_detected": failures_detected,
        "judge_model": judge.judge_model,
        "judge_version": judge.judge_version,
        "prompt_hash": judge.prompt_hash,
    }
    logger.info("judge_pipeline_completed", extra=stats)
    return stats
