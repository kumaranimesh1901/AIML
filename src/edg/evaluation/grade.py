"""Checklist grading, release-item pass/fail grading, and judge flip-rate measurement."""

from __future__ import annotations

import random
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from edg.config import AppConfig
from edg.db.models import (
    BenchmarkRelease,
    EvalResult,
    Judgement,
    StateSnapshot,
    TestCase,
)
from edg.judge.evidence import assemble_evidence
from edg.judge.llm_judge import HybridJudge, _preload_policies
from edg.judge.verifier import text_mentions_amount, text_mentions_date, text_mentions_status
from edg.labeling.severity import resolve_severity_for_interaction
from edg.llm.cache import LLMCache
from edg.llm.client import BaseLLMClient, create_llm_client
from edg.world.taxonomy import TaxonomyConfig, load_taxonomy

_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_AMOUNT_RE = re.compile(r"\$(\d+(?:\.\d{1,2})?)")
_QUOTED_STATUS_RE = re.compile(r"'([a-z_]+)'")


def evaluate_checklist(
    response_text: str,
    expected_checklist_json: dict[str, Any],
) -> dict[str, Any]:
    """Deterministically evaluate response satisfaction against must_do and must_not_do items."""
    resp_low = response_text.lower()
    must_do = list(expected_checklist_json.get("must_do") or [])
    must_not_do = list(expected_checklist_json.get("must_not_do") or [])

    must_do_results: list[dict[str, Any]] = []
    for item in must_do:
        item_low = item.lower()
        satisfied = True

        # Check explicit dates, amounts, and quoted statuses if present in the must_do item
        # Facts may be written in any spelling: "2026-11-25" / "November 25", "$89.99" / "89.99 USD",
        # "out_for_delivery" / "out for delivery"
        dates = _DATE_RE.findall(item)
        if dates and not any(text_mentions_date(response_text, d) for d in dates):
            satisfied = False

        amounts = _AMOUNT_RE.findall(item)
        if amounts and not any(text_mentions_amount(response_text, float(a)) for a in amounts):
            satisfied = False

        statuses = _QUOTED_STATUS_RE.findall(item_low)
        if statuses and not any(text_mentions_status(response_text, st) for st in statuses):
            satisfied = False

        # Key policy phrase checks
        if "carrier investigation" in item_low and "investigation" not in resp_low:
            satisfied = False
        if "cannot be cancelled" in item_low and "cannot be cancelled" not in resp_low:
            satisfied = False
        if "past the 30-day" in item_low and "30-day" not in resp_low and "not eligible" not in resp_low:
            satisfied = False
        if "free prepaid return" in item_low and "prepaid" not in resp_low:
            satisfied = False
        if "2-business-day" in item_low and "2-business-day" not in resp_low and "queue" not in resp_low:
            satisfied = False

        must_do_results.append({"item": item, "satisfied": satisfied})

    must_not_do_results: list[dict[str, Any]] = []
    for item in must_not_do:
        item_low = item.lower()
        violated = False
        if "immediate refund" in item_low and "approve an immediate refund" in resp_low:
            violated = True
        if "already-shipped order has been cancelled" in item_low and "submitted a cancellation" in resp_low:
            violated = True
        if "standard refund past the 30-day" in item_low and "process a standard refund" in resp_low:
            violated = True
        if "invent tracking" in item_low and "trk-" in resp_low:
            violated = True
        must_not_do_results.append({"item": item, "satisfied": not violated})

    all_items = [*must_do_results, *must_not_do_results]
    total = len(all_items)
    satisfied_cnt = sum(1 for r in all_items if r["satisfied"])
    rate = round(satisfied_cnt / total, 4) if total > 0 else 1.0

    return {
        "must_do_results": must_do_results,
        "must_not_do_results": must_not_do_results,
        "satisfied": satisfied_cnt,
        "total": total,
        "satisfaction_rate": rate,
    }


def compute_case_passed(
    *,
    judgement: Judgement,
    checklist_result: dict[str, Any],
    intent: str | None,
    hard_state_tags: list[str] | None,
    taxonomy: TaxonomyConfig,
    checklist_pass_threshold: float = 0.80,
) -> bool:
    """Compute binary pass/fail for a replayed benchmark item.

    Pass requires: correctness == pass, relevance == pass, no critical/high failure,
    and checklist satisfaction >= checklist_pass_threshold.
    """
    if judgement.correctness_verdict != "pass" or judgement.relevance_verdict != "pass":
        return False

    failures = list(judgement.failure_types or [])
    if failures:
        if "judge_error" in failures:
            return False
        _area, sev = resolve_severity_for_interaction(
            failure_types=failures,
            intent=intent,
            hard_state_tags=hard_state_tags,
            taxonomy=taxonomy,
        )
        if sev in {"critical", "high"}:
            return False

    if float(checklist_result.get("satisfaction_rate", 0.0)) < checklist_pass_threshold:
        return False

    return True


def compute_judge_flip_rate(
    session: Session,
    config: AppConfig,
    *,
    release_id: str,
    model_id: str,
    run_id: str,
    llm_client: BaseLLMClient | None = None,
) -> dict[str, Any]:
    """Re-judge a random 5% sample twice and measure how often the pass/fail verdict flips."""
    release = session.get(BenchmarkRelease, release_id)
    if release is None:
        raise ValueError(f"Release '{release_id}' not found.")

    eval_rows = list(
        session.scalars(
            select(EvalResult)
            .where(EvalResult.release_id == release_id, EvalResult.model_id == model_id)
            .order_by(EvalResult.test_case_id)
        ).all()
    )
    if not eval_rows:
        return {"n_sampled": 0, "flips": 0, "flip_rate": 0.0}

    sample_n = max(1, round(len(eval_rows) * config.evaluation.flip_rate_sample_share))
    rng = random.Random(config.seed + 77)
    sample = rng.sample(eval_rows, k=min(sample_n, len(eval_rows)))

    client = llm_client
    if client is None:
        cache = LLMCache(engine=session.get_bind()) if config.llm.cache_enabled else None
        effective_provider = "fake" if config.llm.provider == "fake" else config.judge.provider
        client = create_llm_client(
            config.llm,
            provider=effective_provider,
            default_model=config.judge.model_name,
            cache=cache,
        )

    cfg_frozen = config.model_copy(deep=True)
    cfg_frozen.judge.judge_version = release.judge_version
    judge = HybridJudge(config=cfg_frozen, llm_client=client)
    preloaded = _preload_policies(session, config.world.policy_version)
    taxonomy = load_taxonomy()

    flips = 0
    for er in sample:
        tc = session.get(TestCase, er.test_case_id)
        if tc is None:
            continue
        snap = session.get(StateSnapshot, tc.snapshot_id)
        evidence = assemble_evidence(
            session,
            snapshot_id=tc.snapshot_id,
            query=tc.input_query,
            reference_answer=tc.reference_answer,
            policy_version=tc.policy_version,
            top_k_policies=config.judge.top_k_policies,
            preloaded_policies=preloaded,
        )
        repeat_jdg, _ = judge.judge_one(
            session=session,
            interaction_id=f"{tc.source_interaction_id}",
            query=tc.input_query,
            response=er.response,
            evidence=evidence,
            run_id=run_id,
            bypass_cache=True,
            persist=False,
        )
        repeat_passed = compute_case_passed(
            judgement=repeat_jdg,
            checklist_result=er.checklist_results_json,
            intent=tc.intent,
            hard_state_tags=list(snap.hard_state_tags or []) if snap else [],
            taxonomy=taxonomy,
            checklist_pass_threshold=config.evaluation.checklist_pass_threshold,
        )
        if repeat_passed != er.passed:
            flips += 1

    return {
        "n_sampled": len(sample),
        "flips": flips,
        "flip_rate": round(flips / len(sample), 4) if sample else 0.0,
    }
