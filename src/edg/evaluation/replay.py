"""Replay a model version on a frozen benchmark release and grade with the release's frozen judge."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from edg.config import AppConfig
from edg.db.models import (
    BenchmarkRelease,
    EvalResult,
    ReleaseItem,
    StateSnapshot,
    TestCase,
)
from edg.evaluation.grade import compute_case_passed, evaluate_checklist
from edg.judge.evidence import assemble_evidence
from edg.judge.llm_judge import HybridJudge, _preload_policies
from edg.llm.cache import LLMCache
from edg.llm.client import BaseLLMClient, create_llm_client
from edg.logging import get_logger
from edg.sut.assistant import CustomerSupportAssistant
from edg.sut.tools import SnapshotToolBox
from edg.sut.versions import register_model_versions
from edg.world.taxonomy import load_taxonomy

logger = get_logger(__name__)


def replay_and_grade_release(
    session: Session,
    config: AppConfig,
    *,
    release_id: str,
    model_version_name: str,
    run_id: str,
    sut_client: BaseLLMClient | None = None,
    judge_client: BaseLLMClient | None = None,
) -> dict[str, Any]:
    """Replay a SUT model version on every item in a frozen release and grade with the release's judge."""
    release = session.get(BenchmarkRelease, release_id)
    if release is None:
        raise ValueError(f"Benchmark release '{release_id}' not found.")

    if model_version_name not in config.sut.versions:
        raise ValueError(
            f"Unknown model version '{model_version_name}'. Available: {list(config.sut.versions)}"
        )

    register_model_versions(session, config, run_id=run_id)
    ver_cfg = config.sut.versions[model_version_name]
    cache = LLMCache(engine=session.get_bind()) if config.llm.cache_enabled else None

    resolved_sut_client = sut_client
    if resolved_sut_client is None:
        eff_sut_provider = "fake" if config.llm.provider == "fake" else ver_cfg.provider
        resolved_sut_client = create_llm_client(
            config.llm,
            provider=eff_sut_provider,
            default_model=ver_cfg.base_model,
            cache=cache,
        )

    resolved_judge_client = judge_client
    if resolved_judge_client is None:
        eff_judge_provider = "fake" if config.llm.provider == "fake" else config.judge.provider
        resolved_judge_client = create_llm_client(
            config.llm,
            provider=eff_judge_provider,
            default_model=config.judge.model_name,
            cache=cache,
        )

    # Pin judge to the release's frozen judge_version
    judge_cfg = config.model_copy(deep=True)
    judge_cfg.judge.judge_version = release.judge_version
    judge = HybridJudge(config=judge_cfg, llm_client=resolved_judge_client)
    assistant = CustomerSupportAssistant(
        version_config=ver_cfg,
        llm_client=resolved_sut_client,
        max_tool_calls=config.sut.max_tool_calls,
    )
    taxonomy = load_taxonomy()
    preloaded_policies = _preload_policies(session, config.world.policy_version)

    items = session.execute(
        select(ReleaseItem, TestCase, StateSnapshot)
        .join(TestCase, TestCase.test_case_id == ReleaseItem.test_case_id)
        .join(StateSnapshot, StateSnapshot.snapshot_id == TestCase.snapshot_id)
        .where(ReleaseItem.release_id == release_id)
        .order_by(ReleaseItem.test_case_id)
    ).all()

    existing_tc_ids = set(
        session.scalars(
            select(EvalResult.test_case_id).where(
                EvalResult.release_id == release_id,
                EvalResult.model_id == model_version_name,
            )
        ).all()
    )

    evaluated_new = 0
    passed_count = 0

    for _ri, tc, snap in items:
        if tc.test_case_id in existing_tc_ids:
            continue

        toolbox = SnapshotToolBox(
            snap,
            session=session,
            retrieval_enabled=ver_cfg.retrieval_enabled,
            policy_cache=preloaded_policies,
        )
        ans = assistant.answer(query=tc.input_query, order_id=snap.order_id, toolbox=toolbox)

        evidence = assemble_evidence(
            session,
            snapshot_id=tc.snapshot_id,
            query=tc.input_query,
            reference_answer=tc.reference_answer,
            policy_version=tc.policy_version,
            top_k_policies=config.judge.top_k_policies,
            preloaded_policies=preloaded_policies,
        )
        jdg, _ = judge.judge_one(
            session=session,
            interaction_id=tc.source_interaction_id,
            query=tc.input_query,
            response=ans.answer,
            evidence=evidence,
            run_id=run_id,
            bypass_cache=True,  # Always grade the newly replayed response freshly
            persist=True,
        )
        session.flush()

        chk_res = evaluate_checklist(ans.answer, tc.expected_checklist_json)
        passed = compute_case_passed(
            judgement=jdg,
            checklist_result=chk_res,
            intent=tc.intent,
            hard_state_tags=list(snap.hard_state_tags or []),
            taxonomy=taxonomy,
            checklist_pass_threshold=config.evaluation.checklist_pass_threshold,
        )
        if passed:
            passed_count += 1

        session.add(
            EvalResult(
                result_id=f"EVL-{uuid.uuid4().hex[:12]}",
                release_id=release_id,
                model_id=model_version_name,
                test_case_id=tc.test_case_id,
                response=ans.answer,
                judgement_id=jdg.judgement_id,
                passed=passed,
                checklist_results_json=chk_res,
                run_id=run_id,
            )
        )
        evaluated_new += 1

    session.flush()
    stats = {
        "release_id": release_id,
        "model_id": model_version_name,
        "judge_version": release.judge_version,
        "total_release_items": len(items),
        "evaluated_new": evaluated_new,
        "passed_new": passed_count,
        "pass_rate_new": round(passed_count / evaluated_new, 4) if evaluated_new else None,
    }
    logger.info("release_replay_completed", extra=stats)
    return stats
