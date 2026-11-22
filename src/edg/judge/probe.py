"""Judge probe: re-judge a small sample of an existing pipeline DB without writing anything.

Each real-model run-all takes hours, so judge behaviour (and alternative judge models) is checked
here on a few dozen already-logged interactions in minutes. Nothing is persisted: no judgements,
and the LLM cache is off. For every item it reports where each failure label came from: the LLM
itself, or the deterministic code rules (contradicted claim -> factual_error, concrete
unverified claim -> hallucination).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from edg.config import AppConfig
from edg.db.models import InteractionClean, InteractionRaw, JudgeSample
from edg.judge.evidence import assemble_evidence
from edg.judge.llm_judge import HybridJudge, _preload_policies
from edg.llm.client import BaseLLMClient, _strip_json_fences, create_llm_client
from edg.sut.failure_injector import CONTROLLED_SOURCE_TAG

DIMS = ("correctness", "relevance", "helpfulness", "completeness", "safety")


def _llm_failure_types(raw_output: str) -> list[str] | None:
    try:
        parsed = json.loads(_strip_json_fences(raw_output or ""))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    return list(parsed.get("failure_types") or [])


def probe_judge(
    session: Session,
    config: AppConfig,
    *,
    n: int = 15,
    stratum: str = "random_control",
    judge_model: str | None = None,
    seed: int = 7,
    llm_client: BaseLLMClient | None = None,
) -> dict[str, Any]:
    """Re-judge `n` interactions of `stratum` ("random_control", "priority", "cluster", "controlled")."""
    cfg = config.model_copy(deep=True)
    if judge_model:
        cfg.judge.model_name = judge_model
    cfg.judge.judge_version = f"{cfg.judge.judge_version}-probe"

    if stratum == "controlled":
        ids = list(
            session.scalars(
                select(InteractionRaw.interaction_id)
                .where(InteractionRaw.source_tag == CONTROLLED_SOURCE_TAG)
                .order_by(InteractionRaw.interaction_id)
            ).all()
        )
    else:
        ids = list(
            session.scalars(
                select(JudgeSample.interaction_id)
                .where(JudgeSample.stratum == stratum)
                .order_by(JudgeSample.interaction_id)
            ).all()
        )
    if not ids:
        raise ValueError(f"No interactions found for stratum '{stratum}'. Run the pipeline first.")
    sample = random.Random(seed).sample(ids, k=min(n, len(ids)))

    client = llm_client
    if client is None:
        provider = "fake" if cfg.llm.provider == "fake" else cfg.judge.provider
        client = create_llm_client(cfg.llm, provider=provider, default_model=cfg.judge.model_name, cache=None)
    judge = HybridJudge(config=cfg, llm_client=client)
    policies = _preload_policies(session, cfg.world.policy_version)

    items: list[dict[str, Any]] = []
    for iid in sample:
        raw = session.get(InteractionRaw, iid)
        if raw is None:
            continue
        clean = session.get(InteractionClean, iid)
        query = clean.clean_query if (clean and clean.valid) else raw.user_query
        response = clean.clean_response if (clean and clean.valid) else raw.ai_response
        evidence = assemble_evidence(
            session,
            snapshot_id=raw.snapshot_id,
            query=query,
            reference_answer=raw.reference_answer,
            policy_version=cfg.world.policy_version,
            top_k_policies=cfg.judge.top_k_policies,
            preloaded_policies=policies,
        )
        jdg, _ = judge.judge_one(
            session=None,
            interaction_id=iid,
            query=query,
            response=response,
            evidence=evidence,
            run_id="probe",
            bypass_cache=True,
            persist=False,
        )
        final = list(jdg.failure_types or [])
        from_llm = _llm_failure_types(jdg.raw_output)
        checks = Counter(f"{c['claim_type']}:{c['status']}" for c in (jdg.claim_checks_json or []))
        items.append(
            {
                "interaction_id": iid,
                "injected_failure": raw.injected_failure,
                "query": query[:200],
                "response": response[:400],
                "verdicts": {d: f"{getattr(jdg, f'{d}_verdict')}/{getattr(jdg, f'{d}_score')}" for d in DIMS},
                "failure_types": final,
                "failure_types_from_llm": from_llm,
                "added_by_code_rules": [f for f in final if from_llm is not None and f not in from_llm],
                "claim_checks": dict(checks),
                "confidence": jdg.confidence,
                "explanation": (jdg.explanation or "")[:300],
            }
        )

    n_items = len(items)
    flagged = [i for i in items if i["failure_types"]]
    llm_flagged = [i for i in items if i["failure_types_from_llm"]]
    errors = [i for i in items if "judge_error" in i["failure_types"]]
    only_code = [i for i in flagged if not i["failure_types_from_llm"] and i not in errors]
    summary: dict[str, Any] = {
        "judge_model": cfg.judge.model_name,
        "stratum": stratum,
        "n": n_items,
        "flagged_any_failure": len(flagged),
        "flagged_by_llm": len(llm_flagged),
        "flagged_only_by_code_rules": len(only_code),
        "judge_errors": len(errors),
        "failure_type_counts": dict(Counter(f for i in items for f in i["failure_types"]).most_common()),
        "llm_failure_type_counts": dict(
            Counter(f for i in items for f in (i["failure_types_from_llm"] or [])).most_common()
        ),
        "code_rule_additions": dict(Counter(f for i in items for f in i["added_by_code_rules"]).most_common()),
        "verdict_fail_counts": {
            d: sum(1 for i in items if i["verdicts"][d].startswith("fail")) for d in DIMS
        },
    }
    if stratum == "controlled":
        hits = sum(1 for i in items if i["injected_failure"] in i["failure_types"])
        summary["controlled_recall"] = round(hits / n_items, 4) if n_items else None
    return {"summary": summary, "items": items}
