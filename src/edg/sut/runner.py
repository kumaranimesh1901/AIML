"""End-to-end SUT execution runner: binds queries, runs assistant, logs interactions_raw, injects controlled tier."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from edg.config import AppConfig
from edg.db.models import InteractionRaw, PolicyDocument, StateSnapshot
from edg.llm.cache import LLMCache
from edg.llm.client import BaseLLMClient, create_llm_client
from edg.logging import get_logger
from edg.queries.binder import BoundQuery, bind_queries_to_snapshots
from edg.queries.bitext_loader import load_bitext_queries
from edg.queries.hard_query_generator import generate_hard_queries
from edg.sut.assistant import CustomerSupportAssistant
from edg.sut.failure_injector import inject_controlled_failures
from edg.sut.tools import SnapshotToolBox
from edg.sut.versions import register_model_versions

logger = get_logger(__name__)


def _preload_policies(session: Session, policy_version: str) -> list[tuple[str, str, str, Any]]:
    import numpy as np

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


def run_sut_pipeline(
    session: Session,
    config: AppConfig,
    *,
    run_id: str,
    version_name: str = "v1",
    llm_client: BaseLLMClient | None = None,
    prefer_hf_bitext: bool = True,
    inject_failures: bool = True,
) -> dict[str, Any]:
    """Run query binding, SUT answering, raw logging, and controlled failure injection."""
    if version_name not in config.sut.versions:
        raise ValueError(f"Unknown SUT version '{version_name}'. Available: {list(config.sut.versions)}")

    register_model_versions(session, config, run_id=run_id)
    ver_cfg = config.sut.versions[version_name]

    client = llm_client
    if client is None:
        cache = LLMCache(engine=session.get_bind()) if config.llm.cache_enabled else None
        effective_provider = config.llm.provider if config.llm.provider == "fake" else ver_cfg.provider
        client = create_llm_client(
            config.llm,
            provider=effective_provider,
            default_model=ver_cfg.base_model,
            cache=cache,
        )

    bitext_raw = load_bitext_queries(
        sample_size=config.queries.bitext_sample_size,
        seed=config.seed,
        prefer_hf=prefer_hf_bitext,
    )
    bound_bitext = bind_queries_to_snapshots(session, bitext_raw, seed=config.seed, id_prefix="QRY")
    bound_hard = generate_hard_queries(
        session,
        count=config.queries.hard_query_count,
        seed=config.seed,
        llm_client=client,
        generator_model=config.queries.hard_query_generator_model,
    )
    all_queries: list[BoundQuery] = [*bound_bitext, *bound_hard]

    snapshots_by_id = {
        s.snapshot_id: s for s in session.scalars(select(StateSnapshot)).all()
    }
    policy_cache = _preload_policies(session, config.world.policy_version)

    assistant = CustomerSupportAssistant(
        version_config=ver_cfg,
        llm_client=client,
        max_tool_calls=config.sut.max_tool_calls,
    )

    natural_inserted = 0
    errors = 0
    source_tag_counts: dict[str, int] = {}

    for bq in all_queries:
        interaction_id = f"INT-{version_name}-{bq.query_id}"
        if session.get(InteractionRaw, interaction_id) is not None:
            continue

        snap = snapshots_by_id[bq.snapshot_id]
        toolbox = SnapshotToolBox(
            snap,
            session=session,
            retrieval_enabled=ver_cfg.retrieval_enabled,
            policy_cache=policy_cache,
        )
        result = assistant.answer(query=bq.user_query, order_id=bq.order_id, toolbox=toolbox)
        if result.error:
            errors += 1

        ts = datetime.now(UTC)
        session.add(
            InteractionRaw(
                interaction_id=interaction_id,
                conversation_id=f"conv_{interaction_id}",
                turn_index=0,
                user_query=bq.user_query,
                ai_response=result.answer,
                model_version=ver_cfg.name,
                timestamp=ts,
                source_tag=bq.source_tag,
                intent_label=bq.intent,
                snapshot_id=bq.snapshot_id,
                reference_answer=bq.reference_answer,
                injected_failure=None,
                raw_json={
                    "query_id": bq.query_id,
                    "order_id": bq.order_id,
                    "category": bq.category,
                    "generator_model": bq.generator_model,
                    "hard_state_tags": bq.hard_state_tags,
                    "tool_calls": result.tool_calls,
                    "latency_ms": round(result.latency_ms, 2),
                    "error": result.error,
                },
                run_id=run_id,
            )
        )
        natural_inserted += 1
        source_tag_counts[bq.source_tag] = source_tag_counts.get(bq.source_tag, 0) + 1

    session.flush()

    controlled_stats: dict[str, Any] = {"controlled_injected": 0, "by_failure_type": {}}
    if inject_failures:
        controlled_stats = inject_controlled_failures(
            session,
            run_id=run_id,
            model_version=ver_cfg.name,
            injection_rate=config.sut.failure_injection_rate,
            seed=config.seed,
            llm_client=client,
            rewrite_model=ver_cfg.base_model,
        )

    counts: dict[str, Any] = {
        "model_version": ver_cfg.name,
        "bitext_queries": len(bound_bitext),
        "hard_queries": len(bound_hard),
        "natural_inserted": natural_inserted,
        "errors": errors,
        "source_tag_counts": source_tag_counts,
        **controlled_stats,
    }
    logger.info("sut_pipeline_completed", extra=counts)
    return counts
