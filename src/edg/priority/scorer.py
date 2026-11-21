"""Weighted priority scoring and ranking persisted to priority_scores."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from edg.config import AppConfig
from edg.db.models import (
    ClusteringRun,
    ClusterMembership,
    Embedding,
    InteractionClean,
    InteractionRaw,
    PriorityScore,
    StateSnapshot,
)
from edg.logging import get_logger
from edg.priority.signals import (
    compute_multi_intent_signal,
    compute_novelty_signal,
    compute_policy_boundary_signal,
    compute_response_drift_signal,
    compute_small_cluster_signal,
    compute_state_conflict_signal,
)
from edg.sut.failure_injector import CONTROLLED_SOURCE_TAG

logger = get_logger(__name__)

_DEFAULT_WEIGHTS: dict[str, float] = {
    "novelty": 0.20,
    "small_cluster": 0.15,
    "multi_intent": 0.15,
    "state_conflict": 0.20,
    "policy_boundary": 0.15,
    "response_drift": 0.15,
}


def _get_latest_clustering_run_id(session: Session) -> str:
    latest = session.scalars(
        select(ClusteringRun).order_by(ClusteringRun.created_at.desc()).limit(1)
    ).first()
    if latest is None:
        raise ValueError("No clustering_runs found in database; run `edg cluster run` first.")
    return latest.clustering_run_id


def compute_and_store_priority_scores(
    session: Session,
    config: AppConfig,
    *,
    run_id: str,
    clustering_run_id: str | None = None,
) -> dict[str, Any]:
    """Compute all 6 normalized signals, weighted total_score, and rank for natural-tier rows."""
    c_run_id = clustering_run_id or _get_latest_clustering_run_id(session)
    model_name = config.embeddings.model_name

    rows = session.execute(
        select(
            ClusterMembership.interaction_id,
            ClusterMembership.cluster_id,
            ClusterMembership.glosh_score,
            InteractionClean.clean_query,
            StateSnapshot.snapshot_json,
            StateSnapshot.hard_state_tags,
            Embedding.vector,
        )
        .join(InteractionClean, InteractionClean.interaction_id == ClusterMembership.interaction_id)
        .join(InteractionRaw, InteractionRaw.interaction_id == ClusterMembership.interaction_id)
        .join(StateSnapshot, StateSnapshot.snapshot_id == InteractionRaw.snapshot_id)
        .join(
            Embedding,
            (Embedding.interaction_id == ClusterMembership.interaction_id)
            & (Embedding.kind == "response")
            & (Embedding.model_name == model_name),
        )
        .where(
            ClusterMembership.clustering_run_id == c_run_id,
            InteractionClean.valid.is_(True),
            InteractionRaw.source_tag != CONTROLLED_SOURCE_TAG,
            InteractionRaw.injected_failure.is_(None),
        )
        .order_by(ClusterMembership.interaction_id)
    ).all()

    if not rows:
        raise ValueError("No eligible clustered interactions found for priority scoring.")

    # Precompute GLOSH 90th percentile, cluster sizes, and cluster mean response vectors
    glosh_vals = np.asarray([float(r.glosh_score) for r in rows], dtype=np.float32)
    p90_glosh = float(np.percentile(glosh_vals, 90)) if len(glosh_vals) > 0 else 0.0

    cluster_sizes: dict[int, int] = defaultdict(int)
    cluster_resp_vecs: dict[int, list[np.ndarray]] = defaultdict(list)
    for r in rows:
        cid = int(r.cluster_id)
        cluster_sizes[cid] += 1
        cluster_resp_vecs[cid].append(np.asarray(r.vector, dtype=np.float32))

    non_noise_sizes = [sz for cid, sz in cluster_sizes.items() if cid != -1 and sz > 1]
    max_inv_log = max((1.0 / np.log1p(sz) for sz in non_noise_sizes), default=1.0)

    cluster_mean_resp: dict[int, np.ndarray] = {}
    for cid, vecs in cluster_resp_vecs.items():
        mean_v = np.mean(np.vstack(vecs), axis=0)
        norm = float(np.linalg.norm(mean_v))
        cluster_mean_resp[cid] = mean_v / norm if norm > 0 else mean_v

    weights = dict(_DEFAULT_WEIGHTS)
    weights.update(config.priority.weights or {})
    weight_sum = sum(weights.values()) or 1.0

    scored_items: list[dict[str, Any]] = []
    for r in rows:
        cid = int(r.cluster_id)
        novelty = compute_novelty_signal(cid, float(r.glosh_score), p90_glosh)
        small_cluster = compute_small_cluster_signal(cid, cluster_sizes[cid], float(max_inv_log))
        multi_intent = compute_multi_intent_signal(r.clean_query)
        state_conflict = compute_state_conflict_signal(
            r.clean_query, r.snapshot_json, list(r.hard_state_tags or [])
        )
        policy_boundary = compute_policy_boundary_signal(
            r.snapshot_json, list(r.hard_state_tags or [])
        )
        response_drift = compute_response_drift_signal(
            np.asarray(r.vector, dtype=np.float32), cluster_mean_resp.get(cid)
        )

        total_score = (
            weights["novelty"] * novelty
            + weights["small_cluster"] * small_cluster
            + weights["multi_intent"] * multi_intent
            + weights["state_conflict"] * state_conflict
            + weights["policy_boundary"] * policy_boundary
            + weights["response_drift"] * response_drift
        ) / weight_sum
        total_score = float(max(0.0, min(1.0, total_score)))

        scored_items.append(
            {
                "interaction_id": r.interaction_id,
                "novelty": round(novelty, 4),
                "small_cluster": round(small_cluster, 4),
                "multi_intent": round(multi_intent, 4),
                "state_conflict": round(state_conflict, 4),
                "policy_boundary": round(policy_boundary, 4),
                "response_drift": round(response_drift, 4),
                "total_score": round(total_score, 4),
            }
        )

    # Sort descending by total_score, tie-break deterministically by interaction_id
    scored_items.sort(key=lambda d: (-d["total_score"], d["interaction_id"]))

    # Replace existing priority scores cleanly
    session.execute(delete(PriorityScore))
    for rank, item in enumerate(scored_items, start=1):
        session.add(
            PriorityScore(
                interaction_id=item["interaction_id"],
                novelty=item["novelty"],
                small_cluster=item["small_cluster"],
                multi_intent=item["multi_intent"],
                state_conflict=item["state_conflict"],
                policy_boundary=item["policy_boundary"],
                response_drift=item["response_drift"],
                total_score=item["total_score"],
                rank=rank,
                run_id=run_id,
            )
        )
    session.flush()

    totals = [d["total_score"] for d in scored_items]
    stats = {
        "clustering_run_id": c_run_id,
        "scored_count": len(scored_items),
        "min_score": min(totals),
        "mean_score": round(float(np.mean(totals)), 4),
        "max_score": max(totals),
    }
    logger.info("priority_scores_computed", extra=stats)
    return stats
