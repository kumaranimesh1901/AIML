"""Three-part weighted judge sampler: random_control, cluster-stratified, and priority top-k."""

from __future__ import annotations

import random
import uuid
from collections import defaultdict
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from edg.config import AppConfig
from edg.db.models import ClusterMembership, JudgeSample, PriorityScore
from edg.logging import get_logger
from edg.priority.scorer import _get_latest_clustering_run_id, compute_and_store_priority_scores

logger = get_logger(__name__)


def build_judge_sample(
    session: Session,
    config: AppConfig,
    *,
    run_id: str,
    clustering_run_id: str | None = None,
) -> dict[str, Any]:
    """Build the three-part judge sample (random_control, cluster, priority) with sampling weights."""
    c_run_id = clustering_run_id or _get_latest_clustering_run_id(session)

    rows = session.execute(
        select(
            PriorityScore.interaction_id,
            PriorityScore.rank,
            PriorityScore.total_score,
            ClusterMembership.cluster_id,
        )
        .join(
            ClusterMembership,
            (ClusterMembership.interaction_id == PriorityScore.interaction_id)
            & (ClusterMembership.clustering_run_id == c_run_id),
        )
        .order_by(PriorityScore.interaction_id)
    ).all()

    if not rows:
        raise ValueError("No priority_scores found; run priority scoring first.")

    n_pop = len(rows)
    target_total = min(config.priority.judge_sample_size, n_pop)

    n_random = round(target_total * config.priority.random_control_share)
    n_stratified = round(target_total * config.priority.stratified_share)
    n_priority = max(0, target_total - n_random - n_stratified)

    # 1. random_control: true uniform random sample without replacement from full population
    rng_random = random.Random(config.seed)
    all_ids_sorted = [r.interaction_id for r in rows]
    random_ids = rng_random.sample(all_ids_sorted, k=min(n_random, n_pop))
    random_weight = round(n_pop / len(random_ids), 4) if random_ids else 1.0

    selected: dict[str, tuple[str, float]] = {
        iid: ("random_control", random_weight) for iid in random_ids
    }

    # 2. cluster (stratified): equal quota per cluster (noise -1 treated as one stratum)
    rng_strat = random.Random(config.seed + 1)
    by_cluster_all: dict[int, list[str]] = defaultdict(list)
    by_cluster_rem: dict[int, list[str]] = defaultdict(list)
    for r in rows:
        cid = int(r.cluster_id)
        by_cluster_all[cid].append(r.interaction_id)
        if r.interaction_id not in selected:
            by_cluster_rem[cid].append(r.interaction_id)

    for cid in by_cluster_rem:
        rng_strat.shuffle(by_cluster_rem[cid])

    cluster_ids_sorted = sorted(by_cluster_rem.keys())
    strat_picked_by_cluster: dict[int, list[str]] = defaultdict(list)
    strat_total = 0

    # Round-robin across strata so every cluster gets equal representation
    while strat_total < n_stratified:
        made_progress = False
        for cid in cluster_ids_sorted:
            if strat_total >= n_stratified:
                break
            if by_cluster_rem[cid]:
                picked = by_cluster_rem[cid].pop(0)
                strat_picked_by_cluster[cid].append(picked)
                strat_total += 1
                made_progress = True
        if not made_progress:
            break

    for cid, picked_list in strat_picked_by_cluster.items():
        stratum_pop = len(by_cluster_all[cid])
        w = round(stratum_pop / len(picked_list), 4)
        for iid in picked_list:
            selected[iid] = ("cluster", w)

    # 3. priority: top-ranked remaining items not already chosen
    remaining_ranked = sorted(
        [r for r in rows if r.interaction_id not in selected],
        key=lambda r: (r.rank, r.interaction_id),
    )
    priority_picks = remaining_ranked[:n_priority]
    priority_weight = (
        round(len(remaining_ranked) / len(priority_picks), 4) if priority_picks else 1.0
    )
    for r in priority_picks:
        selected[r.interaction_id] = ("priority", priority_weight)

    # Persist into judge_samples
    session.execute(delete(JudgeSample))
    stratum_counts: dict[str, int] = {"random_control": 0, "cluster": 0, "priority": 0}
    for iid in sorted(selected.keys()):
        stratum, weight = selected[iid]
        stratum_counts[stratum] += 1
        session.add(
            JudgeSample(
                sample_id=f"SMP-{uuid.uuid5(uuid.NAMESPACE_DNS, f'{run_id}:{iid}').hex[:12]}",
                interaction_id=iid,
                stratum=stratum,
                sampling_weight=weight,
                run_id=run_id,
            )
        )
    session.flush()

    stats = {
        "population_size": n_pop,
        "sampled_total": len(selected),
        "stratum_counts": stratum_counts,
        "random_control_weight": random_weight,
    }
    logger.info("judge_sample_built", extra=stats)
    return stats


def run_priority_and_sampling(
    session: Session,
    config: AppConfig,
    *,
    run_id: str,
    clustering_run_id: str | None = None,
) -> dict[str, Any]:
    """Run priority scoring followed by three-part judge sampling."""
    score_stats = compute_and_store_priority_scores(
        session, config, run_id=run_id, clustering_run_id=clustering_run_id
    )
    sample_stats = build_judge_sample(
        session,
        config,
        run_id=run_id,
        clustering_run_id=score_stats["clustering_run_id"],
    )
    return {**score_stats, **sample_stats}
