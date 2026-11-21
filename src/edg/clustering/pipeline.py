"""End-to-end clustering pipeline: 10D UMAP + HDBSCAN on representatives, 2D plot, duplicate propagation."""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from edg.clustering.describe import describe_cluster, select_central_examples
from edg.clustering.hdbscan_cluster import grid_search_hdbscan
from edg.clustering.intent_mapping import map_cluster_intent
from edg.clustering.reducer import reduce_for_clustering, reduce_for_plot
from edg.clustering.stability import compute_stability_ari
from edg.config import AppConfig
from edg.db.models import (
    Cluster,
    ClusteringRun,
    ClusterMembership,
    Embedding,
    InteractionClean,
    InteractionRaw,
    StateSnapshot,
)
from edg.llm.client import BaseLLMClient
from edg.logging import get_logger
from edg.metrics.clustering_metrics import summarize_clustering_metrics
from edg.sut.failure_injector import CONTROLLED_SOURCE_TAG

logger = get_logger(__name__)


def run_clustering_pipeline(
    session: Session,
    config: AppConfig,
    *,
    run_id: str,
    model_name: str | None = None,
    llm_client: BaseLLMClient | None = None,
) -> dict[str, Any]:
    """Run UMAP + HDBSCAN on representative natural-tier query embeddings and propagate to duplicates."""
    resolved_model = model_name or config.embeddings.model_name

    # Fetch all valid natural-tier cleaned rows (excluding controlled failure tier)
    rows = session.execute(
        select(
            InteractionClean.interaction_id,
            InteractionClean.clean_query,
            InteractionClean.dup_group_id,
            InteractionClean.is_representative,
            InteractionRaw.intent_label,
            InteractionRaw.source_tag,
            StateSnapshot.hard_state_tags,
            Embedding.vector,
        )
        .join(InteractionRaw, InteractionRaw.interaction_id == InteractionClean.interaction_id)
        .join(StateSnapshot, StateSnapshot.snapshot_id == InteractionRaw.snapshot_id)
        .join(
            Embedding,
            (Embedding.interaction_id == InteractionClean.interaction_id)
            & (Embedding.kind == "query")
            & (Embedding.model_name == resolved_model),
        )
        .where(
            InteractionClean.valid.is_(True),
            InteractionRaw.source_tag != CONTROLLED_SOURCE_TAG,
            InteractionRaw.injected_failure.is_(None),
        )
        .order_by(InteractionClean.interaction_id)
    ).all()

    if not rows:
        raise ValueError(
            f"No valid query embeddings found for model '{resolved_model}'. Run `edg embed run` first."
        )

    reps = [r for r in rows if r.is_representative]
    dups = [r for r in rows if not r.is_representative]

    rep_vectors = np.asarray([r.vector for r in reps], dtype=np.float32)

    # 1. Separate 10D clustering space and 2D plot space
    reduced_10d = reduce_for_clustering(rep_vectors, config.clustering, seed=config.seed)
    reduced_2d = reduce_for_plot(rep_vectors, config.clustering, seed=config.seed)

    # 2. HDBSCAN grid search strictly on 10D space
    best = grid_search_hdbscan(
        reduced_10d,
        min_cluster_sizes=config.clustering.hdbscan_min_cluster_sizes,
        min_samples_list=config.clustering.hdbscan_min_samples,
        max_noise_share=config.clustering.max_noise_share,
    )

    # 3. Multi-seed stability ARI
    ari = compute_stability_ari(
        rep_vectors,
        config.clustering,
        min_cluster_size=best.min_cluster_size,
        min_samples=best.min_samples,
        seeds=config.clustering.stability_seeds,
    )

    clustering_run_id = f"clrun_{uuid.uuid4().hex[:12]}"
    session.add(
        ClusteringRun(
            clustering_run_id=clustering_run_id,
            umap_params={
                "n_components": config.clustering.umap_n_components,
                "n_neighbors": config.clustering.umap_n_neighbors,
                "min_dist": config.clustering.umap_min_dist,
                "metric": config.clustering.umap_metric,
            },
            hdbscan_params={
                "min_cluster_size": best.min_cluster_size,
                "min_samples": best.min_samples,
            },
            seed=config.seed,
            dbcv=round(best.dbcv, 4),
            n_clusters=best.n_clusters,
            noise_share=round(best.noise_share, 4),
            ari_vs_other_seeds=ari,
            run_id=run_id,
        )
    )
    session.flush()

    # Group representative indices by cluster_id
    members_by_cluster: dict[int, list[int]] = defaultdict(list)
    rep_membership_by_dup_group: dict[str, tuple[int, float, float, float, float]] = {}

    for idx, rep in enumerate(reps):
        cid = int(best.labels[idx])
        prob = float(best.probabilities[idx])
        glosh = float(best.glosh_scores[idx])
        x2d = float(reduced_2d[idx, 0])
        y2d = float(reduced_2d[idx, 1])

        members_by_cluster[cid].append(idx)
        if rep.dup_group_id:
            rep_membership_by_dup_group[rep.dup_group_id] = (cid, prob, glosh, x2d, y2d)

        session.add(
            ClusterMembership(
                interaction_id=rep.interaction_id,
                clustering_run_id=clustering_run_id,
                cluster_id=cid,
                probability=prob,
                glosh_score=glosh,
                umap2d_x=x2d,
                umap2d_y=y2d,
                run_id=run_id,
            )
        )

    # Propagate representative's cluster membership to duplicates
    propagated_dups = 0
    for dup in dups:
        if dup.dup_group_id and dup.dup_group_id in rep_membership_by_dup_group:
            cid, prob, glosh, x2d, y2d = rep_membership_by_dup_group[dup.dup_group_id]
        else:
            cid, prob, glosh, x2d, y2d = -1, 0.0, 0.0, 0.0, 0.0
        session.add(
            ClusterMembership(
                interaction_id=dup.interaction_id,
                clustering_run_id=clustering_run_id,
                cluster_id=cid,
                probability=prob,
                glosh_score=glosh,
                umap2d_x=x2d,
                umap2d_y=y2d,
                run_id=run_id,
            )
        )
        propagated_dups += 1

    # Describe and intent-map every discovered cluster (excluding noise -1)
    purities: list[float] = []
    candidate_new_intents = 0
    for cid in sorted(members_by_cluster.keys()):
        if cid == -1:
            continue
        idxs = members_by_cluster[cid]
        c_queries = [reps[i].clean_query for i in idxs]
        c_vecs = rep_vectors[idxs]
        central = select_central_examples(c_queries, c_vecs, max_examples=15)
        desc = describe_cluster(central, llm_client=llm_client)

        c_intents = [reps[i].intent_label for i in idxs]
        c_hard_flags = [
            bool(reps[i].hard_state_tags) and reps[i].source_tag == "synthetic_natural"
            for i in idxs
        ]
        mapping = map_cluster_intent(
            c_intents,
            is_hard_query_flags=c_hard_flags,
            purity_threshold=config.clustering.intent_purity_threshold,
        )
        purities.append(mapping.purity)
        if mapping.is_new_intent:
            candidate_new_intents += 1

        session.add(
            Cluster(
                cluster_id=cid,
                clustering_run_id=clustering_run_id,
                size=len(idxs),
                label=desc.label,
                description=desc.description,
                mapped_intent=mapping.mapped_intent,
                is_new_intent=mapping.is_new_intent,
                central_examples_json=central,
                run_id=run_id,
            )
        )

    session.flush()

    metrics = summarize_clustering_metrics(
        vectors_nd=reduced_10d,
        labels=best.labels,
        dbcv=best.dbcv,
        ari_vs_other_seeds=ari,
        cluster_purities=purities,
        candidate_new_intents=candidate_new_intents,
    )
    result: dict[str, Any] = {
        "clustering_run_id": clustering_run_id,
        "representatives_clustered": len(reps),
        "duplicates_propagated": propagated_dups,
        "best_min_cluster_size": best.min_cluster_size,
        "best_min_samples": best.min_samples,
        **metrics,
    }
    logger.info("clustering_pipeline_completed", extra=result)
    return result
