"""HDBSCAN clustering with DBCV (relative_validity) grid search and GLOSH outlier scores."""

from __future__ import annotations

from dataclasses import dataclass

import hdbscan
import numpy as np


@dataclass(frozen=True)
class HDBSCANResult:
    labels: np.ndarray
    probabilities: np.ndarray
    glosh_scores: np.ndarray
    dbcv: float
    n_clusters: int
    noise_share: float
    min_cluster_size: int
    min_samples: int


def fit_hdbscan(
    vectors_nd: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: int,
) -> HDBSCANResult:
    """Fit HDBSCAN on a high-dimensional (>2D) embedding/UMAP space.

    Explicitly rejects 2D plot projections to enforce clustering in the 10D space.
    """
    if vectors_nd.ndim != 2:
        raise ValueError(f"Expected 2D array (n_samples, n_features), got shape {vectors_nd.shape}")
    if vectors_nd.shape[1] == 2:
        raise ValueError(
            "HDBSCAN must never run on the 2D plot projection; pass the 10D clustering space."
        )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        gen_min_span_tree=True,
    )
    clusterer.fit(vectors_nd.astype(np.float64))

    labels = np.asarray(clusterer.labels_, dtype=np.int32)
    probs = np.asarray(clusterer.probabilities_, dtype=np.float32)
    glosh = np.nan_to_num(np.asarray(clusterer.outlier_scores_, dtype=np.float32), nan=0.0)
    dbcv = float(getattr(clusterer, "relative_validity_", 0.0) or 0.0)
    if np.isnan(dbcv):
        dbcv = 0.0

    n_clusters = len({int(lbl) for lbl in labels if int(lbl) != -1})
    noise_share = float(np.mean(labels == -1)) if len(labels) > 0 else 0.0

    return HDBSCANResult(
        labels=labels,
        probabilities=probs,
        glosh_scores=glosh,
        dbcv=dbcv,
        n_clusters=n_clusters,
        noise_share=noise_share,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )


def grid_search_hdbscan(
    vectors_nd: np.ndarray,
    *,
    min_cluster_sizes: list[int],
    min_samples_list: list[int],
    max_noise_share: float = 0.35,
) -> HDBSCANResult:
    """Grid-search HDBSCAN hyperparameters, selecting by DBCV with a soft noise_share constraint."""
    if vectors_nd.shape[1] == 2:
        raise ValueError(
            "HDBSCAN grid search must never run on the 2D plot projection."
        )

    n_samples = vectors_nd.shape[0]
    candidates: list[HDBSCANResult] = []
    for mcs in min_cluster_sizes:
        if mcs >= n_samples:
            continue
        for ms in min_samples_list:
            if ms > mcs:
                continue
            res = fit_hdbscan(vectors_nd, min_cluster_size=mcs, min_samples=ms)
            candidates.append(res)

    if not candidates:
        # Fallback for very small fixtures
        fallback_mcs = max(2, min(5, n_samples // 3))
        fallback_ms = max(2, min(3, fallback_mcs))
        return fit_hdbscan(vectors_nd, min_cluster_size=fallback_mcs, min_samples=fallback_ms)

    # Prefer candidates with >= 1 cluster and noise_share <= max_noise_share, maximizing DBCV
    feasible = [
        c for c in candidates if c.n_clusters >= 1 and c.noise_share <= max_noise_share
    ]
    if feasible:
        return max(feasible, key=lambda c: (c.dbcv, c.n_clusters, -c.noise_share))

    non_empty = [c for c in candidates if c.n_clusters >= 1]
    if non_empty:
        return max(non_empty, key=lambda c: (c.dbcv, -c.noise_share))

    return max(candidates, key=lambda c: c.dbcv)
