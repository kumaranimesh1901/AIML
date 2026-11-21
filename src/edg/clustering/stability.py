"""Multi-seed clustering stability via pairwise Adjusted Rand Index (ARI) on non-noise points."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.metrics import adjusted_rand_score

from edg.clustering.hdbscan_cluster import fit_hdbscan
from edg.clustering.reducer import reduce_for_clustering
from edg.config import ClusteringConfig


def pairwise_non_noise_ari(labels_a: np.ndarray, labels_b: np.ndarray) -> float | None:
    """Compute ARI strictly on points that are non-noise (label != -1) in both runs."""
    mask = (labels_a != -1) & (labels_b != -1)
    if int(mask.sum()) < 2:
        return None
    return float(adjusted_rand_score(labels_a[mask], labels_b[mask]))


def compute_stability_ari(
    vectors: np.ndarray,
    config: ClusteringConfig,
    *,
    min_cluster_size: int,
    min_samples: int,
    seeds: list[int] | None = None,
) -> float | None:
    """Rerun 10D UMAP + HDBSCAN across seeds and return mean pairwise non-noise ARI."""
    run_seeds = seeds or config.stability_seeds
    if len(run_seeds) < 2:
        return None

    label_runs: list[np.ndarray] = []
    for seed in run_seeds:
        reduced_10d = reduce_for_clustering(vectors, config, seed=seed)
        res = fit_hdbscan(
            reduced_10d,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
        label_runs.append(res.labels)

    scores: list[float] = []
    for lbl_a, lbl_b in combinations(label_runs, 2):
        ari = pairwise_non_noise_ari(lbl_a, lbl_b)
        if ari is not None:
            scores.append(ari)

    if not scores:
        return None
    return round(float(np.mean(scores)), 4)
