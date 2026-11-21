"""Clustering quality metrics (DBCV, ARI, intent purity, silhouette strictly on non-noise points)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import silhouette_score


def compute_silhouette_non_noise(vectors_nd: np.ndarray, labels: np.ndarray) -> float | None:
    """Compute silhouette score strictly on non-noise points (label != -1)."""
    mask = labels != -1
    non_noise_labels = labels[mask]
    unique_clusters = set(int(x) for x in non_noise_labels)
    if len(unique_clusters) < 2 or int(mask.sum()) <= len(unique_clusters):
        return None
    return round(float(silhouette_score(vectors_nd[mask], non_noise_labels)), 4)


def summarize_clustering_metrics(
    *,
    vectors_nd: np.ndarray,
    labels: np.ndarray,
    dbcv: float | None,
    ari_vs_other_seeds: float | None,
    cluster_purities: list[float],
    candidate_new_intents: int,
) -> dict[str, Any]:
    """Assemble full clustering metrics dictionary."""
    n_total = len(labels)
    n_noise = int(np.sum(labels == -1))
    n_clusters = len({int(x) for x in labels if int(x) != -1})
    noise_share = round(n_noise / n_total, 4) if n_total > 0 else 0.0
    sil = compute_silhouette_non_noise(vectors_nd, labels)
    mean_purity = round(float(np.mean(cluster_purities)), 4) if cluster_purities else None

    return {
        "n_clusters": n_clusters,
        "noise_share": noise_share,
        "dbcv": round(dbcv, 4) if dbcv is not None else None,
        "ari_vs_other_seeds": round(ari_vs_other_seeds, 4) if ari_vs_other_seeds is not None else None,
        "mean_intent_purity": mean_purity,
        "candidate_new_intents": candidate_new_intents,
        "silhouette_non_noise": sil,
    }
