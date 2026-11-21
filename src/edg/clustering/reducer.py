"""Separate UMAP projections for clustering (10D) and dashboard plotting (2D)."""

from __future__ import annotations

import warnings

import numpy as np
import umap

from edg.config import ClusteringConfig


def reduce_umap(
    vectors: np.ndarray,
    *,
    n_components: int,
    n_neighbors: int = 30,
    min_dist: float = 0.0,
    metric: str = "cosine",
    seed: int = 42,
) -> np.ndarray:
    """Run seeded UMAP reduction, safely clamping neighborhood size for small inputs."""
    n_samples = vectors.shape[0]
    if n_samples < 4:
        raise ValueError(f"UMAP requires at least 4 samples, got {n_samples}")

    eff_neighbors = max(2, min(n_neighbors, n_samples - 1))
    eff_components = min(n_components, n_samples - 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(
            n_components=eff_components,
            n_neighbors=eff_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=seed,
            n_jobs=1,
        )
        reduced = reducer.fit_transform(vectors)
    return np.asarray(reduced, dtype=np.float32)


def reduce_for_clustering(
    vectors: np.ndarray,
    config: ClusteringConfig,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Compute the 10D (configured) UMAP space used strictly for HDBSCAN clustering."""
    if config.umap_n_components <= 2:
        raise ValueError("Clustering UMAP space must have n_components > 2 (default 10)")
    return reduce_umap(
        vectors,
        n_components=config.umap_n_components,
        n_neighbors=config.umap_n_neighbors,
        min_dist=config.umap_min_dist,
        metric=config.umap_metric,
        seed=seed,
    )


def reduce_for_plot(
    vectors: np.ndarray,
    config: ClusteringConfig,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Compute the separate 2D UMAP projection used strictly for dashboard visualization."""
    return reduce_umap(
        vectors,
        n_components=2,
        n_neighbors=config.umap_n_neighbors,
        min_dist=config.umap2d_min_dist,
        metric=config.umap_metric,
        seed=seed,
    )
