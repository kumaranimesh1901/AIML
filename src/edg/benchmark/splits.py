"""Cluster-disjoint dev/heldout splitting and mined/random_control slicing for benchmark releases."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from edg.db.models import TestCase


def assign_slice(test_case: TestCase) -> str:
    """Assign 'random_control' if sourced from the random_control stratum, else 'mined'."""
    prov: dict[str, Any] = test_case.provenance_json or {}
    if prov.get("stratum") == "random_control":
        return "random_control"
    return "mined"


def split_test_cases_by_cluster(
    test_cases: list[TestCase],
    *,
    heldout_ratio: float = 0.5,
    seed: int = 42,
) -> dict[str, tuple[str, str]]:
    """Split test cases into (slice, split) such that NO cluster spans dev and heldout.

    Returns:
        Mapping `test_case_id -> (slice, split)` where slice in {'mined', 'random_control'}
        and split in {'dev', 'heldout'}.
    """
    if not test_cases:
        return {}

    by_cluster: dict[int, list[TestCase]] = defaultdict(list)
    for tc in test_cases:
        cid = int(tc.cluster_id) if tc.cluster_id is not None else -1
        by_cluster[cid].append(tc)

    rng = random.Random(seed)
    cids = sorted(by_cluster.keys())
    rng.shuffle(cids)

    total = len(test_cases)
    target_heldout = round(total * heldout_ratio)
    dev_count = 0
    heldout_count = 0

    assignments: dict[str, tuple[str, str]] = {}
    dev_clusters: set[int] = set()
    heldout_clusters: set[int] = set()

    for cid in cids:
        group = by_cluster[cid]
        # Assign the entire cluster to dev or heldout to keep ratio balanced and clusters disjoint
        if len(cids) == 1:
            split_name = "heldout"
        elif heldout_count < target_heldout and (
            heldout_count <= dev_count or dev_count >= (total - target_heldout)
        ):
            split_name = "heldout"
        else:
            split_name = "dev"

        if split_name == "heldout":
            heldout_count += len(group)
            heldout_clusters.add(cid)
        else:
            dev_count += len(group)
            dev_clusters.add(cid)

        for tc in group:
            assignments[tc.test_case_id] = (assign_slice(tc), split_name)

    assert dev_clusters.isdisjoint(heldout_clusters), "Cluster leakage between dev and heldout!"
    return assignments
