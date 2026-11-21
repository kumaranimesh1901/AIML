"""Within-cluster test case deduplication (cosine >= 0.92 on input_query + identical checklist)."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np

from edg.world.policies import hash_embed_texts


def _checklist_signature(checklist_json: dict[str, Any]) -> str:
    normalized = {
        "must_do": sorted(str(x).strip().lower() for x in (checklist_json.get("must_do") or [])),
        "must_not_do": sorted(str(x).strip().lower() for x in (checklist_json.get("must_not_do") or [])),
    }
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def dedupe_test_cases_within_cluster(
    candidates: list[dict[str, Any]],
    *,
    cosine_threshold: float = 0.92,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Filter duplicate candidate test cases within the same cluster.

    Two candidates in the same cluster are considered duplicates when their `input_query`
    embedding cosine similarity is >= `cosine_threshold` AND their normalized checklists are identical.
    """
    if not candidates:
        return [], 0

    embedder = embed_fn or (lambda texts: hash_embed_texts(texts, dim=384))
    by_cluster: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cand in candidates:
        cid = int(cand["cluster_id"]) if cand.get("cluster_id") is not None else -1
        by_cluster[cid].append(cand)

    kept: list[dict[str, Any]] = []
    merged_count = 0

    for _cid, group in sorted(by_cluster.items()):
        queries = [c["input_query"] for c in group]
        vecs = embedder(queries)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        vecs = vecs / norms

        kept_in_cluster: list[tuple[np.ndarray, str]] = []
        for idx, cand in enumerate(group):
            v = vecs[idx]
            sig = _checklist_signature(cand["expected_checklist_json"])
            is_dup = False
            for k_vec, k_sig in kept_in_cluster:
                if sig == k_sig and float(np.dot(v, k_vec)) >= cosine_threshold:
                    is_dup = True
                    break
            if is_dup:
                merged_count += 1
            else:
                kept_in_cluster.append((v, sig))
                kept.append(cand)

    return kept, merged_count
