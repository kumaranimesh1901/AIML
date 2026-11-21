"""Exact and near-duplicate grouping within the same intent."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable

import numpy as np

from edg.world.policies import hash_embed_texts


def _exact_hash(intent: str, text: str) -> str:
    return hashlib.sha256(f"{intent}::{text}".encode()).hexdigest()[:16]


def assign_duplicate_groups(
    items: list[tuple[str, str, str]],
    *,
    cosine_threshold: float = 0.95,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    group_prefix: str = "DUP",
) -> dict[str, tuple[str, int, bool]]:
    """Group exact and near-duplicate queries within the same intent.

    Args:
        items: list of `(interaction_id, intent_label, clean_query)`.
        cosine_threshold: minimum cosine similarity for near-duplicate grouping.
        group_prefix: prefix for generated group ids (keeps separately-deduped tiers disjoint).
        embed_fn: optional embedding function returning L2-normalized `(N, D)` float32 array.

    Returns:
        Mapping `interaction_id -> (dup_group_id, dup_count, is_representative)`.
    """
    if not items:
        return {}

    by_intent: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for interaction_id, intent, query in items:
        by_intent[intent or "unknown"].append((interaction_id, query))

    embedder = embed_fn or (lambda texts: hash_embed_texts(texts, dim=384))
    results: dict[str, tuple[str, int, bool]] = {}

    for intent, group_items in sorted(by_intent.items()):
        # Step 1: exact dedupe by hash within intent
        exact_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for iid, q in group_items:
            h = _exact_hash(intent, q)
            exact_groups[h].append((iid, q))

        # Keep one representative text per exact group for near-dedupe
        rep_hashes = sorted(exact_groups.keys())
        rep_texts = [exact_groups[h][0][1] for h in rep_hashes]

        # Step 2: near-duplicate clustering over exact representatives using cosine similarity
        vecs = embedder(rep_texts)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        vecs = vecs / norms

        # Greedy leader clustering: deterministic order by rep_hashes
        clusters: list[tuple[str, np.ndarray, list[str]]] = []
        for idx, h in enumerate(rep_hashes):
            v = vecs[idx]
            matched_idx: int | None = None
            for c_idx, (_gid, leader_v, _members) in enumerate(clusters):
                if float(np.dot(v, leader_v)) >= cosine_threshold:
                    matched_idx = c_idx
                    break
            if matched_idx is not None:
                clusters[matched_idx][2].append(h)
            else:
                gid = f"{group_prefix}-{intent[:16]}-{h[:10]}"
                clusters.append((gid, v, [h]))

        # Assign dup_group_id, dup_count, is_representative
        for gid, _leader_v, member_hashes in clusters:
            member_ids: list[str] = []
            for h in member_hashes:
                member_ids.extend(iid for iid, _q in exact_groups[h])
            total_count = len(member_ids)
            for pos, iid in enumerate(member_ids):
                is_rep = pos == 0
                results[iid] = (gid, total_count if is_rep else 1, is_rep)

    return results
