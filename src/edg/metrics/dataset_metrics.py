"""Benchmark dataset quality metrics (coverage, slices, splits, trust tiers, approval rate)."""

from __future__ import annotations

from collections import Counter
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from edg.db.models import BenchmarkRelease, ReleaseItem, TestCase


def compute_dataset_metrics(session: Session, release_id: str) -> dict[str, Any]:
    """Compute coverage and composition metrics for a frozen benchmark release."""
    release = session.get(BenchmarkRelease, release_id)
    if release is None:
        raise ValueError(f"Release '{release_id}' not found.")

    rows = session.execute(
        select(ReleaseItem, TestCase)
        .join(TestCase, TestCase.test_case_id == ReleaseItem.test_case_id)
        .where(ReleaseItem.release_id == release_id)
    ).all()

    by_slice: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    by_intent: Counter[str] = Counter()
    by_severity: Counter[str] = Counter()
    by_failure_type: Counter[str] = Counter()
    clusters: set[int] = set()

    for ri, tc in rows:
        by_slice[ri.slice] += 1
        by_split[ri.split] += 1
        by_intent[tc.intent or "unknown"] += 1
        by_severity[tc.severity] += 1
        if tc.cluster_id is not None:
            clusters.add(int(tc.cluster_id))
        for ft in tc.original_failure_types or []:
            by_failure_type[ft] += 1

    all_tcs = list(session.scalars(select(TestCase)).all())
    approved_total = sum(1 for t in all_tcs if t.status == "approved")
    human_gold = sum(1 for t in all_tcs if t.trust_tier == "gold")

    return {
        "release_id": release_id,
        "content_hash": release.content_hash,
        "judge_version": release.judge_version,
        "total_cases": len(rows),
        "by_slice": dict(by_slice),
        "by_split": dict(by_split),
        "unique_intents": len(by_intent),
        "unique_clusters": len(clusters),
        "by_severity": dict(by_severity),
        "by_failure_type": dict(by_failure_type),
        "pool_total_cases": len(all_tcs),
        "pool_approval_rate": round(approved_total / len(all_tcs), 4) if all_tcs else 0.0,
        "pool_gold_count": human_gold,
    }
