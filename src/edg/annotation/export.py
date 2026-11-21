"""Export stratified cluster-disjoint judge_dev and judge_test annotation sheets."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from edg.config import AppConfig
from edg.db.models import (
    ClusteringRun,
    ClusterMembership,
    InteractionClean,
    InteractionRaw,
    JudgeSample,
    StateSnapshot,
)
from edg.logging import get_logger

logger = get_logger(__name__)

ANNOTATION_CSV_COLUMNS = [
    "interaction_id",
    "split",
    "cluster_id",
    "stratum",
    "user_query",
    "ai_response",
    "reference_answer",
    "snapshot_json",
    "annotator_id",
    "evidence_status",
    "correctness_verdict",
    "correctness_score",
    "relevance_verdict",
    "relevance_score",
    "helpfulness_verdict",
    "helpfulness_score",
    "completeness_verdict",
    "completeness_score",
    "safety_verdict",
    "safety_score",
    "failure_types",
    "notes",
]


def _split_clusters_disjoint(
    items_by_cluster: dict[int, list[Any]],
    *,
    dev_target: int,
    test_target: int,
    seed: int,
) -> tuple[list[Any], list[Any]]:
    """Assign whole clusters to judge_dev or judge_test so NO cluster spans both splits."""
    rng = random.Random(seed)
    cids = sorted(items_by_cluster.keys())
    rng.shuffle(cids)

    total_target = max(1, dev_target + test_target)
    dev_ratio = dev_target / total_target

    dev_items: list[Any] = []
    test_items: list[Any] = []

    for cid in cids:
        cluster_rows = list(items_by_cluster[cid])
        rng.shuffle(cluster_rows)
        # Assign entire cluster to whichever split is further below its target ratio
        dev_fill = len(dev_items) / max(1, dev_target)
        test_fill = len(test_items) / max(1, test_target)
        if dev_fill <= test_fill and (len(dev_items) < dev_target or dev_ratio >= 0.5):
            dev_items.extend(cluster_rows)
        else:
            test_items.extend(cluster_rows)

    return dev_items[:dev_target], test_items[:test_target]


def export_annotation_sheets(
    session: Session,
    config: AppConfig,
    *,
    output_dir: str | Path = "data/annotations",
    clustering_run_id: str | None = None,
) -> dict[str, Any]:
    """Export cluster-disjoint judge_dev.csv and judge_test.csv sheets."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    c_run_id = clustering_run_id
    if c_run_id is None:
        latest = session.scalars(
            select(ClusteringRun).order_by(ClusteringRun.created_at.desc()).limit(1)
        ).first()
        if latest is None:
            raise ValueError("No clustering_runs found; run `edg cluster run` first.")
        c_run_id = latest.clustering_run_id

    rows = session.execute(
        select(
            JudgeSample.interaction_id,
            JudgeSample.stratum,
            ClusterMembership.cluster_id,
            InteractionClean.clean_query,
            InteractionClean.clean_response,
            InteractionRaw.reference_answer,
            StateSnapshot.snapshot_json,
        )
        .join(
            ClusterMembership,
            (ClusterMembership.interaction_id == JudgeSample.interaction_id)
            & (ClusterMembership.clustering_run_id == c_run_id),
        )
        .join(InteractionClean, InteractionClean.interaction_id == JudgeSample.interaction_id)
        .join(InteractionRaw, InteractionRaw.interaction_id == JudgeSample.interaction_id)
        .join(StateSnapshot, StateSnapshot.snapshot_id == InteractionRaw.snapshot_id)
        .order_by(JudgeSample.interaction_id)
    ).all()

    if not rows:
        raise ValueError("No judge_samples found; run `edg priority run` first.")

    by_cluster: dict[int, list[Any]] = defaultdict(list)
    for r in rows:
        by_cluster[int(r.cluster_id)].append(r)

    dev_rows, test_rows = _split_clusters_disjoint(
        by_cluster,
        dev_target=config.judge.dev_annotation_size,
        test_target=config.judge.test_annotation_size,
        seed=config.seed,
    )

    dev_clusters = {int(r.cluster_id) for r in dev_rows}
    test_clusters = {int(r.cluster_id) for r in test_rows}
    assert dev_clusters.isdisjoint(test_clusters), "Cluster leakage between judge_dev and judge_test!"

    dev_csv = out_path / "judge_dev.csv"
    test_csv = out_path / "judge_test.csv"

    for split_name, split_rows, csv_path in (
        ("judge_dev", dev_rows, dev_csv),
        ("judge_test", test_rows, test_csv),
    ):
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=ANNOTATION_CSV_COLUMNS)
            writer.writeheader()
            for r in split_rows:
                writer.writerow(
                    {
                        "interaction_id": r.interaction_id,
                        "split": split_name,
                        "cluster_id": int(r.cluster_id),
                        "stratum": r.stratum,
                        "user_query": r.clean_query,
                        "ai_response": r.clean_response,
                        "reference_answer": r.reference_answer or "",
                        "snapshot_json": json.dumps(r.snapshot_json, sort_keys=True),
                        "annotator_id": "",
                        "evidence_status": "sufficient",
                        "correctness_verdict": "",
                        "correctness_score": "",
                        "relevance_verdict": "",
                        "relevance_score": "",
                        "helpfulness_verdict": "",
                        "helpfulness_score": "",
                        "completeness_verdict": "",
                        "completeness_score": "",
                        "safety_verdict": "",
                        "safety_score": "",
                        "failure_types": "",
                        "notes": "",
                    }
                )

    stats = {
        "judge_dev_path": str(dev_csv),
        "judge_test_path": str(test_csv),
        "judge_dev_count": len(dev_rows),
        "judge_test_count": len(test_rows),
        "judge_dev_clusters": len(dev_clusters),
        "judge_test_clusters": len(test_clusters),
    }
    logger.info("annotation_sheets_exported", extra=stats)
    return stats
