"""Immutable versioned benchmark release creation and content hashing."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import event, select
from sqlalchemy.orm import Session

from edg.benchmark.splits import split_test_cases_by_cluster
from edg.config import AppConfig
from edg.db.models import BenchmarkRelease, ReleaseItem, TestCase
from edg.logging import get_logger

logger = get_logger(__name__)


class ReleaseImmutabilityError(PermissionError):
    """Raised when attempting to overwrite, update, or delete an immutable benchmark release."""


@event.listens_for(BenchmarkRelease, "before_update")
def _block_release_update(_mapper: Any, _conn: Any, target: BenchmarkRelease) -> None:
    raise ReleaseImmutabilityError(
        f"BenchmarkRelease '{target.release_id}' is immutable and cannot be modified."
    )


@event.listens_for(BenchmarkRelease, "before_delete")
def _block_release_delete(_mapper: Any, _conn: Any, target: BenchmarkRelease) -> None:
    raise ReleaseImmutabilityError(
        f"BenchmarkRelease '{target.release_id}' is immutable and cannot be deleted."
    )


@event.listens_for(ReleaseItem, "before_update")
def _block_release_item_update(_mapper: Any, _conn: Any, target: ReleaseItem) -> None:
    raise ReleaseImmutabilityError(
        f"ReleaseItem in release '{target.release_id}' is immutable and cannot be modified."
    )


@event.listens_for(ReleaseItem, "before_delete")
def _block_release_item_delete(_mapper: Any, _conn: Any, target: ReleaseItem) -> None:
    raise ReleaseImmutabilityError(
        f"ReleaseItem in release '{target.release_id}' is immutable and cannot be deleted."
    )


def _compute_release_content_hash(
    test_cases: list[TestCase],
    assignments: dict[str, tuple[str, str]],
) -> str:
    payload = []
    for tc in sorted(test_cases, key=lambda x: x.test_case_id):
        slc, spl = assignments[tc.test_case_id]
        payload.append(
            {
                "test_case_id": tc.test_case_id,
                "slice": slc,
                "split": spl,
                "snapshot_id": tc.snapshot_id,
                "policy_version": tc.policy_version,
                "input_query": tc.input_query,
                "expected_checklist_json": tc.expected_checklist_json,
            }
        )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def create_benchmark_release(
    session: Session,
    config: AppConfig,
    *,
    release_id: str,
    run_id: str,
    description: str | None = None,
    heldout_ratio: float = 0.5,
) -> dict[str, Any]:
    """Freeze approved gold & silver test cases into an immutable BenchmarkRelease."""
    if session.get(BenchmarkRelease, release_id) is not None:
        raise ReleaseImmutabilityError(
            f"Benchmark release '{release_id}' already exists and is immutable. Create a new version instead."
        )

    approved_cases = list(
        session.scalars(
            select(TestCase)
            .where(
                TestCase.status == "approved",
                TestCase.trust_tier.in_(["gold", "silver"]),
            )
            .order_by(TestCase.test_case_id)
        ).all()
    )
    if not approved_cases:
        raise ValueError(
            "No approved gold/silver test cases found; run `edg testcases generate` first."
        )

    assignments = split_test_cases_by_cluster(
        approved_cases, heldout_ratio=heldout_ratio, seed=config.seed
    )
    content_hash = _compute_release_content_hash(approved_cases, assignments)

    slice_counts = {"mined": 0, "random_control": 0}
    split_counts = {"dev": 0, "heldout": 0}
    tier_counts = {"gold": 0, "silver": 0}
    for tc in approved_cases:
        slc, spl = assignments[tc.test_case_id]
        slice_counts[slc] = slice_counts.get(slc, 0) + 1
        split_counts[spl] = split_counts.get(spl, 0) + 1
        tier_counts[tc.trust_tier] = tier_counts.get(tc.trust_tier, 0) + 1

    counts_json = {
        "total_cases": len(approved_cases),
        "by_slice": slice_counts,
        "by_split": split_counts,
        "by_trust_tier": tier_counts,
    }

    release = BenchmarkRelease(
        release_id=release_id,
        frozen_at=datetime.now(UTC),
        content_hash=content_hash,
        judge_version=config.judge.judge_version,
        description=description or f"Frozen benchmark release {release_id}",
        counts_json=counts_json,
        run_id=run_id,
    )
    session.add(release)
    session.flush()

    for tc in approved_cases:
        slc, spl = assignments[tc.test_case_id]
        session.add(
            ReleaseItem(
                release_id=release_id,
                test_case_id=tc.test_case_id,
                slice=slc,
                split=spl,
                run_id=run_id,
            )
        )
    session.flush()

    result = {
        "release_id": release_id,
        "content_hash": content_hash,
        "judge_version": config.judge.judge_version,
        **counts_json,
    }
    logger.info("benchmark_release_created", extra=result)
    return result
