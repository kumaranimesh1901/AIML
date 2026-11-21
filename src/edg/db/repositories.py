"""Repository helpers and pipeline_runs lifecycle tracking."""

from __future__ import annotations

import subprocess
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import ColumnElement, select
from sqlalchemy.orm import Session

from edg.config import AppConfig
from edg.db.models import EvalResult, Judgement, PipelineRun, utcnow
from edg.logging import get_logger

logger = get_logger(__name__)


def get_git_commit() -> str:
    """Return current short git commit SHA, or 'unknown' if not inside a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class PipelineRunRepository:
    """CRUD helper for the pipeline_runs tracking table."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def start_run(self, step: str, config: AppConfig, run_id: str | None = None) -> PipelineRun:
        run = PipelineRun(
            run_id=run_id or f"run_{uuid.uuid4().hex[:12]}",
            step=step,
            config_hash=config.config_hash(),
            git_commit=get_git_commit(),
            started_at=utcnow(),
            status="running",
            counts_json={},
        )
        self.session.add(run)
        self.session.flush()
        return run

    def finish_run(
        self,
        run: PipelineRun,
        *,
        status: str = "completed",
        counts: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> PipelineRun:
        run.finished_at = utcnow()
        run.status = status
        if counts is not None:
            run.counts_json = counts
        if error is not None:
            run.error = error
        self.session.flush()
        return run

    def get(self, run_id: str) -> PipelineRun | None:
        return self.session.get(PipelineRun, run_id)


@contextmanager
def track_run(
    session: Session,
    step: str,
    config: AppConfig,
) -> Iterator[tuple[PipelineRun, dict[str, Any]]]:
    """Context manager that records a pipeline step in pipeline_runs.

    Yields `(run, counts)` where callers can mutate `counts` with step metrics.
    """
    repo = PipelineRunRepository(session)
    run = repo.start_run(step=step, config=config)
    counts: dict[str, Any] = {}
    logger.info(
        "pipeline_step_started",
        extra={"run_id": run.run_id, "step": step, "config_hash": run.config_hash},
    )
    try:
        yield run, counts
        repo.finish_run(run, status="completed", counts=counts)
        logger.info(
            "pipeline_step_completed",
            extra={"run_id": run.run_id, "step": step, "counts": counts},
        )
    except Exception as exc:
        run_id = run.run_id
        config_hash = run.config_hash
        git_commit = run.git_commit
        started_at = run.started_at
        err_msg = f"{type(exc).__name__}: {exc}"
        session.rollback()
        existing = session.get(PipelineRun, run_id)
        if existing is None:
            existing = PipelineRun(
                run_id=run_id,
                step=step,
                config_hash=config_hash,
                git_commit=git_commit,
                started_at=started_at,
                finished_at=utcnow(),
                status="failed",
                counts_json=counts,
                error=err_msg,
            )
            session.add(existing)
        else:
            existing.finished_at = utcnow()
            existing.status = "failed"
            existing.counts_json = counts
            existing.error = err_msg
        session.commit()
        logger.error(
            "pipeline_step_failed",
            extra={"run_id": run_id, "step": step, "error": str(exc)},
        )
        raise


def source_judgements_only() -> ColumnElement[bool]:
    """SQL filter selecting judgements of ORIGINAL logged responses only.

    Release replays persist judgements of a *newly generated* model response under the source
    test case's interaction_id. Those rows are linked from `eval_results.judgement_id` and must
    never be treated as judgements of the logged `interactions_raw.ai_response` (calibration
    pairing, judge caching, labeling, test-case mining).
    """
    replay_ids = select(EvalResult.judgement_id).where(EvalResult.judgement_id.is_not(None))
    return Judgement.judgement_id.notin_(replay_ids)
