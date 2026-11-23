"""Read-only database queries shared by dashboard pages. Every number shown comes from here or
from a page's own query; nothing is hard-coded. Results are cached for a short time so pages
switch quickly, and refresh on their own while a pipeline run is filling the database."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from db_helper import get_database_url, get_engine
from sqlalchemy import func, select

from edg.db.models import (
    BenchmarkRelease,
    Cluster,
    ClusteringRun,
    EvalResult,
    InteractionClean,
    InteractionRaw,
    Judgement,
    JudgeSample,
    Label,
    Order,
    PriorityScore,
    RegressionReport,
    TestCase,
)
from edg.db.repositories import source_judgements_only
from edg.db.session import get_session
from edg.metrics.judge_metrics import compute_controlled_tier_recall
from edg.sut.failure_injector import CONTROLLED_SOURCE_TAG

TTL = 30  # seconds

# Plain-English pipeline stages, in order, with the table that proves each one ran
STAGES: list[tuple[str, str, Any]] = [
    ("Store built", "orders", Order.order_id),
    ("Conversations logged", "conversations", InteractionRaw.interaction_id),
    ("Cleaned", "clean conversations", InteractionClean.interaction_id),
    ("Topics found", "clustering run(s)", ClusteringRun.clustering_run_id),
    ("Edge cases ranked", "scored", PriorityScore.interaction_id),
    ("Answers judged", "judgements", Judgement.judgement_id),
    ("Failures labelled", "labels", Label.label_id),
    ("Test cases built", "test cases", TestCase.test_case_id),
    ("Benchmark frozen", "release(s)", BenchmarkRelease.release_id),
    ("v1 vs v2 compared", "statistics", RegressionReport.report_id),
]


def _url() -> str:
    return get_database_url()


@st.cache_data(ttl=TTL, show_spinner=False)
def stage_counts(_db: str) -> list[tuple[str, str, int]]:
    with get_session(get_engine()) as s:
        return [(name, unit, int(s.scalar(select(func.count(col))) or 0)) for name, unit, col in STAGES]


@st.cache_data(ttl=TTL, show_spinner=False)
def conversation_counts(_db: str) -> dict[str, int]:
    with get_session(get_engine()) as s:
        rows = s.execute(
            select(InteractionRaw.source_tag, InteractionRaw.model_version, func.count())
            .group_by(InteractionRaw.source_tag, InteractionRaw.model_version)
        ).all()
    natural = sum(n for tag, _, n in rows if tag != CONTROLLED_SOURCE_TAG)
    planted = sum(n for tag, _, n in rows if tag == CONTROLLED_SOURCE_TAG)
    return {"natural": natural, "planted": planted}


@st.cache_data(ttl=TTL, show_spinner=False)
def latest_clustering(_db: str) -> dict[str, Any] | None:
    with get_session(get_engine()) as s:
        run = s.scalars(select(ClusteringRun).order_by(ClusteringRun.created_at.desc()).limit(1)).first()
        if run is None:
            return None
        new_intents = int(
            s.scalar(
                select(func.count(Cluster.id)).where(
                    Cluster.clustering_run_id == run.clustering_run_id, Cluster.is_new_intent.is_(True)
                )
            )
            or 0
        )
        return {
            "run_id": run.clustering_run_id,
            "n_clusters": run.n_clusters,
            "noise_share": run.noise_share,
            "dbcv": run.dbcv,
            "ari": run.ari_vs_other_seeds,
            "new_intents": new_intents,
        }


@st.cache_data(ttl=TTL, show_spinner=False)
def sampled_labels(_db: str) -> pd.DataFrame:
    """One row per judged sample: stratum, priority rank and whether the judge found a failure."""
    with get_session(get_engine()) as s:
        rows = s.execute(
            select(JudgeSample.stratum, PriorityScore.rank, Label.failure_detected)
            .join(Label, Label.interaction_id == JudgeSample.interaction_id)
            .outerjoin(PriorityScore, PriorityScore.interaction_id == JudgeSample.interaction_id)
        ).all()
    return pd.DataFrame(rows, columns=["stratum", "rank", "failure"])


def lift_table(df: pd.DataFrame, ks: tuple[int, ...] = (50, 100)) -> pd.DataFrame:
    """Failure rate of the top-k priority-ranked samples vs the unbiased random-control sample."""
    if df.empty:
        return pd.DataFrame(columns=["group", "failure_rate", "n"])
    rc = df[df["stratum"] == "random_control"]
    rows = [{"group": "Random sample", "failure_rate": rc["failure"].mean() if len(rc) else None, "n": len(rc)}]
    ranked = df.dropna(subset=["rank"]).sort_values("rank")
    for k in ks:
        top = ranked.head(k)
        if len(top):
            rows.append({"group": f"Top {k} edge cases", "failure_rate": top["failure"].mean(), "n": len(top)})
    return pd.DataFrame(rows)


@st.cache_data(ttl=TTL, show_spinner=False)
def controlled_recall(_db: str) -> dict[str, Any]:
    """Share of deliberately planted failures that the judge caught with the right label."""
    with get_session(get_engine()) as s:
        rows = s.execute(
            select(InteractionRaw.injected_failure, Judgement.failure_types)
            .join(Judgement, Judgement.interaction_id == InteractionRaw.interaction_id)
            .where(
                InteractionRaw.source_tag == CONTROLLED_SOURCE_TAG,
                InteractionRaw.injected_failure.is_not(None),
                source_judgements_only(),
            )
        ).all()
    return compute_controlled_tier_recall([(str(r[0]), list(r[1] or [])) for r in rows])


@st.cache_data(ttl=TTL, show_spinner=False)
def failure_type_counts(_db: str) -> pd.DataFrame:
    """Failure types found by the judge in real (not planted) conversations."""
    with get_session(get_engine()) as s:
        rows = s.execute(
            select(Label.failure_types)
            .join(InteractionRaw, InteractionRaw.interaction_id == Label.interaction_id)
            .where(InteractionRaw.source_tag != CONTROLLED_SOURCE_TAG, Label.failure_detected.is_(True))
        ).all()
    counts = Counter(ft for (fts,) in rows for ft in (fts or []) if ft != "judge_error")
    return pd.DataFrame(sorted(counts.items(), key=lambda kv: -kv[1]), columns=["failure_type", "count"])


@st.cache_data(ttl=TTL, show_spinner=False)
def headline_comparison(_db: str) -> dict[str, Any] | None:
    """Overall pass rates of baseline vs candidate on the latest release that has a comparison."""
    with get_session(get_engine()) as s:
        rows = s.execute(
            select(RegressionReport, BenchmarkRelease.frozen_at)
            .join(BenchmarkRelease, BenchmarkRelease.release_id == RegressionReport.release_id)
            .where(RegressionReport.scope == "overall", RegressionReport.metric == "pass_rate")
            .order_by(BenchmarkRelease.frozen_at.desc())
        ).all()
        if not rows:
            return None
        r = rows[0][0]
        n_cases = int(
            s.scalar(
                select(func.count(func.distinct(EvalResult.test_case_id))).where(
                    EvalResult.release_id == r.release_id, EvalResult.model_id == r.candidate_model_id
                )
            )
            or 0
        )
        return {
            "release_id": r.release_id,
            "baseline": r.baseline_model_id,
            "candidate": r.candidate_model_id,
            "baseline_value": r.baseline_value,
            "candidate_value": r.candidate_value,
            "diff": r.diff,
            "ci_low": r.ci_low,
            "ci_high": r.ci_high,
            "p_adjusted": r.p_adjusted,
            "n_cases": n_cases,
        }


@st.cache_data(ttl=TTL, show_spinner=False)
def release_case_count(_db: str) -> tuple[str, int] | None:
    with get_session(get_engine()) as s:
        rel = s.scalars(select(BenchmarkRelease).order_by(BenchmarkRelease.frozen_at.desc()).limit(1)).first()
        if rel is None:
            return None
        return rel.release_id, int((rel.counts_json or {}).get("total_cases") or 0)


def human_check_report() -> dict[str, Any] | None:
    """The real two-annotator check of the judge (`edg calibrate human-import`), if it was run.

    It lives in a JSON file, not the database, and is only shown when it was scored against the
    same judge version that produced this database's judgements."""
    path = Path("data/exports/human_agreement.json")
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if report.get("source") != "real_human_annotation":
        return None
    return report


db = _url
