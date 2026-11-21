"""Page 1: Pipeline Overview, stage counts, weighted failure rate, latest run and release."""

from __future__ import annotations

import sys
from pathlib import Path

_DASH_DIR = str(Path(__file__).resolve().parents[1])
if _DASH_DIR not in sys.path:
    sys.path.insert(0, _DASH_DIR)

import pandas as pd
import plotly.express as px
import streamlit as st
from db_helper import get_engine
from sqlalchemy import func, select

from edg.db.models import (
    BenchmarkRelease,
    Cluster,
    Embedding,
    EvalResult,
    InteractionClean,
    InteractionRaw,
    Judgement,
    JudgeSample,
    Label,
    Order,
    PipelineRun,
    StateSnapshot,
    TestCase,
)
from edg.db.session import get_session

st.set_page_config(page_title="Overview | EDG", layout="wide")
st.title("1. Pipeline Overview")

engine = get_engine()
with get_session(engine) as session:
    stage_counts = {
        "Orders": session.scalar(select(func.count(Order.order_id))) or 0,
        "Snapshots": session.scalar(select(func.count(StateSnapshot.snapshot_id))) or 0,
        "Raw Interactions": session.scalar(select(func.count(InteractionRaw.interaction_id))) or 0,
        "Clean Interactions": session.scalar(select(func.count(InteractionClean.interaction_id))) or 0,
        "Embeddings": session.scalar(select(func.count(Embedding.embedding_id))) or 0,
        "Clusters": session.scalar(select(func.count(Cluster.id))) or 0,
        "Judge Samples": session.scalar(select(func.count(JudgeSample.sample_id))) or 0,
        "Judgements": session.scalar(select(func.count(Judgement.judgement_id))) or 0,
        "Labels": session.scalar(select(func.count(Label.label_id))) or 0,
        "Test Cases": session.scalar(select(func.count(TestCase.test_case_id))) or 0,
        "Releases": session.scalar(select(func.count(BenchmarkRelease.release_id))) or 0,
        "Eval Results": session.scalar(select(func.count(EvalResult.result_id))) or 0,
    }

    # Weighted failure rate over judge_samples + separate random_control unbiased rate
    sample_label_rows = session.execute(
        select(JudgeSample.stratum, JudgeSample.sampling_weight, Label.failure_detected)
        .join(Label, Label.interaction_id == JudgeSample.interaction_id)
    ).all()

    latest_run = session.scalars(
        select(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(1)
    ).first()
    latest_release = session.scalars(
        select(BenchmarkRelease).order_by(BenchmarkRelease.frozen_at.desc()).limit(1)
    ).first()

total_weight = sum(float(r.sampling_weight) for r in sample_label_rows)
weighted_fail = sum(
    float(r.sampling_weight) for r in sample_label_rows if r.failure_detected
)
weighted_fail_rate = (weighted_fail / total_weight) if total_weight > 0 else None

rc_rows = [r for r in sample_label_rows if r.stratum == "random_control"]
rc_fail_rate = (
    sum(1 for r in rc_rows if r.failure_detected) / len(rc_rows) if rc_rows else None
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Raw Interactions", stage_counts["Raw Interactions"])
col2.metric("Test Cases", stage_counts["Test Cases"])
col3.metric(
    "Weighted Failure Rate",
    f"{weighted_fail_rate:.1%}" if weighted_fail_rate is not None else "N/A",
)
col4.metric(
    "Random Control Failure Rate",
    f"{rc_fail_rate:.1%}" if rc_fail_rate is not None else "N/A",
)

df_stages = pd.DataFrame(
    [{"stage": k, "count": int(v)} for k, v in stage_counts.items()]
)
fig = px.bar(df_stages, x="stage", y="count", title="Records per Pipeline Stage")
st.plotly_chart(fig, use_container_width=True)

c_left, c_right = st.columns(2)
with c_left:
    st.subheader("Latest Pipeline Run")
    if latest_run is not None:
        st.json(
            {
                "run_id": latest_run.run_id,
                "step": latest_run.step,
                "status": latest_run.status,
                "config_hash": latest_run.config_hash,
                "git_commit": latest_run.git_commit,
                "counts": latest_run.counts_json,
            }
        )
    else:
        st.info("No pipeline runs found.")

with c_right:
    st.subheader("Latest Benchmark Release")
    if latest_release is not None:
        st.json(
            {
                "release_id": latest_release.release_id,
                "judge_version": latest_release.judge_version,
                "content_hash": latest_release.content_hash,
                "frozen_at": str(latest_release.frozen_at),
                "counts": latest_release.counts_json,
            }
        )
    else:
        st.info("No benchmark releases created yet.")
