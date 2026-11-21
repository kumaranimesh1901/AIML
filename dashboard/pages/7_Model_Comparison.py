"""Page 7: Model Comparison with CIs, per-category regression highlights, and failing case drill-down."""

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
from sqlalchemy import select

from edg.db.models import BenchmarkRelease, EvalResult, Judgement, RegressionReport, TestCase
from edg.db.session import get_session

st.set_page_config(page_title="Model Comparison | EDG", layout="wide")
st.title("7. Model Comparison & Regression Detection")

engine = get_engine()
with get_session(engine) as session:
    releases = list(
        session.scalars(select(BenchmarkRelease).order_by(BenchmarkRelease.frozen_at.desc())).all()
    )

if not releases:
    st.info("No benchmark releases found. Run `edg release create v1.0` first.")
    st.stop()

chosen_rel = st.selectbox("Benchmark Release", [r.release_id for r in releases])

with get_session(engine) as session:
    reports = list(
        session.scalars(
            select(RegressionReport).where(RegressionReport.release_id == chosen_rel)
        ).all()
    )

if not reports:
    st.info(f"No regression reports found for `{chosen_rel}`. Run `edg eval compare {chosen_rel}` first.")
    st.stop()

pairs = sorted({(r.baseline_model_id, r.candidate_model_id) for r in reports})
pair_labels = [f"{b} vs {c}" for b, c in pairs]
chosen_pair_idx = st.selectbox("Model Pair (Baseline vs Candidate)", range(len(pairs)), format_func=lambda i: pair_labels[i])
baseline_id, candidate_id = pairs[chosen_pair_idx]

pair_reports = [
    r
    for r in reports
    if r.baseline_model_id == baseline_id and r.candidate_model_id == candidate_id
]
df_rep = pd.DataFrame(
    [
        {
            "scope": r.scope,
            "metric": r.metric,
            "baseline": r.baseline_value,
            "candidate": r.candidate_value,
            "diff": r.diff,
            "ci_low": r.ci_low,
            "ci_high": r.ci_high,
            "p_value": r.p_value,
            "p_adjusted": r.p_adjusted,
            "flagged": r.flagged,
        }
        for r in pair_reports
    ]
)

# Headline summary table: overall, slice:mined, slice:random_control
st.subheader("Headline Rates (Overall, Mined Slice, and Random Control Slice Separately)")
headline = df_rep[
    df_rep["scope"].isin(["overall", "slice:mined", "slice:random_control"])
    & (df_rep["metric"] == "pass_rate")
]
st.dataframe(headline, use_container_width=True)

# Per-category chart highlighting flagged regressions
metric_choice = st.selectbox("Metric for Category Chart", sorted(df_rep["metric"].unique().tolist()))
cat_df = df_rep[
    (df_rep["metric"] == metric_choice)
    & (df_rep["diff"].notna())
].copy()
cat_df["status"] = cat_df["flagged"].map({True: "Flagged Regression", False: "Not Flagged"})

if not cat_df.empty:
    fig = px.bar(
        cat_df,
        x="scope",
        y="diff",
        color="status",
        color_discrete_map={"Flagged Regression": "#d32f2f", "Not Flagged": "#1976d2"},
        title=f"Paired Difference ({candidate_id} - {baseline_id}) on {metric_choice}",
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Full Statistical Comparison Table")
st.dataframe(df_rep, use_container_width=True)

# Drill-down into failing cases for candidate model
st.subheader(f"Failing Cases for `{candidate_id}` on `{chosen_rel}`")
with get_session(engine) as session:
    fail_rows = session.execute(
        select(EvalResult, TestCase, Judgement)
        .join(TestCase, TestCase.test_case_id == EvalResult.test_case_id)
        .outerjoin(Judgement, Judgement.judgement_id == EvalResult.judgement_id)
        .where(
            EvalResult.release_id == chosen_rel,
            EvalResult.model_id == candidate_id,
            EvalResult.passed.is_(False),
        )
        .order_by(EvalResult.test_case_id)
    ).all()

if fail_rows:
    df_fail = pd.DataFrame(
        [
            {
                "test_case_id": tc.test_case_id,
                "intent": tc.intent,
                "severity": tc.severity,
                "input_query": tc.input_query,
                "response": er.response,
                "checklist_rate": (er.checklist_results_json or {}).get("satisfaction_rate"),
                "failure_types": ", ".join((jdg.failure_types or []) if jdg else []),
                "explanation": jdg.explanation if jdg else "",
            }
            for er, tc, jdg in fail_rows
        ]
    )
    st.dataframe(df_fail, use_container_width=True)
else:
    st.success(f"Zero failing cases for `{candidate_id}` on `{chosen_rel}`.")
