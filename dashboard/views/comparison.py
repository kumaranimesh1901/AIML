"""v1 vs v2: replay both chatbot versions on the frozen benchmark and test whether the difference is real."""

from __future__ import annotations

import sys
from pathlib import Path

_DASH_DIR = str(Path(__file__).resolve().parents[1])
if _DASH_DIR not in sys.path:
    sys.path.insert(0, _DASH_DIR)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import queries as q
import streamlit as st
from db_helper import get_engine
from sqlalchemy import select
from ui import COLORS, kpi, load_or_wait, not_ready, page_header, pct, show_chart

from edg.db.models import BenchmarkRelease, EvalResult, Judgement, RegressionReport, TestCase
from edg.db.session import get_session

METRICS = {
    "pass_rate": "Overall pass",
    "correctness_pass_rate": "Correctness",
    "relevance_pass_rate": "Relevance",
    "helpfulness_pass_rate": "Helpfulness",
    "completeness_pass_rate": "Completeness",
    "safety_pass_rate": "Safety",
    "hallucination_rate": "Hallucination (lower is better)",
    "critical_failure_rate": "Critical failure (lower is better)",
}

page_header(
    "v1 vs v2",
    "Both chatbot versions answer every test case in the frozen benchmark, the same judge grades "
    "them, and paired statistics (bootstrap CIs, corrected p-values) say whether a change is real.",
)


@st.cache_data(ttl=q.TTL, show_spinner=False)
def _reports(_db: str) -> pd.DataFrame:
    with get_session(get_engine()) as s:
        rows = s.execute(
            select(RegressionReport, BenchmarkRelease.frozen_at).join(
                BenchmarkRelease, BenchmarkRelease.release_id == RegressionReport.release_id
            )
        ).all()
    return pd.DataFrame(
        [
            {
                "release": r.release_id,
                "frozen_at": frozen,
                "baseline": r.baseline_model_id,
                "candidate": r.candidate_model_id,
                "scope": r.scope,
                "metric": r.metric,
                "base": r.baseline_value,
                "cand": r.candidate_value,
                "diff": r.diff,
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
                "p_adj": r.p_adjusted,
                "flagged": bool(r.flagged),
            }
            for r, frozen in rows
        ]
    )


rep = load_or_wait(lambda: _reports(q.db()))
if rep.empty:
    not_ready("The version comparison", "benchmark replay and compare")

combos = rep.sort_values("frozen_at", ascending=False)[["release", "baseline", "candidate"]].drop_duplicates()
release, baseline, candidate = combos.iloc[0]
if len(combos) > 1:
    labels = [f"{r} · {b} vs {c}" for r, b, c in combos.itertuples(index=False)]
    choice = st.selectbox("Comparison", range(len(labels)), format_func=lambda i: labels[i])
    release, baseline, candidate = combos.iloc[choice]
pair = rep[(rep["release"] == release) & (rep["baseline"] == baseline) & (rep["candidate"] == candidate)]


def _row(scope: str, metric: str = "pass_rate") -> pd.Series | None:
    hit = pair[(pair["scope"] == scope) & (pair["metric"] == metric)]
    return hit.iloc[0] if len(hit) else None


overall = _row("overall")
k1, k2, k3, k4 = st.columns(4)
if overall is not None and pd.notna(overall["base"]):
    kpi(k1, f"{baseline} pass rate", pct(overall["base"], 1))
    kpi(k2, f"{candidate} pass rate", pct(overall["cand"], 1))
    better = (overall["diff"] or 0) >= 0
    kpi(
        k3,
        "Difference",
        f"{overall['diff'] * 100:+.1f} pts",
        f"95% CI {overall['ci_low'] * 100:+.1f} to {overall['ci_high'] * 100:+.1f} pts"
        if pd.notna(overall["ci_low"])
        else "",
        trend="up" if better else "down",
    )
    significant = pd.notna(overall["p_adj"]) and overall["p_adj"] < 0.05
    kpi(
        k4,
        "Is it real?",
        "Yes" if significant else "Not proven",
        ("adjusted p < 0.001" if overall["p_adj"] < 0.001 else f"adjusted p = {overall['p_adj']:.3f}")
        if pd.notna(overall["p_adj"])
        else "",
    )
st.write("")

left, right = st.columns(2)
with left:
    dims = pair[(pair["scope"] == "overall") & pair["metric"].isin(METRICS)].copy()
    dims = dims.dropna(subset=["base"])
    if not dims.empty:
        dims["label"] = dims["metric"].map(METRICS)
        dims = dims.set_index("metric").loc[[m for m in METRICS if m in set(dims["metric"])]].reset_index()
        fig = go.Figure(
            [
                go.Bar(name=baseline, y=dims["label"], x=dims["base"], orientation="h", marker_color=COLORS["v1"]),
                go.Bar(name=candidate, y=dims["label"], x=dims["cand"], orientation="h", marker_color=COLORS["v2"]),
            ]
        )
        fig.update_layout(barmode="group", title="Every metric, side by side")
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(tickformat=".0%")
        show_chart(fig, height=420)
with right:
    slices = [("slice:mined", "Edge-case slice"), ("slice:random_control", "Random slice"), ("overall", "All cases")]
    srows = [(lbl, _row(sc)) for sc, lbl in slices]
    sdf = pd.DataFrame(
        [{"slice": lbl, "version": v, "pass_rate": r[col]} for lbl, r in srows if r is not None and pd.notna(r["base"]) for v, col in ((baseline, "base"), (candidate, "cand"))]
    )
    if not sdf.empty:
        fig = px.bar(
            sdf,
            x="slice",
            y="pass_rate",
            color="version",
            barmode="group",
            text=sdf["pass_rate"].map(lambda v: pct(v)),
            color_discrete_sequence=[COLORS["v1"], COLORS["v2"]],
            title="Pass rate on edge cases vs a random slice",
        )
        fig.update_yaxes(tickformat=".0%", range=[0, 1.1])
        fig.update_traces(textposition="outside")
        show_chart(fig, height=420)

st.subheader("Where the versions differ")
metric = st.segmented_control(
    "Metric",
    [m for m in METRICS if m in set(pair["metric"])],
    default="pass_rate",
    format_func=lambda m: METRICS[m].replace(" (lower is better)", ""),
    label_visibility="collapsed",
)
cats = pair[(pair["metric"] == (metric or "pass_rate")) & pair["scope"].str.startswith(("intent:", "severity:")) & pair["diff"].notna()].copy()
if cats.empty:
    st.caption("Not enough test cases per topic to compare (each topic needs a minimum number of cases).")
else:
    kind, name = cats["scope"].str.split(":", n=1).str[0], cats["scope"].str.split(":", n=1).str[1]
    cats["group"] = kind.map({"intent": "topic", "severity": "severity"}) + ": " + name.str.replace("_", " ")
    cats["result"] = cats.apply(
        lambda r: "Significant regression" if r["flagged"] else ("Better" if r["diff"] > 0 else "Worse or same"), axis=1
    )
    cats["plus"] = (cats["ci_high"] - cats["diff"]).clip(lower=0)
    cats["minus"] = (cats["diff"] - cats["ci_low"]).clip(lower=0)
    cats = cats.sort_values("diff")
    fig = px.bar(
        cats,
        x="diff",
        y="group",
        color="result",
        orientation="h",
        error_x="plus",
        error_x_minus="minus",
        color_discrete_map={"Significant regression": COLORS["fail"], "Better": COLORS["pass"], "Worse or same": COLORS["neutral"]},
        title=f"Change from {baseline} to {candidate}, with 95% CI",
    )
    fig.update_xaxes(tickformat="+.0%")
    show_chart(fig, height=max(300, 34 * len(cats) + 120))
    n_flag = int(cats["flagged"].sum())
    st.caption(
        f"{n_flag} significant regression(s) after correcting for multiple comparisons."
        if n_flag
        else "No topic got significantly worse after correcting for multiple comparisons."
    )

with st.expander("Full statistics table"):
    st.dataframe(
        pair[["scope", "metric", "base", "cand", "diff", "ci_low", "ci_high", "p_adj", "flagged"]].rename(
            columns={"base": baseline, "cand": candidate, "p_adj": "p (adjusted)"}
        ),
        use_container_width=True,
        hide_index=True,
    )

st.subheader(f"Test cases {candidate} still fails")


@st.cache_data(ttl=q.TTL, show_spinner=False)
def _fails(_db: str, release_id: str, model_id: str) -> pd.DataFrame:
    with get_session(get_engine()) as s:
        rows = s.execute(
            select(EvalResult, TestCase, Judgement)
            .join(TestCase, TestCase.test_case_id == EvalResult.test_case_id)
            .outerjoin(Judgement, Judgement.judgement_id == EvalResult.judgement_id)
            .where(EvalResult.release_id == release_id, EvalResult.model_id == model_id, EvalResult.passed.is_(False))
            .order_by(EvalResult.test_case_id)
        ).all()
    return pd.DataFrame(
        [
            {
                "question": tc.input_query,
                "answer": er.response,
                "severity": tc.severity,
                "problems": ", ".join(f.replace("_", " ") for f in ((jdg.failure_types or []) if jdg else [])),
                "why": jdg.explanation if jdg else "",
            }
            for er, tc, jdg in rows
        ]
    )


fails = _fails(q.db(), release, candidate)
if fails.empty:
    st.success(f"{candidate} passes every test case in {release}.")
else:
    st.caption(f"{len(fails)} failing case(s). Open one to read the answer and the judge's reason.")
    for _, f in fails.head(25).iterrows():
        title = f["question"][:100] + ("…" if len(f["question"]) > 100 else "")
        with st.expander(f"{f['severity'].upper()} · {title}"):
            with st.chat_message("user"):
                st.markdown(f["question"])
            with st.chat_message("assistant"):
                st.markdown(f["answer"])
            if f["problems"]:
                st.markdown(f"**Problems:** {f['problems']}")
            st.markdown(f"**Why:** {f['why']}")
    if len(fails) > 25:
        st.caption(f"Showing 25 of {len(fails)}.")
