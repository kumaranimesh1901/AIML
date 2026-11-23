"""Test Dataset: the replayable test cases built from failures, and the frozen benchmark release."""

from __future__ import annotations

import sys
from pathlib import Path

_DASH_DIR = str(Path(__file__).resolve().parents[1])
if _DASH_DIR not in sys.path:
    sys.path.insert(0, _DASH_DIR)

import pandas as pd
import plotly.express as px
import queries as q
import streamlit as st
from db_helper import get_engine
from sqlalchemy import select
from ui import COLORS, SEVERITY_COLORS, SEVERITY_ORDER, kpi, load_or_wait, not_ready, page_header, show_chart

from edg.db.models import BenchmarkRelease, TestCase
from edg.db.session import get_session

page_header(
    "Test Dataset",
    "Failed and risky conversations become test cases: the question, the order it is about, and a "
    "checklist of what a good answer must and must not do. Approved cases are frozen into a benchmark.",
)


@st.cache_data(ttl=q.TTL, show_spinner=False)
def _load(_db: str) -> tuple[pd.DataFrame, list[dict]]:
    with get_session(get_engine()) as s:
        cases = s.scalars(select(TestCase).order_by(TestCase.test_case_id)).all()
        releases = s.scalars(select(BenchmarkRelease).order_by(BenchmarkRelease.frozen_at.desc())).all()
        df = pd.DataFrame(
            [
                {
                    "id": tc.test_case_id,
                    "Question": tc.input_query,
                    "Topic": (tc.intent or "unknown").replace("_", " "),
                    "Severity": tc.severity,
                    "Tier": tc.trust_tier,
                    "Status": tc.status,
                    "Problems": ", ".join(f.replace("_", " ") for f in (tc.original_failure_types or [])),
                    "Must do": len((tc.expected_checklist_json or {}).get("must_do") or []),
                    "Must not do": len((tc.expected_checklist_json or {}).get("must_not_do") or []),
                }
                for tc in cases
            ]
        )
        rels = [
            {"id": r.release_id, "frozen_at": r.frozen_at, "judge": r.judge_version, "counts": r.counts_json or {}}
            for r in releases
        ]
    return df, rels


df, releases = load_or_wait(lambda: _load(q.db()))
if df.empty:
    not_ready("Test cases", "test-case generation")

approved = df[df["Status"] == "approved"]
rel = releases[0] if releases else None
k1, k2, k3, k4 = st.columns(4)
kpi(k1, "Test cases", f"{len(df):,}", f"{len(approved):,} approved")
kpi(k2, "Topics covered", f"{df['Topic'].nunique():,}")
kpi(k3, "High or critical", f"{int(df['Severity'].isin(['high', 'critical']).sum()):,}", "severity")
kpi(
    k4,
    "Frozen benchmark",
    rel["id"] if rel else "–",
    f"{rel['counts'].get('total_cases', 0):,} cases · judge {rel['judge']}" if rel else "not frozen yet",
)
st.write("")

left, right = st.columns(2)
with left:
    sev = df["Severity"].value_counts().reindex(SEVERITY_ORDER).dropna().reset_index()
    sev.columns = ["Severity", "cases"]
    fig = px.bar(sev, x="Severity", y="cases", color="Severity", color_discrete_map=SEVERITY_COLORS, title="Cases by severity")
    show_chart(fig, legend=False)
with right:
    top = df["Topic"].value_counts().head(10).sort_values().reset_index()
    top.columns = ["Topic", "cases"]
    fig = px.bar(top, x="cases", y="Topic", orientation="h", color_discrete_sequence=[COLORS["primary"]], title="Most-tested topics")
    show_chart(fig, legend=False)

if rel:
    with st.expander(f"What's inside benchmark {rel['id']}"):
        counts = rel["counts"]
        c1, c2, c3 = st.columns(3)
        for col, key, title in ((c1, "by_slice", "Source"), (c2, "by_split", "Split"), (c3, "by_trust_tier", "Trust tier")):
            col.markdown(f"**{title}**")
            for k, v in (counts.get(key) or {}).items():
                col.markdown(f"{k.replace('_', ' ')}: {v:,}")
        st.caption(
            "Mined cases come from the edge-case ranking; random-control cases keep the benchmark "
            "representative. Frozen releases can't change, so later chatbot versions are compared fairly."
        )

st.subheader("Browse test cases")
f1, f2, f3 = st.columns(3)
sev_sel = f1.multiselect("Severity", [s for s in SEVERITY_ORDER if s in set(df["Severity"])], placeholder="Any severity")
status_sel = f2.multiselect("Status", sorted(df["Status"].unique()), placeholder="Any status")
search = f3.text_input("Search", placeholder="Search questions")
view = df
if sev_sel:
    view = view[view["Severity"].isin(sev_sel)]
if status_sel:
    view = view[view["Status"].isin(status_sel)]
if search:
    view = view[view["Question"].str.contains(search, case=False, na=False)]

table = st.dataframe(
    view[["Question", "Topic", "Severity", "Status", "Problems"]],
    use_container_width=True,
    hide_index=True,
    height=360,
    on_select="rerun",
    selection_mode="single-row",
    column_config={"Question": st.column_config.TextColumn(width="large")},
)
d1, d2, _ = st.columns([1, 1, 3])
d1.download_button("Download CSV", view.to_csv(index=False).encode("utf-8"), "edg_test_cases.csv", "text/csv", icon=":material/download:")
d2.download_button("Download JSON", view.to_json(orient="records", indent=2).encode("utf-8"), "edg_test_cases.json", "application/json", icon=":material/download:")

picked = table.selection.rows if table and table.selection else []
if not picked:
    st.caption("Select a row to see its checklist and review it.")
    st.stop()

tc_id = view.iloc[picked[0]]["id"]
with get_session(get_engine()) as session:
    tc = session.get(TestCase, tc_id)
if tc is None:
    st.stop()

with st.container(border=True):
    st.markdown(f"**Customer question:** {tc.input_query}")
    checklist = tc.expected_checklist_json or {}
    a, b = st.columns(2)
    a.markdown("**A good answer must**")
    for item in checklist.get("must_do") or []:
        a.markdown(f":material/check: {item}")
    b.markdown("**A good answer must not**")
    for item in checklist.get("must_not_do") or []:
        b.markdown(f":material/close: {item}")
    if tc.reference_answer:
        st.markdown(f"**Reference answer:** {tc.reference_answer}")

with st.expander("Review this test case (approve, reject or edit its checklist)"):
    must_do = st.text_area("Must do (one per line)", "\n".join(checklist.get("must_do") or []))
    must_not = st.text_area("Must not do (one per line)", "\n".join(checklist.get("must_not_do") or []))
    new_checklist = {
        "must_do": [x.strip() for x in must_do.splitlines() if x.strip()],
        "must_not_do": [x.strip() for x in must_not.splitlines() if x.strip()],
    }
    b1, b2, b3 = st.columns(3)
    action = None
    if b1.button("Approve as gold", type="primary", use_container_width=True):
        action = "approve"
    if b2.button("Reject", use_container_width=True):
        action = "reject"
    if b3.button("Save checklist", use_container_width=True):
        action = "save"
    if action:
        with get_session(get_engine()) as session:
            obj = session.get(TestCase, tc_id)
            if obj is not None:
                if action == "approve":
                    obj.status, obj.trust_tier = "approved", "gold"
                if action == "reject":
                    obj.status = "rejected"
                if action in ("approve", "save"):
                    obj.expected_checklist_json = new_checklist
        _load.clear()
        st.success({"approve": "Approved as gold.", "reject": "Rejected.", "save": "Checklist saved."}[action])
