"""Page 6: Test-case browser, approve/reject/edit checklist, releases, coverage, and CSV/JSON export."""

from __future__ import annotations

import json
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

from edg.db.models import BenchmarkRelease, TestCase
from edg.db.session import get_session
from edg.metrics.dataset_metrics import compute_dataset_metrics

st.set_page_config(page_title="Dataset | EDG", layout="wide")
st.title("6. Evaluation Dataset & Benchmark Releases")

engine = get_engine()
with get_session(engine) as session:
    cases = list(session.scalars(select(TestCase).order_by(TestCase.test_case_id)).all())
    releases = list(
        session.scalars(select(BenchmarkRelease).order_by(BenchmarkRelease.frozen_at.desc())).all()
    )

if not cases:
    st.info("No test cases found. Run `edg testcases generate` first.")
    st.stop()

df = pd.DataFrame(
    [
        {
            "test_case_id": tc.test_case_id,
            "status": tc.status,
            "trust_tier": tc.trust_tier,
            "severity": tc.severity,
            "intent": tc.intent or "unknown",
            "cluster_id": tc.cluster_id,
            "input_query": tc.input_query,
            "failure_types": ", ".join(tc.original_failure_types or []),
        }
        for tc in cases
    ]
)

f1, f2, f3 = st.columns(3)
status_sel = f1.multiselect("Status", sorted(df["status"].unique().tolist()))
tier_sel = f2.multiselect("Trust Tier", sorted(df["trust_tier"].unique().tolist()))
sev_sel = f3.multiselect("Severity", sorted(df["severity"].unique().tolist()))

filtered = df.copy()
if status_sel:
    filtered = filtered[filtered["status"].isin(status_sel)]
if tier_sel:
    filtered = filtered[filtered["trust_tier"].isin(tier_sel)]
if sev_sel:
    filtered = filtered[filtered["severity"].isin(sev_sel)]

st.dataframe(filtered, use_container_width=True)

# Export buttons
e1, e2 = st.columns(2)
e1.download_button(
    "Download Filtered CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="edg_test_cases.csv",
    mime="text/csv",
)
e2.download_button(
    "Download Filtered JSON",
    data=filtered.to_json(orient="records", indent=2).encode("utf-8"),
    file_name="edg_test_cases.json",
    mime="application/json",
)

# Approve / Reject / Edit checklist
if not filtered.empty:
    st.subheader("Review & Edit Test Case")
    chosen_tc_id = st.selectbox("Test Case ID", filtered["test_case_id"].tolist())
    with get_session(engine) as session:
        tc_obj = session.get(TestCase, chosen_tc_id)

    if tc_obj is not None:
        st.markdown(f"**Query:** {tc_obj.input_query}")
        must_do_text = st.text_area(
            "must_do (one per line)",
            value="\n".join(tc_obj.expected_checklist_json.get("must_do") or []),
        )
        must_not_text = st.text_area(
            "must_not_do (one per line)",
            value="\n".join(tc_obj.expected_checklist_json.get("must_not_do") or []),
        )
        b1, b2, b3 = st.columns(3)
        if b1.button("Approve (Promote to Gold)"):
            with get_session(engine) as session:
                obj = session.get(TestCase, chosen_tc_id)
                if obj is not None:
                    obj.status = "approved"
                    obj.trust_tier = "gold"
                    obj.expected_checklist_json = {
                        "must_do": [x.strip() for x in must_do_text.splitlines() if x.strip()],
                        "must_not_do": [x.strip() for x in must_not_text.splitlines() if x.strip()],
                    }
            st.success(f"Approved {chosen_tc_id} as gold.")
        if b2.button("Reject"):
            with get_session(engine) as session:
                obj = session.get(TestCase, chosen_tc_id)
                if obj is not None:
                    obj.status = "rejected"
            st.warning(f"Rejected {chosen_tc_id}.")
        if b3.button("Save Checklist Edits"):
            with get_session(engine) as session:
                obj = session.get(TestCase, chosen_tc_id)
                if obj is not None:
                    obj.expected_checklist_json = {
                        "must_do": [x.strip() for x in must_do_text.splitlines() if x.strip()],
                        "must_not_do": [x.strip() for x in must_not_text.splitlines() if x.strip()],
                    }
            st.info(f"Updated checklist for {chosen_tc_id}.")

st.divider()
st.subheader("Frozen Benchmark Releases & Coverage")
if releases:
    rel_ids = [r.release_id for r in releases]
    chosen_rel = st.selectbox("Inspect Release", rel_ids)
    with get_session(engine) as session:
        ds_metrics = compute_dataset_metrics(session, chosen_rel)
    st.json(json.loads(json.dumps(ds_metrics)))

    sev_df = pd.DataFrame(
        [{"severity": k, "count": v} for k, v in (ds_metrics.get("by_severity") or {}).items()]
    )
    if not sev_df.empty:
        fig = px.bar(sev_df, x="severity", y="count", title=f"Severity Coverage ({chosen_rel})")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No frozen benchmark releases yet. Run `edg release create v1.0`.")
