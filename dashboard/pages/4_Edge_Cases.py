"""Page 4: Edge Cases, priority ranking with all 6 signals, and top-k vs random_control lift."""

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

from edg.db.models import InteractionClean, JudgeSample, Label, PriorityScore
from edg.db.session import get_session

st.set_page_config(page_title="Edge Cases | EDG", layout="wide")
st.title("4. Edge Cases & Priority Ranking")

engine = get_engine()
with get_session(engine) as session:
    rows = session.execute(
        select(PriorityScore, InteractionClean.clean_query, JudgeSample.stratum, Label.failure_detected)
        .join(InteractionClean, InteractionClean.interaction_id == PriorityScore.interaction_id)
        .outerjoin(JudgeSample, JudgeSample.interaction_id == PriorityScore.interaction_id)
        .outerjoin(Label, Label.interaction_id == PriorityScore.interaction_id)
        .order_by(PriorityScore.rank)
    ).all()

if not rows:
    st.info("No priority scores found. Run `edg priority run` first.")
    st.stop()

df = pd.DataFrame(
    [
        {
            "rank": ps.rank,
            "interaction_id": ps.interaction_id,
            "total_score": ps.total_score,
            "novelty": ps.novelty,
            "small_cluster": ps.small_cluster,
            "multi_intent": ps.multi_intent,
            "state_conflict": ps.state_conflict,
            "policy_boundary": ps.policy_boundary,
            "response_drift": ps.response_drift,
            "stratum": stratum or "unsampled",
            "failure_detected": lbl_fail,
            "query": query,
        }
        for ps, query, stratum, lbl_fail in rows
    ]
)

# Compute failure rate in top-k vs random_control (lift)
labelled = df[df["failure_detected"].notna()].copy()
rc_df = labelled[labelled["stratum"] == "random_control"]
rc_rate = float(rc_df["failure_detected"].mean()) if not rc_df.empty else 0.0

lift_rows = [{"group": "Random Control", "failure_rate": round(rc_rate, 4), "n": len(rc_df)}]
for k in (50, 100, 500):
    top_k = labelled.nsmallest(min(k, len(labelled)), "rank")
    if not top_k.empty:
        lift_rows.append(
            {
                "group": f"Top-{k}",
                "failure_rate": round(float(top_k["failure_detected"].mean()), 4),
                "n": len(top_k),
            }
        )

df_lift = pd.DataFrame(lift_rows)
fig = px.bar(
    df_lift,
    x="group",
    y="failure_rate",
    text="n",
    title="Failure Rate: Priority Top-K vs. Random Control",
)
st.plotly_chart(fig, use_container_width=True)

min_score = st.slider("Minimum priority score", 0.0, 1.0, 0.0, 0.05)
filtered = df[df["total_score"] >= min_score]
st.dataframe(filtered, use_container_width=True)
