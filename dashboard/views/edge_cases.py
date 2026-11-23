"""Edge Cases: questions ranked by how likely the chatbot is to get them wrong."""

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
from ui import COLORS, kpi, load_or_wait, not_ready, page_header, pct, show_chart

from edg.db.models import InteractionClean, JudgeSample, Label, PriorityScore
from edg.db.session import get_session

SIGNALS = {
    "novelty": "Unusual wording",
    "small_cluster": "Rare topic",
    "multi_intent": "Several requests at once",
    "state_conflict": "Conflicts with the order record",
    "policy_boundary": "Near a policy limit",
    "response_drift": "Answer drifts from the question",
}

page_header(
    "Edge Cases",
    "Each question gets a priority score from six risk signals. The judge then grades the "
    "riskiest ones first, and they fail far more often than a random sample.",
)


@st.cache_data(ttl=q.TTL, show_spinner=False)
def _load(_db: str) -> pd.DataFrame:
    with get_session(get_engine()) as s:
        rows = s.execute(
            select(PriorityScore, InteractionClean.clean_query, JudgeSample.stratum, Label.failure_detected)
            .join(InteractionClean, InteractionClean.interaction_id == PriorityScore.interaction_id)
            .outerjoin(JudgeSample, JudgeSample.interaction_id == PriorityScore.interaction_id)
            .outerjoin(Label, Label.interaction_id == PriorityScore.interaction_id)
            .order_by(PriorityScore.rank)
        ).all()
    return pd.DataFrame(
        [
            {
                "Rank": ps.rank,
                "Question": query,
                "Score": ps.total_score,
                **{k: getattr(ps, k) for k in SIGNALS},
                "stratum": stratum,
                "Judge": {True: "Fail", False: "Pass"}.get(fail, "Not judged"),
            }
            for ps, query, stratum, fail in rows
        ]
    )


df = load_or_wait(lambda: _load(q.db()))
if df.empty:
    not_ready("Edge cases", "priority scoring")

lift = q.lift_table(q.sampled_labels(q.db()))
rc = lift[lift["group"] == "Random sample"]["failure_rate"]
rc_rate = float(rc.iloc[0]) if len(rc) and pd.notna(rc.iloc[0]) else None
top50 = lift[lift["group"] == "Top 50 edge cases"]["failure_rate"]
top_rate = float(top50.iloc[0]) if len(top50) else None

k1, k2, k3 = st.columns(3)
kpi(k1, "Questions scored", f"{len(df):,}")
kpi(k2, "Top 50 failure rate", pct(top_rate), f"vs {pct(rc_rate)} for a random sample")
kpi(
    k3,
    "Lift",
    f"{top_rate / rc_rate:.1f}×" if (top_rate is not None and rc_rate) else "–",
    "more failures found per answer checked",
)
st.write("")

left, right = st.columns(2)
with left:
    if lift["n"].sum() > 0:
        fig = px.bar(
            lift,
            x="group",
            y="failure_rate",
            text=lift["failure_rate"].map(pct),
            color="group",
            color_discrete_sequence=[COLORS["neutral"], COLORS["fail"], COLORS["partial"]],
            custom_data=["n"],
            title="How often the chatbot fails",
        )
        fig.update_traces(textposition="outside", hovertemplate="%{x}: %{y:.1%} of %{customdata[0]} answers<extra></extra>")
        fig.update_yaxes(tickformat=".0%", range=[0, max(0.1, float(lift["failure_rate"].max() or 0) * 1.25)])
        show_chart(fig, legend=False)
    else:
        st.info("Failure rates appear once the judge has graded the sample.", icon=":material/hourglass_top:")
with right:
    top_k = st.select_slider("Compare the top", options=[25, 50, 100, 200], value=50)
    top = df.head(top_k)[list(SIGNALS)].mean()
    rest = df[list(SIGNALS)].mean()
    sig = pd.DataFrame(
        {
            "signal": [SIGNALS[k] for k in SIGNALS] * 2,
            "value": list(top.values) + list(rest.values),
            "group": [f"Top {top_k}"] * len(SIGNALS) + ["All questions"] * len(SIGNALS),
        }
    )
    fig = px.bar(
        sig,
        x="value",
        y="signal",
        color="group",
        barmode="group",
        orientation="h",
        color_discrete_sequence=[COLORS["fail"], COLORS["neutral"]],
        title="What makes them risky (average signal, 0 to 1)",
    )
    show_chart(fig)

st.subheader("Riskiest questions")
min_score = st.slider("Minimum priority score", 0.0, 1.0, 0.0, 0.05)
view = df[df["Score"] >= min_score]
st.dataframe(
    view[["Rank", "Question", "Score", "Judge"]],
    use_container_width=True,
    hide_index=True,
    height=420,
    column_config={
        "Score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
        "Question": st.column_config.TextColumn(width="large"),
    },
)
with st.expander("All six signal scores"):
    st.dataframe(
        view[["Rank", "Question", *SIGNALS]].rename(columns=SIGNALS),
        use_container_width=True,
        hide_index=True,
    )
