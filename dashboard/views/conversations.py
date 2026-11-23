"""Conversations: browse what customers asked, how the chatbot answered, and what the judge said."""

from __future__ import annotations

import sys
from pathlib import Path

_DASH_DIR = str(Path(__file__).resolve().parents[1])
if _DASH_DIR not in sys.path:
    sys.path.insert(0, _DASH_DIR)

import pandas as pd
import queries as q
import streamlit as st
from db_helper import get_engine
from sqlalchemy import select
from ui import COLORS, chip, kpi, load_or_wait, not_ready, page_header, pct, verdict_chips

from edg.db.models import InteractionClean, InteractionRaw, Judgement, Label, StateSnapshot
from edg.db.repositories import source_judgements_only
from edg.db.session import get_session
from edg.judge.schema import DIMENSIONS
from edg.sut.failure_injector import CONTROLLED_SOURCE_TAG

page_header(
    "Conversations",
    "Every logged customer question with the chatbot's answer. Pick a row to read the whole "
    "conversation and the judge's verdict.",
)

engine = get_engine()


@st.cache_data(ttl=q.TTL, show_spinner=False)
def _load(_db: str) -> pd.DataFrame:
    with get_session(get_engine()) as session:
        rows = session.execute(
            select(InteractionRaw, InteractionClean, Label)
            .outerjoin(InteractionClean, InteractionClean.interaction_id == InteractionRaw.interaction_id)
            .outerjoin(Label, Label.interaction_id == InteractionRaw.interaction_id)
            .order_by(InteractionRaw.interaction_id)
        ).all()
    return pd.DataFrame(
        [
            {
                "id": raw.interaction_id,
                "Question": clean.clean_query if (clean and clean.valid) else raw.user_query,
                "Answer": clean.clean_response if (clean and clean.valid) else raw.ai_response,
                "Version": raw.model_version,
                "Topic": (raw.intent_label or "unknown").replace("_", " "),
                "Verdict": ("Fail" if lbl.failure_detected else "Pass") if lbl else "Not judged",
                "Severity": lbl.severity if (lbl and lbl.failure_detected) else "",
                "Planted": raw.source_tag == CONTROLLED_SOURCE_TAG,
            }
            for raw, clean, lbl in rows
        ]
    )


df = load_or_wait(lambda: _load(q.db()))
if df.empty:
    not_ready("Conversations", "SUT run")

# ---- Filters -----------------------------------------------------------------------------------------
f1, f2, f3 = st.columns([2, 1.6, 1.2])
search = f1.text_input("Search questions", placeholder="e.g. refund, cancel, charged twice", label_visibility="collapsed")
verdict = f2.segmented_control(
    "Verdict", ["All", "Fail", "Pass", "Not judged"], default="All", label_visibility="collapsed"
)
topics = f3.multiselect("Topic", sorted(df["Topic"].unique()), placeholder="All topics", label_visibility="collapsed")
show_planted = st.toggle(
    "Include answers with deliberately planted mistakes",
    value=False,
    help="The pipeline plants known mistakes in some answers to test whether the judge catches them.",
)

view = df if show_planted else df[~df["Planted"]]
if search:
    view = view[view["Question"].str.contains(search, case=False, na=False)]
if verdict and verdict != "All":
    view = view[view["Verdict"] == verdict]
if topics:
    view = view[view["Topic"].isin(topics)]

judged = view[view["Verdict"] != "Not judged"]
k1, k2, k3 = st.columns(3)
kpi(k1, "Conversations shown", f"{len(view):,}", f"of {len(df):,} logged")
kpi(k2, "Judged", f"{len(judged):,}", "the judge grades a sample, not every answer")
kpi(k3, "Failure rate", pct((judged["Verdict"] == "Fail").mean() if len(judged) else None), "among judged answers shown")
st.write("")

table = st.dataframe(
    view[["Question", "Topic", "Version", "Verdict", "Severity"]],
    use_container_width=True,
    hide_index=True,
    height=360,
    on_select="rerun",
    selection_mode="single-row",
    column_config={"Question": st.column_config.TextColumn(width="large")},
)

picked = table.selection.rows if table and table.selection else []
if not picked:
    st.caption("Select a row to open the conversation.")
    st.stop()

row = view.iloc[picked[0]]
with get_session(engine) as session:
    raw = session.get(InteractionRaw, row["id"])
    snap = session.get(StateSnapshot, raw.snapshot_id) if raw else None
    jdg = session.scalars(
        select(Judgement)
        .where(Judgement.interaction_id == row["id"], source_judgements_only())
        .order_by(Judgement.created_at.desc())
        .limit(1)
    ).first()

st.divider()
left, right = st.columns([3, 2])
with left:
    st.subheader("Conversation")
    with st.chat_message("user"):
        st.markdown(row["Question"])
    with st.chat_message("assistant"):
        st.markdown(row["Answer"])
        st.caption(f"Chatbot {row['Version']}")
    if raw is not None and raw.injected_failure:
        st.warning(f"This answer had a **{raw.injected_failure.replace('_', ' ')}** planted on purpose.", icon=":material/science:")
with right:
    st.subheader("Judge's verdict")
    if jdg is None:
        st.info("This conversation wasn't in the sample the judge graded.", icon=":material/info:")
    else:
        failed = bool(jdg.failure_types) or any(getattr(jdg, f"{d}_verdict") == "fail" for d in DIMENSIONS)
        st.markdown(
            chip("Fail", COLORS["fail"]) if failed else chip("Pass", COLORS["pass"]),
            unsafe_allow_html=True,
        )
        st.markdown(verdict_chips({d: getattr(jdg, f"{d}_verdict") for d in DIMENSIONS}), unsafe_allow_html=True)
        if jdg.failure_types:
            st.markdown("**Problems:** " + ", ".join(f.replace("_", " ") for f in jdg.failure_types))
        st.markdown(f"**Why:** {jdg.explanation}")
        st.caption(f"Judge confidence {jdg.confidence:.0%}")

with st.expander("Order record the chatbot and judge could see"):
    if snap is not None:
        st.json(snap.snapshot_json, expanded=1)
with st.expander("Technical details"):
    if raw is not None:
        st.json(
            {
                "interaction_id": raw.interaction_id,
                "source_tag": raw.source_tag,
                "snapshot_id": raw.snapshot_id,
                "run_id": raw.run_id,
                "judge": f"{jdg.judge_model} ({jdg.judge_version})" if jdg else None,
                "evidence_status": jdg.evidence_status if jdg else None,
            }
        )
