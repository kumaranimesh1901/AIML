"""Page 2: Interaction Explorer with search, filters, snapshot, judgements, and lineage drill-down."""

from __future__ import annotations

import sys
from pathlib import Path

_DASH_DIR = str(Path(__file__).resolve().parents[1])
if _DASH_DIR not in sys.path:
    sys.path.insert(0, _DASH_DIR)

import pandas as pd
import streamlit as st
from db_helper import get_engine
from sqlalchemy import select

from edg.db.models import InteractionClean, InteractionRaw, Judgement, Label, StateSnapshot
from edg.db.session import get_session

st.set_page_config(page_title="Interaction Explorer | EDG", layout="wide")
st.title("2. Interaction Explorer")

engine = get_engine()
with get_session(engine) as session:
    rows = session.execute(
        select(InteractionRaw, InteractionClean, Label)
        .outerjoin(InteractionClean, InteractionClean.interaction_id == InteractionRaw.interaction_id)
        .outerjoin(Label, Label.interaction_id == InteractionRaw.interaction_id)
        .order_by(InteractionRaw.interaction_id)
    ).all()

if not rows:
    st.info("No interactions found in database.")
    st.stop()

df = pd.DataFrame(
    [
        {
            "interaction_id": raw.interaction_id,
            "model_version": raw.model_version,
            "source_tag": raw.source_tag,
            "intent": raw.intent_label or "unknown",
            "snapshot_id": raw.snapshot_id,
            "failure_detected": bool(lbl.failure_detected) if lbl else False,
            "severity": lbl.severity if lbl else "unlabelled",
            "query": clean.clean_query if (clean and clean.valid) else raw.user_query,
            "response": clean.clean_response if (clean and clean.valid) else raw.ai_response,
            "run_id": raw.run_id,
        }
        for raw, clean, lbl in rows
    ]
)

f1, f2, f3, f4 = st.columns(4)
search_term = f1.text_input("Search query", "")
source_filter = f2.multiselect("Source tag", sorted(df["source_tag"].unique().tolist()))
intent_filter = f3.multiselect("Intent", sorted(df["intent"].unique().tolist()))
only_failures = f4.checkbox("Failures only", value=False)

filtered = df.copy()
if search_term:
    filtered = filtered[filtered["query"].str.contains(search_term, case=False, na=False)]
if source_filter:
    filtered = filtered[filtered["source_tag"].isin(source_filter)]
if intent_filter:
    filtered = filtered[filtered["intent"].isin(intent_filter)]
if only_failures:
    filtered = filtered[filtered["failure_detected"]]

st.caption(f"Showing {len(filtered)} of {len(df)} interactions")
st.dataframe(
    filtered[["interaction_id", "model_version", "source_tag", "intent", "severity", "failure_detected", "query"]],
    use_container_width=True,
)

if not filtered.empty:
    selected_id = st.selectbox("Drill down into interaction", filtered["interaction_id"].tolist())
    with get_session(engine) as session:
        raw_obj = session.get(InteractionRaw, selected_id)
        snap_obj = session.get(StateSnapshot, raw_obj.snapshot_id) if raw_obj else None
        jdg_list = list(
            session.scalars(
                select(Judgement).where(Judgement.interaction_id == selected_id)
            ).all()
        )

    if raw_obj is not None:
        left, right = st.columns(2)
        with left:
            st.subheader("Query & Response")
            st.markdown(f"**User Query:** {raw_obj.user_query}")
            st.markdown(f"**AI Response (`{raw_obj.model_version}`):** {raw_obj.ai_response}")
            if raw_obj.reference_answer:
                st.markdown(f"**Reference Answer:** {raw_obj.reference_answer}")
            st.subheader("Lineage")
            st.json(
                {
                    "interaction_id": raw_obj.interaction_id,
                    "conversation_id": raw_obj.conversation_id,
                    "snapshot_id": raw_obj.snapshot_id,
                    "source_tag": raw_obj.source_tag,
                    "injected_failure": raw_obj.injected_failure,
                    "run_id": raw_obj.run_id,
                }
            )
        with right:
            st.subheader("Bound State Snapshot")
            if snap_obj is not None:
                st.json(snap_obj.snapshot_json)
            st.subheader("Judgements")
            for j in jdg_list:
                st.json(
                    {
                        "judgement_id": j.judgement_id,
                        "judge_model": j.judge_model,
                        "judge_version": j.judge_version,
                        "evidence_status": j.evidence_status,
                        "correctness": [j.correctness_verdict, j.correctness_score],
                        "relevance": [j.relevance_verdict, j.relevance_score],
                        "helpfulness": [j.helpfulness_verdict, j.helpfulness_score],
                        "completeness": [j.completeness_verdict, j.completeness_score],
                        "safety": [j.safety_verdict, j.safety_score],
                        "failure_types": j.failure_types,
                        "confidence": j.confidence,
                        "explanation": j.explanation,
                    }
                )
