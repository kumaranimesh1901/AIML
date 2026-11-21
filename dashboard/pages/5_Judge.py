"""Page 5: Judge verdicts, evidence & claim checks side-by-side, agreement metrics, and annotation form."""

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

from edg.annotation.agreement import _evaluate_split
from edg.annotation.ui_helpers import save_single_annotation
from edg.db.models import HumanAnnotation, InteractionRaw, Judgement, StateSnapshot
from edg.db.session import get_session
from edg.judge.schema import DIMENSIONS
from edg.world.taxonomy import load_taxonomy

st.set_page_config(page_title="Judge | EDG", layout="wide")
st.title("5. Hybrid LLM Judge & Human Calibration")

engine = get_engine()
with get_session(engine) as session:
    judgements = list(
        session.scalars(select(Judgement).order_by(Judgement.created_at.desc()).limit(200)).all()
    )
    ann_count = len(list(session.scalars(select(HumanAnnotation.annotation_id)).all()))

if not judgements:
    st.info("No judgements found. Run `edg judge run` first.")
    st.stop()

st.subheader("Inspect Judgement, Evidence & Claim Checks")
j_ids = [j.judgement_id for j in judgements]
chosen_jid = st.selectbox("Select Judgement", j_ids)

with get_session(engine) as session:
    jdg = session.get(Judgement, chosen_jid)
    raw = session.get(InteractionRaw, jdg.interaction_id) if jdg else None
    snap = session.get(StateSnapshot, raw.snapshot_id) if raw else None

if jdg is not None and raw is not None:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Interaction:** `{raw.interaction_id}` | **Model:** `{raw.model_version}`")
        st.markdown(f"**User Query:** {raw.user_query}")
        st.markdown(f"**AI Response:** {raw.ai_response}")
        st.markdown(f"**Judge Model:** `{jdg.judge_model}` (`{jdg.judge_version}`)")
        st.markdown(f"**Evidence Status:** `{jdg.evidence_status}` | **Confidence:** `{jdg.confidence:.2f}`")
        st.markdown(f"**Failure Types:** `{jdg.failure_types}`")
        st.markdown(f"**Explanation:** {jdg.explanation}")

        dim_df = pd.DataFrame(
            [
                {
                    "dimension": d,
                    "verdict": getattr(jdg, f"{d}_verdict"),
                    "score": getattr(jdg, f"{d}_score"),
                }
                for d in DIMENSIONS
            ]
        )
        st.table(dim_df)
    with c2:
        st.markdown("**Deterministic Claim Checks**")
        st.json(jdg.claim_checks_json)
        st.markdown("**Bound State Snapshot**")
        if snap is not None:
            st.json(snap.snapshot_json)

st.divider()
st.subheader("Human-Judge Agreement & Controlled-Tier Recall")
split_choice = st.selectbox("Calibration Split", ["judge_dev", "judge_test"])
with get_session(engine) as session:
    cal_report = _evaluate_split(session, split=split_choice)

st.caption(f"Total human annotations in DB: {ann_count}")
if cal_report["agreement"]["n_paired"] > 0:
    ord_rows = [
        {"dimension": d, **vals}
        for d, vals in cal_report["agreement"]["ordinal_by_dimension"].items()
    ]
    st.dataframe(pd.DataFrame(ord_rows), use_container_width=True)
else:
    st.info(f"No human annotations imported for split `{split_choice}` yet.")

ctrl_rec = cal_report["controlled_tier_recall"]
if ctrl_rec["total_controlled"] > 0:
    st.markdown(f"**Controlled-Tier Overall Recall:** `{ctrl_rec['overall_recall']:.1%}` (n={ctrl_rec['total_controlled']})")
    st.dataframe(
        pd.DataFrame(
            [{"failure_type": k, **v} for k, v in ctrl_rec["per_failure_type"].items()]
        ),
        use_container_width=True,
    )

st.divider()
st.subheader("Annotate Selected Interaction")
if raw is not None:
    taxonomy = load_taxonomy()
    with st.form("human_annotation_form"):
        annotator = st.text_input("Annotator ID", value="annotator_1")
        ann_split = st.selectbox("Split", ["judge_dev", "judge_test"], key="form_split")
        verdicts: dict[str, str] = {}
        scores: dict[str, int | None] = {}
        cols = st.columns(len(DIMENSIONS))
        for idx, dim in enumerate(DIMENSIONS):
            with cols[idx]:
                verdicts[dim] = st.selectbox(
                    f"{dim} verdict", ["pass", "partial", "fail", "not_assessable"], key=f"v_{dim}"
                )
                scores[dim] = st.number_input(
                    f"{dim} score", min_value=1, max_value=5, value=5, key=f"s_{dim}"
                )
        ft_selected = st.multiselect("Failure Types", sorted(taxonomy.failure_types.keys()))
        notes = st.text_area("Notes", "")
        submitted = st.form_submit_button("Save Annotation")
        if submitted:
            with get_session(engine) as session:
                save_single_annotation(
                    session,
                    interaction_id=raw.interaction_id,
                    annotator_id=annotator,
                    split=ann_split,
                    verdicts=verdicts,
                    scores={k: (None if verdicts[k] == "not_assessable" else int(v)) for k, v in scores.items()},
                    failure_types=ft_selected,
                    notes=notes,
                )
            st.success(f"Saved annotation for {raw.interaction_id}.")
