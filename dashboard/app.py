"""Streamlit main entrypoint for Automated Evaluation Dataset Generator (EDG)."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from db_helper import get_database_url, get_engine
from sqlalchemy import select

from edg.db.models import PipelineRun
from edg.db.session import get_session

st.set_page_config(page_title="EDG Dashboard", layout="wide")
st.title("Automated Evaluation Dataset Generator (EDG)")
st.caption(f"Connected database: `{get_database_url()}` — all values read live from database tables.")

engine = get_engine()
with get_session(engine) as session:
    recent_runs = list(
        session.scalars(
            select(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(10)
        ).all()
    )

st.subheader("Recent Pipeline Runs")
if recent_runs:
    df_runs = pd.DataFrame(
        [
            {
                "run_id": r.run_id,
                "step": r.step,
                "status": r.status,
                "config_hash": r.config_hash[:12],
                "git_commit": r.git_commit,
                "started_at": str(r.started_at),
            }
            for r in recent_runs
        ]
    )
    st.dataframe(df_runs, use_container_width=True)
else:
    st.info("No pipeline runs recorded yet. Run `edg run-all --small` to populate the database.")
