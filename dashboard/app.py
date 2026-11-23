"""Streamlit entrypoint for the EDG dashboard: page navigation and shared sidebar."""

from __future__ import annotations

import sys
from pathlib import Path

_DASH_DIR = Path(__file__).resolve().parent
# The dashboard helpers, and the `edg` package itself (so a `streamlit` from outside the
# project's virtual environment can still find it)
for _path in (_DASH_DIR.parent / "src", _DASH_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import streamlit as st

try:
    from db_helper import get_database_url
    from model_check import render_model_status

    from edg.config import load_config
except ImportError as exc:  # streamlit started from a Python without the project's packages
    st.error(
        f"This Streamlit is running on a Python that lacks the project's packages ({exc}). "
        "Start it from the project's virtual environment instead:\n\n"
        "```\nsource .venv/bin/activate\npython -m streamlit run dashboard/app.py\n```"
    )
    st.stop()

st.set_page_config(
    page_title="EDG · NovaMart AI Support Evaluation",
    page_icon=":material/insights:",
    layout="wide",
)


def _page(file: str, title: str, icon: str, **kwargs: object) -> st.Page:
    return st.Page(str(_DASH_DIR / "views" / file), title=title, icon=icon, **kwargs)


navigation = st.navigation(
    {
        "Overview": [
            _page("home.py", "Home", ":material/home:", default=True),
            _page("chat.py", "Chat Assistant", ":material/forum:"),
        ],
        "Support": [
            _page("support.py", "Support Desk", ":material/support_agent:"),
        ],
        "How the pipeline works": [
            _page("conversations.py", "Conversations", ":material/chat_bubble:"),
            _page("topics.py", "Topics & Clusters", ":material/bubble_chart:"),
            _page("edge_cases.py", "Edge Cases", ":material/warning:"),
            _page("judge.py", "AI Judge", ":material/gavel:"),
        ],
        "Results": [
            _page("dataset.py", "Test Dataset", ":material/fact_check:"),
            _page("comparison.py", "v1 vs v2", ":material/compare_arrows:"),
        ],
    }
)

with st.sidebar:
    st.caption(f"Data: `{Path(get_database_url().removeprefix('sqlite:///')).name}` · every number is read live")
    render_model_status(load_config(), compact=True)

navigation.run()
