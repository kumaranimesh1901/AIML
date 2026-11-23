"""Shared database engine helper for Streamlit dashboard pages."""

from __future__ import annotations

import os

import streamlit as st
from sqlalchemy import Engine

from edg.config import load_config
from edg.db.demo import seed_from_demo
from edg.db.session import create_db_engine


def get_database_url() -> str:
    env_url = os.getenv("EDG_DATABASE_URL")
    if env_url:
        return env_url
    return load_config().database.url


@st.cache_resource
def get_dashboard_engine(database_url: str) -> Engine:
    # A fresh clone that skipped `edg db init` still opens with the bundled real results
    seed_from_demo(database_url)
    return create_db_engine(database_url, echo=False)


def get_engine() -> Engine:
    return get_dashboard_engine(get_database_url())
