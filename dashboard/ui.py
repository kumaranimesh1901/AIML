"""Shared look and feel for the dashboard: page header, KPI cards, chart styling, empty states."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import plotly.graph_objects as go
import streamlit as st
from sqlalchemy.exc import OperationalError

# One palette for every chart, so v1/v2 and pass/fail mean the same colour on every page
COLORS = {
    "primary": "#4F46E5",
    "v1": "#94A3B8",
    "v2": "#4F46E5",
    "pass": "#16A34A",
    "fail": "#DC2626",
    "partial": "#F59E0B",
    "neutral": "#CBD5E1",
    "noise": "#D1D5DB",
}
VERDICT_COLORS = {
    "pass": COLORS["pass"],
    "partial": COLORS["partial"],
    "fail": COLORS["fail"],
    "not_assessable": COLORS["neutral"],
}
SEVERITY_ORDER = ["none", "low", "medium", "high", "critical"]
SEVERITY_COLORS = {
    "none": "#CBD5E1",
    "low": "#93C5FD",
    "medium": "#F59E0B",
    "high": "#F97316",
    "critical": "#DC2626",
}

_CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1300px;}
h1 {font-weight: 700; letter-spacing: -0.02em;}
.edg-sub {color: #64748B; font-size: 1.02rem; margin-top: -0.6rem; margin-bottom: 1.2rem;}
.edg-card {border: 1px solid rgba(100,116,139,0.25); border-radius: 14px; padding: 16px 18px;
           background: rgba(148,163,184,0.06); min-height: 124px; margin-bottom: 8px;}
.edg-card .lbl {color: #64748B; font-size: 0.82rem; text-transform: uppercase; letter-spacing: .04em;}
.edg-card .val {font-size: 1.85rem; font-weight: 700; line-height: 1.25; margin-top: 2px;}
.edg-card .hint {color: #64748B; font-size: 0.85rem; margin-top: 2px;}
.edg-card .up {color: #16A34A; font-weight: 600;}
.edg-card .down {color: #DC2626; font-weight: 600;}
.edg-chip {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.8rem;
           font-weight: 600; margin: 0 6px 4px 0; color: white;}
</style>
"""


def page_header(title: str, subtitle: str) -> None:
    """Page title with a one-line plain-English explanation underneath."""
    st.markdown(_CSS, unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="edg-sub">{subtitle}</div>', unsafe_allow_html=True)


def kpi(col: Any, label: str, value: str, hint: str = "", trend: str | None = None) -> None:
    """A KPI card. `trend` is 'up' or 'down' to colour the hint green or red."""
    hint_html = f'<div class="hint"><span class="{trend}">{hint}</span></div>' if trend else (
        f'<div class="hint">{hint}</div>' if hint else ""
    )
    col.markdown(
        f'<div class="edg-card"><div class="lbl">{label}</div><div class="val">{value}</div>{hint_html}</div>',
        unsafe_allow_html=True,
    )


def chip(text: str, color: str) -> str:
    return f'<span class="edg-chip" style="background:{color}">{text}</span>'


def verdict_chips(verdicts: dict[str, str]) -> str:
    return "".join(
        chip(f"{dim}: {v.replace('_', ' ')}", VERDICT_COLORS.get(v, COLORS["neutral"]))
        for dim, v in verdicts.items()
    )


def pct(value: float | None, digits: int = 0) -> str:
    return "–" if value is None else f"{value:.{digits}%}"


def num(value: float | int | None) -> str:
    return "–" if value is None else f"{value:,}"


def style_fig(fig: go.Figure, *, height: int = 360, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 48, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"size": 13},
        title={"font": {"size": 15}},
        showlegend=legend,
        legend={"orientation": "h", "yanchor": "top", "y": -0.08, "xanchor": "left", "x": 0, "title": None},
        hoverlabel={"font_size": 13},
    )
    fig.update_xaxes(showgrid=False, title=None)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.25)", title=None)
    return fig


def show_chart(fig: go.Figure, **kwargs: Any) -> None:
    st.plotly_chart(style_fig(fig, **kwargs), use_container_width=True, config={"displayModeBar": False})


def not_ready(what: str, step: str) -> None:
    """Friendly empty state for a page whose pipeline step has not run yet, then stop the page."""
    st.info(
        f"**{what} isn't available yet.** It appears once the pipeline reaches the *{step}* step. "
        "If a run is in progress, reload this page later.",
        icon=":material/hourglass_top:",
    )
    st.stop()


def load_or_wait(loader: Callable[[], Any]) -> Any:
    """Run a DB read; if SQLite is busy (a pipeline run is writing), ask the user to reload."""
    try:
        return loader()
    except OperationalError as exc:
        st.warning(
            f"The database is busy (a pipeline run may be writing to it): {exc.orig}. Reload in a moment.",
            icon=":material/sync:",
        )
        st.stop()
