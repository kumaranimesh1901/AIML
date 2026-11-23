"""Home: what the system does, the headline results, and how far the pipeline has got."""

from __future__ import annotations

import sys
from pathlib import Path

_DASH_DIR = str(Path(__file__).resolve().parents[1])
if _DASH_DIR not in sys.path:
    sys.path.insert(0, _DASH_DIR)

import plotly.express as px
import queries as q
import streamlit as st
from ui import COLORS, kpi, load_or_wait, page_header, pct, show_chart

page_header(
    "NovaMart AI Support: Automated Evaluation",
    "Turns a support chatbot's conversation logs into a tested benchmark: it groups customer "
    "questions into topics, finds the risky edge cases, has an AI judge grade the answers, and "
    "compares chatbot versions.",
)

db = q.db()
stages = load_or_wait(lambda: q.stage_counts(db))

# ---- Pipeline progress ---------------------------------------------------------------------------
done = sum(1 for _, _, n in stages if n > 0)
with st.container(border=True):
    left, right = st.columns([3, 1])
    left.markdown(f"**Pipeline progress:** {done} of {len(stages)} steps have data")
    right.page_link("views/chat.py", label="Try the chatbot", icon=":material/forum:")
    st.progress(done / len(stages))
    cols = st.columns(5)
    for i, (name, unit, n) in enumerate(stages):
        icon = ":material/check_circle:" if n > 0 else ":material/radio_button_unchecked:"
        detail = f"{n:,} {unit}" if n > 0 else "waiting"
        cols[i % 5].markdown(f"{icon} **{name}**  \n:gray[{detail}]")

if stages[1][2] == 0:
    st.info(
        "No conversations yet. Once the pipeline logs the chatbot's answers, results appear here. "
        "You can already chat with the assistant on the **Chat Assistant** page once the store is built.",
        icon=":material/hourglass_top:",
    )
    st.stop()

# ---- Headline numbers ------------------------------------------------------------------------------
conv = q.conversation_counts(db)
clus = q.latest_clustering(db)
lift = q.lift_table(q.sampled_labels(db))
recall = q.controlled_recall(db)
comp = q.headline_comparison(db)
release = q.release_case_count(db)

st.subheader("Headline results")
c1, c2, c3, c4, c5 = st.columns(5)
kpi(c1, "Conversations analysed", f"{conv['natural']:,}", f"+{conv['planted']:,} with planted mistakes")
kpi(
    c2,
    "Topics discovered",
    f"{clus['n_clusters']:,}" if clus else "–",
    f"{clus['new_intents']} not in the original intent list" if clus else "after clustering",
)
rc = lift[lift["group"] == "Random sample"]["failure_rate"]
top = lift[lift["group"].str.startswith("Top 50")]["failure_rate"]
rc_rate = float(rc.iloc[0]) if len(rc) and rc.iloc[0] is not None else None
top_rate = float(top.iloc[0]) if len(top) else None
kpi(
    c3,
    "Edge-case hit rate",
    f"{top_rate / rc_rate:.1f}×" if (top_rate is not None and rc_rate) else "–",
    f"top 50 fail {pct(top_rate)} vs random {pct(rc_rate)}" if top_rate is not None else "after judging",
)
kpi(
    c4,
    "Planted mistakes caught",
    pct(recall["overall_recall"]) if recall["total_controlled"] else "–",
    f"of {recall['total_controlled']} planted failures" if recall["total_controlled"] else "after judging",
)
if comp and comp["baseline_value"] is not None and comp["candidate_value"] is not None:
    delta = comp["candidate_value"] - comp["baseline_value"]
    kpi(
        c5,
        f"{comp['candidate']} pass rate",
        pct(comp["candidate_value"]),
        f"{delta * 100:+.0f} pts vs {comp['baseline']} ({pct(comp['baseline_value'])})",
        trend="up" if delta >= 0 else "down",
    )
else:
    kpi(c5, "Version comparison", "–", "after the benchmark replay")

# ---- Charts ----------------------------------------------------------------------------------------
left, right = st.columns(2)
with left:
    ft = q.failure_type_counts(db)
    if not ft.empty:
        ft["failure_type"] = ft["failure_type"].str.replace("_", " ")
        fig = px.bar(
            ft.sort_values("count"),
            x="count",
            y="failure_type",
            orientation="h",
            title="What goes wrong in real answers",
            color_discrete_sequence=[COLORS["fail"]],
        )
        fig.update_traces(hovertemplate="%{y}: %{x} answers<extra></extra>")
        show_chart(fig, legend=False)
    else:
        st.info("Failure types appear after the judge and labelling steps.", icon=":material/hourglass_top:")

with right:
    if comp and comp["baseline_value"] is not None:
        fig = px.bar(
            x=[comp["baseline"], comp["candidate"]],
            y=[comp["baseline_value"], comp["candidate_value"]],
            text=[pct(comp["baseline_value"], 1), pct(comp["candidate_value"], 1)],
            color=[comp["baseline"], comp["candidate"]],
            color_discrete_sequence=[COLORS["v1"], COLORS["v2"]],
            title=f"Pass rate on the frozen benchmark ({comp['n_cases']} test cases)",
        )
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        fig.update_traces(textposition="outside", hovertemplate="%{x}: %{y:.1%}<extra></extra>")
        show_chart(fig, legend=False)
        st.page_link("views/comparison.py", label="See where the versions differ", icon=":material/arrow_forward:")
    elif not lift.empty and lift["n"].sum() > 0:
        fig = px.bar(
            lift,
            x="group",
            y="failure_rate",
            text=lift["failure_rate"].map(lambda v: pct(v)),
            title="Failure rate: edge cases vs a random sample",
            color_discrete_sequence=[COLORS["primary"]],
        )
        fig.update_yaxes(tickformat=".0%")
        show_chart(fig, legend=False)
    else:
        st.info("The version comparison appears after the benchmark replay.", icon=":material/hourglass_top:")

if release:
    st.caption(f"Benchmark release `{release[0]}` holds {release[1]:,} approved test cases.")
