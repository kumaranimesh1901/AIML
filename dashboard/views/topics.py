"""Topics & Clusters: the groups of similar customer questions the pipeline discovered."""

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

from edg.db.models import Cluster, ClusteringRun, ClusterMembership, InteractionClean, InteractionRaw
from edg.db.session import get_session

page_header(
    "Topics & Clusters",
    "Customer questions are turned into embeddings and grouped by meaning (UMAP + HDBSCAN). Each "
    "dot is a question; questions that fit no group are grey.",
)


@st.cache_data(ttl=q.TTL, show_spinner=False)
def _runs(_db: str) -> pd.DataFrame:
    with get_session(get_engine()) as s:
        runs = s.scalars(select(ClusteringRun).order_by(ClusteringRun.created_at.desc())).all()
        return pd.DataFrame(
            [
                {
                    "run": r.clustering_run_id,
                    "clusters": r.n_clusters,
                    "noise": r.noise_share,
                    "dbcv": r.dbcv,
                    "ari": r.ari_vs_other_seeds,
                    "seed": r.seed,
                }
                for r in runs
            ]
        )


@st.cache_data(ttl=q.TTL, show_spinner=False)
def _clusters(_db: str, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    with get_session(get_engine()) as s:
        clusters = s.scalars(
            select(Cluster).where(Cluster.clustering_run_id == run_id).order_by(Cluster.size.desc())
        ).all()
        members = s.execute(
            select(
                ClusterMembership.cluster_id,
                ClusterMembership.umap2d_x,
                ClusterMembership.umap2d_y,
                InteractionClean.clean_query,
                InteractionRaw.intent_label,
            )
            .join(InteractionClean, InteractionClean.interaction_id == ClusterMembership.interaction_id)
            .join(InteractionRaw, InteractionRaw.interaction_id == ClusterMembership.interaction_id)
            .where(ClusterMembership.clustering_run_id == run_id)
        ).all()
    df_c = pd.DataFrame(
        [
            {
                "cluster_id": c.cluster_id,
                "Topic": c.label or f"Cluster {c.cluster_id}",
                "Questions": c.size,
                "Matches intent": (c.mapped_intent or "").replace("_", " "),
                "New topic": bool(c.is_new_intent),
                "description": c.description or "",
            }
            for c in clusters
        ]
    )
    labels = dict(zip(df_c["cluster_id"], df_c["Topic"], strict=False)) if not df_c.empty else {}
    df_p = pd.DataFrame(
        [
            {
                "cluster_id": int(m.cluster_id),
                "Topic": "No group" if int(m.cluster_id) == -1 else labels.get(int(m.cluster_id), f"Cluster {m.cluster_id}"),
                "x": float(m.umap2d_x),
                "y": float(m.umap2d_y),
                "question": m.clean_query,
                "intent": (m.intent_label or "unknown").replace("_", " "),
            }
            for m in members
        ]
    )
    return df_c, df_p


runs = load_or_wait(lambda: _runs(q.db()))
if runs.empty:
    not_ready("Topics", "clustering")

run_id = runs.iloc[0]["run"]
if len(runs) > 1:
    with st.expander("Clustering run"):
        run_id = st.selectbox("Run", runs["run"].tolist(), label_visibility="collapsed")
run = runs[runs["run"] == run_id].iloc[0]
df_c, df_p = _clusters(q.db(), run_id)

k1, k2, k3, k4 = st.columns(4)
kpi(k1, "Topics found", f"{int(run['clusters']):,}", f"{int(df_c['New topic'].sum()) if not df_c.empty else 0} new vs the intent list")
kpi(k2, "Questions grouped", pct(1 - float(run["noise"])), "the rest fit no topic")
kpi(k3, "Stability", f"{run['ari']:.2f}" if pd.notna(run["ari"]) else "–", "same groups across 3 seeds (ARI, 1 = identical)")
kpi(k4, "Cluster quality", f"{run['dbcv']:.2f}" if pd.notna(run["dbcv"]) else "–", "DBCV, higher is better (max 1)")
st.write("")

if df_p.empty:
    st.info("No cluster members recorded for this run.")
    st.stop()

if len(df_c) > 5:
    top_n = st.slider("Topics to highlight", 5, min(30, len(df_c)), min(12, len(df_c)))
else:
    top_n = len(df_c)
top_topics = set(df_c.head(top_n)["Topic"])
plot = df_p.copy()
plot["Group"] = plot["Topic"].where(plot["Topic"].isin(top_topics) | (plot["cluster_id"] == -1), "Other topics")
order = list(df_c.head(top_n)["Topic"]) + ["Other topics", "No group"]
fig = px.scatter(
    plot,
    x="x",
    y="y",
    color="Group",
    category_orders={"Group": order},
    color_discrete_map={"No group": COLORS["noise"], "Other topics": "#E2E8F0"},
    hover_data={"question": True, "intent": True, "x": False, "y": False, "Group": False, "Topic": True},
    title="Map of customer questions (similar questions sit close together)",
)
fig.update_traces(marker={"size": 7, "opacity": 0.85})
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False, showgrid=False)
show_chart(fig, height=520)

left, right = st.columns([1, 1])
with left:
    bars = df_c.head(top_n).sort_values("Questions")
    fig = px.bar(
        bars,
        x="Questions",
        y="Topic",
        orientation="h",
        color="New topic",
        color_discrete_map={True: COLORS["partial"], False: COLORS["primary"]},
        title="Largest topics (orange = not in the original intent list)",
    )
    show_chart(fig, height=max(320, 26 * len(bars)), legend=False)
with right:
    st.markdown("##### Look inside a topic")
    pick = st.selectbox("Topic", df_c["Topic"].tolist(), label_visibility="collapsed")
    info = df_c[df_c["Topic"] == pick].iloc[0]
    st.caption(
        f"{info['Questions']} questions · "
        + (f"matches intent *{info['Matches intent']}*" if info["Matches intent"] else "no matching intent")
        + (" · **new topic**" if info["New topic"] else "")
    )
    if info["description"]:
        st.markdown(info["description"])
    samples = df_p[df_p["cluster_id"] == info["cluster_id"]]["question"].head(8)
    for s in samples:
        st.markdown(f"- {s}")
