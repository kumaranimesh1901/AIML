"""Page 3: Clusters (2D UMAP scatter with noise in grey, cluster descriptions, purity, DBCV & ARI)."""

from __future__ import annotations

import sys
from pathlib import Path

_DASH_DIR = str(Path(__file__).resolve().parents[1])
if _DASH_DIR not in sys.path:
    sys.path.insert(0, _DASH_DIR)

import pandas as pd
import plotly.express as px
import streamlit as st
from db_helper import get_engine
from sqlalchemy import select

from edg.db.models import Cluster, ClusteringRun, ClusterMembership, InteractionClean, InteractionRaw
from edg.db.session import get_session

st.set_page_config(page_title="Clusters | EDG", layout="wide")
st.title("3. Semantic Clusters & Stability")

engine = get_engine()
with get_session(engine) as session:
    runs = list(
        session.scalars(select(ClusteringRun).order_by(ClusteringRun.created_at.desc())).all()
    )

if not runs:
    st.info("No clustering runs found. Run `edg cluster run` first.")
    st.stop()

run_ids = [r.clustering_run_id for r in runs]
selected_run_id = st.selectbox("Clustering Run", run_ids)

df_runs = pd.DataFrame(
    [
        {
            "clustering_run_id": r.clustering_run_id,
            "n_clusters": r.n_clusters,
            "noise_share": r.noise_share,
            "dbcv": r.dbcv,
            "ari_vs_other_seeds": r.ari_vs_other_seeds,
            "seed": r.seed,
        }
        for r in runs
    ]
)
st.subheader("Clustering Run Metrics")
st.dataframe(df_runs, use_container_width=True)

with get_session(engine) as session:
    clusters = list(
        session.scalars(
            select(Cluster)
            .where(Cluster.clustering_run_id == selected_run_id)
            .order_by(Cluster.size.desc())
        ).all()
    )
    members = session.execute(
        select(
            ClusterMembership.interaction_id,
            ClusterMembership.cluster_id,
            ClusterMembership.probability,
            ClusterMembership.glosh_score,
            ClusterMembership.umap2d_x,
            ClusterMembership.umap2d_y,
            InteractionClean.clean_query,
            InteractionRaw.intent_label,
        )
        .join(InteractionClean, InteractionClean.interaction_id == ClusterMembership.interaction_id)
        .join(InteractionRaw, InteractionRaw.interaction_id == ClusterMembership.interaction_id)
        .where(ClusterMembership.clustering_run_id == selected_run_id)
    ).all()

if members:
    df_pts = pd.DataFrame(
        [
            {
                "interaction_id": m.interaction_id,
                "cluster": "Noise (-1)" if int(m.cluster_id) == -1 else f"Cluster {m.cluster_id}",
                "cluster_id": int(m.cluster_id),
                "x": float(m.umap2d_x),
                "y": float(m.umap2d_y),
                "glosh": float(m.glosh_score),
                "intent": m.intent_label or "unknown",
                "query": m.clean_query,
            }
            for m in members
        ]
    )
    color_map = {"Noise (-1)": "#9e9e9e"}
    fig = px.scatter(
        df_pts,
        x="x",
        y="y",
        color="cluster",
        color_discrete_map=color_map,
        hover_data=["interaction_id", "intent", "glosh", "query"],
        title="2D UMAP Projection (Plot Space Only; Clustering Ran in 10D)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Discovered Clusters")
    df_clusters = pd.DataFrame(
        [
            {
                "cluster_id": c.cluster_id,
                "size": c.size,
                "label": c.label,
                "mapped_intent": c.mapped_intent,
                "is_new_intent": c.is_new_intent,
                "description": c.description,
            }
            for c in clusters
        ]
    )
    st.dataframe(df_clusters, use_container_width=True)

    # Drill-down to individual records in selected cluster
    avail_cids = sorted(df_pts["cluster_id"].unique().tolist())
    chosen_cid = st.selectbox("Inspect cluster members", avail_cids)
    st.dataframe(
        df_pts[df_pts["cluster_id"] == chosen_cid][["interaction_id", "intent", "glosh", "query"]],
        use_container_width=True,
    )
