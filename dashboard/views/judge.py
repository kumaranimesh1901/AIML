"""AI Judge: how the evidence-based judge grades answers, and how far it can be trusted."""

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
from ui import (
    COLORS,
    VERDICT_COLORS,
    chip,
    kpi,
    load_or_wait,
    not_ready,
    page_header,
    pct,
    show_chart,
    verdict_chips,
)

from edg.annotation.human_check import interpret_kappa
from edg.db.models import InteractionRaw, Judgement, StateSnapshot
from edg.db.repositories import source_judgements_only
from edg.db.session import get_session
from edg.judge.schema import DIMENSIONS
from edg.sut.failure_injector import CONTROLLED_SOURCE_TAG

page_header(
    "AI Judge",
    "A local LLM judge grades each sampled answer on five dimensions, using only the order record "
    "and store policies as evidence. Code checks every date, amount and status the answer claims.",
)


@st.cache_data(ttl=q.TTL, show_spinner=False)
def _load(_db: str) -> pd.DataFrame:
    with get_session(get_engine()) as s:
        rows = s.execute(
            select(Judgement, InteractionRaw.source_tag, InteractionRaw.user_query, InteractionRaw.ai_response)
            .join(InteractionRaw, InteractionRaw.interaction_id == Judgement.interaction_id)
            .where(source_judgements_only())
            .order_by(Judgement.created_at.desc())
        ).all()
    out = []
    seen: set[str] = set()
    for j, tag, query, answer in rows:
        if j.interaction_id in seen:
            continue  # latest judgement per answer
        seen.add(j.interaction_id)
        failed = bool(j.failure_types) or any(getattr(j, f"{d}_verdict") == "fail" for d in DIMENSIONS)
        out.append(
            {
                "judgement_id": j.judgement_id,
                "interaction_id": j.interaction_id,
                "planted": tag == CONTROLLED_SOURCE_TAG,
                "question": query,
                "answer": answer,
                "failed": failed,
                "error": "judge_error" in (j.failure_types or []),
                "failure_types": [f for f in (j.failure_types or []) if f != "judge_error"],
                "model": j.judge_model,
                "version": j.judge_version,
                **{d: getattr(j, f"{d}_verdict") for d in DIMENSIONS},
            }
        )
    return pd.DataFrame(out)


df = load_or_wait(lambda: _load(q.db()))
if df.empty:
    not_ready("Judge results", "judge")

real = df[~df["planted"]]
lift = q.lift_table(q.sampled_labels(q.db()))
rc = lift[lift["group"] == "Random sample"]["failure_rate"]
recall = q.controlled_recall(q.db())

k1, k2, k3, k4 = st.columns(4)
kpi(k1, "Answers judged", f"{len(real):,}", f"by {df['model'].iloc[0]} ({df['version'].iloc[0]})")
kpi(
    k2,
    "Failure rate",
    pct(float(rc.iloc[0])) if len(rc) and pd.notna(rc.iloc[0]) else "–",
    "on an unbiased random sample",
)
kpi(
    k3,
    "Planted mistakes caught",
    pct(recall["overall_recall"]) if recall["total_controlled"] else "–",
    f"right label on {recall['total_controlled']} planted failures" if recall["total_controlled"] else "",
)
kpi(k4, "Judge errors", pct(df["error"].mean(), 1), "no valid verdict after retries")
st.write("")

left, right = st.columns(2)
with left:
    dims = (
        real[list(DIMENSIONS)]
        .melt(var_name="dimension", value_name="verdict")
        .value_counts()
        .reset_index(name="answers")
    )
    fig = px.bar(
        dims,
        x="answers",
        y="dimension",
        color="verdict",
        orientation="h",
        color_discrete_map=VERDICT_COLORS,
        category_orders={"verdict": ["pass", "partial", "fail", "not_assessable"], "dimension": list(DIMENSIONS)},
        title="Verdicts on real answers, by dimension",
    )
    show_chart(fig)
with right:
    per = recall.get("per_failure_type") or {}
    if per:
        rec = pd.DataFrame(
            [{"type": k.replace("_", " "), "recall": v["recall"], "n": v["n"], "caught": v["detected"]} for k, v in per.items()]
        ).sort_values("recall")
        fig = px.bar(
            rec,
            x="recall",
            y="type",
            orientation="h",
            text=rec["recall"].map(pct),
            custom_data=["caught", "n"],
            color_discrete_sequence=[COLORS["primary"]],
            title="Planted mistakes caught, by type",
        )
        fig.update_traces(hovertemplate="%{y}: %{customdata[0]} of %{customdata[1]}<extra></extra>")
        fig.update_xaxes(tickformat=".0%", range=[0, 1.1])
        show_chart(fig, legend=False)
    else:
        st.info("No planted mistakes in this run.", icon=":material/info:")

# ---- Real human check (two people graded a blind sample) -----------------------------------------
report = q.human_check_report()
st.subheader("Does the judge agree with people?")
if report is None:
    st.caption(
        "Not measured for this database yet. Run `edg calibrate human-export`, have two people fill the "
        "sheet, then `edg calibrate human-import` to show real agreement here."
    )
else:
    pairs = report.get("pairs", {})
    cons = pairs.get("judge_vs_human_consensus", {})
    hh = pairs.get("annotator_1_vs_annotator_2", {})
    h1, h2, h3 = st.columns(3)
    kpi(
        h1,
        "Judge vs people (where both agree)",
        f"κ {cons['kappa']:.2f}" if cons.get("kappa") is not None else "–",
        f"{interpret_kappa(cons.get('kappa'))} · n={cons.get('n', 0)}",
    )
    kpi(
        h2,
        "Person vs person",
        f"κ {hh['kappa']:.2f}" if hh.get("kappa") is not None else "–",
        f"{interpret_kappa(hh.get('kappa'))} · n={hh.get('n', 0)}",
    )
    rates = [report.get("judge_fail_rate_on_sheet"), report.get("annotator_1_fail_rate"), report.get("annotator_2_fail_rate")]
    kpi(h3, "Fail rate: judge / A1 / A2", " / ".join(pct(r) for r in rates), f"on {report.get('items_in_sheet', 0)} blind items")
    note = f"Two people graded a blind sample on {str(report.get('generated_at', ''))[:10]} against judge {report.get('judge_model')} ({report.get('judge_version')})."
    if report.get("judge_version") != df["version"].iloc[0]:
        note += " That was a different judge version from the one in this database."
    st.caption(note + " Cohen's κ: 1 is perfect agreement, 0 is chance.")

# ---- Inspect one graded answer ---------------------------------------------------------------------
st.subheader("Inspect a graded answer")
c1, c2 = st.columns([1, 2])
which = c1.segmented_control("Show", ["Failed", "Passed", "Planted"], default="Failed", label_visibility="collapsed")
pool = {
    "Failed": real[real["failed"] & ~real["error"]],
    "Passed": real[~real["failed"]],
    "Planted": df[df["planted"]],
}.get(which or "Failed", real)
types = sorted({t for ts in pool["failure_types"] for t in ts})
type_pick = c2.multiselect("Problem type", types, placeholder="Any problem type", label_visibility="collapsed")
if type_pick:
    pool = pool[pool["failure_types"].map(lambda ts: any(t in ts for t in type_pick))]
if pool.empty:
    st.info("Nothing matches these filters.")
    st.stop()

pick = st.selectbox(
    "Answer",
    pool.index.tolist(),
    format_func=lambda i: (pool.loc[i, "question"][:110] + ("…" if len(pool.loc[i, "question"]) > 110 else "")),
)
row = pool.loc[pick]
with get_session(get_engine()) as session:
    jdg = session.get(Judgement, row["judgement_id"])
    raw = session.get(InteractionRaw, row["interaction_id"])
    snap = session.get(StateSnapshot, raw.snapshot_id) if raw else None

left, right = st.columns([3, 2])
with left:
    with st.chat_message("user"):
        st.markdown(row["question"])
    with st.chat_message("assistant"):
        st.markdown(row["answer"])
    if raw is not None and raw.injected_failure:
        st.caption(f"Planted mistake: **{raw.injected_failure.replace('_', ' ')}**")
with right:
    st.markdown(chip("Fail", COLORS["fail"]) if row["failed"] else chip("Pass", COLORS["pass"]), unsafe_allow_html=True)
    st.markdown(verdict_chips({d: row[d] for d in DIMENSIONS}), unsafe_allow_html=True)
    if row["failure_types"]:
        st.markdown("**Problems:** " + ", ".join(t.replace("_", " ") for t in row["failure_types"]))
    if jdg is not None:
        st.markdown(f"**Why:** {jdg.explanation}")
        st.caption(f"Confidence {jdg.confidence:.0%} · evidence {jdg.evidence_status}")

if jdg is not None:
    with st.expander("Facts the answer claimed, checked against the order record"):
        checks = pd.DataFrame(jdg.claim_checks_json or [])
        if checks.empty:
            st.caption("No checkable facts in this answer.")
        else:
            st.dataframe(checks, use_container_width=True, hide_index=True)
with st.expander("Order record (the judge's evidence)"):
    if snap is not None:
        st.json(snap.snapshot_json, expanded=1)
