"""Support Desk: escalated chats, the executive's corrected answer, and re-evaluation analytics."""

from __future__ import annotations

import sys
from pathlib import Path

_DASH_DIR = str(Path(__file__).resolve().parents[1])
if _DASH_DIR not in sys.path:
    sys.path.insert(0, _DASH_DIR)

import pandas as pd
import plotly.express as px
import streamlit as st
from db_helper import get_database_url, get_engine
from sqlalchemy import select
from ui import COLORS, chip, kpi, load_or_wait, page_header, pct, show_chart

from edg.config import load_config
from edg.db.models import ChatTurn, StateSnapshot, SupportTicket
from edg.db.session import get_session
from edg.judge.retriever import retrieve_policy_chunks
from edg.llm.client import LLMError
from edg.support import SupportError, reevaluate_ticket, resolve_ticket, support_metrics
from edg.support.service import ensure_support_tables
from edg.sut.chat import ChatUnavailableError, load_policies, missing_models, ollama_status

TRIGGER_LABELS = {
    "customer": "Customer unhappy",
    "ai_judge": "AI-detected error",
    "customer_and_ai_judge": "Customer + AI",
}
TRIGGER_COLORS = {"customer": "#F59E0B", "ai_judge": "#DC2626", "customer_and_ai_judge": "#7C3AED"}
PRIORITY_COLORS = {"urgent": "#DC2626", "high": "#F97316", "normal": "#64748B"}
REEVAL_LABELS = {
    "not_started": "Not sent",
    "queued": "Waiting for re-evaluation",
    "validated": "Validated, benchmark candidate",
    "failed_validation": "Judge still found problems",
    "error": "Judge error",
}

page_header(
    "Support Desk",
    "Chats escalate here when the customer is unhappy or the AI quality check finds a serious "
    "error. Executives send a corrected answer to the customer, and the judge re-checks it before "
    "it can become a benchmark test case.",
)

config = load_config()
engine = get_engine()


@st.cache_resource(show_spinner=False)
def _schema_ready(database_url: str) -> bool:
    ensure_support_tables(get_engine())
    return True


load_or_wait(lambda: _schema_ready(get_database_url()))

def _metrics() -> dict:
    with get_session(engine) as session:
        return support_metrics(session)


metrics = load_or_wait(_metrics)
if metrics["total_interactions"] == 0:
    st.info(
        "No customer chats yet. Conversations from the **Chat Assistant** page appear here, with their "
        "feedback and any escalations.",
        icon=":material/forum:",
    )
    st.page_link("views/chat.py", label="Open the Chat Assistant", icon=":material/forum:")
    st.stop()

# ---- Analytics -----------------------------------------------------------------------------------
st.subheader("Customer experience vs AI quality")
r1 = st.columns(4)
kpi(r1[0], "Customer interactions", f"{metrics['total_interactions']:,}", f"{metrics['no_feedback']} without feedback")
kpi(r1[1], "Satisfied · Dissatisfied", f"{metrics['satisfied']} · {metrics['dissatisfied']}", "customer feedback")
kpi(r1[2], "Customer satisfaction", pct(metrics["csat"]), "of answers that got feedback")
kpi(
    r1[3],
    "AI-detected failures",
    f"{metrics['ai_failures']:,}",
    f"of {metrics['ai_evaluated']} answers checked by the judge",
)
r2 = st.columns(4)
kpi(r2[0], "Customer-triggered escalations", f"{metrics['customer_escalations']:,}", f"{metrics['ai_escalations']} triggered by the AI check")
kpi(r2[1], "Tickets pending", f"{metrics['tickets_pending']:,}", "waiting for an executive", trend="down" if metrics["tickets_pending"] else None)
kpi(r2[2], "Tickets resolved", f"{metrics['tickets_resolved']:,}", "corrected answer sent")
kpi(
    r2[3],
    "Corrections re-evaluated",
    f"{metrics['sent_for_reevaluation']:,}",
    f"{metrics['reeval_validated']} validated · {metrics['reeval_queued']} queued",
)
st.write("")

left, right = st.columns(2)
with left:
    mat = pd.DataFrame(metrics["feedback_vs_ai"])
    if not mat.empty:
        mat["AI check"] = mat["ai_failure"].map({True: "AI found a problem", False: "AI check passed"}).fillna("Not checked")
        mat["Customer"] = mat["feedback"].map({"satisfied": "Satisfied", "dissatisfied": "Dissatisfied", "none": "No feedback"})
        fig = px.bar(
            mat,
            x="Customer",
            y="count",
            color="AI check",
            category_orders={"Customer": ["Satisfied", "Dissatisfied", "No feedback"]},
            color_discrete_map={"AI found a problem": COLORS["fail"], "AI check passed": COLORS["pass"], "Not checked": COLORS["neutral"]},
            title="Satisfaction and correctness are separate signals",
        )
        fig.update_yaxes(tickformat=",d", rangemode="tozero", dtick=1 if mat["count"].max() <= 10 else None)
        show_chart(fig)
        st.caption(
            "Red inside *Satisfied* = customers happy with a wrong answer (escalated by the AI check). "
            "Green inside *Dissatisfied* = correct answers the customer still needed help with."
        )


@st.cache_data(ttl=5, show_spinner=False)
def _tickets(_v: int) -> pd.DataFrame:
    with get_session(engine) as session:
        rows = session.execute(
            select(SupportTicket, ChatTurn).join(ChatTurn, ChatTurn.turn_id == SupportTicket.turn_id)
            .order_by(SupportTicket.created_at.desc())
        ).all()
    return pd.DataFrame(
        [
            {
                "Ticket": t.ticket_id,
                "Status": t.status.capitalize(),
                "Trigger": TRIGGER_LABELS.get(t.trigger, t.trigger),
                "trigger": t.trigger,
                "Priority": t.priority,
                "Order": c.order_id,
                "Question": c.user_query,
                "Re-evaluation": REEVAL_LABELS.get(t.reeval_status, t.reeval_status),
                "Opened": t.created_at,
            }
            for t, c in rows
        ]
    )


version = st.session_state.setdefault("support_version", 0)
tickets = _tickets(version)
with right:
    if not tickets.empty:
        agg = tickets.groupby(["Trigger", "Status"]).size().reset_index(name="tickets")
        fig = px.bar(
            agg,
            x="Trigger",
            y="tickets",
            color="Status",
            color_discrete_map={"Pending": COLORS["partial"], "Resolved": COLORS["pass"]},
            title="Tickets by what triggered them",
        )
        fig.update_yaxes(tickformat=",d", rangemode="tozero", dtick=1 if agg["tickets"].max() <= 10 else None)
        show_chart(fig)
    else:
        st.info("No escalations yet: every answer so far was accepted and passed the AI check.", icon=":material/check_circle:")

# ---- Ticket queue --------------------------------------------------------------------------------
st.subheader("Ticket queue")
if tickets.empty:
    st.stop()
f1, f2 = st.columns([1, 2])
status = f1.segmented_control("Status", ["Pending", "Resolved", "All"], default="Pending", label_visibility="collapsed")
trig = f2.segmented_control(
    "Trigger", ["All triggers", *TRIGGER_LABELS.values()], default="All triggers", label_visibility="collapsed"
)
view = tickets
if status and status != "All":
    view = view[view["Status"] == status]
if trig and trig != "All triggers":
    view = view[view["Trigger"] == trig]
opened = st.session_state.get("support_open_ticket")
if view.empty and not opened:
    st.success("Nothing in this queue.", icon=":material/inbox:")
    st.stop()

table = st.dataframe(
    view[["Ticket", "Priority", "Trigger", "Order", "Question", "Status", "Re-evaluation"]],
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    column_config={"Question": st.column_config.TextColumn(width="large")},
    key=f"queue_{abs(hash(tuple(view['Ticket'])))}",  # a changed queue resets the selection
)
picked = table.selection.rows if table and table.selection else []
if picked:
    ticket_id = view.iloc[picked[0]]["Ticket"]
    st.session_state["support_open_ticket"] = ticket_id
elif opened in set(tickets["Ticket"]):
    ticket_id = opened  # keep a ticket open after it leaves the filtered queue (e.g. just resolved)
else:
    st.caption("Select a ticket to open it.")
    st.stop()
with get_session(engine) as session:
    ticket = session.get(SupportTicket, ticket_id)
    turn = session.get(ChatTurn, ticket.turn_id)
    snapshot = session.get(StateSnapshot, turn.snapshot_id)
    policy_chunks = retrieve_policy_chunks(
        session,
        query=turn.user_query,
        policy_version=snapshot.policy_version,
        top_k=config.judge.top_k_policies,
        preloaded_docs=load_policies(session, snapshot.policy_version),
    )

st.divider()
st.markdown(
    f"### Ticket {ticket.ticket_id} &nbsp; "
    + chip(ticket.priority, PRIORITY_COLORS.get(ticket.priority, "#64748B"))
    + chip(TRIGGER_LABELS.get(ticket.trigger, ticket.trigger), TRIGGER_COLORS.get(ticket.trigger, "#64748B"))
    + chip(ticket.status, COLORS["pass"] if ticket.status == "resolved" else COLORS["partial"]),
    unsafe_allow_html=True,
)
st.caption(f"Why it was escalated: {ticket.escalation_reason}")

left, right = st.columns([3, 2])
with left:
    st.markdown("##### Conversation")
    for q, a in turn.history_json or []:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)
    with st.chat_message("user"):
        st.markdown(turn.user_query)
    with st.chat_message("assistant"):
        st.markdown(turn.ai_response)
        st.caption(f"AI chatbot {turn.model_version} ({turn.base_model}) · this is the escalated answer")
    if ticket.corrected_answer:
        with st.chat_message("human", avatar=":material/support_agent:"):
            st.markdown(f"**{ticket.executive_name}** (sent to the customer)")
            st.markdown(ticket.corrected_answer)
with right:
    st.markdown("##### Customer feedback")
    if turn.customer_feedback == "dissatisfied":
        st.markdown(":red[:material/thumb_down: Not satisfied]")
        st.markdown(f"> {turn.feedback_comment}" if turn.feedback_comment else ":gray[No comment left]")
    elif turn.customer_feedback == "satisfied":
        st.markdown(":green[:material/thumb_up: Satisfied]  \n:gray[Escalated by the AI check anyway: satisfaction is not proof of correctness.]")
    else:
        st.markdown(":gray[No feedback yet]")
    st.markdown("##### AI quality check")
    if turn.eval_status == "done":
        st.markdown(
            (f":red[Problem found · {turn.ai_severity} severity]" if turn.ai_failure_detected else ":green[Passed]")
            + f" · confidence {turn.ai_confidence:.0%}"
        )
        if turn.ai_failure_types:
            st.markdown("**Problems:** " + ", ".join(t.replace("_", " ") for t in turn.ai_failure_types))
        st.markdown(f"**Judge:** {turn.ai_explanation}")
        contradicted = [c for c in turn.ai_claim_checks_json or [] if c.get("status") == "contradicted"]
        for c in contradicted:
            st.markdown(f":red[:material/close:] {c.get('claim_text')}  \n:gray[{c.get('evidence_detail')}]")
    else:
        st.markdown(f":gray[{turn.eval_status}]")

with st.expander(f"Order {turn.order_id} record", expanded=False):
    st.json(snapshot.snapshot_json, expanded=1)
with st.expander("Relevant store policies", expanded=False):
    for p in policy_chunks:
        st.markdown(f"**{p.get('title')}**  \n{p.get('body')}")

# ---- Resolve -------------------------------------------------------------------------------------
if ticket.status == "pending":
    st.markdown("##### Reply to the customer")
    with st.form(f"resolve_{ticket.ticket_id}"):
        name = st.text_input("Your name", value=st.session_state.get("exec_name", ""), placeholder="Support executive")
        answer = st.text_area(
            "Corrected answer (sent to the customer's chat)",
            height=160,
            placeholder="Write the correct answer using the order record and policies above.",
        )
        notes = st.text_area("Internal notes (optional)", height=80, placeholder="What was wrong with the AI answer?")
        if st.form_submit_button("Send to customer and resolve", type="primary", icon=":material/send:"):
            try:
                with get_session(engine) as session:
                    resolve_ticket(session, ticket.ticket_id, executive_name=name, corrected_answer=answer, notes=notes)
                st.session_state["exec_name"] = name
                st.session_state["support_version"] = version + 1
                st.rerun()
            except SupportError as exc:
                st.error(str(exc))
    st.stop()

# ---- Re-evaluation -------------------------------------------------------------------------------
st.markdown("##### Re-evaluation of the human correction")
st.caption(REEVAL_LABELS.get(ticket.reeval_status, ticket.reeval_status))
if ticket.reeval_status in ("queued", "failed_validation", "error"):
    reachable, installed, _err = ollama_status(config.llm.ollama_base_url)
    absent = missing_models(installed, [config.judge.model_name, config.llm.helper_model]) if reachable else []
    if not reachable:
        st.warning("Start Ollama to re-evaluate (`ollama serve`), or run `edg support reevaluate` later.")
    elif absent:
        st.warning("Missing models: " + ", ".join(f"`ollama pull {m}`" for m in absent))
    elif st.button(f"Re-evaluate with the judge ({config.judge.model_name})", type="primary", icon=":material/gavel:"):
        with st.spinner("The judge is checking the corrected answer, then drafting a test checklist..."):
            try:
                reevaluate_ticket(engine, config, ticket.ticket_id)
            except (SupportError, ChatUnavailableError, LLMError) as exc:
                st.error(str(exc))
        st.session_state["support_version"] = version + 1
        st.rerun()
if ticket.reeval_json:
    r = ticket.reeval_json
    st.markdown(
        " · ".join(f"**{d}** {v}" for d, v in (r.get("verdicts") or {}).items())
        + f"  \n**Judge:** {r.get('explanation')}"
    )
if ticket.promoted_test_case_id:
    st.success(
        f"Validated. Added as benchmark candidate **{ticket.promoted_test_case_id}**, with the corrected "
        "answer as its reference. Approve it on the Test Dataset page to include it in the next release.",
        icon=":material/verified:",
    )
    st.page_link("views/dataset.py", label="Open Test Dataset", icon=":material/fact_check:")
