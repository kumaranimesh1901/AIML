"""Chat Assistant: chat with the NovaMart support assistant (the SUT) on a real local Ollama model.

The assistant answers with the same tool-calling loop the pipeline uses, bound to one order's
state snapshot, and each reply can be graded live by the frozen hybrid judge. Nothing from this
page is written to the database.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DASH_DIR = str(Path(__file__).resolve().parents[1])
if _DASH_DIR not in sys.path:
    sys.path.insert(0, _DASH_DIR)

import pandas as pd
import streamlit as st
from db_helper import get_database_url, get_engine
from sqlalchemy.exc import OperationalError
from ui import kpi, page_header

from edg.config import load_config
from edg.db.session import get_session
from edg.judge.schema import DIMENSIONS
from edg.llm.client import LLMError
from edg.sut.chat import (
    ChatUnavailableError,
    chat_reply,
    count_snapshots,
    find_order_ids,
    grade_reply,
    list_order_choices,
    load_order_snapshot,
    load_policies,
    missing_models,
    ollama_status,
    summarize_snapshot,
)

# Example questions per hard-state tag, so a demo can jump straight to the tricky cases
EXAMPLES_BY_TAG = {
    "duplicate_charge": "I was charged twice for this order. Can I get my money back?",
    "delivered_disputed": "The tracking says delivered but I never got my package. What now?",
    "past_return_window": "Can I still return this order for a refund?",
    "boundary_return_window": "Am I still within the return window for this order?",
    "partial_shipment": "I only received part of my order. Where is the rest?",
    "cancel_after_shipping": "Please cancel this order, I don't want it anymore.",
    "refund_approved_not_issued": "My refund was approved but I haven't received it. When will it arrive?",
    "wrong_item_delivered": "I received the wrong item. How do I get the right one?",
}
GENERIC_EXAMPLES = ["Where is my order and when will it arrive?", "Can I return this order?"]

page_header(
    "Chat Assistant",
    "Ask the NovaMart support chatbot a customer question about an order. It looks the order up with "
    "its tools and answers on a real local Ollama model. Chats are not saved.",
)

config = load_config()
engine = get_engine()


@st.cache_resource(show_spinner=False)
def _policies(database_url: str, policy_version: str):  # noqa: ANN202
    with get_session(get_engine()) as session:
        return load_policies(session, policy_version)


# ---- Real-model check: Ollama must be running and have the models ----------------------------
if config.llm.provider == "fake":
    st.error(
        "The chat assistant only runs on real models, but the LLM provider is set to `fake`. "
        "Unset `EDG_LLM_PROVIDER` and reload this page."
    )
    st.stop()

reachable, installed, ollama_error = ollama_status(config.llm.ollama_base_url)
if not reachable:
    st.error(
        f"Can't reach Ollama at `{config.llm.ollama_base_url}`. Start it with `ollama serve` "
        "(or open the Ollama app) and reload this page."
    )
    st.caption(f"Details: {ollama_error}")
    st.stop()

# ---- The store must be built far enough to have order snapshots ------------------------------
try:
    with get_session(engine) as session:
        n_snapshots = count_snapshots(session)
except OperationalError as exc:
    st.warning(f"The database is busy or not ready yet ({exc.orig}). Reload in a moment.")
    st.stop()

if n_snapshots == 0:
    st.info(
        "No orders in the database yet. The chat needs the store snapshot from the `world_build` "
        "step; run `edg world build` (or wait for the running pipeline to get past it) and reload."
    )
    st.stop()

# ---- Sidebar: assistant version, order, grading ----------------------------------------------
versions = list(config.sut.versions)
with st.sidebar:
    st.header("Chat settings")
    version_name = st.selectbox(
        "Assistant version",
        versions,
        index=len(versions) - 1,
        format_func=lambda v: (
            f"{v} · {config.sut.versions[v].base_model}"
            + (" · policy search" if config.sut.versions[v].retrieval_enabled else "")
        ),
    )
    with get_session(engine) as session:
        all_choices = list_order_choices(session, limit=100000)
    tag_options = sorted({t for _, tags in all_choices for t in tags})
    tag_filter = st.selectbox("Order type", ["any", *tag_options], format_func=lambda t: t.replace("_", " "))
    choices = [c for c in all_choices if tag_filter == "any" or tag_filter in c[1]][:500]
    order_ids = [oid for oid, _ in choices]
    tags_by_order = dict(all_choices)

    pending = st.session_state.pop("chat_switch_order", None)
    if pending and pending in tags_by_order and pending not in order_ids:
        order_ids.insert(0, pending)
    if pending in order_ids:
        st.session_state["chat_order"] = pending
    if st.session_state.get("chat_order") not in order_ids:
        st.session_state["chat_order"] = order_ids[0]
    order_id = st.selectbox(
        "Order",
        order_ids,
        key="chat_order",
        format_func=lambda o: f"{o}  ({', '.join(tags_by_order.get(o) or []) or 'normal'})",
    )
    auto_grade = st.toggle(
        f"Grade each reply with the judge ({config.judge.model_name})",
        value=False,
        help="Slower: the judge checks the reply against the order snapshot and store policies.",
    )
    if st.button("New chat", icon=":material/refresh:", use_container_width=True):
        st.session_state["chat_messages"] = []
        for key in [k for k in st.session_state if str(k).startswith("examples_")]:
            del st.session_state[key]

with get_session(engine) as session:
    snapshot = load_order_snapshot(session, order_id)
if snapshot is None:
    st.warning(f"Order {order_id} has no snapshot yet. Pick another order.")
    st.stop()

ver_cfg = config.sut.versions[version_name]
needed = [ver_cfg.base_model] + ([config.judge.model_name] if auto_grade else [])
absent = missing_models(installed, needed)
if absent:
    st.error(
        "Ollama is running but doesn't have "
        + ", ".join(f"`{m}`" for m in absent)
        + ". Pull it with "
        + " and ".join(f"`ollama pull {m}`" for m in absent)
        + ", then reload."
    )
    st.stop()

policies = _policies(get_database_url(), snapshot.policy_version)

# A new order or version starts a new conversation, since the tools are bound to one order
chat_key = (order_id, version_name)
if st.session_state.get("chat_key") != chat_key:
    st.session_state["chat_key"] = chat_key
    st.session_state["chat_messages"] = []
    st.session_state.pop(f"examples_{order_id}", None)
messages: list[dict] = st.session_state.setdefault("chat_messages", [])

summary = summarize_snapshot(snapshot.snapshot_json)
c1, c2, c3, c4 = st.columns(4)
kpi(c1, "Order", str(summary["order_id"]), ", ".join(t.replace("_", " ") for t in snapshot.hard_state_tags or []) or "normal order")
kpi(c2, "Status", str(summary["status"]).replace("_", " "), f"placed {summary['placed_at']}" if summary["placed_at"] else "")
kpi(c3, "Total", f"${summary['total_amount']:.2f}" if summary["total_amount"] is not None else "n/a", f"{summary['items']} item(s)")
kpi(c4, "Shipments · Returns", f"{summary['shipments']} · {summary['returns']}", f"{summary['payments']} payment(s)")
with st.expander("What the chatbot can see for this order"):
    st.json(snapshot.snapshot_json, expanded=1)


def _render_grade(jdg) -> None:  # noqa: ANN001
    if "judge_error" in (jdg.failure_types or []):
        st.warning(jdg.explanation)
        return
    failed = [d for d in DIMENSIONS if getattr(jdg, f"{d}_verdict") == "fail"]
    label = "Judge: pass" if not jdg.failure_types and not failed else "Judge: problems found"
    with st.expander(f"{label} · confidence {jdg.confidence:.2f}", expanded=bool(jdg.failure_types)):
        st.table(
            pd.DataFrame(
                [
                    {
                        "dimension": d,
                        "verdict": getattr(jdg, f"{d}_verdict"),
                        "score": getattr(jdg, f"{d}_score"),
                    }
                    for d in DIMENSIONS
                ]
            )
        )
        if jdg.failure_types:
            st.markdown("**Failure types:** " + ", ".join(f"`{f}`" for f in jdg.failure_types))
        st.markdown(f"**Why:** {jdg.explanation}")


def _render_assistant(msg: dict) -> None:
    st.markdown(msg["content"])
    tools = msg.get("tool_calls") or []
    meta = f"{msg['version']} · {msg['model']} · {msg['latency_ms'] / 1000:.1f}s · {len(tools)} tool call(s)"
    if msg.get("error"):
        meta += f" · error: {msg['error']}"
    st.caption(meta)
    if tools:
        with st.expander("Tools the assistant used"):
            for t in tools:
                st.markdown(f"`{t['tool']}({', '.join(f'{k}={v!r}' for k, v in t['args'].items())})`")
                st.json(t["result"], expanded=False)
    if msg.get("grade") is not None:
        _render_grade(msg["grade"])


def _grade_into(msg: dict, question: str) -> None:
    with st.spinner(f"Judge {config.judge.model_name} is grading the reply..."):
        try:
            msg["grade"] = grade_reply(
                config=config,
                snapshot=snapshot,
                message=question,
                reply=msg["content"],
                policies=policies,
            )
        except (ChatUnavailableError, LLMError) as exc:
            st.error(f"Grading failed: {exc}")


for idx, msg in enumerate(messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            _render_assistant(msg)
            if msg.get("grade") is None and st.button("Grade this reply", key=f"grade_{idx}"):
                _grade_into(msg, messages[idx - 1]["content"])
                st.rerun()

prompt = st.chat_input(f"Message the NovaMart assistant about order {order_id}")
if prompt:
    mentioned = [o for o in find_order_ids(prompt) if o != order_id]
    if mentioned:
        if mentioned[0] in tags_by_order:
            st.session_state["chat_switch_order"] = mentioned[0]
            st.session_state["chat_pending_prompt"] = prompt
            st.rerun()
        st.warning(f"Order {mentioned[0]} isn't in the store. The assistant will answer about {order_id}.")

prompt = prompt or st.session_state.pop("chat_pending_prompt", None)
if not messages and not prompt:
    examples = [EXAMPLES_BY_TAG[t] for t in snapshot.hard_state_tags or [] if t in EXAMPLES_BY_TAG]
    examples += [e for e in GENERIC_EXAMPLES if e not in examples]
    st.markdown("**Try a question** (or type your own below)")
    clicked = st.pills("Example questions", examples[:4], key=f"examples_{order_id}", label_visibility="collapsed")
    if clicked:
        prompt = clicked
if prompt:
    history = [
        (messages[i]["content"], messages[i + 1]["content"])
        for i in range(0, len(messages) - 1, 2)
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant"
    ]
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner(f"{version_name} ({ver_cfg.base_model}) is checking order {order_id}..."):
            try:
                result = chat_reply(
                    config=config,
                    version_name=version_name,
                    snapshot=snapshot,
                    message=prompt,
                    history=history,
                    policies=policies,
                )
            except (ChatUnavailableError, LLMError) as exc:
                messages.pop()
                st.error(f"The assistant couldn't answer: {exc}")
                st.stop()
        reply = {
            "role": "assistant",
            "content": result.answer,
            "version": version_name,
            "model": ver_cfg.base_model,
            "latency_ms": result.latency_ms,
            "tool_calls": result.tool_calls,
            "error": result.error,
            "grade": None,
        }
        if auto_grade and not result.error:
            _grade_into(reply, prompt)
        messages.append(reply)
        _render_assistant(reply)
