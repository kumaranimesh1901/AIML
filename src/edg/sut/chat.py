"""Live chat with the NovaMart support assistant (SUT), used by the dashboard's Chat Assistant page.

It reuses the pipeline's own assistant, snapshot-bound tools and hybrid judge, so a chat reply is
produced exactly the way logged interactions are. Chat never writes to the database: no
interactions, no judgements and no LLM cache rows, so it is safe to use while a pipeline run is
writing to the same SQLite file. It only talks to real models (Ollama or an OpenAI-compatible
server you host); the fake provider is refused.
"""

from __future__ import annotations

import re
from typing import Any

import httpx
import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from edg.config import AppConfig
from edg.db.models import Judgement, PolicyDocument, StateSnapshot
from edg.judge.evidence import assemble_evidence
from edg.judge.llm_judge import HybridJudge
from edg.llm.client import BaseLLMClient, create_llm_client
from edg.sut.assistant import AssistantResult, CustomerSupportAssistant
from edg.sut.tools import SnapshotToolBox

CHAT_RUN_ID = "chat_live"
_ORDER_ID_RE = re.compile(r"\bORD-\d{4,}\b", re.IGNORECASE)


class ChatUnavailableError(RuntimeError):
    """Raised when the live chat cannot run with a real model (e.g. the fake provider is set)."""


def find_order_ids(text: str) -> list[str]:
    """Order ids such as `ORD-000123` mentioned in a customer message, in order of appearance."""
    return list(dict.fromkeys(m.group(0).upper() for m in _ORDER_ID_RE.finditer(text)))


def ollama_status(base_url: str, *, timeout_seconds: float = 3.0) -> tuple[bool, list[str], str | None]:
    """Return `(reachable, installed_model_names, error)` for a local Ollama server."""
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout_seconds)
        resp.raise_for_status()
        names = [str(m.get("name", "")) for m in resp.json().get("models", [])]
        return True, [n for n in names if n], None
    except (httpx.HTTPError, ValueError) as exc:
        return False, [], str(exc)


def missing_models(installed: list[str], wanted: list[str]) -> list[str]:
    """Models from `wanted` that Ollama does not have (`qwen2.5` matches `qwen2.5:latest`)."""
    have = set(installed)
    have |= {n.removesuffix(":latest") for n in installed}
    return [m for m in dict.fromkeys(wanted) if m not in have and f"{m}:latest" not in have]


def build_live_client(config: AppConfig, *, provider: str, model: str) -> BaseLLMClient:
    """A real-model client with no response cache, so every chat reply is generated live."""
    if provider == "fake" or config.llm.provider == "fake":
        raise ChatUnavailableError(
            "The chat assistant only runs on real models. Unset EDG_LLM_PROVIDER=fake "
            "(or set llm.provider to ollama) and start Ollama."
        )
    return create_llm_client(config.llm, provider=provider, default_model=model, cache=None)


def count_snapshots(session: Session) -> int:
    return int(session.scalar(select(func.count()).select_from(StateSnapshot)) or 0)


def list_order_choices(
    session: Session, *, hard_state_tag: str | None = None, limit: int = 300
) -> list[tuple[str, list[str]]]:
    """`(order_id, hard_state_tags)` pairs for the order picker, hard-state orders first."""
    rows = session.execute(
        select(StateSnapshot.order_id, StateSnapshot.hard_state_tags).order_by(StateSnapshot.order_id)
    ).all()
    seen: dict[str, list[str]] = {}
    for order_id, tags in rows:
        seen.setdefault(order_id, list(tags or []))
    pairs = list(seen.items())
    if hard_state_tag:
        pairs = [p for p in pairs if hard_state_tag in p[1]]
    pairs.sort(key=lambda p: (not p[1], p[0]))
    return pairs[:limit]


def load_order_snapshot(session: Session, order_id: str) -> StateSnapshot | None:
    """The latest state snapshot of an order (attributes stay readable after the session closes)."""
    return session.scalars(
        select(StateSnapshot)
        .where(StateSnapshot.order_id == order_id)
        .order_by(StateSnapshot.created_at.desc())
        .limit(1)
    ).first()


def load_policies(session: Session, policy_version: str) -> list[tuple[str, str, str, np.ndarray]]:
    """Policy chunks with unit-normalised embeddings, the shape the SUT toolbox and judge expect."""
    loaded: list[tuple[str, str, str, np.ndarray]] = []
    for doc in session.scalars(select(PolicyDocument).where(PolicyDocument.version == policy_version)):
        if doc.embedding:
            vec = np.asarray(doc.embedding, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            loaded.append((doc.policy_id, doc.title, doc.body, vec))
    return loaded


def chat_reply(
    *,
    config: AppConfig,
    version_name: str,
    snapshot: StateSnapshot,
    message: str,
    history: list[tuple[str, str]] | None = None,
    policies: list[tuple[str, str, str, np.ndarray]] | None = None,
    llm_client: BaseLLMClient | None = None,
) -> AssistantResult:
    """Answer one customer message with SUT `version_name`, using tools bound to `snapshot`."""
    if version_name not in config.sut.versions:
        raise ValueError(f"Unknown SUT version '{version_name}'. Available: {list(config.sut.versions)}")
    ver = config.sut.versions[version_name]
    client = llm_client or build_live_client(config, provider=ver.provider, model=ver.base_model)
    toolbox = SnapshotToolBox(
        snapshot,
        retrieval_enabled=ver.retrieval_enabled,
        policy_cache=policies or [],
    )
    assistant = CustomerSupportAssistant(
        version_config=ver,
        llm_client=client,
        max_tool_calls=config.sut.max_tool_calls,
    )
    return assistant.answer(
        query=message,
        order_id=snapshot.order_id,
        toolbox=toolbox,
        history=history,
    )


def grade_reply(
    *,
    config: AppConfig,
    snapshot: StateSnapshot,
    message: str,
    reply: str,
    policies: list[tuple[str, str, str, np.ndarray]] | None = None,
    llm_client: BaseLLMClient | None = None,
) -> Judgement:
    """Grade one chat reply with the frozen hybrid judge. The judgement is not saved anywhere."""
    client = llm_client or build_live_client(
        config, provider=config.judge.provider, model=config.judge.model_name
    )
    judge = HybridJudge(config=config, llm_client=client)
    evidence = assemble_evidence(
        None,
        snapshot_id=snapshot.snapshot_id,
        query=message,
        policy_version=snapshot.policy_version,
        top_k_policies=config.judge.top_k_policies,
        snapshot_override=snapshot.snapshot_json,
        preloaded_policies=policies or [],
    )
    judgement, _ = judge.judge_one(
        session=None,
        interaction_id=f"CHAT-{snapshot.order_id}",
        query=message,
        response=reply,
        evidence=evidence,
        run_id=CHAT_RUN_ID,
        bypass_cache=True,
        persist=False,
    )
    return judgement


def summarize_snapshot(snapshot_json: dict[str, Any]) -> dict[str, Any]:
    """A short, display-friendly view of an order snapshot for the chat sidebar."""
    order = snapshot_json.get("order", {})
    return {
        "order_id": order.get("order_id"),
        "status": order.get("status"),
        "total_amount": order.get("total_amount"),
        "placed_at": str(order.get("placed_at") or order.get("created_at") or "")[:10] or None,
        "items": len(snapshot_json.get("items", [])),
        "shipments": len(snapshot_json.get("shipments", [])),
        "payments": len(snapshot_json.get("payments", [])),
        "returns": len(snapshot_json.get("refunds_returns", [])),
    }
