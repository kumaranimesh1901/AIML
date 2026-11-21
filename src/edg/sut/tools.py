"""Snapshot-isolated tools available to the customer-support assistant (SUT)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from edg.db.models import PolicyDocument, StateSnapshot
from edg.world.policies import hash_embed_texts


class SnapshotAccessError(PermissionError):
    """Raised when a tool attempts to access an order outside its bound snapshot."""


class SnapshotToolBox:
    """Read-only toolbox strictly bound to a single order's StateSnapshot."""

    def __init__(
        self,
        snapshot: StateSnapshot,
        *,
        session: Session | None = None,
        retrieval_enabled: bool = False,
        policy_cache: list[tuple[str, str, str, np.ndarray]] | None = None,
    ) -> None:
        self._snapshot = snapshot
        self._data: dict[str, Any] = snapshot.snapshot_json
        self._bound_order_id = snapshot.order_id
        self._policy_version = snapshot.policy_version
        self._session = session
        self.retrieval_enabled = retrieval_enabled
        self._policy_cache = policy_cache
        self.tool_trace: list[dict[str, Any]] = []

    @property
    def bound_order_id(self) -> str:
        return self._bound_order_id

    @property
    def snapshot_id(self) -> str:
        return self._snapshot.snapshot_id

    def _check_order_scope(self, order_id: str) -> None:
        if order_id != self._bound_order_id:
            raise SnapshotAccessError(
                f"Access denied: tool is bound to order '{self._bound_order_id}', "
                f"cannot read order '{order_id}'"
            )

    def get_order(self, order_id: str) -> dict[str, Any]:
        self._check_order_scope(order_id)
        result = {
            "order": self._data["order"],
            "items": self._data.get("items", []),
            "reference_date": self._data.get("reference_date"),
        }
        self.tool_trace.append({"tool": "get_order", "args": {"order_id": order_id}, "result": result})
        return result

    def get_shipment(self, order_id: str) -> dict[str, Any]:
        self._check_order_scope(order_id)
        result = {"order_id": order_id, "shipments": self._data.get("shipments", [])}
        self.tool_trace.append({"tool": "get_shipment", "args": {"order_id": order_id}, "result": result})
        return result

    def get_payments(self, order_id: str) -> dict[str, Any]:
        self._check_order_scope(order_id)
        result = {"order_id": order_id, "payments": self._data.get("payments", [])}
        self.tool_trace.append({"tool": "get_payments", "args": {"order_id": order_id}, "result": result})
        return result

    def get_returns(self, order_id: str) -> dict[str, Any]:
        self._check_order_scope(order_id)
        result = {"order_id": order_id, "refunds_returns": self._data.get("refunds_returns", [])}
        self.tool_trace.append({"tool": "get_returns", "args": {"order_id": order_id}, "result": result})
        return result

    def _load_policies(self) -> list[tuple[str, str, str, np.ndarray]]:
        if self._policy_cache is not None:
            return self._policy_cache
        if self._session is None:
            return []
        docs = list(
            self._session.scalars(
                select(PolicyDocument).where(PolicyDocument.version == self._policy_version)
            ).all()
        )
        loaded: list[tuple[str, str, str, np.ndarray]] = []
        for doc in docs:
            if doc.embedding:
                vec = np.asarray(doc.embedding, dtype=np.float32)
                norm = float(np.linalg.norm(vec))
                if norm > 0:
                    vec = vec / norm
                loaded.append((doc.policy_id, doc.title, doc.body, vec))
        self._policy_cache = loaded
        return loaded

    def search_policy(self, query: str, top_k: int = 3) -> dict[str, Any]:
        if not self.retrieval_enabled:
            result: dict[str, Any] = {
                "enabled": False,
                "chunks": [],
                "message": "Policy retrieval is disabled for this assistant version.",
            }
            self.tool_trace.append(
                {"tool": "search_policy", "args": {"query": query, "top_k": top_k}, "result": result}
            )
            return result

        policies = self._load_policies()
        if not policies:
            result = {"enabled": True, "chunks": []}
            self.tool_trace.append(
                {"tool": "search_policy", "args": {"query": query, "top_k": top_k}, "result": result}
            )
            return result

        dim = len(policies[0][3])
        q_vec = hash_embed_texts([query], dim=dim)[0]
        scored: list[tuple[float, str, str, str]] = []
        for pid, title, body, pvec in policies:
            # Combine vector cosine similarity with lightweight keyword overlap for robust retrieval
            cos = float(np.dot(q_vec, pvec))
            q_words = set(query.lower().split())
            p_words = set(f"{title} {body}".lower().split())
            overlap = len(q_words & p_words) / max(1, len(q_words))
            score = 0.5 * cos + 0.5 * overlap
            scored.append((score, pid, title, body))

        scored.sort(key=lambda x: x[0], reverse=True)
        chunks = [
            {"policy_id": pid, "title": title, "body": body, "score": round(score, 4)}
            for score, pid, title, body in scored[:top_k]
        ]
        result = {"enabled": True, "chunks": chunks}
        self.tool_trace.append(
            {"tool": "search_policy", "args": {"query": query, "top_k": top_k}, "result": result}
        )
        return result

    def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a tool call by name safely."""
        try:
            if tool_name == "get_order":
                return self.get_order(str(arguments.get("order_id", self._bound_order_id)))
            if tool_name == "get_shipment":
                return self.get_shipment(str(arguments.get("order_id", self._bound_order_id)))
            if tool_name == "get_payments":
                return self.get_payments(str(arguments.get("order_id", self._bound_order_id)))
            if tool_name == "get_returns":
                return self.get_returns(str(arguments.get("order_id", self._bound_order_id)))
            if tool_name == "search_policy":
                return self.search_policy(
                    query=str(arguments.get("query", "")),
                    top_k=int(arguments.get("top_k", 3)),
                )
            return {"error": f"Unknown tool: {tool_name}"}
        except SnapshotAccessError as exc:
            err = {"error": str(exc)}
            self.tool_trace.append({"tool": tool_name, "args": arguments, "result": err})
            return err
