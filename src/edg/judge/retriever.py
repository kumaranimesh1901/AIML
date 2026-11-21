"""Policy chunk retrieval over embedded policy_documents (pgvector on Postgres, cosine on SQLite)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from edg.db.models import PolicyDocument
from edg.world.policies import hash_embed_texts


def retrieve_policy_chunks(
    session: Session,
    *,
    query: str,
    policy_version: str = "v1",
    top_k: int = 3,
    preloaded_docs: list[tuple[str, str, str, np.ndarray]] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the top-k most relevant policy chunks for a query."""
    if preloaded_docs is not None:
        docs_list = preloaded_docs
    else:
        rows = list(
            session.scalars(
                select(PolicyDocument).where(PolicyDocument.version == policy_version)
            ).all()
        )
        docs_list = []
        for doc in rows:
            if doc.embedding:
                vec = np.asarray(doc.embedding, dtype=np.float32)
                norm = float(np.linalg.norm(vec))
                if norm > 0:
                    vec = vec / norm
                docs_list.append((doc.policy_id, doc.title, doc.body, vec))

    if not docs_list:
        return []

    dim = len(docs_list[0][3])
    q_vec = hash_embed_texts([query], dim=dim)[0]
    q_words = set(query.lower().split())

    scored: list[tuple[float, str, str, str]] = []
    for pid, title, body, pvec in docs_list:
        cos = float(np.dot(q_vec, pvec))
        p_words = set(f"{title} {body}".lower().split())
        overlap = len(q_words & p_words) / max(1, len(q_words))
        score = 0.5 * cos + 0.5 * overlap
        scored.append((score, pid, title, body))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "policy_id": pid,
            "title": title,
            "body": body,
            "score": round(score, 4),
        }
        for score, pid, title, body in scored[:top_k]
    ]
