"""Persistent (DB-backed) and in-memory embedding cache keyed by (text_hash, model_name)."""

from __future__ import annotations

import hashlib

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from edg.db.models import Embedding


def compute_text_hash(text: str) -> str:
    """Compute a deterministic SHA-256 hash of a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Cache for embedding vectors keyed by `(text_hash, model_name)`."""

    def __init__(self, session: Session | None = None) -> None:
        self._session = session
        self._memory: dict[tuple[str, str], np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    def preload_for_model(self, model_name: str) -> int:
        """Warm the in-memory cache from the embeddings table for a given model."""
        if self._session is None:
            return 0
        rows = self._session.execute(
            select(Embedding.text_hash, Embedding.vector).where(Embedding.model_name == model_name)
        ).all()
        for t_hash, vec in rows:
            if vec is not None:
                self._memory[(t_hash, model_name)] = np.asarray(vec, dtype=np.float32)
        return len(rows)

    def get(self, text_hash: str, model_name: str) -> np.ndarray | None:
        key = (text_hash, model_name)
        if key in self._memory:
            self.hits += 1
            return self._memory[key]
        self.misses += 1
        return None

    def set(self, text_hash: str, model_name: str, vector: np.ndarray) -> None:
        self._memory[(text_hash, model_name)] = np.asarray(vector, dtype=np.float32)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total > 0 else 0.0
