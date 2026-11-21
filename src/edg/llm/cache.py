"""Persistent (DB-backed) and in-memory cache for LLM responses."""

from __future__ import annotations

import hashlib
import json

from sqlalchemy import Connection, Engine, insert, select

from edg.db.models import LLMCacheEntry, utcnow
from edg.db.session import get_active_session, get_session


def make_prompt_hash(system_prompt: str | None, prompt: str) -> str:
    """Compute a deterministic SHA-256 hash of system and user prompts."""
    payload = json.dumps(
        {"system_prompt": system_prompt or "", "prompt": prompt},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_cache_key(
    *,
    provider: str,
    model: str,
    prompt_hash: str,
    temperature: float,
    schema_name: str | None = None,
) -> str:
    """Compute a deterministic cache key for an LLM request."""
    payload = json.dumps(
        {
            "provider": provider,
            "model": model,
            "prompt_hash": prompt_hash,
            "temperature": round(temperature, 4),
            "schema_name": schema_name or "",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LLMCache:
    """Cache for LLM responses supporting both in-memory and SQLAlchemy DB backends."""

    def __init__(self, engine: Engine | Connection | None = None) -> None:
        self._engine = engine
        self._memory: dict[str, tuple[str, float]] = {}

    def _active_connection(self) -> Connection | None:
        """Connection of an open pipeline session on this same engine, if one exists.

        Writing through it (instead of opening a second connection) avoids SQLite's
        single-writer `database is locked` error while a pipeline step's transaction is open.
        Cache rows then commit or roll back together with that step.
        """
        session = get_active_session()
        if session is None or self._engine is None:
            return None
        try:
            bind = session.get_bind()
        except Exception:
            return None
        if bind is self._engine:
            return session.connection()
        return None

    def get(self, cache_key: str) -> tuple[str, float] | None:
        """Return `(response_text, latency_ms)` if cached, else None."""
        if cache_key in self._memory:
            return self._memory[cache_key]
        if self._engine is None:
            return None

        stmt = select(LLMCacheEntry.response_text, LLMCacheEntry.latency_ms).where(
            LLMCacheEntry.cache_key == cache_key
        )
        conn = self._active_connection()
        if conn is not None:
            row = conn.execute(stmt).first()
        else:
            with get_session(self._engine) as session:
                row = session.execute(stmt).first()
        if row is None:
            return None
        result = (str(row[0]), float(row[1]))
        self._memory[cache_key] = result
        return result

    def set(
        self,
        *,
        cache_key: str,
        provider: str,
        model: str,
        prompt_hash: str,
        response_text: str,
        latency_ms: float,
    ) -> None:
        """Store an LLM response in memory and (if configured) in the database."""
        self._memory[cache_key] = (response_text, latency_ms)
        if self._engine is None:
            return

        row = {
            "cache_key": cache_key,
            "provider": provider,
            "model": model,
            "prompt_hash": prompt_hash,
            "response_text": response_text,
            "latency_ms": latency_ms,
            "created_at": utcnow(),
        }
        exists_stmt = select(LLMCacheEntry.cache_key).where(LLMCacheEntry.cache_key == cache_key)
        conn = self._active_connection()
        if conn is not None:
            if conn.execute(exists_stmt).first() is None:
                conn.execute(insert(LLMCacheEntry).values(**row))
            return
        with get_session(self._engine) as session:
            if session.execute(exists_stmt).first() is None:
                session.execute(insert(LLMCacheEntry).values(**row))
