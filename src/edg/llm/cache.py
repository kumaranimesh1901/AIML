"""Persistent (DB-backed) and in-memory cache for LLM responses."""

from __future__ import annotations

import hashlib
import json

from sqlalchemy import Connection, Engine

from edg.db.models import LLMCacheEntry
from edg.db.session import get_session


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

    def get(self, cache_key: str) -> tuple[str, float] | None:
        """Return `(response_text, latency_ms)` if cached, else None."""
        if cache_key in self._memory:
            return self._memory[cache_key]

        if self._engine is not None:
            with get_session(self._engine) as session:
                entry = session.get(LLMCacheEntry, cache_key)
                if entry is not None:
                    result = (entry.response_text, entry.latency_ms)
                    self._memory[cache_key] = result
                    return result
        return None

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
        if self._engine is not None:
            with get_session(self._engine) as session:
                existing = session.get(LLMCacheEntry, cache_key)
                if existing is None:
                    session.add(
                        LLMCacheEntry(
                            cache_key=cache_key,
                            provider=provider,
                            model=model,
                            prompt_hash=prompt_hash,
                            response_text=response_text,
                            latency_ms=latency_ms,
                        )
                    )
