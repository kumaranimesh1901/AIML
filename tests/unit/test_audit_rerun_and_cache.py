"""Regression tests for bugs reported from a real local run (Ollama + SQLite).

1. With a real (non-fake) LLM provider, every DB-cache write happened on a *separate*
   connection while the pipeline session held SQLite's write lock, so each call failed with
   `database is locked`; the cache failure was then retried as if the LLM call had failed.
2. Re-running `edg run-all` / `edg world build` on an already-populated database crashed with an
   opaque `UNIQUE constraint failed: customers.customer_id`.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import Engine, func, select

from edg.config import AppConfig
from edg.db.models import LLMCacheEntry, PipelineRun
from edg.db.repositories import track_run
from edg.db.session import get_session
from edg.llm.cache import LLMCache
from edg.llm.client import BaseLLMClient, LLMRequest
from edg.pipeline.orchestrator import run_full_pipeline
from edg.world.store_generator import WorldAlreadyBuiltError, generate_store_world


class _StubHTTPClient(BaseLLMClient):
    """Behaves like a real provider (Ollama/OpenAI): plain _send, no fake shortcuts."""

    provider = "stub_http"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_model="stub-model", retry_base_delay_seconds=0.0, **kwargs)
        self.calls = 0

    def _send(self, request: LLMRequest) -> tuple[str, dict[str, Any]]:
        self.calls += 1
        return '{"ok": true}', {}


def test_db_cache_write_inside_open_pipeline_transaction_does_not_lock(
    db_engine: Engine, small_config: AppConfig
) -> None:
    client = _StubHTTPClient(cache=LLMCache(engine=db_engine))
    with get_session(db_engine) as session:
        # track_run flushes a pipeline_runs INSERT -> this session now holds the SQLite write lock
        with track_run(session, step="sut_run_v1", config=small_config):
            resp = client.generate("hello", system_prompt="sys")
            assert resp.text == '{"ok": true}'
            assert client.calls == 1, "a cache write failure must never trigger LLM retries"
            # Same request again is served from cache, not re-sent
            again = client.generate("hello", system_prompt="sys")
            assert again.cached is True
            assert client.calls == 1

    with get_session(db_engine) as session:
        assert session.scalar(select(func.count()).select_from(LLMCacheEntry)) == 1

    # A fresh cache object (new process) reads the persisted entry
    fresh = _StubHTTPClient(cache=LLMCache(engine=db_engine))
    assert fresh.generate("hello", system_prompt="sys").cached is True
    assert fresh.calls == 0


def test_cache_backend_failure_is_not_treated_as_llm_failure(db_engine: Engine) -> None:
    class _BrokenCache(LLMCache):
        def set(self, **kwargs: Any) -> None:  # noqa: D401
            raise RuntimeError("disk full")

    client = _StubHTTPClient(cache=_BrokenCache(engine=db_engine))
    resp = client.generate("hello")
    assert resp.text == '{"ok": true}'
    assert client.calls == 1


def test_world_build_on_populated_db_fails_with_clear_error(
    db_engine: Engine, small_config: AppConfig
) -> None:
    kwargs = {"prefer_hf_catalogue": False, "use_sentence_transformers": False}
    with get_session(db_engine) as session:
        with track_run(session, step="world_build", config=small_config) as (run, _c):
            generate_store_world(session, small_config, run_id=run.run_id, **kwargs)
    with pytest.raises(WorldAlreadyBuiltError, match="already contains a store world"):
        with get_session(db_engine) as session:
            with track_run(session, step="world_build", config=small_config) as (run, _c):
                generate_store_world(session, small_config, run_id=run.run_id, **kwargs)


def test_run_all_on_populated_db_fails_fast_before_any_step(
    db_engine: Engine, small_config: AppConfig, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    kwargs = {"prefer_hf_catalogue": False, "use_sentence_transformers": False}
    with get_session(db_engine) as session:
        with track_run(session, step="world_build", config=small_config) as (run, _c):
            generate_store_world(session, small_config, run_id=run.run_id, **kwargs)
        runs_before = session.scalar(select(func.count()).select_from(PipelineRun))

    with pytest.raises(WorldAlreadyBuiltError, match="empty database"):
        run_full_pipeline(
            small_config,
            annotations_dir=tmp_path / "ann",
            freeze_registry_path=tmp_path / "ann" / "frozen.json",
            summary_output_path=None,
        )
    with get_session(db_engine) as session:
        # No failed/partial pipeline_runs rows were added by the refused run
        assert session.scalar(select(func.count()).select_from(PipelineRun)) == runs_before
