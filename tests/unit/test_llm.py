"""Unit tests for the LLM client, fake LLM, retry logic, JSON validation, and cache."""

from __future__ import annotations

import json
from typing import Literal

import pytest
from pydantic import BaseModel, Field
from sqlalchemy import Engine

from edg.config import LLMConfig
from edg.llm import (
    FakeLLMClient,
    LLMCache,
    LLMRetryExhaustedError,
    LLMValidationError,
    OllamaClient,
    OpenAICompatibleClient,
    create_llm_client,
)


class SampleVerdict(BaseModel):
    verdict: Literal["pass", "partial", "fail", "not_assessable"]
    score: int = Field(ge=1, le=5)
    reason: str
    tags: list[str] = Field(default_factory=list)


def test_fake_llm_returns_schema_valid_json() -> None:
    client = FakeLLMClient(cache_enabled=False)
    parsed, resp = client.generate_json("Evaluate this response", SampleVerdict)
    assert isinstance(parsed, SampleVerdict)
    assert parsed.verdict in {"pass", "partial", "fail", "not_assessable"}
    assert 1 <= parsed.score <= 5
    assert resp.provider == "fake"
    assert resp.cached is False


def test_retry_logic_recovers_from_transient_failures() -> None:
    client = FakeLLMClient(
        default_response="recovered response",
        fail_times=2,
        max_retries=3,
        retry_base_delay_seconds=0.0,
        cache_enabled=False,
    )
    resp = client.generate("hello")
    assert resp.text == "recovered response"
    assert client.call_count == 3


def test_retry_logic_raises_after_max_retries() -> None:
    client = FakeLLMClient(
        always_fail=True,
        max_retries=2,
        retry_base_delay_seconds=0.0,
        cache_enabled=False,
    )
    with pytest.raises(LLMRetryExhaustedError):
        client.generate("hello")
    assert client.call_count == 3


def test_generate_json_retries_on_invalid_json_then_succeeds() -> None:
    valid_payload = json.dumps(
        {"verdict": "pass", "score": 5, "reason": "Matches snapshot.", "tags": ["ok"]}
    )
    client = FakeLLMClient(
        responses=[
            "not json at all",
            '{"verdict": "bad_enum", "score": 99}',
            f"```json\n{valid_payload}\n```",
        ],
        max_retries=3,
        retry_base_delay_seconds=0.0,
        cache_enabled=False,
    )
    parsed, _resp = client.generate_json("judge this", SampleVerdict)
    assert parsed.verdict == "pass"
    assert parsed.score == 5
    assert client.call_count == 3


def test_generate_json_raises_validation_error_when_exhausted() -> None:
    client = FakeLLMClient(
        responses=["bad1", "bad2", "bad3"],
        max_retries=2,
        retry_base_delay_seconds=0.0,
        cache_enabled=False,
    )
    with pytest.raises(LLMValidationError):
        client.generate_json("judge this", SampleVerdict)
    assert client.call_count == 3


def test_llm_cache_hits_in_memory_and_db(db_engine: Engine) -> None:
    cache = LLMCache(engine=db_engine)
    client1 = FakeLLMClient(default_response="cached answer", cache=cache, cache_enabled=True)

    r1 = client1.generate("same prompt", system_prompt="sys")
    assert r1.cached is False
    assert r1.text == "cached answer"
    assert client1.call_count == 1

    # Second call on same client hits memory cache
    r2 = client1.generate("same prompt", system_prompt="sys")
    assert r2.cached is True
    assert r2.text == "cached answer"
    assert client1.call_count == 1

    # Fresh cache instance pointing to same DB hits DB cache
    cache2 = LLMCache(engine=db_engine)
    client2 = FakeLLMClient(default_response="different answer", cache=cache2, cache_enabled=True)
    r3 = client2.generate("same prompt", system_prompt="sys")
    assert r3.cached is True
    assert r3.text == "cached answer"
    assert client2.call_count == 0


def test_create_llm_client_factory() -> None:
    cfg = LLMConfig(provider="ollama")
    assert isinstance(create_llm_client(cfg), OllamaClient)
    assert isinstance(create_llm_client(cfg, provider="openai_compatible"), OpenAICompatibleClient)
    assert isinstance(create_llm_client(cfg, provider="fake"), FakeLLMClient)
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_client(cfg, provider="unknown")
