"""Regression tests: structured JSON output with real local models (reported from an Ollama run).

With qwen2.5:3b most hard-query generations failed `HardQueryOutput` validation (the model replied
`{"response": ...}` / `{"intent": ...}`) and fell back after 4 tries, because:
  * the Ollama client only sent `format: "json"`, never the JSON schema (Ollama structured outputs);
  * the judge prompt asked for JSON "matching LLMJudgeOutput" without ever showing that schema;
  * validation retries re-sent the identical prompt, which at temperature 0 reproduces the
    identical invalid output, so retries could never recover;
  * the SUT tool-calling loop had the same unconstrained `format: "json"`.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from edg.config import SUTVersionConfig
from edg.judge.schema import LLMJudgeOutput
from edg.llm.client import OllamaClient
from edg.queries.hard_query_generator import HardQueryOutput
from edg.sut.assistant import CustomerSupportAssistant


class _RecordingOllama:
    """In-process stand-in for the Ollama /api/chat endpoint."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.payloads: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        self.payloads.append(payload)
        content = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})


@pytest.fixture
def fake_ollama(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    def _install(replies: list[str]) -> _RecordingOllama:
        server = _RecordingOllama(replies)
        real_client = httpx.Client

        def _client(*args: Any, **kwargs: Any) -> httpx.Client:
            kwargs["transport"] = httpx.MockTransport(server.handler)
            return real_client(*args, **kwargs)

        monkeypatch.setattr(httpx, "Client", _client)
        return server

    return _install


def _client() -> OllamaClient:
    return OllamaClient(default_model="qwen2.5:3b", max_retries=3, retry_base_delay_seconds=0.0)


def _last_user_prompt(payload: dict[str, Any]) -> str:
    return next(m["content"] for m in reversed(payload["messages"]) if m["role"] == "user")


def test_generate_json_sends_json_schema_as_ollama_format(fake_ollama) -> None:  # type: ignore[no-untyped-def]
    server = fake_ollama(['{"query": "Where is order ORD-000004? It says delivered."}'])
    parsed, _ = _client().generate_json("write a query", HardQueryOutput, temperature=0.7)
    assert parsed.query.startswith("Where is order")
    fmt = server.payloads[0]["format"]
    assert isinstance(fmt, dict), "Ollama must receive the JSON schema, not just 'json'"
    assert fmt["properties"].keys() == {"query"}
    assert fmt["required"] == ["query"]


def test_prompt_shows_the_exact_schema_for_providers_without_enforcement(fake_ollama) -> None:  # type: ignore[no-untyped-def]
    valid = LLMJudgeOutput.model_validate(
        {
            **{d: {"verdict": "pass", "score": 5} for d in ("correctness", "relevance", "helpfulness", "completeness", "safety")},
            "failure_types": [],
            "explanation": "All claims supported.",
            "confidence": 0.9,
        }
    ).model_dump_json()
    server = fake_ollama([valid])
    _client().generate_json("judge this", LLMJudgeOutput, system_prompt="Return ONLY valid JSON matching LLMJudgeOutput.")
    prompt = _last_user_prompt(server.payloads[0])
    for field in ("correctness", "safety", "failure_types", "explanation", "confidence", "verdict"):
        assert f'"{field}"' in prompt


def test_validation_retry_tells_the_model_what_was_wrong(fake_ollama) -> None:  # type: ignore[no-untyped-def]
    # Typical small-model reply, then a corrected one
    server = fake_ollama(['{"response": "Hi, my order never arrived"}', '{"query": "My order ORD-1 never arrived, help!"}'])
    parsed, _ = _client().generate_json("write a query", HardQueryOutput, temperature=0.0)
    assert parsed.query.startswith("My order")
    assert len(server.payloads) == 2
    first, second = (_last_user_prompt(p) for p in server.payloads)
    assert first != second, "a temperature-0 retry with an identical prompt can never recover"
    assert '{"response": "Hi, my order never arrived"}' in second
    assert "query" in second and "Field required" in second


def test_sut_tool_loop_constrains_ollama_output_to_action_schema(fake_ollama) -> None:  # type: ignore[no-untyped-def]
    server = fake_ollama(['{"action": "final_answer", "answer": "Your order ORD-1 shipped on 2026-11-01."}'])
    version = SUTVersionConfig(name="v1", provider="ollama", base_model="qwen2.5:1.5b", system_prompt="Be helpful.")

    class _Toolbox:
        tool_trace: list[dict[str, Any]] = []

        def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
            return {}

    ans = CustomerSupportAssistant(version_config=version, llm_client=_client(), max_tool_calls=2).answer(
        query="Where is my order?", order_id="ORD-1", toolbox=_Toolbox()  # type: ignore[arg-type]
    )
    assert ans.answer == "Your order ORD-1 shipped on 2026-11-01."
    fmt = server.payloads[0]["format"]
    assert isinstance(fmt, dict)
    assert "action" in json.dumps(fmt) and "final_answer" in json.dumps(fmt) and "tool_call" in json.dumps(fmt)


# ---------------------------------------------------------------------------
# Pre-run review: Ollama's default context window (2048-4096 tokens) is smaller than a judge
# prompt (rubric + taxonomy + schema + snapshot + policies + reference + claim checks, about
# 3-5k tokens). Ollama silently drops the start of an over-long prompt, so the judge lost its
# rubric and failure-type rules. The client now sets num_ctx and warns when the window fills.
# ---------------------------------------------------------------------------


def test_ollama_requests_set_a_context_window_large_enough_for_the_judge(fake_ollama) -> None:  # type: ignore[no-untyped-def]
    from edg.config import load_config
    from edg.llm.client import create_llm_client

    server = fake_ollama(['{"query": "Where is ORD-1?"}'])
    cfg = load_config(small=True, overrides={"llm": {"provider": "ollama", "cache_enabled": False}})
    client = create_llm_client(cfg.llm)
    client.generate(prompt="hi", model="llama3.1:8b")
    assert server.payloads[0]["options"]["num_ctx"] >= 8192


def test_ollama_warns_when_the_prompt_filled_the_context_window(monkeypatch: pytest.MonkeyPatch, caplog) -> None:  # type: ignore[no-untyped-def]
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"content": "ok"}, "prompt_eval_count": 4096, "eval_count": 10},
        )

    monkeypatch.setattr(httpx, "Client", lambda *a, **k: real_client(*a, **{**k, "transport": httpx.MockTransport(handler)}))
    client = OllamaClient(default_model="llama3.1:8b", num_ctx=4096, cache_enabled=False)
    with caplog.at_level("WARNING"):
        client.generate(prompt="x" * 100)
    assert any("ollama_context_window_full" in r.getMessage() for r in caplog.records)
