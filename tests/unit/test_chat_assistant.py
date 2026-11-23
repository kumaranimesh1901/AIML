"""Tests for the live chat assistant (dashboard page 8) and its `edg.sut.chat` helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from edg.config import load_config
from edg.db.models import StateSnapshot
from edg.llm.client import BaseLLMClient, LLMRequest
from edg.sut.assistant import AssistantResult, CustomerSupportAssistant
from edg.sut.chat import (
    ChatUnavailableError,
    build_live_client,
    chat_reply,
    find_order_ids,
    grade_reply,
    missing_models,
)
from edg.sut.tools import SnapshotToolBox

CHAT_PAGE = Path(__file__).resolve().parents[2] / "dashboard" / "pages" / "8_Chat_Assistant.py"


class _StubClient(BaseLLMClient):
    """Stands in for a real HTTP model (provider is not `fake`), replaying scripted replies."""

    provider = "stub_http"

    def __init__(self, replies: list[str]) -> None:
        super().__init__(default_model="stub", max_retries=0, retry_base_delay_seconds=0.0)
        self.replies = replies
        self.requests: list[LLMRequest] = []

    def _send(self, request: LLMRequest) -> tuple[str, dict[str, Any]]:
        self.requests.append(request)
        return self.replies.pop(0), {}


def _snapshot() -> StateSnapshot:
    return StateSnapshot(
        snapshot_id="SNAP-000007",
        order_id="ORD-000007",
        policy_version="v1",
        hard_state_tags=["partial_shipment"],
        snapshot_json={
            "order": {"order_id": "ORD-000007", "status": "shipped", "total_amount": 42.5},
            "items": [],
            "payments": [],
            "shipments": [{"shipment_id": "SHP-1", "carrier": "UPS", "actual_delivery": None}],
            "refunds_returns": [],
        },
        run_id="run_x",
    )


def _config():
    return load_config(small=True, overrides={"llm": {"provider": "ollama"}})


def test_find_order_ids_and_missing_models() -> None:
    assert find_order_ids("where is ord-000123 and ORD-000123, also ORD-000999?") == [
        "ORD-000123",
        "ORD-000999",
    ]
    assert missing_models(["qwen2.5:7b", "llama3:latest"], ["qwen2.5:7b", "llama3", "qwen2.5:14b"]) == [
        "qwen2.5:14b"
    ]


def test_live_client_refuses_fake_provider() -> None:
    with pytest.raises(ChatUnavailableError):
        build_live_client(load_config(small=True, overrides={"llm": {"provider": "fake"}}), provider="ollama", model="m")
    with pytest.raises(ChatUnavailableError):
        build_live_client(_config(), provider="fake", model="m")
    client = build_live_client(_config(), provider="ollama", model="qwen2.5:7b")
    assert client.provider == "ollama" and client.cache is None


def test_pipeline_prompt_unchanged_without_history() -> None:
    cfg = _config()
    client = _StubClient([json.dumps({"action": "final_answer", "answer": "Hi."})])
    assistant = CustomerSupportAssistant(version_config=cfg.sut.versions["v1"], llm_client=client)
    assistant.answer(query="Where is it?", order_id="ORD-000007", toolbox=SnapshotToolBox(_snapshot()))
    assert client.requests[0].prompt == "Customer query (Order ORD-000007): Where is it?"


def test_chat_reply_uses_tools_and_history() -> None:
    cfg = _config()
    client = _StubClient(
        [
            json.dumps({"action": "tool_call", "tool": "get_shipment", "arguments": {"order_id": "ORD-000007"}}),
            json.dumps({"action": "final_answer", "answer": "Your UPS package SHP-1 is on its way."}),
        ]
    )
    result = chat_reply(
        config=cfg,
        version_name="v2",
        snapshot=_snapshot(),
        message="And when will it arrive?",
        history=[("Where is my order?", "It has shipped.")],
        llm_client=client,
    )
    assert isinstance(result, AssistantResult)
    assert result.answer == "Your UPS package SHP-1 is on its way."
    assert [t["tool"] for t in result.tool_calls] == ["get_shipment"]
    first_prompt = client.requests[0].prompt
    assert "Customer: Where is my order?\nAssistant: It has shipped." in first_prompt
    assert first_prompt.endswith("Customer query (Order ORD-000007): And when will it arrive?")
    assert client.requests[0].model == cfg.sut.versions["v2"].base_model


def test_grade_reply_runs_judge_without_a_session() -> None:
    cfg = _config()
    judge_json = json.dumps(
        {
            **{d: {"verdict": "pass", "score": 5} for d in ("correctness", "relevance", "helpfulness", "completeness", "safety")},
            "failure_types": [],
            "explanation": "Matches the snapshot.",
            "confidence": 0.9,
        }
    )
    client = _StubClient(['{"claims": []}', judge_json])
    jdg = grade_reply(
        config=cfg,
        snapshot=_snapshot(),
        message="Where is my order?",
        reply="Your order has shipped.",
        llm_client=client,
    )
    assert jdg.correctness_verdict == "pass"
    assert jdg.failure_types == []
    assert jdg.evidence_status == "partial"  # snapshot present, no policies passed
    assert client.requests[-1].model == cfg.judge.model_name


def test_chat_page_explains_when_ollama_is_down(monkeypatch: pytest.MonkeyPatch, small_pipeline_db: str) -> None:
    monkeypatch.setenv("EDG_DATABASE_URL", small_pipeline_db)
    monkeypatch.setattr("edg.sut.chat.ollama_status", lambda *_a, **_k: (False, [], "connection refused"))
    at = AppTest.from_file(str(CHAT_PAGE), default_timeout=30)
    at.run()
    assert not at.exception
    assert any("ollama serve" in e.value for e in at.error)


def test_chat_page_answers_a_message(monkeypatch: pytest.MonkeyPatch, small_pipeline_db: str) -> None:
    monkeypatch.setenv("EDG_DATABASE_URL", small_pipeline_db)
    cfg = load_config()
    models = [v.base_model for v in cfg.sut.versions.values()] + [cfg.judge.model_name]
    monkeypatch.setattr("edg.sut.chat.ollama_status", lambda *_a, **_k: (True, models, None))
    seen: dict[str, Any] = {}

    def _fake_reply(**kwargs: Any) -> AssistantResult:
        seen.update(kwargs)
        return AssistantResult(answer="It is on the way.", model_version=kwargs["version_name"], latency_ms=1200.0)

    monkeypatch.setattr("edg.sut.chat.chat_reply", _fake_reply)
    at = AppTest.from_file(str(CHAT_PAGE), default_timeout=30)
    at.run()
    assert not at.exception
    at.chat_input[0].set_value("Where is my order?").run()
    assert not at.exception
    assert seen["message"] == "Where is my order?"
    assert seen["snapshot"].order_id == at.sidebar.selectbox[2].value
    assert any("It is on the way." in m.value for m in at.markdown)
