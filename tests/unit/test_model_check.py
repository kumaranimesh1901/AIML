"""The Ollama model check and download used by the dashboard and `edg doctor` (edg.sut.chat)."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from edg.config import load_config
from edg.sut.chat import check_models, pull_model, required_models


@pytest.fixture
def ollama() -> Iterator[tuple[str, set[str]]]:
    have = {"qwen2.5:7b:latest"} | {"qwen2.5:7b"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_GET(self) -> None:  # noqa: N802
            body = json.dumps({"models": [{"name": n} for n in sorted(have)]}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.end_headers()
            if req["model"] == "bad:model":
                self.wfile.write(b'{"error": "pull model manifest: file does not exist"}\n')
                return
            for msg in ({"status": "pulling manifest"}, {"status": "pulling x", "total": 10, "completed": 10}, {"status": "success"}):
                self.wfile.write((json.dumps(msg) + "\n").encode())
            have.add(req["model"])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", have
    server.shutdown()


def _config(url: str):  # noqa: ANN202
    return load_config(overrides={"llm": {"provider": "ollama", "ollama_base_url": url}})


def test_required_models_cover_both_versions_and_the_judge() -> None:
    models = dict(required_models(load_config()))
    assert {"qwen2.5:1.5b", "qwen2.5:7b", "qwen2.5:14b"} <= set(models)
    assert "v1" in models["qwen2.5:1.5b"] and "v2" in models["qwen2.5:7b"] and "judge" in models["qwen2.5:14b"]


def test_check_then_pull_a_missing_model(ollama: tuple[str, set[str]]) -> None:
    url, _ = ollama
    status = check_models(_config(url))
    assert status["reachable"]
    missing = {m["model"] for m in status["models"] if not m["installed"]}
    assert missing == {"qwen2.5:1.5b", "qwen2.5:14b"}
    msgs = list(pull_model(url, "qwen2.5:1.5b"))
    assert msgs[-1]["status"] == "success"
    after = {m["model"] for m in check_models(_config(url))["models"] if not m["installed"]}
    assert after == {"qwen2.5:14b"}
    with pytest.raises(RuntimeError, match="does not exist"):
        list(pull_model(url, "bad:model"))


def test_unreachable_ollama_marks_every_model_missing() -> None:
    status = check_models(_config("http://127.0.0.1:9"))
    assert not status["reachable"] and status["error"]
    assert not any(m["installed"] for m in status["models"])
