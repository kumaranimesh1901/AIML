"""Bundled results for a fresh clone (edg.db.demo): snapshot, seed, and never overwrite data."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from edg.db import demo

REPO = Path(__file__).resolve().parents[2]
EDG = [sys.executable, "-m", "edg.cli"]
ENV = {**os.environ, "EDG_LLM_PROVIDER": "fake"}


def _count(path: Path, table: str) -> int:
    con = sqlite3.connect(path)
    try:
        return int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])  # noqa: S608
    finally:
        con.close()


def _layout(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (path / "configs").symlink_to(REPO / "configs")
    monkeypatch.chdir(path)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A working directory laid out like the repo, with a built store in data/edg.db."""
    _layout(tmp_path, monkeypatch)
    url = "sqlite:///data/edg.db"
    for args in (["db", "init", "--empty"], ["world", "build", "--small", "--synthetic-catalogue", "--no-st"]):
        subprocess.run([*EDG, *args, "--db-url", url], check=True, capture_output=True, env=ENV, cwd=tmp_path)
    con = sqlite3.connect(tmp_path / "data/edg.db")
    con.execute(
        "INSERT INTO llm_cache (cache_key, provider, model, prompt_hash, response_text, latency_ms, created_at)"
        " VALUES ('k', 'ollama', 'm', 'h', 'r', 0, '2026-01-01')"
    )
    con.commit()
    con.close()
    (tmp_path / "data/exports").mkdir(parents=True)
    (tmp_path / "data/exports/human_agreement.json").write_text('{"source": "real_human_annotation"}')
    return tmp_path


def test_snapshot_then_fresh_clone_init_loads_it(repo: Path) -> None:
    out = demo.make_snapshot("sqlite:///data/edg.db")
    assert out == demo.DEMO_DB_PATH and out.exists()
    assert _count(out, "llm_cache") == 0 and _count(out, "orders") > 0
    assert (repo / "data/demo/human_agreement.json").exists()
    orders = _count(out, "orders")

    # A fresh clone: no working database and no exports yet
    (repo / "data/edg.db").unlink()
    (repo / "data/exports/human_agreement.json").unlink()
    res = subprocess.run([*EDG, "db", "init"], check=True, capture_output=True, text=True, env=ENV, cwd=repo)
    assert "bundled results" in res.stdout
    assert _count(repo / "data/edg.db", "orders") == orders
    assert {"chat_turns", "support_tickets"} <= {
        r[0] for r in sqlite3.connect(repo / "data/edg.db").execute("SELECT name FROM sqlite_master")
    }
    assert (repo / "data/exports/human_agreement.json").exists()


def test_seed_never_replaces_data_a_custom_url_or_an_empty_request(repo: Path) -> None:
    demo.make_snapshot("sqlite:///data/edg.db")
    assert demo.seed_from_demo("sqlite:///data/edg.db") is False  # already holds a store
    assert demo.seed_from_demo(f"sqlite:///{repo / 'other.db'}") is False  # not the default DB
    assert demo.seed_from_demo("postgresql+psycopg://u:p@localhost/edg") is False

    (repo / "data/edg.db").unlink()
    subprocess.run([*EDG, "db", "init", "--empty"], check=True, capture_output=True, env=ENV, cwd=repo)
    assert _count(repo / "data/edg.db", "orders") == 0
    assert demo.seed_from_demo("sqlite:///data/edg.db") is True  # an empty schema is filled
    assert _count(repo / "data/edg.db", "orders") > 0


def test_snapshot_refuses_a_database_without_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _layout(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        demo.make_snapshot("sqlite:///data/edg.db")
    subprocess.run([*EDG, "db", "init", "--empty"], check=True, capture_output=True, env=ENV, cwd=tmp_path)
    with pytest.raises(ValueError):
        demo.make_snapshot("sqlite:///data/edg.db")
