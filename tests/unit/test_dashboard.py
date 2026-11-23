"""Phase 10 acceptance tests: Streamlit AppTest on all 8 pages + grep check for zero literal metrics."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

PAGE_FILES = [
    DASHBOARD_DIR / "app.py",
    DASHBOARD_DIR / "views" / "home.py",
    DASHBOARD_DIR / "views" / "conversations.py",
    DASHBOARD_DIR / "views" / "topics.py",
    DASHBOARD_DIR / "views" / "edge_cases.py",
    DASHBOARD_DIR / "views" / "judge.py",
    DASHBOARD_DIR / "views" / "dataset.py",
    DASHBOARD_DIR / "views" / "comparison.py",
    DASHBOARD_DIR / "views" / "chat.py",
]

# Patterns that would indicate hardcoded/literal metric numbers in dashboard code
_LITERAL_METRIC_RE = re.compile(
    r'st\.metric\(\s*["\'][^"\']+["\']\s*,\s*(?:["\']\d+(?:\.\d+)?%?["\']|\d+(?:\.\d+)?)\s*[,\)]'
)
_ILLUSTRATIVE_NUMBERS_RE = re.compile(r"\b(?:78%|91%|0\.78|0\.91)\b")


def test_no_hardcoded_or_illustrative_metrics_in_dashboard_code() -> None:
    """Phase 10 grep-based acceptance test: zero literal metric numbers in dashboard code."""
    for py_file in sorted(DASHBOARD_DIR.rglob("*.py")):
        source = py_file.read_text(encoding="utf-8")
        assert not _LITERAL_METRIC_RE.search(source), (
            f"Hardcoded literal metric found in {py_file}"
        )
        assert not _ILLUSTRATIVE_NUMBERS_RE.search(source), (
            f"Illustrative doc number found in {py_file}"
        )


def run_page(page_path: Path) -> AppTest:
    """Run one page the way the app does: through the navigation in app.py."""
    at = AppTest.from_file(str(DASHBOARD_DIR / "app.py"), default_timeout=30)
    if page_path.name != "app.py":
        at.switch_page(str(page_path.relative_to(DASHBOARD_DIR)))
    at.run()
    return at


@pytest.mark.parametrize("page_path", PAGE_FILES, ids=lambda p: p.name)
def test_dashboard_page_loads_without_errors(
    page_path: Path, monkeypatch: pytest.MonkeyPatch, small_pipeline_db: str
) -> None:
    """Phase 10 acceptance test: every Streamlit page loads cleanly against the small DB."""
    monkeypatch.setenv("EDG_DATABASE_URL", small_pipeline_db)

    at = run_page(page_path)
    assert not at.exception, f"{page_path.name} raised exceptions: {at.exception}"


@pytest.fixture(scope="module")
def world_only_db(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A database part-way through a run: the store is built, nothing else yet."""
    import os
    import subprocess
    import sys

    db_url = f"sqlite:///{tmp_path_factory.mktemp('partial') / 'edg_partial.db'}"
    edg = [sys.executable, "-m", "edg.cli"]
    subprocess.run([*edg, "db", "init", "--db-url", db_url], check=True, capture_output=True)
    subprocess.run(
        [*edg, "world", "build", "--small", "--synthetic-catalogue", "--no-st", "--db-url", db_url],
        check=True,
        capture_output=True,
        env={**os.environ, "EDG_LLM_PROVIDER": "fake"},
    )
    return db_url


@pytest.mark.parametrize("page_path", PAGE_FILES, ids=lambda p: p.name)
def test_dashboard_page_handles_partly_built_database(
    page_path: Path, monkeypatch: pytest.MonkeyPatch, world_only_db: str
) -> None:
    """While a pipeline run is still early, every page shows a friendly message instead of crashing."""
    monkeypatch.setenv("EDG_DATABASE_URL", world_only_db)
    monkeypatch.setattr("edg.sut.chat.ollama_status", lambda *_a, **_k: (False, [], "down"))
    at = run_page(page_path)
    assert not at.exception, f"{page_path.name} raised exceptions: {at.exception}"
