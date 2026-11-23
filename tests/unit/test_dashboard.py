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
    DASHBOARD_DIR / "pages" / "1_Overview.py",
    DASHBOARD_DIR / "pages" / "2_Interaction_Explorer.py",
    DASHBOARD_DIR / "pages" / "3_Clusters.py",
    DASHBOARD_DIR / "pages" / "4_Edge_Cases.py",
    DASHBOARD_DIR / "pages" / "5_Judge.py",
    DASHBOARD_DIR / "pages" / "6_Dataset.py",
    DASHBOARD_DIR / "pages" / "7_Model_Comparison.py",
    DASHBOARD_DIR / "pages" / "8_Chat_Assistant.py",
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


@pytest.mark.parametrize("page_path", PAGE_FILES, ids=lambda p: p.name)
def test_dashboard_page_loads_without_errors(
    page_path: Path, monkeypatch: pytest.MonkeyPatch, small_pipeline_db: str
) -> None:
    """Phase 10 acceptance test: every Streamlit page loads cleanly against the small DB."""
    monkeypatch.setenv("EDG_DATABASE_URL", small_pipeline_db)

    at = AppTest.from_file(str(page_path), default_timeout=30)
    at.run()
    assert not at.exception, f"{page_path.name} raised exceptions: {at.exception}"
