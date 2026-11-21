"""Phase 11 end-to-end acceptance test: edg run-all --small from empty DB to release, report, and dashboard."""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import func, select
from typer.testing import CliRunner

from edg.cli import app
from edg.db.models import (
    BenchmarkRelease,
    EvalResult,
    HumanAnnotation,
    Judgement,
    RegressionReport,
    TestCase,
)
from edg.db.session import create_db_engine, get_session

runner = CliRunner()


def test_e2e_run_all_small(tmp_path: Path) -> None:
    """Phase 11 acceptance test: empty DB -> populated tables, frozen v1.0, V1 vs V2 report, summary."""
    db_url = f"sqlite:///{tmp_path / 'e2e_small.db'}"
    ann_dir = tmp_path / "annotations"
    summary_path = tmp_path / "results_summary.md"

    res = runner.invoke(
        app,
        [
            "run-all",
            "--small",
            "--db-url",
            db_url,
            "--provider",
            "fake",
            "--no-hf",
            "--no-st",
            "--no-spacy",
            "--release-id",
            "v1.0",
            "--annotations-dir",
            str(ann_dir),
            "--summary-out",
            str(summary_path),
        ],
    )
    assert res.exit_code == 0, res.output
    data = json.loads(res.stdout)

    assert data["release"]["release_id"] == "v1.0"
    assert data["release"]["total_cases"] > 0
    assert data["comparison"]["baseline_model_id"] == "v1"
    assert data["comparison"]["candidate_model_id"] == "v2"
    assert summary_path.exists()

    summary_text = summary_path.read_text(encoding="utf-8")
    assert "v1.0" in summary_text
    assert "judge_test" in summary_text
    assert "slice:mined" in summary_text
    assert "slice:random_control" in summary_text

    engine = create_db_engine(db_url)
    try:
        with get_session(engine) as session:
            assert session.get(BenchmarkRelease, "v1.0") is not None
            assert (session.scalar(select(func.count(Judgement.judgement_id))) or 0) > 0
            assert (session.scalar(select(func.count(HumanAnnotation.annotation_id))) or 0) > 0
            assert (session.scalar(select(func.count(TestCase.test_case_id))) or 0) > 0
            assert (session.scalar(select(func.count(EvalResult.result_id))) or 0) >= 2 * data["release"]["total_cases"]
            assert (session.scalar(select(func.count(RegressionReport.report_id))) or 0) > 0
    finally:
        engine.dispose()
