"""Real human check of the judge on an existing run: blind export, import, Cohen's kappa."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from edg.annotation.human_check import (
    HUMAN_CHECK_COLUMNS,
    SECTION_END,
    SECTION_START,
    HumanCheckError,
    _kappa,
    _latest_judge_decisions,
    read_human_check_sheets,
    render_evidence_facts,
    write_human_check_section,
)
from edg.cli import app
from edg.config import load_config
from edg.db.session import create_db_engine, get_session


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HUMAN_CHECK_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def test_export_is_blind_and_import_scores_real_agreement(small_pipeline_db: str, tmp_path: Path) -> None:
    runner = CliRunner()
    sheet = tmp_path / "human_check.csv"
    res = runner.invoke(
        app,
        [
            "calibrate",
            "human-export",
            "--small",
            "--db-url",
            small_pipeline_db,
            "--out",
            str(sheet),
            "--n",
            "20",
        ],
    )
    assert res.exit_code == 0, res.output
    rows = _read(sheet)
    assert len(rows) == 20
    assert (tmp_path / "human_check_instructions.txt").read_text().startswith("How to fill in")
    assert list(rows[0]) == HUMAN_CHECK_COLUMNS
    assert len({r["interaction_id"] for r in rows}) == 20
    assert all(r["customer_query"] and r["ai_answer"] and r["evidence_facts"] for r in rows)
    assert not any("judge" in c or "failure" in c for c in HUMAN_CHECK_COLUMNS), (
        "sheet must not show the judge"
    )

    cfg = load_config(small=True, overrides={"database": {"url": small_pipeline_db}})
    with get_session(create_db_engine(small_pipeline_db)) as s:
        decisions = _latest_judge_decisions(s, cfg.judge.judge_version)
    # Annotator 1 copies the judge, annotator 2 copies it except on two items; both leave item 1 blank.
    for i, r in enumerate(rows):
        judge = "fail" if decisions[r["interaction_id"]]["judge_fail"] else "pass"
        flipped = "pass" if judge == "fail" else "fail"
        if i == 0:
            continue
        r["annotator_1_verdict"] = judge.upper() if i % 2 else judge
        r["annotator_2_verdict"] = flipped if i in (1, 2) else judge
    # Two people fill separate copies; the import merges them.
    a1 = [{**r, "annotator_2_verdict": ""} for r in rows]
    a2 = [{**r, "annotator_1_verdict": ""} for r in rows]
    _write(tmp_path / "a1.csv", a1)
    _write(tmp_path / "a2.csv", a2)
    summary = tmp_path / "results_summary.md"
    summary.write_text("# Results\n\nexisting content\n", encoding="utf-8")
    stem = tmp_path / "human_agreement"
    res = runner.invoke(
        app,
        [
            "calibrate", "human-import", str(tmp_path / "a1.csv"), str(tmp_path / "a2.csv"),
            "--small", "--db-url", small_pipeline_db, "--summary-out", str(summary), "--report-out", str(stem),
        ],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    report = json.loads(stem.with_suffix(".json").read_text())
    assert report["source"] == "real_human_annotation"
    assert report["filled_by"] == {"annotator_1": 19, "annotator_2": 19}
    p = report["pairs"]
    assert p["judge_vs_annotator_1"]["raw_agreement"] == 1.0
    assert p["judge_vs_annotator_2"]["raw_agreement"] == pytest.approx(17 / 19, abs=1e-4)
    assert p["judge_vs_human_consensus"]["n"] == 17
    assert p["judge_vs_human_consensus"]["items_where_annotators_disagree"] == 2
    text = summary.read_text()
    assert text.startswith("# Results") and "existing content" in text
    assert "real human annotation" in text and "Cohen's kappa" in text
    # Re-importing replaces the section instead of adding a second one.
    res = runner.invoke(
        app,
        [
            "calibrate", "human-import", str(tmp_path / "a1.csv"), str(tmp_path / "a2.csv"),
            "--small", "--db-url", small_pipeline_db, "--summary-out", str(summary), "--report-out", str(stem),
        ],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    assert summary.read_text().count(SECTION_START) == 1


def test_import_rejects_a_bad_verdict(tmp_path: Path) -> None:
    sheet = tmp_path / "s.csv"
    _write(
        sheet,
        [{c: "" for c in HUMAN_CHECK_COLUMNS} | {"interaction_id": "INT-1", "annotator_1_verdict": "maybe"}],
    )
    with pytest.raises(HumanCheckError, match="not pass or fail"):
        read_human_check_sheets([sheet])


def test_semicolon_sheet_from_excel_is_read(tmp_path: Path) -> None:
    sheet = tmp_path / "s.csv"
    sheet.write_text(";".join(HUMAN_CHECK_COLUMNS) + "\n1;INT-1;q;a;e;p;Fail;;pass;\n", encoding="utf-8-sig")
    assert read_human_check_sheets([sheet]) == {"INT-1": {"annotator_1": True, "annotator_2": False}}


def test_kappa_values() -> None:
    assert _kappa([True, False] * 10, [True, False] * 10, seed=1)["kappa"] == 1.0
    assert _kappa([True, False] * 10, [False, True] * 10, seed=1)["kappa"] == -1.0
    same = _kappa([True] * 12, [True] * 12, seed=1)
    assert same["kappa"] is None and same["raw_agreement"] == 1.0
    k = _kappa([True, True, False, False] * 5, [True, False, False, False] * 5, seed=1)
    assert k["kappa"] == 0.5 and k["ci95"] is not None and k["ci95"][0] <= 0.5 <= k["ci95"][1]


def test_section_is_replaced_in_place(tmp_path: Path) -> None:
    p = tmp_path / "s.md"
    p.write_text(f"top\n\n{SECTION_START}\nold\n{SECTION_END}\n\nbottom\n", encoding="utf-8")
    write_human_check_section(p, f"{SECTION_START}\nnew\n{SECTION_END}\n")
    assert p.read_text() == f"top\n\n{SECTION_START}\nnew\n{SECTION_END}\n\nbottom\n"


def test_evidence_facts_are_readable() -> None:
    text = render_evidence_facts(
        {
            "reference_date": "2026-11-20T12:00:00Z",
            "order": {"order_id": "ORD-1", "status": "shipped", "total_amount": 89.99},
            "shipments": [{"shipment_id": "SHP-1", "carrier": "UPS", "actual_delivery": None}],
            "refunds_returns": [],
        }
    )
    assert "Order: order_id=ORD-1, status=shipped, total_amount=89.99" in text
    assert "Shipment: shipment_id=SHP-1, carrier=UPS" in text and "actual_delivery" not in text
    assert "Return/refund: none" in text
