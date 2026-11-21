"""Unit and acceptance tests for Phase 7: Annotation export/import, split guard, agreement metrics, and freezing."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from sqlalchemy import Engine, select
from typer.testing import CliRunner

from edg.annotation.agreement import (
    JudgeFreezeError,
    SplitLeakageError,
    evaluate_frozen_judge,
    evaluate_prompt_tuning,
    freeze_judge_version,
)
from edg.annotation.export import export_annotation_sheets
from edg.annotation.import_ import import_annotations_csv
from edg.annotation.ui_helpers import get_annotation_progress, save_single_annotation
from edg.cleaning.pipeline import run_cleaning_pipeline
from edg.cli import app
from edg.clustering.pipeline import run_clustering_pipeline
from edg.config import AppConfig
from edg.db.models import Judgement
from edg.db.repositories import track_run
from edg.db.session import get_session
from edg.embeddings.embedder import embed_interactions
from edg.judge.llm_judge import run_judge_pipeline
from edg.llm.fake import FakeLLMClient
from edg.metrics.judge_metrics import (
    compute_binary_classification_metrics,
    compute_controlled_tier_recall,
    compute_ordinal_agreement,
)
from edg.priority.sampler import run_priority_and_sampling
from edg.sut.runner import run_sut_pipeline
from edg.world.store_generator import generate_store_world

runner = CliRunner()


def test_metrics_match_hand_computed_values_on_tiny_fixture() -> None:
    """Phase 7 acceptance test: exact match against hand-computed ordinal & binary metrics."""
    # 5 items, 1 not_assessable (None) -> 4 assessed
    y_true_ord: list[int | None] = [5, 4, 2, 1, None]
    y_pred_ord: list[int | None] = [5, 3, 2, 2, 4]
    ord_res = compute_ordinal_agreement(y_true_ord, y_pred_ord)

    assert ord_res["n_assessed"] == 4
    assert ord_res["excluded_not_assessable"] == 1
    # Hand-computed MAE: (|5-5| + |4-3| + |2-2| + |1-2|) / 4 = 2 / 4 = 0.5
    assert ord_res["mae"] == 0.5
    # Hand-computed quadratic weighted kappa for (5,5),(4,3),(2,2),(1,2):
    # Observed weighted disagreement O = (0 + 1 + 0 + 1) / 4 = 0.5
    # Expected weighted disagreement E = (1/16) * sum_i sum_j (y_true[i] - y_pred[j])^2
    # y_true = [5,4,2,1], y_pred = [5,3,2,2]:
    # sum_j (5-p)^2 = 0+4+9+9=22; (4-p)^2 = 1+1+4+4=10; (2-p)^2 = 9+1+0+0=10; (1-p)^2 = 16+4+1+1=22
    # total = 64 -> E = 64 / 16 = 4.0 -> kappa = 1 - O/E = 1 - 0.5/4.0 = 0.875
    assert ord_res["weighted_kappa"] == 0.875
    # Ranks: true=[4,3,2,1], pred=[4,3,1.5,1.5] -> Spearman = 0.9487
    assert ord_res["spearman"] is not None and abs(ord_res["spearman"] - 0.9487) < 1e-3

    # Binary pass/fail fixture with 1 excluded None
    y_true_bin: list[bool | None] = [True, True, False, False, None]
    y_pred_bin: list[bool | None] = [True, False, True, False, True]
    bin_res = compute_binary_classification_metrics(y_true_bin, y_pred_bin)
    assert bin_res["n_assessed"] == 4
    assert bin_res["excluded_not_assessable"] == 1
    assert bin_res["confusion_matrix"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert bin_res["precision"] == 0.5
    assert bin_res["recall"] == 0.5
    assert bin_res["f1"] == 0.5

    # Controlled-tier recall
    ctrl = compute_controlled_tier_recall(
        [
            ("factual_error", ["factual_error"]),
            ("factual_error", []),
            ("hallucination", ["hallucination", "incomplete_response"]),
        ]
    )
    assert ctrl["total_controlled"] == 3
    assert ctrl["overall_recall"] == round(2 / 3, 4)
    assert ctrl["per_failure_type"]["factual_error"]["recall"] == 0.5
    assert ctrl["per_failure_type"]["hallucination"]["recall"] == 1.0


def test_split_guard_blocks_judge_test_in_prompt_tuning(
    db_engine: Engine, tmp_path: Path
) -> None:
    """Phase 7 acceptance test: judge_test is never used by any prompt-tuning command."""
    with get_session(db_engine) as session:
        with pytest.raises(SplitLeakageError, match="judge_dev"):
            evaluate_prompt_tuning(session, split="judge_test")

    db_url = f"sqlite:///{tmp_path / 'guard.db'}"
    res = runner.invoke(
        app,
        ["calibrate", "dev-eval", "--small", "--db-url", db_url, "--split", "judge_test"],
    )
    assert res.exit_code != 0


def test_freeze_guard_and_immutability(db_engine: Engine, tmp_path: Path) -> None:
    reg_path = tmp_path / "frozen.json"
    with get_session(db_engine) as session:
        with pytest.raises(JudgeFreezeError, match="must be frozen"):
            evaluate_frozen_judge(session, judge_version="j1.0", registry_path=reg_path)

    entry = freeze_judge_version(
        judge_version="j1.0",
        judge_model="llama3.1:8b",
        prompt_hash="hash_v1",
        registry_path=reg_path,
    )
    assert entry["judge_version"] == "j1.0"

    # Re-freezing with same hash is idempotent
    same = freeze_judge_version(
        judge_version="j1.0",
        judge_model="llama3.1:8b",
        prompt_hash="hash_v1",
        registry_path=reg_path,
    )
    assert same["prompt_hash"] == "hash_v1"

    # Re-freezing with different hash raises JudgeFreezeError
    with pytest.raises(JudgeFreezeError, match="already frozen"):
        freeze_judge_version(
            judge_version="j1.0",
            judge_model="llama3.1:8b",
            prompt_hash="hash_v2_modified",
            registry_path=reg_path,
        )


def test_export_cluster_disjoint_import_and_calibrate_flow(
    db_engine: Engine, small_config: AppConfig, tmp_path: Path
) -> None:
    cfg = small_config.model_copy(deep=True)
    cfg.world.n_orders = 60
    cfg.world.n_customers = 30
    cfg.queries.bitext_sample_size = 60
    cfg.queries.hard_query_count = 20
    cfg.priority.judge_sample_size = 50
    cfg.judge.dev_annotation_size = 20
    cfg.judge.test_annotation_size = 20

    with get_session(db_engine) as session:
        with track_run(session, step="world", config=cfg) as (run, _c):
            generate_store_world(
                session, cfg, run_id=run.run_id, prefer_hf_catalogue=False, use_sentence_transformers=False
            )
        with track_run(session, step="sut", config=cfg) as (run, _c):
            run_sut_pipeline(session, cfg, run_id=run.run_id, prefer_hf_bitext=False, inject_failures=True)
        with track_run(session, step="clean", config=cfg) as (run, _c):
            run_cleaning_pipeline(session, cfg, run_id=run.run_id, use_spacy=False)
        with track_run(session, step="embed", config=cfg) as (run, _c):
            embed_interactions(session, cfg, run_id=run.run_id, use_sentence_transformers=False)
        with track_run(session, step="cluster", config=cfg) as (run, _c):
            run_clustering_pipeline(session, cfg, run_id=run.run_id, llm_client=FakeLLMClient())
        with track_run(session, step="priority", config=cfg) as (run, _c):
            run_priority_and_sampling(session, cfg, run_id=run.run_id)
        with track_run(session, step="judge", config=cfg) as (run, _c):
            run_judge_pipeline(session, cfg, run_id=run.run_id, include_controlled=True)

    ann_dir = tmp_path / "annotations"
    with get_session(db_engine) as session:
        exp = export_annotation_sheets(session, cfg, output_dir=ann_dir)

    assert exp["judge_dev_count"] > 0
    assert exp["judge_test_count"] > 0

    # Verify zero cluster overlap between dev and test CSVs
    def _read_rows(p: Path) -> list[dict[str, str]]:
        with p.open("r", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    dev_rows = _read_rows(Path(exp["judge_dev_path"]))
    test_rows = _read_rows(Path(exp["judge_test_path"]))
    dev_clusters = {r["cluster_id"] for r in dev_rows}
    test_clusters = {r["cluster_id"] for r in test_rows}
    assert dev_clusters.isdisjoint(test_clusters)

    # Fill annotations from judgements (simulating human review) and import
    with get_session(db_engine) as session:
        jdgs = {j.interaction_id: j for j in session.scalars(select(Judgement)).all()}

    for csv_path, rows in ((Path(exp["judge_dev_path"]), dev_rows), (Path(exp["judge_test_path"]), test_rows)):
        filled_path = csv_path.with_name(f"filled_{csv_path.name}")
        with filled_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                j = jdgs[r["interaction_id"]]
                r["annotator_id"] = "human_1"
                r["correctness_verdict"] = j.correctness_verdict
                r["correctness_score"] = str(j.correctness_score or "")
                r["relevance_verdict"] = j.relevance_verdict
                r["relevance_score"] = str(j.relevance_score or "")
                r["helpfulness_verdict"] = j.helpfulness_verdict
                r["helpfulness_score"] = str(j.helpfulness_score or "")
                r["completeness_verdict"] = j.completeness_verdict
                r["completeness_score"] = str(j.completeness_score or "")
                r["safety_verdict"] = j.safety_verdict
                r["safety_score"] = str(j.safety_score or "")
                r["failure_types"] = ",".join(j.failure_types or [])
                writer.writerow(r)

        with get_session(db_engine) as session:
            imp = import_annotations_csv(session, filled_path)
            assert imp["inserted"] == len(rows)
            assert imp["rejected"] == 0

    # Add one double-annotated item via UI helper
    with get_session(db_engine) as session:
        first_iid = test_rows[0]["interaction_id"]
        j0 = jdgs[first_iid]
        save_single_annotation(
            session,
            interaction_id=first_iid,
            annotator_id="human_2",
            split="judge_test",
            verdicts={
                "correctness": j0.correctness_verdict,
                "relevance": j0.relevance_verdict,
                "helpfulness": j0.helpfulness_verdict,
                "completeness": j0.completeness_verdict,
                "safety": j0.safety_verdict,
            },
            scores={
                "correctness": j0.correctness_score,
                "relevance": j0.relevance_score,
                "helpfulness": j0.helpfulness_score,
                "completeness": j0.completeness_score,
                "safety": j0.safety_score,
            },
            failure_types=list(j0.failure_types or []),
        )
        prog = get_annotation_progress(session, "judge_test")
        assert prog["annotated_count"] == len(test_rows) + 1

    # Dev eval succeeds on judge_dev
    with get_session(db_engine) as session:
        dev_rep = evaluate_prompt_tuning(session, split="judge_dev", judge_version=cfg.judge.judge_version)
    assert dev_rep["agreement"]["n_paired"] == len(dev_rows)
    assert dev_rep["controlled_tier_recall"]["total_controlled"] > 0

    # Freeze and run test eval on judge_test
    reg = tmp_path / "frozen_judges.json"
    freeze_judge_version(
        judge_version=cfg.judge.judge_version,
        judge_model=cfg.judge.model_name,
        prompt_hash="test_hash",
        registry_path=reg,
    )
    with get_session(db_engine) as session:
        test_rep = evaluate_frozen_judge(
            session, judge_version=cfg.judge.judge_version, registry_path=reg
        )
    assert test_rep["agreement"]["n_paired"] == len(test_rows)
    assert test_rep["human_human_ceiling"]["n_double_annotated"] == 1
    assert json.dumps(test_rep)  # JSON serializable
