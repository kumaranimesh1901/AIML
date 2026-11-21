"""Single-command end-to-end pipeline orchestrator (`edg run-all`)."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from sqlalchemy import select

from edg.annotation.agreement import (
    DEFAULT_FREEZE_REGISTRY,
    evaluate_frozen_judge,
    evaluate_prompt_tuning,
    freeze_judge_version,
)
from edg.annotation.export import export_annotation_sheets
from edg.annotation.import_ import import_annotations_csv
from edg.annotation.ui_helpers import save_single_annotation
from edg.benchmark.releases import create_benchmark_release
from edg.cleaning.pipeline import run_cleaning_pipeline
from edg.clustering.pipeline import run_clustering_pipeline
from edg.config import AppConfig
from edg.db.models import Judgement
from edg.db.repositories import source_judgements_only, track_run
from edg.db.session import get_session, init_db
from edg.embeddings.embedder import compare_embedding_models, embed_interactions
from edg.evaluation.regression import compare_models_on_release
from edg.evaluation.replay import replay_and_grade_release
from edg.judge.llm_judge import run_judge_pipeline
from edg.judge.prompts import build_judge_system_prompt, compute_prompt_hash, load_judge_rubric
from edg.labeling.rules import run_labeling_pipeline
from edg.llm.cache import LLMCache
from edg.llm.client import create_llm_client
from edg.logging import get_logger
from edg.pipeline.report import write_results_summary_markdown
from edg.priority.sampler import run_priority_and_sampling
from edg.sut.runner import run_sut_pipeline
from edg.testcases.generator import run_testcase_generation
from edg.world.store_generator import generate_store_world

logger = get_logger(__name__)


def _fill_simulated_calibration_csv(
    csv_path: Path,
    judgements_by_id: dict[str, Judgement],
    *,
    annotator_id: str = "sim_annotator_1",
    seed: int = 42,
) -> Path:
    """Create a clearly-tagged simulated human annotation CSV for unattended run-all runs."""
    rng = random.Random(seed)
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    filled_path = csv_path.with_name(f"sim_filled_{csv_path.name}")
    if not rows:
        return filled_path

    with filled_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            jdg = judgements_by_id.get(r["interaction_id"])
            if jdg is None:
                continue
            r["annotator_id"] = annotator_id
            r["evidence_status"] = jdg.evidence_status
            for dim in ("correctness", "relevance", "helpfulness", "completeness", "safety"):
                verdict = getattr(jdg, f"{dim}_verdict")
                score = getattr(jdg, f"{dim}_score")
                # Introduce realistic minor inter-rater variance (~8% +-1 score perturbation)
                if score is not None and rng.random() < 0.08:
                    delta = rng.choice([-1, 1])
                    score = max(1, min(5, int(score) + delta))
                r[f"{dim}_verdict"] = verdict
                r[f"{dim}_score"] = "" if score is None else str(score)
            r["failure_types"] = ",".join(jdg.failure_types or [])
            r["notes"] = "simulated_calibration_annotation"
            writer.writerow(r)
    return filled_path


def run_full_pipeline(
    config: AppConfig,
    *,
    release_id: str = "v1.0",
    prefer_hf: bool = False,
    use_sentence_transformers: bool = False,
    use_spacy: bool = False,
    annotations_dir: str | Path = "data/annotations",
    freeze_registry_path: str | Path = DEFAULT_FREEZE_REGISTRY,
    summary_output_path: str | Path | None = "data/exports/results_summary.md",
) -> dict[str, Any]:
    """Execute the complete EDG pipeline from empty DB to frozen release, V1 vs V2 report, and summary."""
    engine = init_db(config.database.url, use_alembic=True, echo=config.database.echo)
    cache = LLMCache(engine=engine) if config.llm.cache_enabled else None
    client = create_llm_client(config.llm, cache=cache)

    results: dict[str, Any] = {"profile": config.profile, "release_id": release_id}

    # 1. World
    with get_session(engine) as session:
        with track_run(session, step="world_build", config=config) as (run, counts):
            res = generate_store_world(
                session,
                config,
                run_id=run.run_id,
                prefer_hf_catalogue=prefer_hf,
                use_sentence_transformers=use_sentence_transformers,
            )
            counts.update(res)
            results["world"] = res

    # 2. SUT V1 + controlled failure tier
    with get_session(engine) as session:
        with track_run(session, step="sut_run_v1", config=config) as (run, counts):
            res = run_sut_pipeline(
                session,
                config,
                run_id=run.run_id,
                version_name="v1",
                llm_client=client,
                prefer_hf_bitext=prefer_hf,
                inject_failures=True,
            )
            counts.update(res)
            results["sut_v1"] = res

    # 3. Clean
    with get_session(engine) as session:
        with track_run(session, step="clean_run", config=config) as (run, counts):
            res = run_cleaning_pipeline(session, config, run_id=run.run_id, use_spacy=use_spacy)
            counts.update(res)
            results["clean"] = res

    # 4. Embed & compare
    with get_session(engine) as session:
        with track_run(session, step="embed_run", config=config) as (run, counts):
            res = embed_interactions(
                session,
                config,
                run_id=run.run_id,
                use_sentence_transformers=use_sentence_transformers,
            )
            counts.update(res)
            results["embed"] = res
        embed_cmp = compare_embedding_models(
            session, config, use_sentence_transformers=use_sentence_transformers
        )
        results["embed_compare"] = embed_cmp

    # 5. Cluster
    with get_session(engine) as session:
        with track_run(session, step="cluster_run", config=config) as (run, counts):
            res = run_clustering_pipeline(session, config, run_id=run.run_id, llm_client=client)
            counts.update(res)
            results["cluster"] = res

    # 6. Priority & Sampling
    with get_session(engine) as session:
        with track_run(session, step="priority_run", config=config) as (run, counts):
            res = run_priority_and_sampling(session, config, run_id=run.run_id)
            counts.update(res)
            results["priority"] = res

    # 7. Judge
    with get_session(engine) as session:
        with track_run(session, step="judge_run", config=config) as (run, counts):
            res = run_judge_pipeline(
                session, config, run_id=run.run_id, llm_client=client, include_controlled=True
            )
            counts.update(res)
            results["judge"] = res

    # 8. Calibration (export cluster-disjoint dev/test, import simulated calibration set, freeze, test-eval)
    ann_path = Path(annotations_dir)
    with get_session(engine) as session:
        exp = export_annotation_sheets(session, config, output_dir=ann_path)
        jdgs_by_id = {
            j.interaction_id: j
            for j in session.scalars(select(Judgement).where(source_judgements_only())).all()
        }

    dev_filled = _fill_simulated_calibration_csv(
        Path(exp["judge_dev_path"]), jdgs_by_id, annotator_id="sim_annotator_1", seed=config.seed
    )
    test_filled = _fill_simulated_calibration_csv(
        Path(exp["judge_test_path"]), jdgs_by_id, annotator_id="sim_annotator_1", seed=config.seed + 1
    )

    with get_session(engine) as session:
        with track_run(session, step="calibrate_import", config=config) as (run, counts):
            d_imp = import_annotations_csv(session, dev_filled, run_id=run.run_id)
            t_imp = import_annotations_csv(session, test_filled, run_id=run.run_id)
            # Double-annotate up to 50 items on judge_test for human-human ceiling
            with test_filled.open("r", encoding="utf-8", newline="") as fh:
                test_rows = list(csv.DictReader(fh))
            rng_hh = random.Random(config.seed + 2)
            double_n = min(50, len(test_rows))
            for r in test_rows[:double_n]:
                jdg = jdgs_by_id.get(r["interaction_id"])
                if jdg is None:
                    continue
                scores: dict[str, int | None] = {}
                verdicts: dict[str, str] = {}
                for dim in ("correctness", "relevance", "helpfulness", "completeness", "safety"):
                    verdicts[dim] = getattr(jdg, f"{dim}_verdict")
                    sc = getattr(jdg, f"{dim}_score")
                    if sc is not None and rng_hh.random() < 0.06:
                        sc = max(1, min(5, int(sc) + rng_hh.choice([-1, 1])))
                    scores[dim] = sc
                save_single_annotation(
                    session,
                    interaction_id=r["interaction_id"],
                    annotator_id="sim_annotator_2",
                    split="judge_test",
                    verdicts=verdicts,
                    scores=scores,
                    failure_types=list(jdg.failure_types or []),
                    notes="simulated_second_annotator",
                )
            counts.update({"dev_imported": d_imp["inserted"], "test_imported": t_imp["inserted"], "double_annotated": double_n})

        dev_eval = evaluate_prompt_tuning(session, split="judge_dev", judge_version=config.judge.judge_version)
        rubric = load_judge_rubric()
        p_hash = compute_prompt_hash(build_judge_system_prompt(rubric), rubric)
        freeze_judge_version(
            judge_version=config.judge.judge_version,
            judge_model=config.judge.model_name,
            prompt_hash=p_hash,
            registry_path=freeze_registry_path,
        )
        test_eval = evaluate_frozen_judge(
            session, judge_version=config.judge.judge_version, registry_path=freeze_registry_path
        )
        results["calibration"] = {
            "dev_paired": dev_eval["agreement"]["n_paired"],
            "test_paired": test_eval["agreement"]["n_paired"],
            "controlled_recall": test_eval["controlled_tier_recall"]["overall_recall"],
        }

    # 9. Label & Test Cases
    with get_session(engine) as session:
        with track_run(session, step="label_run", config=config) as (run, counts):
            res = run_labeling_pipeline(session, config, run_id=run.run_id)
            counts.update(res)
            results["label"] = res

    with get_session(engine) as session:
        with track_run(session, step="testcases_generate", config=config) as (run, counts):
            res = run_testcase_generation(session, config, run_id=run.run_id, llm_client=client)
            counts.update(res)
            results["testcases"] = res

    # 10. Frozen Release
    with get_session(engine) as session:
        with track_run(session, step=f"release_create_{release_id}", config=config) as (run, counts):
            res = create_benchmark_release(session, config, release_id=release_id, run_id=run.run_id)
            counts.update(res)
            results["release"] = res

    # 11. Replay V1 & V2 and Compare
    for model_ver in ("v1", "v2"):
        with get_session(engine) as session:
            with track_run(session, step=f"eval_replay_{release_id}_{model_ver}", config=config) as (run, counts):
                res = replay_and_grade_release(
                    session,
                    config,
                    release_id=release_id,
                    model_version_name=model_ver,
                    run_id=run.run_id,
                    sut_client=client,
                    judge_client=client,
                )
                counts.update(res)
                results[f"replay_{model_ver}"] = res

    with get_session(engine) as session:
        with track_run(session, step=f"eval_compare_{release_id}_v1_vs_v2", config=config) as (run, counts):
            cmp = compare_models_on_release(
                session,
                config,
                release_id=release_id,
                baseline_model_id="v1",
                candidate_model_id="v2",
                run_id=run.run_id,
            )
            counts.update(
                {
                    "n_shared_cases": cmp["n_shared_cases"],
                    "flagged_regressions_count": cmp["flagged_regressions_count"],
                }
            )
            results["comparison"] = cmp

        if summary_output_path is not None:
            summary_file = write_results_summary_markdown(
                session,
                config,
                release_id=release_id,
                baseline_model_id="v1",
                candidate_model_id="v2",
                output_path=summary_output_path,
                embedding_comparison=embed_cmp,
                judge_flip_info=cmp.get("judge_flip_rate"),
            )
            results["results_summary_path"] = str(summary_file)

    engine.dispose()
    return results
