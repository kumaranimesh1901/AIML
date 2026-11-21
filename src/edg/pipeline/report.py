"""Generate data/exports/results_summary.md strictly from real database rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from edg.annotation.agreement import _evaluate_split
from edg.config import AppConfig
from edg.db.models import (
    BenchmarkRelease,
    ClusteringRun,
    JudgeSample,
    Label,
    PipelineRun,
    PriorityScore,
    RegressionReport,
)
from edg.judge.schema import DIMENSIONS
from edg.metrics.dataset_metrics import compute_dataset_metrics


def _fmt_pct(val: float | None) -> str:
    return f"{val * 100:.2f}%" if val is not None else "N/A"


def _fmt_num(val: float | None, digits: int = 4) -> str:
    return f"{val:.{digits}f}" if val is not None else "N/A"


def _compute_priority_lift(session: Session) -> list[dict[str, Any]]:
    rows = session.execute(
        select(PriorityScore.rank, JudgeSample.stratum, Label.failure_detected)
        .join(JudgeSample, JudgeSample.interaction_id == PriorityScore.interaction_id)
        .join(Label, Label.interaction_id == PriorityScore.interaction_id)
        .order_by(PriorityScore.rank)
    ).all()
    if not rows:
        return []

    rc = [r for r in rows if r.stratum == "random_control"]
    rc_rate = sum(1 for r in rc if r.failure_detected) / len(rc) if rc else 0.0
    out: list[dict[str, Any]] = [
        {"group": "random_control", "n": len(rc), "failure_rate": round(rc_rate, 4), "lift": 1.0}
    ]
    for k in (50, 100, 500):
        top_k = rows[: min(k, len(rows))]
        if not top_k:
            continue
        p_at_k = sum(1 for r in top_k if r.failure_detected) / len(top_k)
        lift = round(p_at_k / rc_rate, 2) if rc_rate > 0 else None
        out.append(
            {
                "group": f"top_{k}",
                "n": len(top_k),
                "failure_rate": round(p_at_k, 4),
                "lift": lift,
            }
        )
    return out


def write_results_summary_markdown(
    session: Session,
    config: AppConfig,
    *,
    release_id: str = "v1.0",
    baseline_model_id: str = "v1",
    candidate_model_id: str = "v2",
    output_path: str | Path = "data/exports/results_summary.md",
    embedding_comparison: dict[str, Any] | None = None,
    judge_flip_info: dict[str, Any] | None = None,
) -> Path:
    """Write a complete results summary markdown report using exclusively DB-backed values."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    runs = list(session.scalars(select(PipelineRun).order_by(PipelineRun.started_at)).all())
    latest_crun = session.scalars(
        select(ClusteringRun).order_by(ClusteringRun.created_at.desc()).limit(1)
    ).first()
    release = session.get(BenchmarkRelease, release_id)
    ds_metrics = compute_dataset_metrics(session, release_id) if release else {}
    test_cal = _evaluate_split(session, split="judge_test", judge_version=config.judge.judge_version)
    priority_lift = _compute_priority_lift(session)

    # Report only the most recent comparison run for this (release, baseline, candidate) triple;
    # earlier `edg eval compare` runs stay in the DB for lineage but must not be double-counted.
    pair_filter = (
        RegressionReport.release_id == release_id,
        RegressionReport.baseline_model_id == baseline_model_id,
        RegressionReport.candidate_model_id == candidate_model_id,
    )
    latest_cmp_run_id = session.scalars(
        select(RegressionReport.run_id)
        .join(PipelineRun, PipelineRun.run_id == RegressionReport.run_id)
        .where(*pair_filter)
        .order_by(PipelineRun.started_at.desc(), RegressionReport.run_id.desc())
        .limit(1)
    ).first()
    reg_rows = list(
        session.scalars(
            select(RegressionReport).where(*pair_filter, RegressionReport.run_id == latest_cmp_run_id)
        ).all()
    )

    lines: list[str] = [
        "# Automated Evaluation Dataset Generator — Results Summary",
        "",
        "> Every number in this report is computed directly from the SQLAlchemy database.",
        "",
        "## Lineage & Run Metadata",
        "",
        f"- **Profile**: `{config.profile}` (seed `{config.seed}`)",
        f"- **Config Hash**: `{config.config_hash()}`",
        f"- **Benchmark Release**: `{release_id}` (content hash `{release.content_hash if release else 'N/A'}`)",
        f"- **Frozen Judge Version**: `{release.judge_version if release else config.judge.judge_version}` (`{config.judge.model_name}`)",
        f"- **Models Compared**: `{baseline_model_id}` (`{config.sut.versions[baseline_model_id].base_model}`) vs `{candidate_model_id}` (`{config.sut.versions[candidate_model_id].base_model}`)",
        f"- **Recorded Pipeline Runs**: {len(runs)}",
        "",
        "## 1. Data Processing",
        "",
        "| Run ID | Step | Status | Key Counts |",
        "|---|---|---|---|",
    ]
    for r in runs:
        summary_counts = ", ".join(
            f"{k}={v}"
            for k, v in (r.counts_json or {}).items()
            if not isinstance(v, (dict, list))
        )[:160]
        lines.append(f"| `{r.run_id}` | `{r.step}` | `{r.status}` | {summary_counts} |")

    lines.extend(["", "## 2. Embedding Model Comparison", ""])
    if embedding_comparison and embedding_comparison.get("models"):
        lines.append(f"Evaluated on `{embedding_comparison.get('n_queries', 0)}` representative queries:")
        lines.append("")
        lines.append("| Model | Within-Intent Recall@5 | Seconds / 1k Texts |")
        lines.append("|---|---|---|")
        for m_name, m_stats in embedding_comparison["models"].items():
            lines.append(
                f"| `{m_name}` | {_fmt_num(m_stats.get('recall_at_5'))} | {_fmt_num(m_stats.get('seconds_per_1k_texts'), 3)} |"
            )
    else:
        lines.append("_No embedding comparison recorded in this run._")

    lines.extend(["", "## 3. Clustering Quality", ""])
    if latest_crun is not None:
        cluster_run_rec = next((r for r in runs if r.step == "cluster_run"), None)
        c_counts = cluster_run_rec.counts_json if cluster_run_rec and cluster_run_rec.counts_json else {}
        lines.extend(
            [
                f"- **Clustering Run ID**: `{latest_crun.clustering_run_id}`",
                f"- **Clusters Discovered**: {latest_crun.n_clusters}",
                f"- **Noise Share**: {_fmt_pct(latest_crun.noise_share)}",
                f"- **DBCV (relative_validity)**: {_fmt_num(latest_crun.dbcv)}",
                f"- **3-Seed Stability (Pairwise Non-Noise ARI)**: {_fmt_num(latest_crun.ari_vs_other_seeds)}",
                f"- **Mean Intent Purity**: {_fmt_num(c_counts.get('mean_intent_purity'))}",
                f"- **Candidate New Intents**: {c_counts.get('candidate_new_intents', 'N/A')}",
                f"- **Silhouette (Non-Noise Points Only)**: {_fmt_num(c_counts.get('silhouette_non_noise'))}",
            ]
        )

    lines.extend(["", "## 4. Priority Scoring & Lift vs. Random Control", ""])
    if priority_lift:
        lines.append("| Group | n | Failure Rate (Precision@k) | Lift vs Random Control |")
        lines.append("|---|---|---|---|")
        for row in priority_lift:
            lines.append(
                f"| `{row['group']}` | {row['n']} | {_fmt_pct(row['failure_rate'])} | {row['lift'] if row['lift'] is not None else 'N/A'}x |"
            )

    lines.extend(["", "## 5. Judge Validation (`judge_test`) & Reliability", ""])
    agr = test_cal["agreement"]
    lines.append(f"- **Paired Items (`judge_test`)**: {agr['n_paired']}")
    lines.append(
        f"- **Double-Annotated Human-Human Ceiling Subset**: {test_cal['human_human_ceiling']['n_double_annotated']} items"
    )
    lines.append("")
    lines.append("| Dimension | Assessed (Excluded N/A) | Weighted Kappa | Spearman | MAE | Pass/Fail F1 | Human-Human Kappa |")
    lines.append("|---|---|---|---|---|---|---|")
    for dim in DIMENSIONS:
        o = agr["ordinal_by_dimension"][dim]
        c = agr["categorical_by_dimension"][dim]
        hh = test_cal["human_human_ceiling"]["by_dimension"][dim]
        lines.append(
            f"| `{dim}` | {o['n_assessed']} ({o['excluded_not_assessable']}) | "
            f"{_fmt_num(o['weighted_kappa'])} | {_fmt_num(o['spearman'])} | {_fmt_num(o['mae'])} | "
            f"{_fmt_num(c['f1'])} | {_fmt_num(hh['weighted_kappa'])} |"
        )

    ctrl = test_cal["controlled_tier_recall"]
    rel = test_cal["reliability"]
    lines.extend(
        [
            "",
            f"- **Controlled-Tier Overall Recall**: {_fmt_pct(ctrl['overall_recall'])} (`n={ctrl['total_controlled']}`)",
            f"- **Abstention Rate**: {_fmt_pct(rel['abstention_rate'])}",
            f"- **Judge Error Rate**: {_fmt_pct(rel['judge_error_rate'])}",
            f"- **Length-vs-Score Spearman (Bias Check)**: {_fmt_num(rel['length_score_spearman'])}",
            f"- **Judge Repeat Flip Rate**: {_fmt_pct(judge_flip_info['flip_rate']) if judge_flip_info else 'N/A'} "
            f"(n={judge_flip_info['n_sampled'] if judge_flip_info else 0})",
        ]
    )

    lines.extend(["", "## 6. Benchmark Dataset Quality (`v1.0`)", ""])
    if ds_metrics:
        lines.extend(
            [
                f"- **Total Frozen Cases**: {ds_metrics['total_cases']}",
                f"- **By Slice**: `mined={ds_metrics['by_slice'].get('mined', 0)}`, `random_control={ds_metrics['by_slice'].get('random_control', 0)}`",
                f"- **By Split (Cluster-Disjoint)**: `dev={ds_metrics['by_split'].get('dev', 0)}`, `heldout={ds_metrics['by_split'].get('heldout', 0)}`",
                f"- **Coverage**: {ds_metrics['unique_intents']} intents, {ds_metrics['unique_clusters']} clusters",
                f"- **By Severity**: {ds_metrics['by_severity']}",
                f"- **Pool Approval Rate**: {_fmt_pct(ds_metrics['pool_approval_rate'])} ({ds_metrics['pool_total_cases']} total pool cases)",
            ]
        )

    lines.extend(
        [
            "",
            f"## 7. Model Comparison (`{baseline_model_id}` vs `{candidate_model_id}` on `{release_id}`)",
            "",
            "Mined and `random_control` slices are reported separately alongside overall metrics:",
            "",
            "| Scope | Metric | Baseline | Candidate | Paired Diff | 95% CI | McNemar p | Holm p_adj | Flagged |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    key_scopes = {"overall", "slice:mined", "slice:random_control", "split:dev", "split:heldout"}
    for rr in sorted(reg_rows, key=lambda x: (x.scope, x.metric)):
        if rr.scope in key_scopes and rr.metric in {
            "pass_rate",
            "correctness_pass_rate",
            "relevance_pass_rate",
            "hallucination_rate",
            "failure_rate",
            "critical_failure_rate",
        }:
            ci_str = f"[{_fmt_num(rr.ci_low)}, {_fmt_num(rr.ci_high)}]" if rr.ci_low is not None else "N/A"
            lines.append(
                f"| `{rr.scope}` | `{rr.metric}` | {_fmt_pct(rr.baseline_value)} | {_fmt_pct(rr.candidate_value)} | "
                f"{_fmt_num(rr.diff)} | {ci_str} | {_fmt_num(rr.p_value, 6)} | {_fmt_num(rr.p_adjusted, 6)} | `{rr.flagged}` |"
            )

    flagged_total = sum(1 for r in reg_rows if r.flagged)
    insufficient_total = sum(1 for r in reg_rows if r.baseline_value is None)
    lines.extend(
        [
            "",
            f"- **Total Scope × Metric Comparisons**: {len(reg_rows)}",
            f"- **Categories Marked Insufficient (`n < {config.evaluation.min_category_size}`)**: {insufficient_total}",
            f"- **Flagged Regressions**: {flagged_total}",
            "",
        ]
    )

    out_file.write_text("\n".join(lines), encoding="utf-8")
    return out_file
