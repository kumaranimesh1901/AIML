"""Paired model comparison and statistically rigorous regression detection."""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from edg.config import AppConfig
from edg.db.models import (
    BenchmarkRelease,
    EvalResult,
    Judgement,
    RegressionReport,
    ReleaseItem,
    StateSnapshot,
    TestCase,
)
from edg.evaluation.grade import compute_judge_flip_rate
from edg.evaluation.stats import (
    bootstrap_rate_ci,
    holm_bonferroni_adjust,
    mcnemar_exact_pvalue,
    paired_bootstrap_ci,
)
from edg.judge.schema import DIMENSIONS
from edg.labeling.severity import resolve_severity_for_interaction
from edg.logging import get_logger
from edg.world.taxonomy import TaxonomyConfig, load_taxonomy

logger = get_logger(__name__)

# Metrics where a DROP (candidate < baseline) is a regression vs error metrics where a RISE is a regression
_HIGHER_IS_BETTER = {
    "pass_rate",
    "correctness_pass_rate",
    "relevance_pass_rate",
    "helpfulness_pass_rate",
    "completeness_pass_rate",
    "safety_pass_rate",
}
_LOWER_IS_BETTER = {
    "hallucination_rate",
    "failure_rate",
    "critical_failure_rate",
}


def _extract_metric_flags(
    er: EvalResult,
    jdg: Judgement,
    tc: TestCase,
    hard_tags: list[str],
    taxonomy: TaxonomyConfig,
) -> dict[str, bool | None]:
    failures = list(jdg.failure_types or [])
    has_failure = len(failures) > 0 or any(
        getattr(jdg, f"{d}_verdict") == "fail" for d in DIMENSIONS
    )
    if has_failure:
        _area, sev = resolve_severity_for_interaction(
            failure_types=failures,
            intent=tc.intent,
            hard_state_tags=hard_tags,
            taxonomy=taxonomy,
        )
        is_critical = sev == "critical"
    else:
        is_critical = False

    out: dict[str, bool | None] = {
        "pass_rate": bool(er.passed),
        "hallucination_rate": "hallucination" in failures,
        "failure_rate": has_failure,
        "critical_failure_rate": is_critical,
    }
    for dim in DIMENSIONS:
        v = getattr(jdg, f"{dim}_verdict")
        out[f"{dim}_pass_rate"] = None if v == "not_assessable" else (v == "pass")
    return out


def is_regression_flagged(
    *,
    metric: str,
    diff: float,
    ci_low: float,
    ci_high: float,
    p_adjusted: float | None,
    alpha: float,
    judge_flip_rate: float,
) -> bool:
    """Flag a regression iff p_adjusted < alpha, CI excludes 0 in the worse direction, and |diff| > flip_rate."""
    if p_adjusted is None or p_adjusted >= alpha:
        return False
    if abs(diff) <= judge_flip_rate:
        return False
    if metric in _HIGHER_IS_BETTER:
        return diff < 0.0 and ci_high < 0.0
    if metric in _LOWER_IS_BETTER:
        return diff > 0.0 and ci_low > 0.0
    return False


def compare_models_on_release(
    session: Session,
    config: AppConfig,
    *,
    release_id: str,
    baseline_model_id: str,
    candidate_model_id: str,
    run_id: str,
    measure_flip_rate: bool = True,
    flip_rate_override: float | None = None,
) -> dict[str, Any]:
    """Compare baseline vs candidate models on a frozen release with paired bootstrap, McNemar, and Holm."""
    release = session.get(BenchmarkRelease, release_id)
    if release is None:
        raise ValueError(f"Release '{release_id}' not found.")

    taxonomy = load_taxonomy()

    def _load_model_map(model_id: str) -> dict[str, tuple[EvalResult, Judgement]]:
        rows = session.execute(
            select(EvalResult, Judgement)
            .join(Judgement, Judgement.judgement_id == EvalResult.judgement_id)
            .where(EvalResult.release_id == release_id, EvalResult.model_id == model_id)
        ).all()
        return {er.test_case_id: (er, jdg) for er, jdg in rows}

    base_map = _load_model_map(baseline_model_id)
    cand_map = _load_model_map(candidate_model_id)
    shared_ids = sorted(set(base_map.keys()) & set(cand_map.keys()))
    if not shared_ids:
        raise ValueError(
            f"No shared eval_results for release='{release_id}', "
            f"baseline='{baseline_model_id}', candidate='{candidate_model_id}'."
        )

    if flip_rate_override is not None:
        flip_info = {"n_sampled": 0, "flips": 0, "flip_rate": float(flip_rate_override)}
    elif measure_flip_rate:
        flip_info = compute_judge_flip_rate(
            session,
            config,
            release_id=release_id,
            model_id=candidate_model_id,
            run_id=run_id,
        )
    else:
        flip_info = {"n_sampled": 0, "flips": 0, "flip_rate": 0.0}
    flip_rate = float(flip_info["flip_rate"])

    # Load ReleaseItem + TestCase + Snapshot metadata for grouping into scopes
    meta_rows = session.execute(
        select(ReleaseItem, TestCase, StateSnapshot.hard_state_tags)
        .join(TestCase, TestCase.test_case_id == ReleaseItem.test_case_id)
        .join(StateSnapshot, StateSnapshot.snapshot_id == TestCase.snapshot_id)
        .where(
            ReleaseItem.release_id == release_id,
            ReleaseItem.test_case_id.in_(shared_ids),
        )
    ).all()

    scopes: dict[str, list[str]] = defaultdict(list)
    flags_base: dict[str, dict[str, bool | None]] = {}
    flags_cand: dict[str, dict[str, bool | None]] = {}

    for ri, tc, hard_tags in meta_rows:
        tid = tc.test_case_id
        ht = list(hard_tags or [])
        b_er, b_jdg = base_map[tid]
        c_er, c_jdg = cand_map[tid]
        flags_base[tid] = _extract_metric_flags(b_er, b_jdg, tc, ht, taxonomy)
        flags_cand[tid] = _extract_metric_flags(c_er, c_jdg, tc, ht, taxonomy)

        scopes["overall"].append(tid)
        scopes[f"slice:{ri.slice}"].append(tid)
        scopes[f"split:{ri.split}"].append(tid)
        scopes[f"intent:{tc.intent or 'unknown'}"].append(tid)
        scopes[f"cluster:{tc.cluster_id if tc.cluster_id is not None else -1}"].append(tid)
        scopes[f"severity:{tc.severity}"].append(tid)

    metrics_list = [
        "pass_rate",
        "correctness_pass_rate",
        "relevance_pass_rate",
        "helpfulness_pass_rate",
        "completeness_pass_rate",
        "safety_pass_rate",
        "hallucination_rate",
        "failure_rate",
        "critical_failure_rate",
    ]
    min_cat = config.evaluation.min_category_size
    n_resamples = config.evaluation.bootstrap_resamples
    alpha = config.evaluation.alpha

    all_entries: list[dict[str, Any]] = []
    # Apply Holm correction per metric across all category scopes
    for metric in metrics_list:
        metric_entries: list[dict[str, Any]] = []
        raw_pvals: list[float | None] = []

        for scope_name in sorted(scopes.keys()):
            tids = scopes[scope_name]
            is_category_scope = scope_name.startswith(("intent:", "cluster:", "severity:"))

            paired_b: list[bool] = []
            paired_c: list[bool] = []
            excluded = 0
            for tid in tids:
                bv = flags_base[tid][metric]
                cv = flags_cand[tid][metric]
                if bv is None or cv is None:
                    excluded += 1
                    continue
                paired_b.append(bool(bv))
                paired_c.append(bool(cv))

            n_assessed = len(paired_b)
            if is_category_scope and n_assessed < min_cat:
                metric_entries.append(
                    {
                        "scope": scope_name,
                        "metric": metric,
                        "n": n_assessed,
                        "excluded_not_assessable": excluded,
                        "status": "insufficient",
                        "baseline_value": None,
                        "baseline_ci": None,
                        "candidate_value": None,
                        "candidate_ci": None,
                        "diff": None,
                        "ci_low": None,
                        "ci_high": None,
                        "p_value": None,
                        "p_adjusted": None,
                        "flagged": False,
                    }
                )
                raw_pvals.append(None)
                continue

            b_arr = [1.0 if x else 0.0 for x in paired_b]
            c_arr = [1.0 if x else 0.0 for x in paired_c]
            b_mean, b_lo, b_hi = bootstrap_rate_ci(b_arr, n_resamples=n_resamples, alpha=alpha, seed=config.seed)
            c_mean, c_lo, c_hi = bootstrap_rate_ci(c_arr, n_resamples=n_resamples, alpha=alpha, seed=config.seed + 1)
            diff, ci_lo, ci_hi = paired_bootstrap_ci(
                b_arr, c_arr, n_resamples=n_resamples, alpha=alpha, seed=config.seed + 2
            )
            pval = round(mcnemar_exact_pvalue(paired_b, paired_c), 6)

            metric_entries.append(
                {
                    "scope": scope_name,
                    "metric": metric,
                    "n": n_assessed,
                    "excluded_not_assessable": excluded,
                    "status": "assessed",
                    "baseline_value": b_mean,
                    "baseline_ci": [b_lo, b_hi],
                    "candidate_value": c_mean,
                    "candidate_ci": [c_lo, c_hi],
                    "diff": diff,
                    "ci_low": ci_lo,
                    "ci_high": ci_hi,
                    "p_value": pval,
                    "p_adjusted": None,
                    "flagged": False,
                }
            )
            raw_pvals.append(pval)

        adjusted = holm_bonferroni_adjust(raw_pvals)
        for entry, p_adj in zip(metric_entries, adjusted, strict=True):
            entry["p_adjusted"] = p_adj
            if entry["status"] == "assessed" and entry["diff"] is not None:
                entry["flagged"] = is_regression_flagged(
                    metric=metric,
                    diff=entry["diff"],
                    ci_low=entry["ci_low"],
                    ci_high=entry["ci_high"],
                    p_adjusted=p_adj,
                    alpha=alpha,
                    judge_flip_rate=flip_rate,
                )

            session.add(
                RegressionReport(
                    report_id=f"REG-{uuid.uuid4().hex[:12]}",
                    release_id=release_id,
                    baseline_model_id=baseline_model_id,
                    candidate_model_id=candidate_model_id,
                    scope=entry["scope"],
                    metric=entry["metric"],
                    baseline_value=entry["baseline_value"],
                    candidate_value=entry["candidate_value"],
                    diff=entry["diff"],
                    ci_low=entry["ci_low"],
                    ci_high=entry["ci_high"],
                    p_value=entry["p_value"],
                    p_adjusted=entry["p_adjusted"],
                    flagged=entry["flagged"],
                    run_id=run_id,
                )
            )
        all_entries.extend(metric_entries)

    session.flush()
    flagged_entries = [e for e in all_entries if e["flagged"]]
    overall_pass = next(
        (e for e in all_entries if e["scope"] == "overall" and e["metric"] == "pass_rate"),
        None,
    )
    mined_pass = next(
        (e for e in all_entries if e["scope"] == "slice:mined" and e["metric"] == "pass_rate"),
        None,
    )
    random_pass = next(
        (e for e in all_entries if e["scope"] == "slice:random_control" and e["metric"] == "pass_rate"),
        None,
    )

    summary = {
        "release_id": release_id,
        "judge_version": release.judge_version,
        "baseline_model_id": baseline_model_id,
        "candidate_model_id": candidate_model_id,
        "config_hash": config.config_hash(),
        "run_id": run_id,
        "n_shared_cases": len(shared_ids),
        "judge_flip_rate": flip_info,
        "overall_pass_rate": overall_pass,
        "mined_slice_pass_rate": mined_pass,
        "random_control_slice_pass_rate": random_pass,
        "total_comparisons": len(all_entries),
        "flagged_regressions_count": len(flagged_entries),
        "flagged_regressions": flagged_entries,
    }
    logger.info(
        "regression_report_completed",
        extra={
            "release_id": release_id,
            "baseline": baseline_model_id,
            "candidate": candidate_model_id,
            "flagged_regressions_count": len(flagged_entries),
        },
    )
    return summary
