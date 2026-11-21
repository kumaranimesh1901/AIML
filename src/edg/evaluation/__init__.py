"""Benchmark replay, frozen-judge & checklist grading, statistics, and regression detection."""

from edg.evaluation.grade import compute_case_passed, compute_judge_flip_rate, evaluate_checklist
from edg.evaluation.regression import compare_models_on_release, is_regression_flagged
from edg.evaluation.replay import replay_and_grade_release
from edg.evaluation.stats import (
    bootstrap_rate_ci,
    holm_bonferroni_adjust,
    mcnemar_exact_pvalue,
    paired_bootstrap_ci,
)

__all__ = [
    "bootstrap_rate_ci",
    "compare_models_on_release",
    "compute_case_passed",
    "compute_judge_flip_rate",
    "evaluate_checklist",
    "holm_bonferroni_adjust",
    "is_regression_flagged",
    "mcnemar_exact_pvalue",
    "paired_bootstrap_ci",
    "replay_and_grade_release",
]
