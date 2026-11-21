"""Judge calibration, agreement, controlled-tier recall, and reliability metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from edg.judge.schema import DIMENSIONS


def compute_ordinal_agreement(
    y_true: list[int | None],
    y_pred: list[int | None],
) -> dict[str, Any]:
    """Compute quadratic weighted Cohen's kappa, Spearman rho, and MAE, excluding None/not_assessable."""
    paired = [
        (int(a), int(b))
        for a, b in zip(y_true, y_pred, strict=True)
        if a is not None and b is not None
    ]
    excluded = len(y_true) - len(paired)
    if not paired:
        return {
            "n_assessed": 0,
            "excluded_not_assessable": excluded,
            "weighted_kappa": None,
            "spearman": None,
            "mae": None,
        }

    arr_true = np.asarray([p[0] for p in paired], dtype=np.float64)
    arr_pred = np.asarray([p[1] for p in paired], dtype=np.float64)
    mae = round(float(np.mean(np.abs(arr_true - arr_pred))), 4)

    if len(set(arr_true.tolist())) < 2 and len(set(arr_pred.tolist())) < 2:
        kappa: float | None = 1.0 if np.array_equal(arr_true, arr_pred) else 0.0
    else:
        raw_k = float(cohen_kappa_score(arr_true.astype(int), arr_pred.astype(int), weights="quadratic"))
        kappa = None if np.isnan(raw_k) else round(raw_k, 4)

    if len(paired) < 2 or np.std(arr_true) == 0 or np.std(arr_pred) == 0:
        rho: float | None = None
    else:
        raw_rho = float(spearmanr(arr_true, arr_pred).statistic)
        rho = None if np.isnan(raw_rho) else round(raw_rho, 4)

    return {
        "n_assessed": len(paired),
        "excluded_not_assessable": excluded,
        "weighted_kappa": kappa,
        "spearman": rho,
        "mae": mae,
    }


def compute_binary_classification_metrics(
    y_true: list[bool | None],
    y_pred: list[bool | None],
) -> dict[str, Any]:
    """Compute precision, recall, F1, and 2x2 confusion matrix, excluding None pairs."""
    paired = [
        (bool(a), bool(b))
        for a, b in zip(y_true, y_pred, strict=True)
        if a is not None and b is not None
    ]
    excluded = len(y_true) - len(paired)
    tp = sum(1 for a, b in paired if a and b)
    fp = sum(1 for a, b in paired if (not a) and b)
    fn = sum(1 for a, b in paired if a and (not b))
    tn = sum(1 for a, b in paired if (not a) and (not b))

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1 = (
        round(2 * precision * recall / (precision + recall), 4)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "n_assessed": len(paired),
        "excluded_not_assessable": excluded,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def verdict_to_pass_bool(verdict: str | None) -> bool | None:
    """Convert dimension verdict to binary pass/fail, returning None for not_assessable."""
    if verdict is None or verdict == "not_assessable":
        return None
    return verdict == "pass"


def compute_judge_agreement_report(
    paired_records: list[dict[str, Any]],
    *,
    failure_type_list: list[str],
) -> dict[str, Any]:
    """Compute full ordinal + categorical agreement between human annotations and judge outputs."""
    ordinal_by_dim: dict[str, Any] = {}
    categorical_by_dim: dict[str, Any] = {}

    for dim in DIMENSIONS:
        h_scores = [r.get(f"human_{dim}_score") for r in paired_records]
        j_scores = [r.get(f"judge_{dim}_score") for r in paired_records]
        ordinal_by_dim[dim] = compute_ordinal_agreement(h_scores, j_scores)

        h_pass = [verdict_to_pass_bool(r.get(f"human_{dim}_verdict")) for r in paired_records]
        j_pass = [verdict_to_pass_bool(r.get(f"judge_{dim}_verdict")) for r in paired_records]
        categorical_by_dim[dim] = compute_binary_classification_metrics(h_pass, j_pass)

    by_failure_type: dict[str, Any] = {}
    for ft in failure_type_list:
        h_has = [ft in (r.get("human_failure_types") or []) for r in paired_records]
        j_has = [ft in (r.get("judge_failure_types") or []) for r in paired_records]
        by_failure_type[ft] = compute_binary_classification_metrics(h_has, j_has)

    return {
        "n_paired": len(paired_records),
        "ordinal_by_dimension": ordinal_by_dim,
        "categorical_by_dimension": categorical_by_dim,
        "by_failure_type": by_failure_type,
    }


def compute_controlled_tier_recall(
    controlled_records: list[tuple[str, list[str]]],
) -> dict[str, Any]:
    """Compute per-injected-failure recall on the synthetic_controlled tier.

    Args:
        controlled_records: list of `(injected_failure, judged_failure_types)`.
    """
    totals: dict[str, int] = {}
    hits: dict[str, int] = {}
    for injected, predicted in controlled_records:
        totals[injected] = totals.get(injected, 0) + 1
        if injected in (predicted or []):
            hits[injected] = hits.get(injected, 0) + 1

    per_type: dict[str, dict[str, Any]] = {}
    for ft in sorted(totals.keys()):
        n = totals[ft]
        h = hits.get(ft, 0)
        per_type[ft] = {"n": n, "detected": h, "recall": round(h / n, 4) if n > 0 else 0.0}

    total_n = sum(totals.values())
    total_h = sum(hits.values())
    return {
        "total_controlled": total_n,
        "overall_recall": round(total_h / total_n, 4) if total_n > 0 else 0.0,
        "per_failure_type": per_type,
    }


def compute_judge_reliability(
    judgement_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute abstention rate, judge_error rate, and length-vs-score correlation (bias check)."""
    n = len(judgement_rows)
    if n == 0:
        return {
            "n_judgements": 0,
            "abstention_rate": 0.0,
            "judge_error_rate": 0.0,
            "length_score_spearman": None,
        }

    abstentions = sum(1 for r in judgement_rows if r.get("correctness_verdict") == "not_assessable")
    errors = sum(1 for r in judgement_rows if "judge_error" in (r.get("failure_types") or []))

    lengths: list[int] = []
    mean_scores: list[float] = []
    for r in judgement_rows:
        scores = [
            r.get(f"{dim}_score")
            for dim in DIMENSIONS
            if r.get(f"{dim}_score") is not None
        ]
        if scores and r.get("response_length") is not None:
            lengths.append(int(r["response_length"]))
            mean_scores.append(float(np.mean(scores)))

    if len(lengths) >= 3 and np.std(lengths) > 0 and np.std(mean_scores) > 0:
        raw_rho = float(spearmanr(lengths, mean_scores).statistic)
        corr: float | None = None if np.isnan(raw_rho) else round(raw_rho, 4)
    else:
        corr = None

    return {
        "n_judgements": n,
        "abstention_rate": round(abstentions / n, 4),
        "judge_error_rate": round(errors / n, 4),
        "length_score_spearman": corr,
    }
