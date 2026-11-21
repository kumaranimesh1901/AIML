"""Statistical tests for model comparison: paired bootstrap CIs, McNemar exact test, Holm correction."""

from __future__ import annotations

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


def bootstrap_rate_ci(
    values: list[float] | np.ndarray,
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute sample mean and percentile bootstrap (1 - alpha) CI."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean_val = float(np.mean(arr))
    if n == 1:
        return round(mean_val, 4), round(mean_val, 4), round(mean_val, 4)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = np.mean(arr[idx], axis=1)
    lo = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return round(mean_val, 4), round(lo, 4), round(hi, 4)


def paired_bootstrap_ci(
    baseline: list[float] | np.ndarray,
    candidate: list[float] | np.ndarray,
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute paired difference `mean(candidate - baseline)` and percentile bootstrap (1 - alpha) CI."""
    base_arr = np.asarray(baseline, dtype=np.float64)
    cand_arr = np.asarray(candidate, dtype=np.float64)
    if len(base_arr) != len(cand_arr):
        raise ValueError("Baseline and candidate arrays must have identical length for paired bootstrap.")
    diffs = cand_arr - base_arr
    return bootstrap_rate_ci(diffs, n_resamples=n_resamples, alpha=alpha, seed=seed)


def mcnemar_exact_pvalue(
    baseline_pass: list[bool],
    candidate_pass: list[bool],
) -> float:
    """Compute exact two-sided McNemar test p-value for paired binary pass/fail outcomes."""
    if len(baseline_pass) != len(candidate_pass):
        raise ValueError("Baseline and candidate pass lists must have identical length.")
    if not baseline_pass:
        return 1.0

    # Contingency table:
    # [[both_pass, base_pass_cand_fail], [base_fail_cand_pass, both_fail]]
    a = sum(1 for b, c in zip(baseline_pass, candidate_pass, strict=True) if b and c)
    b = sum(1 for b, c in zip(baseline_pass, candidate_pass, strict=True) if b and (not c))
    c = sum(1 for b, c in zip(baseline_pass, candidate_pass, strict=True) if (not b) and c)
    d = sum(1 for b, c in zip(baseline_pass, candidate_pass, strict=True) if (not b) and (not c))

    if b + c == 0:
        return 1.0

    res = mcnemar([[a, b], [c, d]], exact=True)
    return float(res.pvalue)


def holm_bonferroni_adjust(p_values: list[float | None]) -> list[float | None]:
    """Apply Holm-Bonferroni multiple-testing correction, preserving None entries."""
    valid_indices = [i for i, p in enumerate(p_values) if p is not None]
    if not valid_indices:
        return [None for _ in p_values]

    valid_pvals = [float(p_values[i]) for i in valid_indices]  # type: ignore[arg-type]
    _reject, adjusted, _sidak, _bonf = multipletests(valid_pvals, method="holm")

    out: list[float | None] = [None] * len(p_values)
    for pos, orig_idx in enumerate(valid_indices):
        out[orig_idx] = round(float(adjusted[pos]), 6)
    return out
