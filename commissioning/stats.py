"""Primary paired comparisons: exact Wilcoxon, Holm, sign-flip, and a class-level bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def holm(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind="mergesort")
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * float(p[index])))
        adjusted[index] = running
    return adjusted.tolist()


def paired_difference(summary: pd.DataFrame, metric: str) -> pd.Series:
    wide = summary.pivot(index="class_name", columns="strategy", values=metric)
    if "kcenter" not in wide.columns or "random" not in wide.columns:
        raise RuntimeError(f"Both strategies are required to compare {metric}.")
    difference = (wide["kcenter"] - wide["random"]).astype(float)
    return difference.sort_index()


def wilcoxon_pvalue(difference: np.ndarray) -> float:
    values = np.asarray(difference, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or np.allclose(values, 0.0):
        return 1.0
    try:
        result = wilcoxon(values, zero_method="wilcox", alternative="two-sided", method="exact")
    except ValueError:
        result = wilcoxon(values, zero_method="wilcox", alternative="two-sided", method="auto")
    return float(result.pvalue)


def signflip_pvalue(difference: np.ndarray) -> float:
    """Exact two-sided paired permutation test of the mean difference. 2^n sign flips."""
    values = np.asarray(difference, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise RuntimeError("Sign-flip expects a finite 1-D vector of paired differences.")
    if values.size > 20:
        raise RuntimeError("Exact sign-flip is limited to 20 or fewer classes.")
    observed = abs(float(np.mean(values)))
    n = int(values.size)
    total = 1 << n
    count = 0
    chunk = 4096
    for start in range(0, total, chunk):
        masks = np.arange(start, min(start + chunk, total), dtype=np.uint32)
        signs = np.ones((masks.size, n), dtype=np.float64)
        for bit in range(n):
            flip = ((masks >> np.uint32(bit)) & np.uint32(1)).astype(bool)
            signs[flip, bit] = -1.0
        means = np.abs(signs @ values / n)
        count += int(np.sum(means + 1e-12 >= observed))
    return count / total


def bootstrap_median_interval(difference: np.ndarray, n_bootstrap: int, seed: int) -> dict:
    values = np.asarray(difference, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(int(n_bootstrap), values.size))
    medians = np.median(values[draws], axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return {
        "median": float(np.median(values)),
        "ci95": [float(low), float(high)],
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }


def _direction_counts(difference: pd.Series, lower_is_better: bool) -> dict:
    values = difference.to_numpy(dtype=float)
    better = values < 0 if lower_is_better else values > 0
    worse = values > 0 if lower_is_better else values < 0
    return {
        "n_kcenter_better": int(np.sum(better)),
        "n_random_better": int(np.sum(worse)),
        "n_ties": int(np.sum(np.isclose(values, 0.0))),
        "per_class": {str(name): float(value) for name, value in difference.items()},
    }


def compare_strategies(summary: pd.DataFrame, margin: float, n_bootstrap: int, seed: int) -> dict:
    tpr = paired_difference(summary, "aulc_tpr")
    fpr = paired_difference(summary, "aulc_fpr")
    p_tpr = wilcoxon_pvalue(tpr.to_numpy())
    p_fpr = wilcoxon_pvalue(fpr.to_numpy())
    adjusted = holm([p_tpr, p_fpr])
    tpr_boot = bootstrap_median_interval(tpr.to_numpy(), n_bootstrap, seed)
    fpr_boot = bootstrap_median_interval(fpr.to_numpy(), n_bootstrap, seed + 1)
    practical = bool(tpr_boot["median"] > 0.0 and fpr_boot["median"] <= float(margin))
    return {
        "difference": "kcenter - random",
        "practical_criterion_passed": practical,
        "fpr_margin": float(margin),
        "aulc_tpr": {
            "wilcoxon_p": p_tpr,
            "holm_p": adjusted[0],
            "permutation_p": signflip_pvalue(tpr.to_numpy()),
            "bootstrap": tpr_boot,
            **_direction_counts(tpr, lower_is_better=False),
        },
        "aulc_fpr": {
            "wilcoxon_p": p_fpr,
            "holm_p": adjusted[1],
            "permutation_p": signflip_pvalue(fpr.to_numpy()),
            "bootstrap": fpr_boot,
            **_direction_counts(fpr, lower_is_better=True),
        },
    }
