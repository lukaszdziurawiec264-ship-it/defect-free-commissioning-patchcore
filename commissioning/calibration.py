"""Threshold from good images only, and the metrics computed on the official test set."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score


def order_statistic_rank(n: int, alpha: float) -> int:
    """1-based rank k = ceil((n + 1) * (1 - alpha)), clipped to 1..n."""
    if n < 1:
        raise RuntimeError("Calibration set is empty.")
    if not 0.0 < alpha < 1.0:
        raise RuntimeError(f"alpha must be in (0, 1), got {alpha}")
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    return min(max(k, 1), n)


def threshold_from_scores(scores: np.ndarray, alpha: float) -> tuple[float, int]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise RuntimeError("Calibration scores must be a finite 1-D vector.")
    k = order_statistic_rank(len(values), alpha)
    ordered = np.sort(values)
    return float(ordered[k - 1]), k


def clopper_pearson_lower(successes: int, n: int, alpha: float = 0.05) -> float:
    """One-sided lower 100*(1-alpha)% Clopper-Pearson bound."""
    if n < 0 or successes < 0 or successes > n:
        raise RuntimeError(f"Invalid binomial counts k={successes}, n={n}")
    if n == 0:
        return float("nan")
    if successes == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, n - successes + 1))


def clopper_pearson_upper(successes: int, n: int, alpha: float = 0.05) -> float:
    """One-sided upper 100*(1-alpha)% Clopper-Pearson bound."""
    if n < 0 or successes < 0 or successes > n:
        raise RuntimeError(f"Invalid binomial counts k={successes}, n={n}")
    if n == 0:
        return float("nan")
    if successes == n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, successes + 1, n - successes))


def false_reject_budget(n_good: int) -> int:
    """Largest k with k / n_good <= 0.05. Integer arithmetic avoids a binary rounding edge."""
    if n_good < 0:
        raise RuntimeError(f"n_good must be non-negative, got {n_good}")
    return (int(n_good) * 5) // 100


def oracle_fpr_at_tpr(scores: np.ndarray, labels: np.ndarray, target_tpr: float = 0.95) -> float:
    """Lowest test-set FPR among thresholds with TPR >= target. Uses the labels; not operational."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if scores.shape != labels.shape or scores.ndim != 1:
        raise RuntimeError("Scores and labels must be 1-D and aligned.")
    if not np.isfinite(scores).all():
        raise RuntimeError("Non-finite test scores.")
    defect = scores[labels == 1]
    good = scores[labels == 0]
    if defect.size == 0 or good.size == 0:
        raise RuntimeError("Oracle FPR needs both good and defect test images.")
    candidates = np.unique(np.concatenate([scores, [scores.min() - 1.0]]))
    best = 1.0
    found = False
    for threshold in candidates:
        tpr = float(np.mean(defect > threshold))
        if tpr + 1e-12 >= target_tpr:
            fpr = float(np.mean(good > threshold))
            if not found or fpr < best:
                best = fpr
                found = True
    if not found:
        raise RuntimeError("No threshold reaches the target TPR.")
    return best


def _meets(count: int, total: int, level: float, at_least: bool) -> bool:
    """Compare count / total with a level in integer arithmetic, so 19/20 >= 0.95 and 3/60 <= 0.05 hold exactly."""
    from fractions import Fraction

    rate = Fraction(int(count), int(total))
    bound = Fraction(str(level))
    return bool(rate >= bound) if at_least else bool(rate <= bound)


def evaluate_scores(
    test_scores: np.ndarray,
    labels: np.ndarray,
    calibration_scores: np.ndarray,
    alpha: float,
    practical_tpr: float,
    practical_fpr: float,
) -> dict:
    """Decision rule: a test image is rejected when its score is strictly above max(C) (for alpha=0.05, n=20)."""
    test_scores = np.asarray(test_scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    calibration_scores = np.asarray(calibration_scores, dtype=np.float64)
    if not np.isfinite(test_scores).all() or not np.isfinite(calibration_scores).all():
        raise RuntimeError("Non-finite scores; refusing to replace them with a sentinel value.")
    threshold, rank = threshold_from_scores(calibration_scores, alpha)
    defect = test_scores[labels == 1]
    good = test_scores[labels == 0]
    if defect.size == 0 or good.size == 0:
        raise RuntimeError("The official test split must contain good and defect images.")
    n_tp = int(np.sum(defect > threshold))
    n_fp = int(np.sum(good > threshold))
    n_defect = int(defect.size)
    n_good = int(good.size)
    tpr = n_tp / n_defect
    fpr = n_fp / n_good
    return {
        "threshold": threshold,
        "threshold_rank": rank,
        "median_calibration": float(np.median(calibration_scores)),
        "tpr": float(tpr),
        "tpr_lower": clopper_pearson_lower(n_tp, n_defect),
        "fpr": float(fpr),
        "fpr_upper": clopper_pearson_upper(n_fp, n_good),
        "n_tp": n_tp,
        "n_fp": n_fp,
        "n_test_defect": n_defect,
        "n_test_good": n_good,
        "auroc": float(roc_auc_score(labels, test_scores)),
        "oracle_fpr": oracle_fpr_at_tpr(test_scores, labels, practical_tpr),
        "criterion": _meets(n_tp, n_defect, practical_tpr, at_least=True)
        and _meets(n_fp, n_good, practical_fpr, at_least=False),
    }
