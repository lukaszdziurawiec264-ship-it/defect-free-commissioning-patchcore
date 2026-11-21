"""12 x 20 x 2 nested trajectories. Each grid point is evaluated under R20 and R60 on the same bank."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from commissioning.calibration import (
    _meets,
    clopper_pearson_lower,
    clopper_pearson_upper,
    false_reject_budget,
    oracle_fpr_at_tpr,
)
from commissioning.io_utils import atomic_parquet, atomic_savez, read_json
from commissioning.patchcore import scores_along_order
from commissioning.selection import kcenter_order, random_order
from commissioning.visa.caches import load_distances, load_gap
from commissioning.visa.config import VisaConfig
from commissioning.visa.data import image_order, load_split, split_seeds
from commissioning.visa.e0 import gate_is_exploratory

LOGGER = logging.getLogger(__name__)


def trajectory_paths(cfg: VisaConfig, class_name: str, rep: int, strategy: str) -> tuple[Path, Path]:
    stem = f"{class_name}_rep{rep:02d}_{strategy}"
    return cfg.work_root / "runs" / "parts" / f"{stem}.parquet", cfg.work_root / "runs" / "scores" / f"{stem}.npz"


def rule_metrics(
    threshold: float,
    defect_scores: np.ndarray,
    good_scores: np.ndarray,
    holdout_scores: np.ndarray,
    practical_tpr: float,
    practical_fpr: float,
) -> dict:
    """Reject when the score is strictly above the threshold."""
    n_tp = int(np.sum(defect_scores > threshold))
    n_fp_off = int(np.sum(good_scores > threshold))
    n_fp_h = int(np.sum(holdout_scores > threshold))
    n_defect, n_good, n_h = len(defect_scores), len(good_scores), len(holdout_scores)
    tpr_ok = _meets(n_tp, n_defect, practical_tpr, at_least=True)
    return {
        "threshold": float(threshold),
        "tpr": n_tp / n_defect,
        "tpr_lower": clopper_pearson_lower(n_tp, n_defect),
        "fpr_off": n_fp_off / n_good,
        "fpr_off_upper": clopper_pearson_upper(n_fp_off, n_good),
        "fpr_h": n_fp_h / n_h,
        "fpr_h_upper": clopper_pearson_upper(n_fp_h, n_h),
        "n_tp": n_tp,
        "n_fp_off": n_fp_off,
        "n_fp_h": n_fp_h,
        "n_test_defect": n_defect,
        "n_test_good": n_good,
        "n_holdout": n_h,
        "budget_off": false_reject_budget(n_good),
        "budget_h": false_reject_budget(n_h),
        "criterion_off": bool(tpr_ok and _meets(n_fp_off, n_good, practical_fpr, at_least=False)),
        "criterion_h": bool(tpr_ok and _meets(n_fp_h, n_h, practical_fpr, at_least=False)),
    }


def evaluate_grid_point(
    cfg: VisaConfig,
    calibration_scores: np.ndarray,
    holdout_scores: np.ndarray,
    test_scores: np.ndarray,
    labels: np.ndarray,
) -> list[dict]:
    values = [calibration_scores, holdout_scores, test_scores]
    if not all(np.isfinite(v).all() for v in values):
        raise RuntimeError("Non-finite scores; refusing to replace them with a sentinel value.")
    if len(calibration_scores) != cfg.n_calibration_large:
        raise RuntimeError(f"Expected {cfg.n_calibration_large} calibration scores, got {len(calibration_scores)}")
    small = calibration_scores[: cfg.n_calibration_small]
    shared = {
        "auroc": float(roc_auc_score(labels, test_scores)),
        "oracle_fpr": oracle_fpr_at_tpr(test_scores, labels, cfg.practical_tpr),
        "median_calibration_c20": float(np.median(small)),
        "median_calibration_c60": float(np.median(calibration_scores)),
    }
    rows = []
    for rule, threshold in (("R20", float(np.max(small))), ("R60", float(np.max(calibration_scores)))):
        metrics = rule_metrics(
            threshold,
            test_scores[labels == 1],
            test_scores[labels == 0],
            holdout_scores,
            cfg.practical_tpr,
            cfg.practical_fpr,
        )
        rows.append({"rule": rule, **shared, **metrics})
    return rows


def run_one(
    cfg: VisaConfig,
    class_name: str,
    rep: int,
    strategy: str,
    exploratory: bool,
    paths: list[str],
    distance: np.ndarray,
    gap: np.ndarray,
) -> None:
    table_path, scores_path = trajectory_paths(cfg, class_name, rep, strategy)
    if table_path.is_file() and scores_path.is_file():
        if set(pd.read_parquet(table_path)["config_hash"].astype(str)) != {cfg.hash}:
            raise RuntimeError(f"Refusing to reuse a trajectory from another config: {table_path}")
        return
    split = load_split(cfg, class_name, rep)
    if paths != image_order(split):
        raise RuntimeError(f"{class_name} rep {rep}: cache rows do not match the split.")
    index = {path: position for position, path in enumerate(paths)}
    calibration = [index[p] for p in split["calibration"]]
    pool = [index[p] for p in split["pool"]]
    holdout = [index[p] for p in split["holdout"]]
    test = [index[p] for p in list(split["test_good"]) + list(split["test_defect"])]
    n_universe = len(split["universe"])
    if max(calibration + pool) >= n_universe or min(holdout + test) < n_universe:
        raise RuntimeError(f"{class_name}: C and P must be universe columns; H and test must not.")
    _split_rng, random_rng, kcenter_rng = split_seeds(cfg, class_name, rep)
    if strategy == "random":
        order = random_order(pool, random_rng)
    elif strategy == "kcenter":
        order = kcenter_order(gap, pool, kcenter_rng)
    else:
        raise RuntimeError(f"Unknown strategy: {strategy}")
    grid = cfg.n_grid
    if grid[-1] != len(pool):
        raise RuntimeError(f"{class_name}: the grid must end at |P| = {len(pool)}")
    queries = np.asarray(calibration + holdout + test, dtype=int)
    stacked = scores_along_order(distance, queries, order, grid)
    n_c, n_h = len(calibration), len(holdout)
    labels = np.array([0] * len(split["test_good"]) + [1] * len(split["test_defect"]), dtype=int)
    rows = []
    for row_index, n in enumerate(grid):
        scores = stacked[row_index]
        for record in evaluate_grid_point(cfg, scores[:n_c], scores[n_c : n_c + n_h], scores[n_c + n_h :], labels):
            rows.append(
                {
                    "class_name": class_name,
                    "group": cfg.group_of(class_name),
                    "rep": int(rep),
                    "strategy": strategy,
                    "N": int(n),
                    "pool_size": len(pool),
                    "config_hash": cfg.hash,
                    "exploratory": bool(exploratory),
                    **record,
                }
            )
    atomic_parquet(table_path, pd.DataFrame(rows))
    atomic_savez(
        scores_path,
        grid=np.asarray(grid, dtype=np.int32),
        scores_calibration=stacked[:, :n_c].astype(np.float32),
        scores_holdout=stacked[:, n_c : n_c + n_h].astype(np.float32),
        scores_test=stacked[:, n_c + n_h :].astype(np.float32),
        test_labels=labels.astype(np.int8),
        order=order.astype(np.int32),
        test_paths=np.asarray(list(split["test_good"]) + list(split["test_defect"])),
        calibration_paths=np.asarray(split["calibration"]),
    )
    LOGGER.info("Finished %s rep %02d %s", class_name, rep, strategy)


def run(cfg: VisaConfig, classes: list[str], exploratory: bool = False, force_exploratory: bool = False) -> None:
    mark = gate_is_exploratory(cfg, classes, exploratory) or force_exploratory
    for class_name in classes:
        pending = [
            (rep, strategy)
            for rep in range(cfg.n_repeats)
            for strategy in cfg.strategies
            if not all(p.is_file() for p in trajectory_paths(cfg, class_name, rep, strategy))
        ]
        if not pending:
            LOGGER.info("All trajectories exist for %s", class_name)
            continue
        distance = load_distances(cfg, class_name)
        gap = load_gap(cfg, class_name)
        if gap["paths"] != list(distance["meta"]["paths"]):
            raise RuntimeError(f"{class_name}: GAP table and distance cache are not aligned.")
        for rep, strategy in pending:
            run_one(cfg, class_name, rep, strategy, mark, gap["paths"], distance["D"], gap["gap"])
        del distance


def load_metrics(cfg: VisaConfig) -> pd.DataFrame:
    parts = sorted((cfg.work_root / "runs" / "parts").glob("*.parquet"))
    if not parts:
        raise FileNotFoundError("No trajectory tables in runs/parts.")
    metrics = pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)
    if set(metrics["config_hash"].astype(str)) != {cfg.hash}:
        raise RuntimeError("Trajectory tables mix config hashes.")
    return metrics


def read_gate(cfg: VisaConfig) -> dict | None:
    from commissioning.visa.e0 import gate_path

    path = gate_path(cfg)
    return read_json(path) if path.is_file() else None
