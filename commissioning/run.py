"""Run the 15 x 20 x 2 nested trajectories and resume from finished parts."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from commissioning.calibration import evaluate_scores
from commissioning.config import Config
from commissioning.data import load_split, split_seeds
from commissioning.distances import load_distance_cache
from commissioning.e0 import gate_path
from commissioning.features import load_gap_table
from commissioning.io_utils import atomic_parquet, atomic_savez, read_json
from commissioning.patchcore import scores_along_order
from commissioning.selection import grid_for_pool, kcenter_order, random_order

LOGGER = logging.getLogger(__name__)


def trajectory_paths(cfg: Config, class_name: str, rep: int, strategy: str) -> tuple:
    stem = f"{class_name}_rep{rep:02d}_{strategy}"
    table = cfg.work_root / "runs" / "parts" / f"{stem}.parquet"
    scores = cfg.work_root / "runs" / "scores" / f"{stem}.npz"
    return table, scores


def gate_is_exploratory(cfg: Config, classes: list[str], exploratory: bool) -> bool:
    path = gate_path(cfg)
    if not path.is_file():
        if exploratory:
            LOGGER.warning("E0 gate is missing. This run is exploratory and cannot be reported as confirmatory.")
            return True
        raise RuntimeError("E0 gate is missing. RQ trajectories stay blocked until e0/gate.json has status PASS.")
    gate = read_json(path)
    if gate.get("config_hash") != cfg.hash:
        raise RuntimeError("E0 gate was written for a different config hash.")
    covered = set(gate.get("classes", []))
    missing = [name for name in classes if name not in covered]
    passed = gate.get("status") == "PASS" and gate.get("part") == "all" and not missing
    if passed:
        return False
    if exploratory:
        LOGGER.warning("E0 gate does not cover this run (%s). Marking it exploratory.", gate.get("status"))
        return True
    if missing:
        raise RuntimeError(f"E0 gate does not cover: {missing}. Run e0 on these classes before RQ.")
    raise RuntimeError(f"E0 gate status is {gate.get('status')}. RQ trajectories stay blocked.")


def _finished(cfg: Config, table_path, scores_path) -> bool:
    if not table_path.is_file() or not scores_path.is_file():
        return False
    frame = pd.read_parquet(table_path)
    hashes = set(frame["config_hash"].astype(str))
    if hashes != {cfg.hash}:
        raise RuntimeError(f"Refusing to reuse a trajectory from another config: {table_path}")
    return True


def _indices(paths: list[str], selected: list[str], class_name: str) -> list[int]:
    index = {path: position for position, path in enumerate(paths)}
    try:
        return [index[path] for path in selected]
    except KeyError as exc:
        raise RuntimeError(f"{class_name}: split path is not in the feature cache: {exc}") from exc


def run_one(
    cfg: Config,
    class_name: str,
    rep: int,
    strategy: str,
    exploratory: bool,
    paths: list[str],
    distance: np.ndarray,
    gap: np.ndarray,
) -> None:
    table_path, scores_path = trajectory_paths(cfg, class_name, rep, strategy)
    if _finished(cfg, table_path, scores_path):
        LOGGER.info("Skipping finished %s rep %02d %s", class_name, rep, strategy)
        return
    split = load_split(cfg, class_name, rep)
    n_train = int(split["n_train"])
    calibration = _indices(paths, split["calibration"], class_name)
    pool = _indices(paths, split["pool"], class_name)
    test_paths = list(split["test_good"]) + [item["path"] for item in split["test_defect"]]
    test = _indices(paths, test_paths, class_name)
    if set(calibration) & set(pool) or max(calibration + pool) >= n_train or min(test) < n_train:
        raise RuntimeError(f"{class_name}: C/P/test indices are not a valid partition.")
    _split_rng, random_rng, kcenter_rng = split_seeds(cfg, class_name, rep)
    if strategy == "random":
        order = random_order(pool, random_rng)
    elif strategy == "kcenter":
        order = kcenter_order(gap, pool, kcenter_rng)
    else:
        raise RuntimeError(f"Unknown strategy: {strategy}")
    grid = grid_for_pool(len(pool), cfg.n_grid)
    queries = np.asarray(calibration + test, dtype=int)
    stacked = scores_along_order(distance, queries, order, grid)
    calibration_scores = stacked[:, : len(calibration)]
    test_scores = stacked[:, len(calibration) :]
    labels = np.array([0] * len(split["test_good"]) + [1] * len(split["test_defect"]), dtype=int)
    rows = []
    for row_index, n in enumerate(grid):
        metrics = evaluate_scores(
            test_scores[row_index],
            labels,
            calibration_scores[row_index],
            cfg.alpha,
            cfg.practical_tpr,
            cfg.practical_fpr,
        )
        rows.append(
            {
                "class_name": class_name,
                "group": cfg.group_of(class_name),
                "rep": int(rep),
                "strategy": strategy,
                "N": int(n),
                "pool_size": int(len(pool)),
                "config_hash": cfg.hash,
                "exploratory": bool(exploratory),
                **metrics,
            }
        )
    atomic_parquet(table_path, pd.DataFrame(rows))
    atomic_savez(
        scores_path,
        grid=np.asarray(grid, dtype=np.int32),
        scores_test=test_scores.astype(np.float32),
        scores_calibration=calibration_scores.astype(np.float32),
        test_labels=labels.astype(np.int8),
        order=order.astype(np.int32),
        test_paths=np.asarray(test_paths),
        calibration_paths=np.asarray(split["calibration"]),
    )
    LOGGER.info("Finished %s rep %02d %s (%d grid points)", class_name, rep, strategy, len(grid))


def run(cfg: Config, classes: list[str], exploratory: bool = False) -> None:
    mark = gate_is_exploratory(cfg, classes, exploratory)
    total = len(classes) * cfg.n_repeats * len(cfg.strategies)
    done = 0
    for class_name in classes:
        distance_cache = load_distance_cache(cfg, class_name)
        gap_table = load_gap_table(cfg, class_name)
        if gap_table["paths"] != list(distance_cache["meta"]["paths"]):
            raise RuntimeError(f"{class_name}: feature and distance caches are not aligned.")
        for rep in range(cfg.n_repeats):
            for strategy in cfg.strategies:
                run_one(
                    cfg,
                    class_name,
                    rep,
                    strategy,
                    mark,
                    gap_table["paths"],
                    distance_cache["D"],
                    gap_table["gap"],
                )
                done += 1
                LOGGER.info("Progress %d/%d", done, total)
