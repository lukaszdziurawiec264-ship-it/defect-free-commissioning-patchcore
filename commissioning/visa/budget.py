"""Section 10: a fixed budget B of good parts from U, split into a random bank and a calibration set."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from commissioning.calibration import _meets, order_statistic_rank
from commissioning.exploratory import pac_rank
from commissioning.io_utils import atomic_parquet
from commissioning.patchcore import scores_along_order
from commissioning.visa.caches import load_distances
from commissioning.visa.config import VisaConfig
from commissioning.visa.data import load_split, row_blocks

LOGGER = logging.getLogger(__name__)

BUDGET_RULES = ("conformal", "pac")


def part_path(cfg: VisaConfig, class_name: str) -> Path:
    return cfg.work_root / "budget" / f"{class_name}.parquet"


def rule_rank(cfg: VisaConfig, rule: str, m: int) -> tuple[int, bool]:
    if rule == "conformal":
        return order_statistic_rank(m, cfg.alpha), True
    k = pac_rank(m, cfg.pac_epsilon, cfg.pac_delta)
    return (m, False) if k is None else (k, True)


def budget_class(cfg: VisaConfig, class_name: str, distance: np.ndarray, split: dict) -> pd.DataFrame:
    blocks = row_blocks(split)
    universe, holdout = blocks["universe"], blocks["holdout"]
    good, defect = blocks["test_good"], blocks["test_defect"]
    n_good, n_defect, n_h = len(good), len(defect), len(holdout)
    grid = [n for n in cfg.budget_grid if n <= max(cfg.budgets) - cfg.budget_min_calibration]
    rows = []
    for rep in range(cfg.n_repeats):
        sequence = np.random.SeedSequence([cfg.seed, cfg.seed_key(class_name), rep, cfg.budget_seed_tag])
        permutation = universe[np.random.default_rng(sequence).permutation(len(universe))]
        # The first bank image is never a query: its only bank member would be itself (+inf).
        queries = np.concatenate([permutation[1:], holdout, good, defect])
        stacked = scores_along_order(distance, queries, permutation[: grid[-1]], grid)
        n_u = len(permutation) - 1
        for row_index, n in enumerate(grid):
            scores = stacked[row_index]
            h_scores = scores[n_u : n_u + n_h]
            good_scores = scores[n_u + n_h : n_u + n_h + n_good]
            defect_scores = scores[n_u + n_h + n_good :]
            for budget in cfg.budgets:
                m = budget - n
                if m < cfg.budget_min_calibration:
                    continue
                ordered = np.sort(scores[n - 1 : budget - 1])
                for rule in BUDGET_RULES:
                    k, feasible = rule_rank(cfg, rule, m)
                    threshold = float(ordered[k - 1])
                    n_tp = int(np.sum(defect_scores > threshold))
                    n_fp_off = int(np.sum(good_scores > threshold))
                    n_fp_h = int(np.sum(h_scores > threshold))
                    tpr_ok = _meets(n_tp, n_defect, cfg.practical_tpr, at_least=True)
                    rows.append(
                        {
                            "class_name": class_name,
                            "group": cfg.group_of(class_name),
                            "rep": rep,
                            "budget": budget,
                            "N": n,
                            "n_calibration": m,
                            "rule": rule,
                            "rank": k,
                            "rule_feasible": feasible,
                            "tpr": n_tp / n_defect,
                            "fpr_off": n_fp_off / n_good,
                            "fpr_h": n_fp_h / n_h,
                            "n_tp": n_tp,
                            "n_fp_off": n_fp_off,
                            "n_fp_h": n_fp_h,
                            "criterion_off": bool(tpr_ok and _meets(n_fp_off, n_good, cfg.practical_fpr, at_least=False)),
                            "criterion_h": bool(tpr_ok and _meets(n_fp_h, n_h, cfg.practical_fpr, at_least=False)),
                            "config_hash": cfg.hash,
                        }
                    )
    return pd.DataFrame(rows)


def run_budget(cfg: VisaConfig, classes: list[str]) -> pd.DataFrame:
    frames = []
    for class_name in classes:
        path = part_path(cfg, class_name)
        if path.is_file():
            frame = pd.read_parquet(path)
            if set(frame["config_hash"].astype(str)) != {cfg.hash}:
                raise RuntimeError(f"Refusing to reuse a budget part from another config: {path}")
            frames.append(frame)
            continue
        LOGGER.info("Budget split %s", class_name)
        split = load_split(cfg, class_name, 0)
        distance = load_distances(cfg, class_name)["D"]
        frame = budget_class(cfg, class_name, distance, split)
        del distance
        atomic_parquet(path, frame)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize_budget(frame: pd.DataFrame, min_passes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["class_name", "group", "budget", "N", "n_calibration", "rule"]
    summary = (
        frame.groupby(keys, sort=True)
        .agg(
            passing_off=("criterion_off", "sum"),
            passing_h=("criterion_h", "sum"),
            median_tpr=("tpr", "median"),
            mean_fpr_off=("fpr_off", "mean"),
            mean_fpr_h=("fpr_h", "mean"),
            rule_feasible=("rule_feasible", "first"),
        )
        .reset_index()
    )
    best = []
    for source in ("off", "h"):
        column = f"passing_{source}"
        for (class_name, group, budget, rule), sub in summary.groupby(["class_name", "group", "budget", "rule"]):
            top = sub.sort_values([column, "median_tpr", "N"], ascending=[False, False, True]).iloc[0]
            best.append(
                {
                    "class_name": class_name,
                    "group": group,
                    "budget": int(budget),
                    "rule": rule,
                    "fpr_source": source,
                    "best_N": int(top["N"]),
                    "best_n_calibration": int(top["n_calibration"]),
                    "passing_reps": int(top[column]),
                    "meets_min_reps": bool(top[column] >= min_passes),
                    "median_tpr": float(top["median_tpr"]),
                }
            )
    return summary, pd.DataFrame(best)


def budget_hypothesis(best: pd.DataFrame) -> dict:
    """Section 10: for B >= 120, does the best split reserve at least 40 images for calibration in most classes?"""
    out = {}
    for (rule, source), sub in best[best["budget"] >= 120].groupby(["rule", "fpr_source"]):
        per_budget = {
            str(int(b)): float(np.mean(part["best_n_calibration"] >= 40)) for b, part in sub.groupby("budget")
        }
        out[f"{rule}_{source}"] = {
            "fraction_of_class_budget_pairs_with_calibration_ge_40": float(np.mean(sub["best_n_calibration"] >= 40)),
            "per_budget": per_budget,
        }
    return out
