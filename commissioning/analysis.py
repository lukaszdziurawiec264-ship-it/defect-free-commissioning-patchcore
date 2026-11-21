"""Aggregate the trajectory table into AULC, N*, FPR summaries, and stopping outcomes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import betabinom

from commissioning.calibration import clopper_pearson_upper, false_reject_budget
from commissioning.config import EXPECTED_COUNTS, Config
from commissioning.io_utils import atomic_parquet, write_json
from commissioning.stats import compare_strategies
from commissioning.stopping import n_stop

MAIN_GRID = (1, 2, 4, 8, 16, 32)


def aulc(ns, values, n_max: int = 32) -> float:
    """Trapezoid on log2(N), divided by log2(n_max) - log2(1). Points above n_max are ignored."""
    if int(n_max) != 32:
        raise RuntimeError("The main AULC normalization is fixed at log2(32) - log2(1).")
    points = {int(n): float(y) for n, y in zip(ns, values)}
    missing = [n for n in MAIN_GRID if n not in points]
    if missing:
        raise RuntimeError(f"AULC is missing main-grid points {missing}.")
    xs = np.log2(np.asarray(MAIN_GRID, dtype=float))
    ys = np.asarray([points[n] for n in MAIN_GRID], dtype=float)
    area = float(np.sum((ys[:-1] + ys[1:]) * 0.5 * np.diff(xs)))
    return area / float(np.log2(n_max) - np.log2(1.0))


def n_star(grid: list[int], n_passed: list[int], min_passes: int) -> dict:
    """Smallest N at which every grid point from there on has at least min_passes successes."""
    if len(grid) != len(n_passed):
        raise RuntimeError("Grid and pass counts must be aligned.")
    ok = [int(count) >= int(min_passes) for count in n_passed]
    for index, n in enumerate(grid):
        if all(ok[index:]):
            return {"n_star": int(n), "censored": False}
    return {"n_star": None, "censored": True}


def stop_category(n_stop_value, n_star_value, censored: bool) -> str:
    if censored:
        return "n_star_not_reached"
    if n_stop_value is None or (isinstance(n_stop_value, float) and np.isnan(n_stop_value)):
        return "rule_not_fired"
    if int(n_stop_value) < int(n_star_value):
        return "premature"
    if int(n_stop_value) == int(n_star_value):
        return "concordant"
    return "excessive"


def load_metrics(cfg: Config) -> pd.DataFrame:
    parts = sorted((cfg.work_root / "runs" / "parts").glob("*.parquet"))
    if not parts:
        raise FileNotFoundError("No trajectory tables in runs/parts. Run the trajectories first.")
    frames = [pd.read_parquet(path) for path in parts]
    metrics = pd.concat(frames, ignore_index=True)
    hashes = set(metrics["config_hash"].astype(str))
    if hashes != {cfg.hash}:
        raise RuntimeError(f"Trajectory tables mix config hashes: {hashes}")
    return metrics


def summarize_classes(metrics: pd.DataFrame, n_max: int, min_passes: int) -> pd.DataFrame:
    rows = []
    grouped = metrics.groupby(["class_name", "group", "strategy"], sort=True)
    for (class_name, group, strategy), sub in grouped:
        tpr_values = []
        fpr_values = []
        for _rep, rep_df in sub.groupby("rep"):
            ordered = rep_df.sort_values("N")
            tpr_values.append(aulc(ordered["N"], ordered["tpr"], n_max))
            fpr_values.append(aulc(ordered["N"], ordered["fpr"], n_max))
        grid = [int(n) for n in sorted(sub["N"].unique())]
        passed = [int(sub.loc[sub["N"] == n, "criterion"].sum()) for n in grid]
        star = n_star(grid, passed, min_passes)
        pool_size = int(sub["pool_size"].iloc[0])
        rows.append(
            {
                "class_name": class_name,
                "group": group,
                "strategy": strategy,
                "aulc_tpr": float(np.mean(tpr_values)),
                "aulc_fpr": float(np.mean(fpr_values)),
                "aulc_tpr_iqr": float(np.subtract(*np.quantile(tpr_values, [0.75, 0.25]))),
                "aulc_fpr_iqr": float(np.subtract(*np.quantile(fpr_values, [0.75, 0.25]))),
                "aulc_tpr_range": float(np.max(tpr_values) - np.min(tpr_values)),
                "aulc_fpr_range": float(np.max(fpr_values) - np.min(fpr_values)),
                "n_star": (float(star["n_star"]) if not star["censored"] else float("nan")),
                "censored": bool(star["censored"]),
                "pool_size": pool_size,
                "n_reps": int(sub["rep"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def fpr_aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    """Spread across repetitions. Repetitions share one test set, so their counts are not pooled."""
    rows = []
    for (class_name, strategy, n), sub in metrics.groupby(["class_name", "strategy", "N"], sort=True):
        fpr = sub["fpr"].to_numpy(dtype=float)
        q1, q3 = np.quantile(fpr, [0.25, 0.75])
        rows.append(
            {
                "class_name": class_name,
                "group": sub["group"].iloc[0],
                "strategy": strategy,
                "N": int(n),
                "mean_fpr": float(np.mean(fpr)),
                "median_fpr": float(np.median(fpr)),
                "iqr_fpr": float(q3 - q1),
                "min_fpr": float(np.min(fpr)),
                "max_fpr": float(np.max(fpr)),
                "fraction_fpr_above_05": float(np.mean(fpr > 0.05)),
                "fraction_meeting_criterion": float(np.mean(sub["criterion"].astype(bool))),
                "n_reps": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def stopping_table(metrics: pd.DataFrame, deltas: list[float]) -> pd.DataFrame:
    rows = []
    for (class_name, strategy, rep), sub in metrics.groupby(["class_name", "strategy", "rep"], sort=True):
        ordered = sub.sort_values("N")
        grid = [int(n) for n in ordered["N"].tolist()]
        medians = ordered["median_calibration"].to_numpy(dtype=float)
        for delta in deltas:
            rows.append(
                {
                    "class_name": class_name,
                    "group": ordered["group"].iloc[0],
                    "strategy": strategy,
                    "rep": int(rep),
                    "delta": float(delta),
                    "n_stop": n_stop(grid, medians, float(delta)),
                    "pool_size": int(ordered["pool_size"].iloc[0]),
                }
            )
    return pd.DataFrame(rows)


def budget_table() -> pd.DataFrame:
    rows = []
    for class_name, counts in EXPECTED_COUNTS.items():
        n_good = int(counts["test_good"])
        budget = false_reject_budget(n_good)
        k_min = budget + 1
        exploratory_probability = float(1.0 - betabinom.cdf(k_min - 1, n_good, 1, 20))
        rows.append(
            {
                "class_name": class_name,
                "n_test_good": n_good,
                "n_train": int(counts["train"]),
                "false_reject_budget": budget,
                "clopper_upper_zero_errors": clopper_pearson_upper(0, n_good),
                "exploratory_p_sample_fpr_above_05": exploratory_probability,
            }
        )
    return pd.DataFrame(rows)


def _median_or_none(values: list[int]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def attach_stopping(summary: pd.DataFrame, stopping: pd.DataFrame, main_delta: float) -> pd.DataFrame:
    main = stopping[np.isclose(stopping["delta"], main_delta)].copy()
    rows = []
    for row in summary.itertuples(index=False):
        sub = main[(main["class_name"] == row.class_name) & (main["strategy"] == row.strategy)]
        fired = [int(v) for v in sub["n_stop"].dropna().tolist()]
        categories = [
            stop_category(None if pd.isna(v) else int(v), None if row.censored else int(row.n_star), bool(row.censored))
            for v in sub["n_stop"].tolist()
        ]
        ratio = []
        excess = []
        if not row.censored:
            ratio = [v / int(row.n_star) for v in fired]
            excess = [max(0, v - int(row.n_star)) for v in fired]
        record = row._asdict()
        record["n_stop_median"] = _median_or_none(fired)
        record["n_stop_ratio_median"] = float(np.median(ratio)) if ratio else None
        record["excess_images_median"] = float(np.median(excess)) if excess else None
        for name in ("premature", "concordant", "excessive", "n_star_not_reached", "rule_not_fired"):
            record[f"fraction_{name}"] = float(np.mean([c == name for c in categories])) if categories else None
        rows.append(record)
    return pd.DataFrame(rows)


def analyze(cfg: Config) -> dict:
    metrics = load_metrics(cfg)
    summary = summarize_classes(metrics, cfg.aulc_n_max, cfg.n_star_min_repeats)
    stopping = stopping_table(metrics, cfg.stopping_deltas)
    summary = attach_stopping(summary, stopping, cfg.stopping_delta_main)
    aggregate = fpr_aggregate(metrics)
    statistics = compare_strategies(
        summary,
        margin=cfg.fpr_aulc_margin,
        n_bootstrap=cfg.bootstrap_replicates,
        seed=cfg.bootstrap_seed,
    )
    out = cfg.work_root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    atomic_parquet(out / "metrics.parquet", metrics)
    summary.to_csv(out / "summary.csv", index=False)
    aggregate.to_csv(out / "fpr_aggregate.csv", index=False)
    stopping.to_csv(out / "stopping.csv", index=False)
    budget_table().to_csv(out / "false_reject_budget.csv", index=False)
    write_json(out / "statistics.json", statistics)
    return {
        "metrics": metrics,
        "summary": summary,
        "fpr_aggregate": aggregate,
        "stopping": stopping,
        "statistics": statistics,
        "budget": budget_table(),
        "analysis_dir": out,
    }


def load_analysis(analysis_dir: Path) -> dict:
    return {
        "metrics": pd.read_parquet(analysis_dir / "metrics.parquet"),
        "summary": pd.read_csv(analysis_dir / "summary.csv"),
        "fpr_aggregate": pd.read_csv(analysis_dir / "fpr_aggregate.csv"),
        "stopping": pd.read_csv(analysis_dir / "stopping.csv"),
        "budget": pd.read_csv(analysis_dir / "false_reject_budget.csv"),
        "statistics": __import__("json").loads((analysis_dir / "statistics.json").read_text(encoding="utf-8")),
    }
