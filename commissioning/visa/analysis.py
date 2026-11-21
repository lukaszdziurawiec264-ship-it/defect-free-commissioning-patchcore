"""Confirmatory analysis of preregistration_visa.md: H1-H3, P1-P3, N*, feasibility, stopping, budget split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import betabinom, binom

from commissioning.analysis import aulc, n_star, stop_category
from commissioning.calibration import false_reject_budget
from commissioning.io_utils import atomic_parquet, write_json
from commissioning.stats import bootstrap_median_interval, compare_strategies
from commissioning.stopping import n_stop
from commissioning.visa.budget import budget_hypothesis, summarize_budget
from commissioning.visa.config import EXPECTED_COUNTS, RULES, VisaConfig
from commissioning.visa.exchange import load_exchange
from commissioning.visa.run import load_metrics

FPR_COLUMN = {"off": "fpr_off", "H": "fpr_h"}
CRITERION_COLUMN = {"off": "criterion_off", "H": "criterion_h"}
THEORY_MEAN = {"R20": 1 / 21, "R60": 1 / 61}
THEORY_EXCEED = {"R20": 0.95**20, "R60": 0.95**60}
P2_BAND = {"R20": (0.03, 0.07), "R60": (0.005, 0.03)}


def summarize(metrics: pd.DataFrame, cfg: VisaConfig) -> pd.DataFrame:
    rows = []
    for (class_name, group, strategy, rule), sub in metrics.groupby(["class_name", "group", "strategy", "rule"], sort=True):
        record = {"class_name": class_name, "group": group, "strategy": strategy, "rule": rule}
        per_rep = {"aulc_tpr": [], "aulc_fpr_off": [], "aulc_fpr_h": []}
        for _rep, rep_df in sub.groupby("rep"):
            ordered = rep_df.sort_values("N")
            per_rep["aulc_tpr"].append(aulc(ordered["N"], ordered["tpr"], cfg.aulc_n_max))
            per_rep["aulc_fpr_off"].append(aulc(ordered["N"], ordered["fpr_off"], cfg.aulc_n_max))
            per_rep["aulc_fpr_h"].append(aulc(ordered["N"], ordered["fpr_h"], cfg.aulc_n_max))
        for key, values in per_rep.items():
            record[key] = float(np.mean(values))
            record[f"{key}_iqr"] = float(np.subtract(*np.quantile(values, [0.75, 0.25])))
        grid = [int(n) for n in sorted(sub["N"].unique())]
        for source, column in CRITERION_COLUMN.items():
            passed = [int(sub.loc[sub["N"] == n, column].sum()) for n in grid]
            star = n_star(grid, passed, cfg.n_star_min_repeats)
            record[f"n_star_{source}"] = float("nan") if star["censored"] else float(star["n_star"])
            record[f"censored_{source}"] = bool(star["censored"])
            record[f"passing_full_pool_{source}"] = passed[-1]
        record["mean_fpr_off"] = float(sub["fpr_off"].mean())
        record["mean_fpr_h"] = float(sub["fpr_h"].mean())
        record["pool_size"] = int(sub["pool_size"].iloc[0])
        record["n_reps"] = int(sub["rep"].nunique())
        rows.append(record)
    return pd.DataFrame(rows)


def _comparison(summary: pd.DataFrame, cfg: VisaConfig, rule: str, source: str) -> dict:
    frame = summary[summary["rule"] == rule][["class_name", "strategy", "aulc_tpr", f"aulc_fpr_{source.lower()}"]]
    frame = frame.rename(columns={f"aulc_fpr_{source.lower()}": "aulc_fpr"})
    return compare_strategies(frame, cfg.fpr_aulc_margin, cfg.bootstrap_replicates, cfg.bootstrap_seed)


def tpr_cost(metrics: pd.DataFrame, cfg: VisaConfig, n: int, strategy: str = "random") -> pd.DataFrame:
    sub = metrics[(metrics["N"] == n) & (metrics["strategy"] == strategy)]
    wide = sub.pivot_table(index=["class_name", "rep"], columns="rule", values="tpr")
    if (wide["R60"] > wide["R20"] + 1e-12).any():
        raise RuntimeError("TPR under R60 exceeded TPR under R20; max(C60) >= max(C20) must hold.")
    per_class = (wide["R20"] - wide["R60"]).groupby("class_name").mean()
    return per_class.rename("tpr_cost").reset_index()


def h3(metrics: pd.DataFrame, cfg: VisaConfig) -> dict:
    full = tpr_cost(metrics, cfg, cfg.pool_size)
    boot = bootstrap_median_interval(full["tpr_cost"].to_numpy(), cfg.bootstrap_replicates, cfg.bootstrap_seed + 2)
    secondary = tpr_cost(metrics, cfg, 32)
    return {
        "quantity": "mean over repetitions of TPR(R20) - TPR(R60) at N = |P|, random strategy",
        "per_class": dict(zip(full["class_name"], full["tpr_cost"].astype(float))),
        "median": boot["median"],
        "bootstrap": boot,
        "margin": cfg.tpr_cost_margin,
        "r60_recommended": bool(boot["median"] <= cfg.tpr_cost_margin),
        "secondary_n32": {
            "per_class": dict(zip(secondary["class_name"], secondary["tpr_cost"].astype(float))),
            "median": float(secondary["tpr_cost"].median()),
        },
    }


def p2_calibration(metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    per_class = (
        metrics.groupby(["class_name", "rule"])
        .agg(mean_fpr_h=("fpr_h", "mean"), fraction_fpr_h_above_05=("fpr_h", lambda x: float(np.mean(x > 0.05))))
        .reset_index()
    )
    out = {}
    for rule in RULES:
        sub = per_class[per_class["rule"] == rule]
        median = float(sub["mean_fpr_h"].median())
        low, high = P2_BAND[rule]
        out[rule] = {
            "median_over_classes_of_mean_fpr_h": median,
            "band": [low, high],
            "inside_band": bool(low <= median <= high),
            "theory_mean": THEORY_MEAN[rule],
            "fraction_fpr_h_above_05_mean_over_classes": float(sub["fraction_fpr_h_above_05"].mean()),
            "theory_fraction_conditional_fpr_above_05": THEORY_EXCEED[rule],
        }
    by_n = (
        metrics.groupby(["rule", "N"])
        .agg(mean_fpr_h=("fpr_h", "mean"), mean_fpr_off=("fpr_off", "mean"),
             fraction_fpr_h_above_05=("fpr_h", lambda x: float(np.mean(x > 0.05))))
        .reset_index()
    )
    out["by_N"] = by_n.to_dict(orient="records")
    return per_class, out


def p3_shift(metrics: pd.DataFrame, exchange: pd.DataFrame | None) -> tuple[pd.DataFrame | None, dict]:
    if exchange is None:
        return None, {"available": False}
    r20 = metrics[metrics["rule"] == "R20"].groupby("class_name").agg(
        mean_fpr_off=("fpr_off", "mean"), mean_fpr_h=("fpr_h", "mean")
    )
    frame = exchange.set_index("class_name").join(r20).reset_index()
    frame["fpr_gap_off_minus_h"] = frame["mean_fpr_off"] - frame["mean_fpr_h"]
    shifted = frame[frame["shift_detected"]]
    return frame, {
        "available": True,
        "classes_with_shift": shifted["class_name"].tolist(),
        "prediction_holds_per_shifted_class": {
            str(row.class_name): bool(row.fpr_gap_off_minus_h > 0) for row in shifted.itertuples()
        },
        "median_gap_shifted": float(shifted["fpr_gap_off_minus_h"].median()) if len(shifted) else None,
        "median_gap_not_shifted": float(frame.loc[~frame["shift_detected"], "fpr_gap_off_minus_h"].median()),
    }


def n_star_table(summary: pd.DataFrame) -> pd.DataFrame:
    keep = ["class_name", "group", "strategy", "rule"]
    for source in ("off", "H"):
        keep += [f"n_star_{source}", f"censored_{source}", f"passing_full_pool_{source}"]
    return summary[keep].sort_values(["rule", "strategy", "class_name"]).reset_index(drop=True)


def p1_feasibility(summary: pd.DataFrame, metrics: pd.DataFrame, cfg: VisaConfig) -> dict:
    out = {}
    for (rule, strategy), sub in summary.groupby(["rule", "strategy"]):
        out[f"{rule}_{strategy}"] = {
            "censored_off": int(sub["censored_off"].sum()),
            "censored_H": int(sub["censored_H"].sum()),
            "n_classes": int(len(sub)),
        }
    full = metrics[(metrics["N"] == cfg.pool_size) & (metrics["strategy"] == "random") & (metrics["rule"] == "R60")]
    tpr_ok = full.groupby("class_name")["tpr"].apply(lambda x: int(np.sum(x >= cfg.practical_tpr)))
    eligible = sorted(tpr_ok[tpr_ok >= cfg.n_star_min_repeats].index)
    r60 = summary[(summary["rule"] == "R60") & (summary["strategy"] == "random")].set_index("class_name")
    out["R60_eligible_by_tpr"] = eligible
    out["R60_eligible_reached_n_star_off"] = {name: bool(not r60.loc[name, "censored_off"]) for name in eligible}
    return out


def ceiling_table(cfg: VisaConfig, metrics: pd.DataFrame) -> pd.DataFrame:
    sizes = metrics.groupby("class_name").agg(n_test_good=("n_test_good", "first"), n_holdout=("n_holdout", "first"))
    rows = []
    for class_name, counts in sizes.iterrows():
        for source, n in (("off", int(counts["n_test_good"])), ("H", int(counts["n_holdout"]))):
            for rule, m in (("R20", cfg.n_calibration_small), ("R60", cfg.n_calibration_large)):
                p = float(betabinom.cdf(false_reject_budget(n), n, 1, m))
                rows.append(
                    {
                        "class_name": class_name,
                        "fpr_source": source,
                        "n_good": n,
                        "budget": false_reject_budget(n),
                        "rule": rule,
                        "p_rep_fpr_ok": p,
                        "expected_passing_reps": cfg.n_repeats * p,
                        "p_at_least_min_reps_if_independent": float(
                            binom.sf(cfg.n_star_min_repeats - 1, cfg.n_repeats, p)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def stopping(metrics: pd.DataFrame, summary: pd.DataFrame, cfg: VisaConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = metrics[metrics["rule"] == "R20"]
    rows = []
    for (class_name, strategy, rep), sub in base.groupby(["class_name", "strategy", "rep"], sort=True):
        ordered = sub.sort_values("N")
        grid = [int(n) for n in ordered["N"]]
        medians = ordered["median_calibration_c20"].to_numpy(dtype=float)
        for delta in cfg.stopping_deltas:
            rows.append(
                {"class_name": class_name, "strategy": strategy, "rep": int(rep), "delta": float(delta),
                 "n_stop": n_stop(grid, medians, float(delta))}
            )
    table = pd.DataFrame(rows)
    main = table[np.isclose(table["delta"], cfg.stopping_delta_main)]
    categories = []
    for rule in RULES:
        lookup = summary[summary["rule"] == rule].set_index(["class_name", "strategy"])
        for (class_name, strategy), sub in main.groupby(["class_name", "strategy"]):
            star = lookup.loc[(class_name, strategy)]
            censored = bool(star["censored_off"])
            labels = [
                stop_category(None if pd.isna(v) else int(v), None if censored else int(star["n_star_off"]), censored)
                for v in sub["n_stop"]
            ]
            record = {"class_name": class_name, "strategy": strategy, "reference": f"N*_{rule}_off"}
            for name in ("premature", "concordant", "excessive", "n_star_not_reached", "rule_not_fired"):
                record[f"fraction_{name}"] = float(np.mean([label == name for label in labels]))
            categories.append(record)
    return table, pd.DataFrame(categories)


def analyze(cfg: VisaConfig) -> dict:
    metrics = load_metrics(cfg)
    expected_rows = len(cfg.classes) * cfg.n_repeats * len(cfg.strategies) * len(cfg.n_grid) * len(RULES)
    if len(metrics) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} trajectory rows, found {len(metrics)}. Finish the run first.")
    summary = summarize(metrics, cfg)
    statistics = {
        "primary": {"H1_H2_R20_official": _comparison(summary, cfg, "R20", "off")},
        "secondary": {
            "R60_official": _comparison(summary, cfg, "R60", "off"),
            "R20_holdout": _comparison(summary, cfg, "R20", "H"),
            "R60_holdout": _comparison(summary, cfg, "R60", "H"),
        },
        "H3": h3(metrics, cfg),
    }
    p2_frame, statistics["P2"] = p2_calibration(metrics)
    exchange = load_exchange(cfg)
    p3_frame, statistics["P3"] = p3_shift(metrics, exchange)
    statistics["P1"] = p1_feasibility(summary, metrics, cfg)
    stop_table, stop_summary = stopping(metrics, summary, cfg)
    out = cfg.work_root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    budget_parts = sorted((cfg.work_root / "budget").glob("*.parquet"))
    if len(budget_parts) == len(cfg.classes):
        budget_frame = pd.concat([pd.read_parquet(p) for p in budget_parts], ignore_index=True)
        budget_summary, budget_best = summarize_budget(budget_frame, cfg.n_star_min_repeats)
        statistics["budget"] = budget_hypothesis(budget_best)
        budget_summary.to_csv(out / "budget_summary.csv", index=False)
        budget_best.to_csv(out / "budget_best.csv", index=False)
    atomic_parquet(out / "metrics.parquet", metrics)
    summary.to_csv(out / "summary.csv", index=False)
    n_star_table(summary).to_csv(out / "n_star.csv", index=False)
    p2_frame.to_csv(out / "p2_calibration.csv", index=False)
    if p3_frame is not None:
        p3_frame.to_csv(out / "p3_shift.csv", index=False)
    ceiling_table(cfg, metrics).to_csv(out / "ceiling.csv", index=False)
    stop_table.to_csv(out / "stopping.csv", index=False)
    stop_summary.to_csv(out / "stopping_summary.csv", index=False)
    statistics["config_hash"] = cfg.hash
    statistics["exploratory_rows"] = int(metrics["exploratory"].sum())
    write_json(out / "statistics.json", json.loads(json.dumps(statistics, default=_json_default)))
    return {"metrics": metrics, "summary": summary, "statistics": statistics, "analysis_dir": out}


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Not JSON serializable: {type(value)}")


def load_analysis(analysis_dir: Path) -> dict:
    frames = {}
    for name in ("summary", "n_star", "p2_calibration", "p3_shift", "ceiling", "stopping_summary", "budget_best"):
        path = analysis_dir / f"{name}.csv"
        if path.is_file():
            frames[name] = pd.read_csv(path)
    frames["metrics"] = pd.read_parquet(analysis_dir / "metrics.parquet")
    frames["statistics"] = json.loads((analysis_dir / "statistics.json").read_text(encoding="utf-8"))
    return frames
