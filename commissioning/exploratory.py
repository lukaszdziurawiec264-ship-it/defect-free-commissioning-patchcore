"""Exploratory analyses declared in exploratory_addendum.md after the confirmatory run.

Nothing here changes N*, AULC, the stopping rule, or the primary tests. Every output
goes to work_root/exploratory and every table row carries exploratory = True.

E1  Feasibility ceiling of the N* criterion for a detector that never misses a defect.
E2  Splitting a fixed budget of good parts between the memory bank and the calibration set.
E3  Exchangeability of train/good and test/good scores under the full training bank.
E4  Test good images that the commissioned threshold rejects in most trajectories.
E5  Why each censored class fails: TPR, FPR, or both.
"""

from __future__ import annotations

import html
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, betabinom, binom, mannwhitneyu
from sklearn.metrics import roc_auc_score

from commissioning.calibration import _meets, false_reject_budget, order_statistic_rank
from commissioning.config import EXPECTED_COUNTS, Config
from commissioning.data import load_split
from commissioning.distances import distance_cache_path, distance_meta_path, load_distance_cache
from commissioning.io_utils import atomic_parquet, read_json, write_json
from commissioning.patchcore import scores_along_order
from commissioning.stats import holm

LOGGER = logging.getLogger(__name__)

EPSILON = 0.05
DELTA = 0.10
RULES = ("conformal", "pac")
BUDGETS = (40, 60, 80, 120, 160, 200)
SPLIT_GRID = (1, 2, 4, 8, 16, 32, 64, 128)
MIN_CALIBRATION = 20
CALIBRATION_SIZES = (20, 30, 40, 45, 60, 80, 100, 150, 200)
# Separate seed stream so E2 cannot reuse or perturb the confirmatory C/P splits.
SEED_TAG = 20261120


def out_dir(cfg: Config) -> Path:
    return cfg.work_root / "exploratory"


# E1 and the calibration-size curve -------------------------------------------------


def exceedance_probability(m: int, k: int, epsilon: float = EPSILON) -> float:
    """P(conditional FPR > epsilon) when the threshold is the k-th smallest of m exchangeable good scores.

    The conditional FPR of that threshold is Beta(m - k + 1, k).
    """
    if not 1 <= k <= m:
        raise RuntimeError(f"Rank {k} is outside 1..{m}")
    return float(beta.sf(epsilon, m - k + 1, k))


@lru_cache(maxsize=None)
def pac_rank(m: int, epsilon: float = EPSILON, delta: float = DELTA) -> int | None:
    """Smallest rank whose conditional FPR exceeds epsilon with probability at most delta."""
    for k in range(1, m + 1):
        if exceedance_probability(m, k, epsilon) <= delta:
            return k
    return None


def rule_rank(rule: str, m: int, alpha: float) -> tuple[int, bool]:
    """Returns (rank, feasible). An infeasible PAC rule falls back to the maximum."""
    if rule == "conformal":
        return order_statistic_rank(m, alpha), True
    if rule == "pac":
        k = pac_rank(m)
        return (m, False) if k is None else (k, True)
    raise RuntimeError(f"Unknown threshold rule: {rule}")


def ideal_pass_probability(n_good: int, m: int, k: int) -> float:
    """P(false rejects <= budget) for a detector with TPR = 1 and exchangeable good scores."""
    return float(betabinom.cdf(false_reject_budget(n_good), n_good, m - k + 1, k))


def ceiling_table(cfg: Config) -> pd.DataFrame:
    rows = []
    for class_name in cfg.classes:
        n_good = int(EXPECTED_COUNTS[class_name]["test_good"])
        n_train = int(EXPECTED_COUNTS[class_name]["train"])
        for rule in RULES:
            for m in CALIBRATION_SIZES:
                if m >= n_train:
                    continue
                k, feasible = rule_rank(rule, m, cfg.alpha)
                p_rep = ideal_pass_probability(n_good, m, k)
                rows.append(
                    {
                        "class_name": class_name,
                        "group": cfg.group_of(class_name),
                        "rule": rule,
                        "n_calibration": int(m),
                        "rank": int(k),
                        "rule_feasible": bool(feasible),
                        "p_conditional_fpr_above_eps": exceedance_probability(m, k),
                        "n_test_good": n_good,
                        "false_reject_budget": false_reject_budget(n_good),
                        "p_rep_fpr_ok": p_rep,
                        "expected_passing_reps": cfg.n_repeats * p_rep,
                        "p_at_least_min_reps_if_independent": float(
                            binom.sf(cfg.n_star_min_repeats - 1, cfg.n_repeats, p_rep)
                        ),
                        "exploratory": True,
                    }
                )
    return pd.DataFrame(rows)


def calibration_size_table(cfg: Config) -> pd.DataFrame:
    rows = []
    for m in range(MIN_CALIBRATION, 201):
        k_conformal = order_statistic_rank(m, cfg.alpha)
        k_pac, feasible = rule_rank("pac", m, cfg.alpha)
        rows.append(
            {
                "n_calibration": m,
                "p_above_eps_max": exceedance_probability(m, m),
                "rank_conformal": k_conformal,
                "p_above_eps_conformal": exceedance_probability(m, k_conformal),
                "rank_pac": k_pac,
                "pac_feasible": feasible,
                "p_above_eps_pac": exceedance_probability(m, k_pac),
                "exploratory": True,
            }
        )
    return pd.DataFrame(rows)


# Shared helpers --------------------------------------------------------------------


def _test_indices(cfg: Config, class_name: str, paths: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    split = load_split(cfg, class_name, 0)
    position = {path: index for index, path in enumerate(paths)}
    good = [position[path] for path in split["test_good"]]
    defect = [position[item["path"]] for item in split["test_defect"]]
    return np.asarray(good, dtype=int), np.asarray(defect, dtype=int), list(split["test_good"])


def _part_is_current(cfg: Config, path: Path) -> bool:
    if not path.is_file():
        return False
    frame = pd.read_parquet(path)
    if set(frame["config_hash"].astype(str)) != {cfg.hash}:
        raise RuntimeError(f"Refusing to reuse an exploratory part from another config: {path}")
    return True


# E2 Budget split -------------------------------------------------------------------


def budget_split_class(cfg: Config, class_name: str, distance: np.ndarray, paths: list[str], n_train: int) -> pd.DataFrame:
    """Random order of all train/good images. Bank = first N of the first B, calibration = the other B - N."""
    good, defect, _ = _test_indices(cfg, class_name, paths)
    test = np.concatenate([good, defect])
    labels = np.array([0] * len(good) + [1] * len(defect), dtype=int)
    n_good, n_defect = len(good), len(defect)
    budgets = [b for b in BUDGETS if b <= n_train]
    if not budgets:
        return pd.DataFrame()
    rows = []
    for rep in range(cfg.n_repeats):
        sequence = np.random.SeedSequence([cfg.seed, cfg.class_index(class_name), rep, SEED_TAG])
        permutation = np.random.default_rng(sequence).permutation(n_train)
        columns = permutation[: max(budgets)]
        reduced = distance[:, :, columns]
        for budget in budgets:
            grid = [n for n in SPLIT_GRID if budget - n >= MIN_CALIBRATION]
            if not grid:
                continue
            # The first bank image is never a query: its only bank member would be itself (+inf).
            queries = np.concatenate([columns[1:budget], test])
            stacked = scores_along_order(reduced, queries, np.arange(budget), grid)
            for row_index, n in enumerate(grid):
                calibration = stacked[row_index, n - 1 : budget - 1]
                test_scores = stacked[row_index, budget - 1 :]
                m = budget - n
                ordered = np.sort(calibration)
                for rule in RULES:
                    k, feasible = rule_rank(rule, m, cfg.alpha)
                    threshold = float(ordered[k - 1])
                    n_tp = int(np.sum(test_scores[labels == 1] > threshold))
                    n_fp = int(np.sum(test_scores[labels == 0] > threshold))
                    tpr_ok = _meets(n_tp, n_defect, cfg.practical_tpr, at_least=True)
                    fpr_ok = _meets(n_fp, n_good, cfg.practical_fpr, at_least=False)
                    rows.append(
                        {
                            "class_name": class_name,
                            "group": cfg.group_of(class_name),
                            "rep": int(rep),
                            "budget": int(budget),
                            "N": int(n),
                            "n_calibration": int(m),
                            "rule": rule,
                            "rank": int(k),
                            "rule_feasible": bool(feasible),
                            "tpr": n_tp / n_defect,
                            "fpr": n_fp / n_good,
                            "n_tp": n_tp,
                            "n_fp": n_fp,
                            "tpr_ok": bool(tpr_ok),
                            "fpr_ok": bool(fpr_ok),
                            "criterion": bool(tpr_ok and fpr_ok),
                            "config_hash": cfg.hash,
                            "exploratory": True,
                        }
                    )
        del reduced
    return pd.DataFrame(rows)


def summarize_budget(frame: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    keys = ["class_name", "group", "budget", "N", "n_calibration", "rule"]
    for key, sub in frame.groupby(keys, sort=True):
        record = dict(zip(keys, key))
        n_good = int(EXPECTED_COUNTS[record["class_name"]]["test_good"])
        k = int(sub["rank"].iloc[0])
        p_ideal = ideal_pass_probability(n_good, int(record["n_calibration"]), k)
        record.update(
            {
                "rank": k,
                "rule_feasible": bool(sub["rule_feasible"].iloc[0]),
                "passing_reps": int(sub["criterion"].sum()),
                "tpr_ok_reps": int(sub["tpr_ok"].sum()),
                "fpr_ok_reps": int(sub["fpr_ok"].sum()),
                "median_tpr": float(sub["tpr"].median()),
                "mean_fpr": float(sub["fpr"].mean()),
                "ideal_expected_passing_reps": cfg.n_repeats * p_ideal,
                "exploratory": True,
            }
        )
        rows.append(record)
    summary = pd.DataFrame(rows)
    best_rows = []
    for (class_name, group, budget, rule), sub in summary.groupby(["class_name", "group", "budget", "rule"], sort=True):
        ranked = sub.sort_values(["passing_reps", "median_tpr", "N"], ascending=[False, False, True])
        top = ranked.iloc[0]
        best_rows.append(
            {
                "class_name": class_name,
                "group": group,
                "budget": int(budget),
                "rule": rule,
                "best_N": int(top["N"]),
                "best_n_calibration": int(top["n_calibration"]),
                "passing_reps": int(top["passing_reps"]),
                "meets_min_reps": bool(top["passing_reps"] >= cfg.n_star_min_repeats),
                "median_tpr": float(top["median_tpr"]),
                "mean_fpr": float(top["mean_fpr"]),
                "exploratory": True,
            }
        )
    return summary, pd.DataFrame(best_rows)


# E3 Exchangeability of train/good and test/good -------------------------------------


def _full_bank_scores(distance: np.ndarray, queries: np.ndarray, chunk: int = 16) -> np.ndarray:
    """Score against all train images. Train queries are leave-one-out through the +inf diagonal."""
    scores = np.empty(len(queries), dtype=np.float64)
    for start in range(0, len(queries), chunk):
        block = distance[queries[start : start + chunk]]
        scores[start : start + chunk] = block.min(axis=-1).max(axis=-1)
    if not np.isfinite(scores).all():
        raise RuntimeError("Non-finite full-bank score")
    return scores


def exchangeability_class(cfg: Config, class_name: str, distance: np.ndarray, paths: list[str], n_train: int) -> dict:
    good, defect, good_paths = _test_indices(cfg, class_name, paths)
    train_scores = _full_bank_scores(distance, np.arange(n_train))
    good_scores = _full_bank_scores(distance, good)
    defect_scores = _full_bank_scores(distance, defect)
    shift_auroc = float(
        roc_auc_score(np.r_[np.zeros(n_train), np.ones(len(good))], np.r_[train_scores, good_scores])
    )
    test_p = mannwhitneyu(good_scores, train_scores, alternative="two-sided").pvalue
    conformal_p = (1.0 + (train_scores[None, :] >= good_scores[:, None]).sum(axis=1)) / (n_train + 1.0)
    n_low = int(np.sum(conformal_p <= cfg.alpha))
    return {
        "summary": {
            "class_name": class_name,
            "group": cfg.group_of(class_name),
            "n_train": int(n_train),
            "n_test_good": int(len(good)),
            "good_shift_auroc": shift_auroc,
            "mannwhitney_p": float(test_p),
            "median_train_good_score": float(np.median(train_scores)),
            "median_test_good_score": float(np.median(good_scores)),
            "fraction_conformal_p_below_alpha": n_low / len(good),
            "binomial_tail_p": float(binom.sf(n_low - 1, len(good), cfg.alpha)),
            "defect_auroc_full_bank": float(
                roc_auc_score(np.r_[np.zeros(len(good)), np.ones(len(defect))], np.r_[good_scores, defect_scores])
            ),
            "config_hash": cfg.hash,
            "exploratory": True,
        },
        "images": pd.DataFrame(
            {
                "class_name": class_name,
                "path": good_paths,
                "full_bank_score": good_scores,
                "conformal_p": conformal_p,
            }
        ),
    }


# E4 Repeatedly rejected good images -------------------------------------------------


def rejection_rates(cfg: Config, class_name: str) -> pd.DataFrame:
    """Share of the 40 trajectories in which each good test image is above max(C)."""
    files = sorted((cfg.work_root / "runs" / "scores").glob(f"{class_name}_rep*_*.npz"))
    if not files:
        return pd.DataFrame()
    counts: dict[str, dict[str, int]] = {}
    n_trajectories = 0
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            grid = [int(n) for n in data["grid"]]
            labels = np.asarray(data["test_labels"], dtype=int)
            test_paths = [str(p) for p in data["test_paths"].tolist()]
            rows = {"full_pool": len(grid) - 1}
            if 32 in grid:
                rows["n32"] = grid.index(32)
            for name, row in rows.items():
                threshold = float(np.max(data["scores_calibration"][row]))
                rejected = np.asarray(data["scores_test"][row]) > threshold
                for test_path, label, flag in zip(test_paths, labels, rejected):
                    if label != 0:
                        continue
                    counts.setdefault(test_path, {}).setdefault(name, 0)
                    counts[test_path][name] += int(flag)
        n_trajectories += 1
    return pd.DataFrame(
        [
            {
                "class_name": class_name,
                "path": test_path,
                "rejection_rate_full_pool": values.get("full_pool", 0) / n_trajectories,
                "rejection_rate_n32": values.get("n32", 0) / n_trajectories,
                "n_trajectories": n_trajectories,
                "exploratory": True,
            }
            for test_path, values in counts.items()
        ]
    )


# E5 Censoring decomposition ---------------------------------------------------------


def censoring_table(cfg: Config, metrics: pd.DataFrame) -> pd.DataFrame:
    frame = metrics.copy()
    frame["tpr_ok"] = [
        _meets(tp, total, cfg.practical_tpr, at_least=True) for tp, total in zip(frame["n_tp"], frame["n_test_defect"])
    ]
    frame["fpr_ok"] = [
        _meets(fp, total, cfg.practical_fpr, at_least=False) for fp, total in zip(frame["n_fp"], frame["n_test_good"])
    ]
    need = cfg.n_star_min_repeats
    rows = []
    for (class_name, strategy), sub in frame.groupby(["class_name", "strategy"], sort=True):
        per_n = sub.groupby("N").agg(
            tpr_ok=("tpr_ok", "sum"), fpr_ok=("fpr_ok", "sum"), both=("criterion", "sum")
        )
        full = per_n.loc[int(sub["pool_size"].iloc[0])]
        tpr_short = int(full["tpr_ok"]) < need
        fpr_short = int(full["fpr_ok"]) < need
        limited = "both" if tpr_short and fpr_short else "tpr" if tpr_short else "fpr" if fpr_short else "none"
        n_good = int(sub["n_test_good"].iloc[0])
        best_n = int(per_n["both"].idxmax())
        rows.append(
            {
                "class_name": class_name,
                "group": cfg.group_of(class_name),
                "strategy": strategy,
                "pool_size": int(sub["pool_size"].iloc[0]),
                "tpr_ok_reps_full_pool": int(full["tpr_ok"]),
                "fpr_ok_reps_full_pool": int(full["fpr_ok"]),
                "passing_reps_full_pool": int(full["both"]),
                "limited_by_full_pool": limited,
                "best_N": best_n,
                "passing_reps_best_N": int(per_n.loc[best_n, "both"]),
                "ideal_expected_fpr_ok_reps": cfg.n_repeats * ideal_pass_probability(n_good, cfg.n_calibration, cfg.n_calibration),
                "exploratory": True,
            }
        )
    return pd.DataFrame(rows)


# Figures and HTML -------------------------------------------------------------------


def _figures(cfg: Config, tables: dict, figures: Path, data_root: Path | None) -> list[tuple[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from commissioning.report import KCENTER_COLOR, RANDOM_COLOR, _model_view, _save, _style

    group_color = {"object": RANDOM_COLOR, "texture": KCENTER_COLOR}
    written: list[tuple[str, str]] = []

    ceiling = tables["ceiling"]
    base = ceiling[(ceiling["rule"] == "conformal") & (ceiling["n_calibration"] == cfg.n_calibration)]
    base = base.sort_values("expected_passing_reps")
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(len(base))
    ax.barh(y, base["expected_passing_reps"], color=[group_color[g] for g in base["group"]], alpha=0.8,
            label="ideal detector, exchangeable good scores")
    larger = ceiling[(ceiling["rule"] == "pac") & (ceiling["n_calibration"] == 60)].set_index("class_name")
    ax.scatter([larger["expected_passing_reps"].get(name, np.nan) for name in base["class_name"]], y, marker="D",
               facecolors="none", edgecolors="#2e7d4f", s=26, zorder=3, label="ideal detector, 60 calibration images, PAC rank")
    censoring = tables.get("censoring")
    if censoring is not None and not censoring.empty:
        observed = censoring.groupby("class_name")["passing_reps_best_N"].max()
        ax.scatter([observed.get(name, np.nan) for name in base["class_name"]], y, color="black", s=18, zorder=3,
                   label="observed, best N and strategy")
    ax.axvline(cfg.n_star_min_repeats, color="#a33b3b", ls="--", lw=1, label=f"N* needs {cfg.n_star_min_repeats}/20")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c.replace('_', ' ')} (n={n})" for c, n in zip(base["class_name"], base["n_test_good"])], fontsize=8)
    ax.set_xlim(0, cfg.n_repeats)
    ax.set_xlabel("Repetitions out of 20 meeting the FPR budget")
    ax.set_title("Feasibility ceiling of N* with threshold = max of 20")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    _save(fig, figures, "figE1_ceiling")
    written.append(("figE1_ceiling", "Expected passing repetitions for a detector with TPR = 1. Bars: objects blue, textures orange. Dots: observed maximum over N and strategies."))

    size = tables["calibration_size"]
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(size["n_calibration"], size["p_above_eps_conformal"], color=RANDOM_COLOR, lw=1.5, label="conformal rank (alpha = 0.05)")
    ax.plot(size["n_calibration"], size["p_above_eps_pac"], color=KCENTER_COLOR, lw=1.5, label=f"PAC rank (delta = {DELTA})")
    ax.plot(size["n_calibration"], size["p_above_eps_max"], color="#333333", lw=1.2, ls="--", zorder=3, label="maximum")
    ax.axhline(DELTA, color="#a33b3b", ls=":", lw=1)
    ax.set_xlabel("Calibration images m")
    ax.set_ylabel(f"P(conditional FPR > {EPSILON})")
    ax.set_title("How many good images the threshold needs")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, figures, "figE2_calibration_size")
    written.append(("figE2_calibration_size", "Distribution-free exceedance probability of each threshold rule under exchangeability."))

    budget = tables.get("budget_summary")
    if budget is not None and not budget.empty:
        sub = budget[budget["rule"] == "pac"]
        budgets = sorted(sub["budget"].unique())
        cols = 3
        rows_n = int(np.ceil(len(budgets) / cols))
        _style()
        fig, axes = plt.subplots(rows_n, cols, figsize=(11, 3.2 * rows_n), sharey=True, squeeze=False)
        for ax, b in zip(axes.ravel(), budgets):
            part = sub[sub["budget"] == b]
            for class_name, line in part.groupby("class_name"):
                line = line.sort_values("N")
                ax.plot(np.log2(line["N"]), line["passing_reps"], color=group_color[line["group"].iloc[0]], lw=1, alpha=0.8)
            ax.axhline(cfg.n_star_min_repeats, color="#a33b3b", ls="--", lw=1)
            ax.set_title(f"B = {b} good parts", fontsize=9)
            ax.set_xlabel("log2 N (bank); calibration = B - N")
        for ax in axes.ravel()[len(budgets):]:
            ax.axis("off")
        axes[0, 0].set_ylabel("Passing repetitions of 20")
        fig.suptitle("Bank versus calibration under a fixed budget, PAC threshold")
        fig.tight_layout()
        _save(fig, figures, "figE3_budget_split")
        written.append(("figE3_budget_split", "One line per class. Objects blue, textures orange. Random bank, PAC rank on the remaining images."))

    exchange = tables.get("exchangeability")
    if exchange is not None and not exchange.empty:
        ordered = exchange.sort_values("good_shift_auroc")
        _style()
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        y = np.arange(len(ordered))
        ax.barh(y, ordered["good_shift_auroc"] - 0.5, left=0.5, color=[group_color[g] for g in ordered["group"]], alpha=0.8)
        for yy, flag in zip(y, ordered["holm_p"] < 0.05):
            if flag:
                ax.text(ordered["good_shift_auroc"].iloc[int(yy)] + 0.01, yy, "*", va="center")
        ax.axvline(0.5, color="#555555", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels([c.replace("_", " ") for c in ordered["class_name"]], fontsize=8)
        ax.set_xlabel("AUROC of test/good versus train/good scores (0.5 = exchangeable)")
        ax.set_title("Train/test shift among good images")
        fig.tight_layout()
        _save(fig, figures, "figE4_good_shift")
        written.append(("figE4_good_shift", "Full training bank; train images scored leave-one-out. * Holm-adjusted Mann-Whitney p < 0.05."))

    rejected = tables.get("rejected_good")
    if data_root is not None and rejected is not None and not rejected.empty:
        top = rejected[rejected["rejection_rate_full_pool"] >= 0.5].sort_values(
            ["class_name", "rejection_rate_full_pool"], ascending=[True, False]
        )
        top = top.groupby("class_name").head(4)
        top = top[[(Path(data_root) / p).is_file() for p in top["path"]]]
        classes = list(dict.fromkeys(top["class_name"]))
        if classes:
            _style()
            fig, axes = plt.subplots(len(classes), 4, figsize=(9, 2.4 * len(classes)), squeeze=False)
            for row, class_name in enumerate(classes):
                items = top[top["class_name"] == class_name]
                for col in range(4):
                    ax = axes[row, col]
                    ax.axis("off")
                    if col < len(items):
                        item = items.iloc[col]
                        ax.imshow(_model_view(Path(data_root) / item["path"], cfg.resize, cfg.center_crop))
                        ax.set_title(f"{Path(item['path']).name}: {item['rejection_rate_full_pool']:.0%}", fontsize=7)
                axes[row, 0].text(-0.08, 0.5, class_name.replace("_", " "), transform=axes[row, 0].transAxes,
                                  rotation=90, va="center", ha="right", fontsize=9)
            fig.suptitle("Good test images rejected in at least half of the trajectories (full pool)")
            fig.tight_layout()
            _save(fig, figures, "figE5_rejected_good")
            written.append(("figE5_rejected_good", "Model view after resize and center crop. Percent: share of the 40 trajectories that reject the image."))
    return written


def _html(cfg: Config, tables: dict, written: list[tuple[str, str]], directory: Path) -> Path:
    from commissioning.report import _html_frame, _inline_figures

    sections = []
    for name, caption in written:
        sections.append(f'<h2>{html.escape(name)}</h2><img src="figures/{html.escape(name)}.png" alt="{html.escape(name)}"><p class="note">{html.escape(caption)}</p>')
    for title, key in (
        ("E5 Censoring decomposition", "censoring"),
        ("E3 Good-image exchangeability", "exchangeability"),
        ("E2 Best split per budget", "budget_best"),
        ("E1 Ceiling at n = 20", "ceiling_base"),
    ):
        frame = tables.get(key)
        if frame is not None and not frame.empty:
            sections.append(f"<h2>{html.escape(title)}</h2>{_html_frame(frame.drop(columns=['exploratory', 'config_hash'], errors='ignore'))}")
    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Exploratory analyses</title>
<style>
body {{ font-family: Georgia, serif; margin: 2rem auto; max-width: 980px; color: #1c1c1c; line-height: 1.45; }}
img {{ max-width: 100%; height: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.8rem; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 0.3rem 0.4rem; text-align: left; }}
.banner {{ background: #fff4e5; border: 1px solid #c47b2b; padding: 0.6rem 0.8rem; }}
.note {{ color: #444; font-size: 0.92rem; }}
</style>
</head>
<body>
<h1>Exploratory analyses (post hoc)</h1>
<p class="banner">Declared in exploratory_addendum.md after the confirmatory run. These results do not change N*, AULC, the stopping rule, or the primary tests. Config hash {html.escape(cfg.hash)}. Epsilon = {EPSILON}, delta = {DELTA}.</p>
{''.join(sections)}
</body>
</html>
"""
    path = directory / "index.html"
    path.write_text(_inline_figures(document, directory), encoding="utf-8")
    return path


# Entry point ------------------------------------------------------------------------


def explore(cfg: Config, classes: list[str]) -> Path:
    directory = out_dir(cfg)
    parts = directory / "parts"
    figures = directory / "figures"
    parts.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}

    tables["ceiling"] = ceiling_table(cfg)
    tables["calibration_size"] = calibration_size_table(cfg)
    tables["ceiling_base"] = tables["ceiling"][
        (tables["ceiling"]["rule"] == "conformal") & (tables["ceiling"]["n_calibration"] == cfg.n_calibration)
    ][["class_name", "n_test_good", "false_reject_budget", "p_rep_fpr_ok", "expected_passing_reps", "p_at_least_min_reps_if_independent"]]

    metrics_path = cfg.work_root / "analysis" / "metrics.parquet"
    if metrics_path.is_file():
        tables["censoring"] = censoring_table(cfg, pd.read_parquet(metrics_path))
    else:
        LOGGER.warning("analysis/metrics.parquet is missing; E5 is skipped.")

    budget_frames, exchange_rows, image_frames, rejection_frames = [], [], [], []
    for class_name in classes:
        budget_path = parts / f"{class_name}_budget.parquet"
        exchange_path = parts / f"{class_name}_exchange.json"
        images_path = parts / f"{class_name}_good_images.parquet"
        need_budget = not _part_is_current(cfg, budget_path)
        need_exchange = not (exchange_path.is_file() and images_path.is_file()
                             and read_json(exchange_path).get("config_hash") == cfg.hash)
        if need_budget or need_exchange:
            if not distance_cache_path(cfg, class_name).is_file() or not distance_meta_path(cfg, class_name).is_file():
                LOGGER.warning("%s: distance cache is missing; E2 and E3 are skipped for this class.", class_name)
            else:
                LOGGER.info("Exploratory E2/E3 for %s", class_name)
                cache = load_distance_cache(cfg, class_name)
                paths = list(cache["meta"]["paths"])
                n_train = int(cache["meta"]["n_train"])
                if need_budget:
                    atomic_parquet(budget_path, budget_split_class(cfg, class_name, cache["D"], paths, n_train))
                if need_exchange:
                    result = exchangeability_class(cfg, class_name, cache["D"], paths, n_train)
                    atomic_parquet(images_path, result["images"])
                    write_json(exchange_path, result["summary"])
                del cache
        if budget_path.is_file():
            budget_frames.append(pd.read_parquet(budget_path))
        if exchange_path.is_file():
            exchange_rows.append(read_json(exchange_path))
        if images_path.is_file():
            image_frames.append(pd.read_parquet(images_path))
        rates = rejection_rates(cfg, class_name)
        if not rates.empty:
            rejection_frames.append(rates)

    if budget_frames:
        frame = pd.concat(budget_frames, ignore_index=True)
        atomic_parquet(directory / "budget_split.parquet", frame)
        tables["budget_summary"], tables["budget_best"] = summarize_budget(frame, cfg)
    if exchange_rows:
        exchange = pd.DataFrame(exchange_rows)
        exchange["holm_p"] = holm(exchange["mannwhitney_p"].tolist())
        tables["exchangeability"] = exchange
    if rejection_frames:
        rejected = pd.concat(rejection_frames, ignore_index=True)
        if image_frames:
            rejected = rejected.merge(pd.concat(image_frames, ignore_index=True), on=["class_name", "path"], how="left")
        tables["rejected_good"] = rejected.sort_values(["class_name", "rejection_rate_full_pool"], ascending=[True, False])

    for name, frame in tables.items():
        frame.to_csv(directory / f"{name}.csv", index=False)
    data_root = cfg.data_root if Path(cfg.data_root).is_dir() else None
    written = _figures(cfg, tables, figures, data_root)
    path = _html(cfg, tables, written, directory)
    LOGGER.info("Exploratory report written to %s", path)
    return path
