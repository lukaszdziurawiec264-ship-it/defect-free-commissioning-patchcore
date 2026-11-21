"""Figures, tables, and a self-contained HTML report for the VisA replication."""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from commissioning.report import KCENTER_COLOR, RANDOM_COLOR, STRATEGY_COLOR, _html_frame, _inline_figures, _save, _style

RULE_STYLE = {"R20": "-", "R60": "--"}
GROUP_COLOR = {"multiple": RANDOM_COLOR, "single": KCENTER_COLOR, "complex": "#2e7d4f"}
THEORY = {"R20": 1 / 21, "R60": 1 / 61}


def _x(frame: pd.DataFrame) -> np.ndarray:
    return np.log2(frame["N"].to_numpy(dtype=float))


def _ticks(ax, grid: list[int]) -> None:
    ax.set_xticks(np.log2(grid))
    ax.set_xticklabels([str(n) for n in grid])
    ax.set_xlabel("N (log scale); last point = |P|")


def figure_tpr(metrics: pd.DataFrame, directory: Path) -> None:
    grid = sorted(metrics["N"].unique())
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, rule in zip(axes, ("R20", "R60")):
        for strategy, color in STRATEGY_COLOR.items():
            sub = metrics[(metrics["rule"] == rule) & (metrics["strategy"] == strategy)]
            per_class = sub.groupby(["N", "class_name"])["tpr"].mean().reset_index()
            band = per_class.groupby("N")["tpr"].agg(
                median="median", q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75)
            ).reset_index()
            x = _x(band)
            ax.plot(x, band["median"], color=color, lw=2, label=strategy)
            ax.fill_between(x, band["q1"], band["q3"], color=color, alpha=0.18, lw=0)
        ax.set_title(f"TPR versus N, threshold {rule}")
        _ticks(ax, grid)
    axes[0].set_ylabel("TPR (median over classes, IQR band)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    _save(fig, directory, "figV1_tpr")


def figure_fpr(metrics: pd.DataFrame, directory: Path) -> None:
    grid = sorted(metrics["N"].unique())
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (column, title) in zip(axes, (("fpr_h", "held-out good H"), ("fpr_off", "official good test"))):
        for rule in ("R20", "R60"):
            sub = metrics[metrics["rule"] == rule].groupby("N")[column].mean().reset_index()
            ax.plot(_x(sub), sub[column], color="#333333", ls=RULE_STYLE[rule], lw=1.8, label=f"{rule}, mean over classes")
            ax.axhline(THEORY[rule], color="#a33b3b", ls=RULE_STYLE[rule], lw=0.9, label=f"theory 1/{round(1 / THEORY[rule])}")
        ax.set_title(f"Realized FPR on {title}")
        _ticks(ax, grid)
    axes[0].set_ylabel("FPR")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, directory, "figV2_fpr")


def figure_tpr_cost(statistics: dict, directory: Path) -> None:
    per_class = pd.Series(statistics["H3"]["per_class"]).sort_values()
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.barh(np.arange(len(per_class)), per_class.to_numpy(), color=RANDOM_COLOR, alpha=0.85)
    ax.axvline(statistics["H3"]["margin"], color="#a33b3b", ls="--", lw=1, label="practical margin")
    ax.axvline(statistics["H3"]["median"], color="#333333", lw=1, label="median over classes")
    ax.set_yticks(np.arange(len(per_class)))
    ax.set_yticklabels(per_class.index, fontsize=8)
    ax.set_xlabel("TPR(R20) - TPR(R60) at N = |P|")
    ax.set_title("H3: TPR cost of 60 instead of 20 calibration images")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, directory, "figV3_tpr_cost")


def figure_forest(statistics: dict, directory: Path) -> None:
    primary = statistics["primary"]["H1_H2_R20_official"]
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, key, label in zip(axes, ("aulc_tpr", "aulc_fpr"), ("ΔAULC_TPR", "ΔAULC_FPR (official)")):
        values = pd.Series(primary[key]["per_class"]).sort_index()
        ax.scatter(values.to_numpy(), np.arange(len(values)), color=KCENTER_COLOR, s=22)
        ax.axvline(0, color="#555555", lw=1)
        ax.set_title(f"{label}, k-center − random")
        ax.set_yticks(np.arange(len(values)))
        ax.set_yticklabels(values.index, fontsize=8)
    fig.tight_layout()
    _save(fig, directory, "figV4_forest")


def figure_shift(p3: pd.DataFrame | None, directory: Path) -> bool:
    if p3 is None or p3.empty:
        return False
    ordered = p3.sort_values("good_shift_auroc")
    _style()
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    y = np.arange(len(ordered))
    ax.barh(y, ordered["good_shift_auroc"] - 0.5, left=0.5, color=[GROUP_COLOR[g] for g in ordered["group"]], alpha=0.85)
    for yy, (auroc, flag) in enumerate(zip(ordered["good_shift_auroc"], ordered["shift_detected"])):
        if flag:
            ax.text(auroc + 0.01, yy, "*", va="center")
    ax.axvline(0.5, color="#555555", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["class_name"], fontsize=8)
    ax.set_xlabel("AUROC of official good versus held-out good scores (0.5 = exchangeable)")
    ax.set_title("Shift among good images; * Holm p < 0.05")
    fig.tight_layout()
    _save(fig, directory, "figV5_shift")
    return True


def figure_budget(best: pd.DataFrame | None, summary_path: Path, directory: Path, min_passes: int) -> bool:
    if not summary_path.is_file():
        return False
    summary = pd.read_csv(summary_path)
    sub = summary[summary["rule"] == "pac"]
    budgets = sorted(sub["budget"].unique())
    _style()
    fig, axes = plt.subplots(1, len(budgets), figsize=(3.2 * len(budgets), 3.4), sharey=True, squeeze=False)
    for ax, budget in zip(axes[0], budgets):
        part = sub[sub["budget"] == budget]
        for _class_name, line in part.groupby("class_name"):
            line = line.sort_values("N")
            ax.plot(np.log2(line["N"]), line["passing_off"], color=GROUP_COLOR[line["group"].iloc[0]], lw=1, alpha=0.85)
        ax.axhline(min_passes, color="#a33b3b", ls="--", lw=1)
        ax.set_title(f"B = {budget}", fontsize=9)
        ax.set_xlabel("log2 N; calibration = B − N")
    axes[0, 0].set_ylabel("Passing repetitions of 20 (official)")
    fig.suptitle("Budget split, PAC threshold. Blue multiple, orange single, green PCB")
    fig.tight_layout()
    _save(fig, directory, "figV6_budget")
    return True


def write_latex(frame: pd.DataFrame, path: Path) -> None:
    def cell(value) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
        return str(value).replace("_", "\\_").replace("Δ", "$\\Delta$")

    lines = ["\\begin{tabular}{" + "l" * len(frame.columns) + "}", "\\toprule"]
    lines.append(" & ".join(cell(c) for c in frame.columns) + " \\\\")
    lines.append("\\midrule")
    lines.extend(" & ".join(cell(v) for v in row) + " \\\\" for row in frame.itertuples(index=False))
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _statistics_table(statistics: dict) -> pd.DataFrame:
    rows = []
    blocks = {"primary R20 official": statistics["primary"]["H1_H2_R20_official"]}
    blocks.update({f"secondary {k}": v for k, v in statistics["secondary"].items()})
    for name, block in blocks.items():
        for metric in ("aulc_tpr", "aulc_fpr"):
            part = block[metric]
            rows.append(
                {
                    "comparison": name,
                    "metric": metric,
                    "median Δ": part["bootstrap"]["median"],
                    "CI95 low": part["bootstrap"]["ci95"][0],
                    "CI95 high": part["bootstrap"]["ci95"][1],
                    "Wilcoxon p": part["wilcoxon_p"],
                    "Holm p": part["holm_p"],
                    "k-center better": part["n_kcenter_better"],
                    "random better": part["n_random_better"],
                    "practical rule": block["practical_criterion_passed"],
                }
            )
    return pd.DataFrame(rows)


def build_report(cfg, frames: dict, gate: dict | None) -> Path:
    directory = cfg.work_root / "report"
    figures = directory / "figures"
    tables = directory / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics, statistics = frames["metrics"], frames["statistics"]
    figure_tpr(metrics, figures)
    figure_fpr(metrics, figures)
    figure_tpr_cost(statistics, figures)
    figure_forest(statistics, figures)
    written = ["figV1_tpr", "figV2_fpr", "figV3_tpr_cost", "figV4_forest"]
    if figure_shift(frames.get("p3_shift"), figures):
        written.append("figV5_shift")
    if figure_budget(frames.get("budget_best"), cfg.work_root / "analysis" / "budget_summary.csv", figures, cfg.n_star_min_repeats):
        written.append("figV6_budget")
    stats_table = _statistics_table(statistics)
    stats_table.to_csv(tables / "statistics.csv", index=False)
    write_latex(stats_table, tables / "statistics.tex")
    sections = [f'<h2>{name}</h2><img src="figures/{name}.png" alt="{name}">' for name in written]
    for title, key in (
        ("N* by rule and FPR source", "n_star"),
        ("P2 calibration on H", "p2_calibration"),
        ("P3 shift", "p3_shift"),
        ("Feasibility ceiling", "ceiling"),
        ("Stopping rule", "stopping_summary"),
        ("Budget split, best N", "budget_best"),
    ):
        frame = frames.get(key)
        if frame is not None and not frame.empty:
            frame.to_csv(tables / f"{key}.csv", index=False)
            sections.append(f"<h2>{html.escape(title)}</h2>{_html_frame(frame.drop(columns=['config_hash'], errors='ignore'))}")
    gate_text = "missing" if gate is None else f"{gate.get('status')} (E0a mean AUROC {gate.get('e0a', {}) and gate['e0a'].get('mean_auroc')})"
    h3 = statistics["H3"]
    p2 = statistics["P2"]
    document = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>VisA replication</title>
<style>
body {{ font-family: Georgia, serif; margin: 2rem auto; max-width: 1000px; color: #1c1c1c; line-height: 1.45; }}
img {{ max-width: 100%; height: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.8rem; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 0.3rem 0.4rem; text-align: left; }}
.banner {{ background: #eef4fb; border: 1px solid #1f4e79; padding: 0.6rem 0.8rem; }}
</style></head><body>
<h1>Confirmatory replication on VisA (preregistration_visa.md)</h1>
<p class="banner">Config hash {html.escape(cfg.hash)}. E0 gate: {html.escape(str(gate_text))}. Exploratory rows: {statistics.get('exploratory_rows')}.</p>
<h2>Primary and secondary comparisons (k-center − random)</h2>{_html_frame(stats_table)}
<h2>H3</h2><p>Median TPR cost of R60 at N = |P|: {h3['median']:.4f} (CI95 {h3['bootstrap']['ci95'][0]:.4f} to {h3['bootstrap']['ci95'][1]:.4f}); margin {h3['margin']}; R60 recommended: {h3['r60_recommended']}.</p>
<h2>P2</h2><pre>{html.escape(json.dumps({k: v for k, v in p2.items() if k != 'by_N'}, indent=2))}</pre>
<h2>P1 and P3</h2><pre>{html.escape(json.dumps({'P1': statistics['P1'], 'P3': statistics['P3'], 'budget': statistics.get('budget')}, indent=2))}</pre>
{''.join(sections)}
</body></html>
"""
    path = directory / "index.html"
    path.write_text(_inline_figures(document, directory), encoding="utf-8")
    return path
