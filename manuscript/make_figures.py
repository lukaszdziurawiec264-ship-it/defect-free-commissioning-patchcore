"""Publication figures from the frozen MVTec and VisA analysis files.

Run from the repository root: python manuscript/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from scipy.stats import betabinom

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figures"
MVTEC = ROOT / "results" / "mvtec"
VISA = ROOT / "results" / "visa"

NAVY = "#1f4e79"
ORANGE = "#c47b2b"
RED = "#a33b3b"
GREEN = "#2e7d4f"
INK = "#1c1c1c"

MAIN_N = [1, 2, 4, 8, 16, 32, 64]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _log_axis(ax, ticks: list[int]) -> None:
    ax.set_xticks([np.log2(n) for n in ticks])
    ax.set_xticklabels([str(n) for n in ticks])
    ax.set_xlabel("Images in the memory bank, N")


def _band(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    grouped = frame.groupby(["group", "strategy", "N", "class_name"], sort=True)[value].mean().reset_index()
    rows = []
    for keys, sub in grouped.groupby(["group", "strategy", "N"], sort=True):
        rows.append(
            {
                "group": keys[0],
                "strategy": keys[1],
                "N": int(keys[2]),
                "median": float(sub[value].median()),
                "mean": float(sub[value].mean()),
                "q1": float(sub[value].quantile(0.25)),
                "q3": float(sub[value].quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def figure_protocol() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(10.8, 4.15))
    ax.set_xlim(0, 124)
    ax.set_ylim(0, 44)
    ax.axis("off")

    def box(x, y, w, h, text, face):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.6",
            facecolor=face, edgecolor=INK, linewidth=0.6,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2, color="white")

    box(2, 16, 24, 12, "Good training\nimages (benchmark)", NAVY)
    box(32, 28, 22, 10, "Calibration C\nnot in the bank", ORANGE)
    box(32, 6, 16, 10, "Pool P", NAVY)
    box(60, 28, 20, 10, "Threshold\nmax of C", RED)
    box(54, 6, 24, 10, "Selection\nrandom or farthest-first", GREEN)
    box(84, 6, 18, 10, "Memory bank\nsize N", NAVY)
    box(86, 28, 34, 10, "Official test\nevaluation only", "#555555")
    ax.annotate("", xy=(32, 33), xytext=(26, 24), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8))
    ax.annotate("", xy=(32, 11), xytext=(26, 20), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8))
    ax.annotate("", xy=(60, 33), xytext=(54, 33), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8))
    ax.annotate("", xy=(54, 11), xytext=(48, 11), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8))
    ax.annotate("", xy=(84, 11), xytext=(78, 11), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8))
    ax.annotate("", xy=(86, 33), xytext=(80, 33), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8))
    ax.annotate("", xy=(100, 28), xytext=(96, 16), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8))
    ax.text(62, 1.6, "Calibration C does not enter the memory bank. On VisA, hold-out H never sets the threshold.", fontsize=7.4, ha="center")
    fig.tight_layout()
    _save(fig, "fig1_protocol")


def figure_mvtec_curves(metrics: pd.DataFrame) -> None:
    frame = metrics[metrics["N"].isin(MAIN_N)].copy()
    tpr = _band(frame, "tpr")
    fpr = _band(frame, "fpr")
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.6), sharex=True)
    panels = [
        (axes[0, 0], tpr, "object", "median", "True-positive rate", "MVTec objects, detection"),
        (axes[0, 1], tpr, "texture", "median", "True-positive rate", "MVTec textures, detection"),
        (axes[1, 0], fpr, "object", "mean", "False-positive rate", "MVTec objects, false rejects"),
        (axes[1, 1], fpr, "texture", "mean", "False-positive rate", "MVTec textures, false rejects"),
    ]
    for ax, table, group, column, ylabel, title in panels:
        for strategy, color in (("random", NAVY), ("kcenter", ORANGE)):
            sub = table[(table["group"] == group) & (table["strategy"] == strategy)].sort_values("N")
            x = np.log2(sub["N"])
            label = "farthest-first" if strategy == "kcenter" else "random"
            ax.plot(x, sub[column], color=color, lw=1.8, label=label)
            ax.fill_between(x, sub["q1"], sub["q3"], color=color, alpha=0.15, lw=0)
        if "True" in ylabel:
            ax.set_ylim(0.40, 1.02)
        elif group == "object":
            ax.set_ylim(0.0, 0.12)
        else:
            ax.set_ylim(0.0, 0.30)
        if "False" in ylabel:
            ax.axhline(1 / 21, color=RED, ls="--", lw=1, label="1/21")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel)
        _log_axis(ax, MAIN_N)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="lower right")
    axes[1, 0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig2_mvtec_curves")


def figure_shift(exchange: pd.DataFrame) -> None:
    ordered = exchange.sort_values("good_shift_auroc")
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    y = np.arange(len(ordered))
    colors = [ORANGE if g == "texture" else NAVY for g in ordered["group"]]
    ax.barh(y, ordered["good_shift_auroc"] - 0.5, left=0.5, color=colors, alpha=0.9)
    for yy, (auroc, flag) in enumerate(zip(ordered["good_shift_auroc"], ordered["holm_p"] < 0.05)):
        if flag:
            ax.text(auroc + 0.008, yy, "*", va="center", fontsize=11)
    ax.axvline(0.5, color="#555555", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in ordered["class_name"]], fontsize=8)
    ax.set_xlabel("Separation of good test images from good training images (0.5 = no shift)")
    ax.set_xlim(0.35, 1.0)
    fig.tight_layout()
    _save(fig, "fig3_mvtec_shift")


def figure_visa_fpr(metrics: pd.DataFrame) -> None:
    frame = metrics.copy()
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    for rule, color, theory, label in (
        ("R20", NAVY, 1 / 21, "maximum of 20"),
        ("R60", ORANGE, 1 / 61, "maximum of 60"),
    ):
        sub = frame[frame["rule"] == rule]
        per_n = sub.groupby("N")["fpr_h"].mean()
        axes[0].plot(np.log2(per_n.index), per_n.values, color=color, lw=1.8, label=label)
        axes[0].axhline(theory, color=color, ls="--", lw=0.9)
        frac = sub.groupby("N")["fpr_h"].apply(lambda s: float(np.mean(s > 0.05)))
        axes[1].plot(np.log2(frac.index), frac.values, color=color, lw=1.8, label=label)
    axes[1].axhline(0.95**20, color=NAVY, ls="--", lw=0.9)
    axes[1].axhline(0.95**60, color=ORANGE, ls="--", lw=0.9)
    axes[0].set_ylabel("False-positive rate on held-out good images")
    axes[1].set_ylabel("Share of runs with rate above 0.05")
    for ax in axes:
        _log_axis(ax, [1, 2, 4, 8, 16, 32, 64, 128, 240])
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_title("Mean rate")
    axes[1].set_title("How often the rate exceeds 5%")
    fig.tight_layout()
    _save(fig, "fig4_visa_fpr")


def figure_tpr_cost(statistics: dict) -> None:
    series = pd.Series(statistics["H3"]["per_class"]).sort_values()
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = [GREEN if v <= 0.03 else NAVY for v in series.to_numpy()]
    ax.barh(np.arange(len(series)), series.to_numpy(), color=colors, alpha=0.9)
    ax.axvline(0.03, color=RED, ls="--", lw=1, label="margin fixed in advance, 0.03")
    ax.axvline(float(statistics["H3"]["median"]), color=INK, lw=1, label="median across classes")
    ax.set_yticks(np.arange(len(series)))
    ax.set_yticklabels([name.replace("_", " ") for name in series.index], fontsize=8)
    ax.set_xlabel("Loss in true-positive rate when the threshold uses 60 images instead of 20")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig5_visa_cost")


def figure_forest(mvtec: dict, visa: dict) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), sharex=False)
    for ax, block, title in (
        (axes[0], mvtec["aulc_tpr"], "MVTec, confirmatory"),
        (axes[1], visa["primary"]["H1_H2_R20_official"]["aulc_tpr"], "VisA, confirmatory"),
    ):
        series = pd.Series(block["per_class"]).sort_values()
        colors = [ORANGE if v > 0 else NAVY for v in series.to_numpy()]
        ax.barh(np.arange(len(series)), series.to_numpy(), color=colors, alpha=0.9)
        ax.axvline(0, color=INK, lw=1)
        ax.set_yticks(np.arange(len(series)))
        ax.set_yticklabels([c.replace("_", " ") for c in series.index], fontsize=7)
        ax.set_xlabel("farthest-first minus random, area under the detection curve")
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    _save(fig, "fig6_kcenter")


def figure_ceiling() -> None:
    """Analytic feasibility of 16/20 passes when the threshold is the maximum of 20 good scores."""
    counts = {
        "bottle": 20, "grid": 21, "metal nut": 22, "capsule": 23, "hazelnut": 40,
        "screw": 41, "transistor": 60, "pill": 26, "carpet": 28, "zipper": 32,
        "leather": 32, "toothbrush": 12, "tile": 33, "cable": 58, "wood": 19,
    }
    rows = []
    for name, n_good in counts.items():
        budget = (5 * n_good) // 100
        prob = float(betabinom.cdf(budget, n_good, 1, 20))
        rows.append((name, n_good, 20 * prob))
    rows.sort(key=lambda item: item[2])
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = np.arange(len(rows))
    ax.barh(y, [item[2] for item in rows], color=NAVY, alpha=0.85)
    ax.axvline(16, color=RED, ls="--", lw=1, label="16 of 20, the N* rule")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{name} (n={n})" for name, n, _ in rows], fontsize=7)
    ax.set_xlabel("Expected runs, of 20, inside the false-reject budget")
    ax.set_xlim(0, 20)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    _save(fig, "figA1_ceiling")


def main() -> None:
    mvtec_metrics = pd.read_parquet(MVTEC / "analysis" / "metrics.parquet")
    exchange = pd.read_csv(MVTEC / "exploratory" / "exchangeability.csv")
    mvtec_stats = json.loads((MVTEC / "analysis" / "statistics.json").read_text(encoding="utf-8"))
    visa_metrics = pd.read_parquet(VISA / "analysis" / "metrics.parquet")
    visa_stats = json.loads((VISA / "analysis" / "statistics.json").read_text(encoding="utf-8"))
    figure_protocol()
    figure_mvtec_curves(mvtec_metrics)
    figure_shift(exchange)
    figure_visa_fpr(visa_metrics)
    figure_tpr_cost(visa_stats)
    figure_forest(mvtec_stats, visa_stats)
    figure_ceiling()
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
