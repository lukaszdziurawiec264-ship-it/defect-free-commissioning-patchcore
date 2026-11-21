"""Publication figures, tables, and a self-contained HTML report."""

from __future__ import annotations

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from commissioning.analysis import stop_category

RANDOM_COLOR = "#1f4e79"
KCENTER_COLOR = "#c47b2b"
STRATEGY_COLOR = {"random": RANDOM_COLOR, "kcenter": KCENTER_COLOR}

# |P| differs by class, so the full pool gets one shared position after the 1..64 grid.
GRID_TICKS = (1, 2, 4, 8, 16, 32, 64)
FULL_POOL_X = float(np.log2(GRID_TICKS[-1]) + 1.0)


def _x_position(n: int, pool_size: int) -> float:
    return FULL_POOL_X if int(n) >= int(pool_size) else float(np.log2(int(n)))


def _with_x(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = metrics.copy()
    frame["x"] = [_x_position(n, pool) for n, pool in zip(frame["N"], frame["pool_size"])]
    return frame


def _grid_axis(ax) -> None:
    ticks = [float(np.log2(n)) for n in GRID_TICKS] + [FULL_POOL_X]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(n) for n in GRID_TICKS] + ["|P|"])
    ax.axvline((ticks[-2] + FULL_POOL_X) / 2.0, color="#bbbbbb", lw=0.8, ls=":")
    ax.set_xlabel("N (log scale); |P| = full pool")


def _save(fig: plt.Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _band(metrics: pd.DataFrame, value: str, group: str) -> pd.DataFrame:
    subset = _with_x(metrics[metrics["group"] == group])
    rows = []
    if subset.empty:
        return pd.DataFrame(columns=["strategy", "x", "median", "q1", "q3", "minimum", "maximum"])
    for (strategy, x), block in subset.groupby(["strategy", "x"]):
        per_rep = block.groupby("rep")[value].median()
        q1, q3 = np.quantile(per_rep, [0.25, 0.75])
        rows.append(
            {
                "strategy": strategy,
                "x": float(x),
                "median": float(per_rep.median()),
                "q1": float(q1),
                "q3": float(q3),
                "minimum": float(per_rep.min()),
                "maximum": float(per_rep.max()),
            }
        )
    return pd.DataFrame(rows)


def _draw_series(ax, x, median, low, high, color: str, label: str | None, lw: float = 2.0) -> None:
    """Grid points as a line with a band; the full pool as a separate point with an interval bar."""
    x = np.asarray(x, dtype=float)
    median = np.asarray(median, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    order = np.argsort(x)
    x, median, low, high = x[order], median[order], low[order], high[order]
    grid = x < FULL_POOL_X
    ax.plot(x[grid], median[grid], color=color, lw=lw, label=label)
    ax.fill_between(x[grid], low[grid], high[grid], color=color, alpha=0.2)
    full = ~grid
    if full.any():
        ax.errorbar(
            x[full],
            median[full],
            yerr=[median[full] - low[full], high[full] - median[full]],
            fmt="o",
            color=color,
            ms=4,
            capsize=3,
            lw=1,
        )


def _draw_band(ax, band: pd.DataFrame, value_key: str = "median") -> None:
    for strategy, color in STRATEGY_COLOR.items():
        sub = band[band["strategy"] == strategy]
        if sub.empty:
            continue
        _draw_series(ax, sub["x"], sub[value_key], sub["q1"], sub["q3"], color, strategy)


def figure_protocol(directory: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.3, 1.6, "train/good"),
        (2.6, 2.5, "C: 20 good\nthreshold + stop"),
        (2.6, 0.6, "P: remaining good\nmemory-bank pool"),
        (6.0, 2.5, "threshold = max(C)"),
        (6.0, 0.6, "nested banks\nrandom or k-center"),
        (9.2, 1.6, "official test\nTPR and FPR"),
    ]
    for x, y, text in boxes:
        patch = FancyBboxPatch((x, y), 2.2, 1.0, boxstyle="round,pad=0.04,rounding_size=0.08",
                                facecolor="#f4f7fb", edgecolor="#1f4e79", lw=1.0)
        ax.add_patch(patch)
        ax.text(x + 1.1, y + 0.5, text, ha="center", va="center", fontsize=8)
    for start, end in [((2.5, 2.1), (2.6, 2.9)), ((2.5, 2.1), (2.6, 1.2)),
                       ((4.8, 3.0), (6.0, 3.0)), ((4.8, 1.1), (6.0, 1.1)),
                       ((8.2, 3.0), (9.6, 2.3)), ((8.2, 1.1), (9.6, 1.9))]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#1f4e79", "lw": 1.0})
    ax.set_title("Defect-free commissioning protocol")
    _save(fig, directory, "fig1_protocol")


def figure_curves(metrics: pd.DataFrame, directory: Path, value: str, name: str, ylabel: str, title: str, hline=None) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True)
    for ax, group, heading in zip(axes, ("object", "texture"), ("Objects", "Textures")):
        band = _band(metrics, value, group)
        _draw_band(ax, band)
        if hline is not None:
            ax.axhline(hline, color="#666666", ls="--", lw=1, label="1/21")
        ax.set_title(heading)
        _grid_axis(ax)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, directory, name)


def figure_class_bands(metrics: pd.DataFrame, directory: Path, value: str, name: str, ylabel: str, use_minmax: bool) -> None:
    _style()
    classes = list(dict.fromkeys(metrics["class_name"]))
    n = max(len(classes), 1)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.2), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, class_name in zip(axes, classes):
        subset = _with_x(metrics[metrics["class_name"] == class_name])
        for strategy, color in STRATEGY_COLOR.items():
            block = subset[subset["strategy"] == strategy]
            xs, med, low, high = [], [], [], []
            for x_value, part in block.groupby("x"):
                series = part[value]
                xs.append(float(x_value))
                med.append(float(series.median()))
                if use_minmax:
                    low.append(float(series.min()))
                    high.append(float(series.max()))
                else:
                    q1, q3 = np.quantile(series, [0.25, 0.75])
                    low.append(float(q1))
                    high.append(float(q3))
            if xs:
                _draw_series(ax, xs, med, low, high, color, None, lw=1.2)
        ax.set_title(class_name.replace("_", " "), fontsize=8)
        ticks = [0.0, 2.0, 4.0, 6.0, FULL_POOL_X]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["1", "4", "16", "64", "|P|"], fontsize=7)
    for ax in axes[len(classes):]:
        ax.axis("off")
    fig.supylabel(ylabel)
    fig.tight_layout()
    _save(fig, directory, name)


def figure_fpr_fraction(aggregate: pd.DataFrame, directory: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    frame = aggregate.copy()
    frame["pool_size"] = frame.groupby("class_name")["N"].transform("max")
    frame = _with_x(frame)
    for strategy, color in STRATEGY_COLOR.items():
        sub = frame[frame["strategy"] == strategy]
        rows = []
        for x_value, block in sub.groupby("x"):
            per_class = block.groupby("class_name")["fraction_fpr_above_05"].median()
            rows.append((float(x_value), float(per_class.median())))
        rows.sort()
        if not rows:
            continue
        grid = [(x, y) for x, y in rows if x < FULL_POOL_X]
        full = [(x, y) for x, y in rows if x >= FULL_POOL_X]
        ax.plot([x for x, _ in grid], [y for _, y in grid], color=color, lw=2, label=strategy)
        if full:
            ax.plot([x for x, _ in full], [y for _, y in full], "o", color=color, ms=5)
    ax.axhline(0.95**20, color="#666666", ls="--", lw=1, label="P(true FPR > 0.05) = 0.358")
    _grid_axis(ax)
    ax.set_ylabel("Fraction of repetitions with FPR > 0.05")
    ax.set_title("How often the realized FPR exceeds 0.05")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, directory, "fig3b_fpr_fraction")


def figure_stopping(summary: pd.DataFrame, stopping: pd.DataFrame, directory: Path) -> None:
    _style()
    main = stopping[np.isclose(stopping["delta"].astype(float), 0.05)].copy()
    merged = main.merge(
        summary[["class_name", "strategy", "n_star", "censored"]],
        on=["class_name", "strategy"],
        how="left",
    )
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    colors = {
        "premature": "#a33b3b",
        "concordant": "#2e7d4f",
        "excessive": "#c47b2b",
        "n_star_not_reached": "#555555",
        "rule_not_fired": "#8d6cab",
    }
    limit = 1.0
    for row in merged.itertuples(index=False):
        censored = bool(row.censored)
        n_stop_value = None if pd.isna(row.n_stop) else int(row.n_stop)
        n_star_value = None if pd.isna(row.n_star) else int(row.n_star)
        category = stop_category(n_stop_value, n_star_value, censored)
        x = row.pool_size if censored or n_star_value is None else n_star_value
        y = row.pool_size if n_stop_value is None else n_stop_value
        limit = max(limit, float(x), float(y))
        ax.scatter(x, y, c=colors[category], s=28, alpha=0.75, linewidths=0)
    upper = limit * 1.05
    ax.plot([1, upper], [1, upper], color="#888888", lw=1, ls="--")
    ax.set_xlabel("N* (pool size when right-censored)")
    ax.set_ylabel("N_stop")
    ax.set_title("Stopping rule against the observed commissioning threshold")
    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, label=name.replace("_", " "))
               for name, color in colors.items()]
    ax.legend(handles=handles, frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, directory, "fig5_nstop")


def figure_forest(statistics: dict, directory: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.2), sharey=False)
    for ax, key, title in zip(
        axes,
        ("aulc_tpr", "aulc_fpr"),
        ("AULC TPR, k-center − random", "AULC FPR, k-center − random"),
    ):
        pairs = statistics[key]["per_class"]
        names = list(pairs)
        values = [pairs[name] for name in names]
        y = np.arange(len(names))
        ax.axvline(0.0, color="#888888", lw=1)
        ax.scatter(values, y, color="#1f4e79", s=28)
        ax.set_yticks(y)
        ax.set_yticklabels([name.replace("_", " ") for name in names], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Paired difference")
    fig.tight_layout()
    _save(fig, directory, "figS1_forest")


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{tabular}{llrrrlr}",
        "\\toprule",
        "Class & Strategy & AULC TPR & AULC FPR & $N^*$ & Censored & $N_{\\mathrm{stop}}$ \\\\",
        "\\midrule",
    ]
    for row in summary.sort_values(["class_name", "strategy"]).itertuples(index=False):
        star = "$>|P|$" if bool(row.censored) else str(int(row.n_star))
        stop = "" if pd.isna(row.n_stop_median) else str(int(round(float(row.n_stop_median))))
        name = str(row.class_name).replace("_", "\\_")
        lines.append(
            f"{name} & {row.strategy} & {row.aulc_tpr:.3f} & {row.aulc_fpr:.3f} & {star} & {bool(row.censored)} & {stop} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _display_summary(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    frame["N*"] = [">|P|" if bool(flag) else str(int(value)) for flag, value in zip(frame["censored"], frame["n_star"])]
    frame["N_stop"] = ["" if pd.isna(value) else str(int(round(float(value)))) for value in frame["n_stop_median"]]
    keep = ["class_name", "group", "strategy", "aulc_tpr", "aulc_tpr_iqr", "aulc_fpr", "aulc_fpr_iqr", "N*",
            "censored", "N_stop", "excess_images_median", "fraction_premature", "fraction_concordant",
            "fraction_excessive", "fraction_n_star_not_reached", "fraction_rule_not_fired"]
    existing = [column for column in keep if column in frame.columns]
    return frame[existing]


def _html_frame(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, float_format=lambda value: f"{value:.3f}", border=0, na_rep="")


def figure_selection_examples(data_root: Path, work_root: Path, directory: Path) -> list[str]:
    """First eight random and k-center images for repetition 0. Skips classes whose files are absent."""
    from PIL import Image

    from commissioning.io_utils import read_json

    written = []
    for table in sorted((work_root / "runs" / "scores").glob("*_rep00_random.npz")):
        class_name = table.name.replace("_rep00_random.npz", "")
        other = work_root / "runs" / "scores" / f"{class_name}_rep00_kcenter.npz"
        meta_path = work_root / "cache" / "dist" / f"{class_name}.json"
        if not other.is_file() or not meta_path.is_file():
            continue
        paths = read_json(meta_path)["paths"]
        with np.load(table, allow_pickle=False) as data:
            first = int(data["order"][0])
        if not (data_root / paths[first]).is_file():
            continue
        _style()
        fig, axes = plt.subplots(2, 8, figsize=(12, 3.4))
        for row, npz_path in enumerate((table, other)):
            with np.load(npz_path, allow_pickle=False) as data:
                order = [int(i) for i in data["order"][:8]]
            for column, index in enumerate(order):
                ax = axes[row, column]
                image_path = data_root / paths[index]
                if image_path.is_file():
                    ax.imshow(Image.open(image_path).convert("RGB"))
                ax.axis("off")
            axes[row, 0].set_ylabel("random" if row == 0 else "k-center", fontsize=8)
        fig.suptitle(class_name.replace("_", " "))
        fig.tight_layout()
        name = f"figS4_selection_{class_name}"
        _save(fig, directory, name)
        written.append(name)
    return written


def _model_view(path: Path, resize: int, crop: int) -> np.ndarray:
    """The image region the backbone sees: resize to resize x resize, then center crop."""
    from PIL import Image

    image = Image.open(path).convert("RGB").resize((resize, resize), Image.BILINEAR)
    offset = (resize - crop) // 2
    return np.asarray(image.crop((offset, offset, offset + crop, offset + crop)))


def figure_heatmaps(data_root: Path, work_root: Path, directory: Path, resize: int = 256, crop: int = 224) -> list[str]:
    """One defect heatmap at N = 1, 8 and the full pool, random trajectory, repetition 0."""
    from PIL import Image

    from commissioning.io_utils import read_json
    from commissioning.patchcore import anomaly_map, patch_scores_for_bank

    written = []
    for table in sorted((work_root / "runs" / "scores").glob("*_rep00_random.npz")):
        class_name = table.name.replace("_rep00_random.npz", "")
        distance_path = work_root / "cache" / "dist" / f"{class_name}.npy"
        meta_path = work_root / "cache" / "dist" / f"{class_name}.json"
        if not distance_path.is_file() or not meta_path.is_file():
            continue
        meta = read_json(meta_path)
        with np.load(table, allow_pickle=False) as data:
            order = [int(i) for i in data["order"]]
            labels = [int(v) for v in data["test_labels"]]
            test_paths = [str(v) for v in data["test_paths"].tolist()]
        defects = [path for path, label in zip(test_paths, labels) if label == 1]
        if not defects or not (data_root / defects[0]).is_file():
            continue
        query = meta["paths"].index(defects[0])
        distance = np.load(distance_path, mmap_mode="r")
        chosen = [n for n in (1, 8, len(order)) if n <= len(order)]
        _style()
        fig, axes = plt.subplots(1, len(chosen), figsize=(3.2 * len(chosen), 3.2))
        axes = np.atleast_1d(axes)
        image = _model_view(data_root / defects[0], resize, crop)
        for ax, n in zip(axes, chosen):
            scores = patch_scores_for_bank(np.array(distance[query]), order[:n])
            ax.imshow(image)
            ax.imshow(anomaly_map(scores, image_size=crop), cmap="jet", alpha=0.45)
            ax.set_title(f"N = {n}", fontsize=9)
            ax.axis("off")
        fig.suptitle(class_name.replace("_", " "))
        fig.tight_layout()
        name = f"figS5_heatmap_{class_name}"
        _save(fig, directory, name)
        written.append(name)
        del distance
    return written


def build_report(
    frames: dict,
    report_dir: Path,
    gate: dict | None = None,
    data_root: Path | None = None,
    work_root: Path | None = None,
    resize: int = 256,
    crop: int = 224,
) -> Path:
    """Write figures, CSV, LaTeX, and report/index.html. Returns the HTML path."""
    report_dir = Path(report_dir)
    figures = report_dir / "figures"
    tables = report_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    metrics = frames["metrics"]
    summary = frames["summary"]
    aggregate = frames["fpr_aggregate"]
    stopping = frames["stopping"]
    statistics = frames["statistics"]
    budget = frames["budget"]

    figure_protocol(figures)
    figure_curves(metrics, figures, "tpr", "fig2_tpr", "TPR", "TPR versus N")
    figure_curves(metrics, figures, "fpr", "fig3_fpr", "FPR", "Realized FPR versus N", hline=1 / 21)
    figure_curves(metrics, figures, "oracle_fpr", "fig4_oracle", "Oracle FPR", "Oracle FPR at TPR = 0.95")
    figure_fpr_fraction(aggregate, figures)
    figure_stopping(summary, stopping, figures)
    figure_forest(statistics, figures)
    figure_class_bands(metrics, figures, "tpr", "figS2_tpr_iqr", "TPR", use_minmax=False)
    figure_class_bands(metrics, figures, "tpr", "figS3_tpr_minmax", "TPR", use_minmax=True)
    selection_names: list[str] = []
    heatmap_names: list[str] = []
    if data_root is not None and work_root is not None:
        selection_names = figure_selection_examples(Path(data_root), Path(work_root), figures)
        heatmap_names = figure_heatmaps(Path(data_root), Path(work_root), figures, resize=resize, crop=crop)

    summary.to_csv(tables / "table1_summary.csv", index=False)
    aggregate.to_csv(tables / "fpr_aggregate.csv", index=False)
    stopping.to_csv(tables / "stopping.csv", index=False)
    budget.to_csv(tables / "false_reject_budget.csv", index=False)
    write_latex_table(summary, tables / "table1_summary.tex")

    exploratory = bool(metrics["exploratory"].astype(bool).any()) if "exploratory" in metrics.columns else False
    gate_status = "not supplied" if gate is None else str(gate.get("status"))
    banner = ""
    if exploratory or gate_status not in {"PASS", "not supplied"}:
        banner = "<p class='banner'>This report contains exploratory output or a gate that is not PASS.</p>"
    tpr = statistics["aulc_tpr"]
    fpr = statistics["aulc_fpr"]
    practical = "passed" if statistics["practical_criterion_passed"] else "not passed"
    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Defect-free commissioning report</title>
<style>
body {{ font-family: Georgia, serif; margin: 2rem auto; max-width: 980px; color: #1c1c1c; line-height: 1.45; }}
img {{ max-width: 100%; height: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 0.3rem 0.4rem; text-align: left; }}
.banner {{ background: #fff4e5; border: 1px solid #c47b2b; padding: 0.6rem 0.8rem; }}
caption, .note {{ color: #444; font-size: 0.92rem; }}
</style>
</head>
<body>
<h1>How Many Good Parts Are Enough?</h1>
<p>Defect-free commissioning of memory-bank anomaly detection. The memory bank and the threshold use only good images. Test labels are not used to choose the threshold, the bank, or the stopping time. N* is an observed commissioning threshold, not a certified sample size.</p>
{banner}
<p>E0 gate: {html.escape(gate_status)}. Practical k-center criterion: {practical}. Difference is k-center minus random.</p>
<h2>Gate and primary comparisons</h2>
<ul>
<li>AULC TPR Wilcoxon p = {tpr['wilcoxon_p']:.4g}, Holm p = {tpr['holm_p']:.4g}, sign-flip p = {tpr['permutation_p']:.4g}. Median difference {tpr['bootstrap']['median']:.4f} (bootstrap 95% interval {tpr['bootstrap']['ci95'][0]:.4f} to {tpr['bootstrap']['ci95'][1]:.4f}). k-center higher on {tpr['n_kcenter_better']} classes.</li>
<li>AULC FPR Wilcoxon p = {fpr['wilcoxon_p']:.4g}, Holm p = {fpr['holm_p']:.4g}, sign-flip p = {fpr['permutation_p']:.4g}. Median difference {fpr['bootstrap']['median']:.4f} (bootstrap 95% interval {fpr['bootstrap']['ci95'][0]:.4f} to {fpr['bootstrap']['ci95'][1]:.4f}). k-center lower on {fpr['n_kcenter_better']} classes.</li>
</ul>
<p class="note">The bootstrap resamples 15 classes. It is an uncertainty display, not a strong inferential claim. Absence of a significant Wilcoxon result is not evidence that the strategies are equivalent.</p>
<h2>Figure 1. Protocol</h2>
<img src="figures/fig1_protocol.png" alt="Protocol diagram">
<p class="note">C is held out of the memory bank. P is the only source of bank images. The same C and P are used for both strategies inside one repetition.</p>
<h2>Figure 2. TPR</h2>
<img src="figures/fig2_tpr.png" alt="TPR versus N">
<p class="note">Line: median across repetitions of the cross-category median. Band: interquartile range of those repetition-level medians. The full pool |P| differs by category, so it is drawn as a separate point with an interquartile bar after the shared 1–64 grid. Toothbrush has |P| = 40 and no N = 64 point. Per-category bands are in the supplement.</p>
<h2>Figure 3. Realized FPR</h2>
<img src="figures/fig3_fpr.png" alt="FPR versus N">
<img src="figures/fig3b_fpr_fraction.png" alt="Fraction of repetitions above 0.05">
<p class="note">The dashed line in the first panel is the marginal level 1/21. The dashed line in the second panel is 0.95^20 ≈ 0.358, the probability that the true conditional FPR exceeds 0.05 when the calibration scores are exchangeable with a new good image. Repetitions share one test set, so their false-reject counts are not pooled into one Clopper-Pearson interval. The beta-binomial column in the budget table is exploratory.</p>
<h2>Figure 4. Oracle FPR at TPR = 0.95</h2>
<img src="figures/fig4_oracle.png" alt="Oracle FPR">
<p class="note">The oracle uses test labels. It is not an operational threshold.</p>
<h2>Figure 5. Stopping</h2>
<img src="figures/fig5_nstop.png" alt="Stopping versus N star">
<p class="note">N_stop ends the second consecutive grid step whose relative change in the median calibration score is below 5%. Right-censored classes are drawn at the pool size. A negative result for this rule does not invalidate RQ1–RQ3. The rule uses the same calibration images as the threshold.</p>
<h2>Supplement. Paired differences</h2>
<img src="figures/figS1_forest.png" alt="Forest plot">
<h2>Table 1</h2>
{_html_frame(_display_summary(summary))}
<h2>False-reject budget on the official good test images</h2>
{_html_frame(budget)}
<h2>Per-category TPR bands</h2>
<img src="figures/figS2_tpr_iqr.png" alt="Per class IQR">
<img src="figures/figS3_tpr_minmax.png" alt="Per class min max">
<h2>Selected images and heatmaps</h2>
{''.join(f'<img src="figures/{html.escape(name)}.png" alt="{html.escape(name)}">' for name in selection_names + heatmap_names)}
<p class="note">Selection sheets show the first eight images of each nested trajectory at repetition 0. Heatmaps use the random trajectory and one defect image. They are illustrations, not a tuning set.</p>
</body>
</html>
"""
    path = report_dir / "index.html"
    path.write_text(_inline_figures(document, report_dir), encoding="utf-8")
    return path


def _inline_figures(document: str, report_dir: Path) -> str:
    """Embed every figures/*.png as a data URI so index.html can be opened on its own."""
    import base64
    import re

    def replace(match: re.Match) -> str:
        relative = html.unescape(match.group(1))
        figure = report_dir / relative
        if not figure.is_file():
            raise RuntimeError(f"Report references a missing figure: {figure}")
        encoded = base64.b64encode(figure.read_bytes()).decode("ascii")
        return f'src="data:image/png;base64,{encoded}"'

    return re.sub(r'src="(figures/[^"]+\.png)"', replace, document)
