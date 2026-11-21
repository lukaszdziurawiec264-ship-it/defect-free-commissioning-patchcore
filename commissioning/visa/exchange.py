"""Section 9: are official good test scores exchangeable with held-out good scores? Bank = all of U."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, mannwhitneyu
from sklearn.metrics import roc_auc_score

from commissioning.io_utils import read_json, write_json
from commissioning.patchcore import scores_for_bank
from commissioning.stats import holm
from commissioning.visa.caches import load_distances
from commissioning.visa.config import VisaConfig
from commissioning.visa.data import load_split, row_blocks

LOGGER = logging.getLogger(__name__)


def part_path(cfg: VisaConfig, class_name: str) -> Path:
    return cfg.work_root / "exchange" / f"{class_name}.json"


def exchange_scores(distance: np.ndarray, split: dict) -> dict[str, np.ndarray]:
    blocks = row_blocks(split)
    bank = blocks["universe"]
    return {name: scores_for_bank(distance, blocks[name], bank) for name in ("holdout", "test_good", "test_defect")}


def exchange_summary(cfg: VisaConfig, class_name: str, scores: dict[str, np.ndarray]) -> dict:
    holdout, good, defect = scores["holdout"], scores["test_good"], scores["test_defect"]
    labels = np.r_[np.zeros(len(holdout)), np.ones(len(good))]
    conformal_p = (1.0 + (holdout[None, :] >= good[:, None]).sum(axis=1)) / (len(holdout) + 1.0)
    n_low = int(np.sum(conformal_p <= cfg.alpha))
    return {
        "class_name": class_name,
        "group": cfg.group_of(class_name),
        "n_holdout": int(len(holdout)),
        "n_test_good": int(len(good)),
        "good_shift_auroc": float(roc_auc_score(labels, np.r_[holdout, good])),
        "mannwhitney_p": float(mannwhitneyu(good, holdout, alternative="two-sided").pvalue),
        "median_holdout_score": float(np.median(holdout)),
        "median_test_good_score": float(np.median(good)),
        "fraction_conformal_p_below_alpha": n_low / len(good),
        "binomial_tail_p": float(binom.sf(n_low - 1, len(good), cfg.alpha)),
        "defect_auroc_bank_u": float(
            roc_auc_score(np.r_[np.zeros(len(good)), np.ones(len(defect))], np.r_[good, defect])
        ),
        "config_hash": cfg.hash,
    }


def run_exchange(cfg: VisaConfig, classes: list[str]) -> pd.DataFrame:
    rows = []
    for class_name in classes:
        path = part_path(cfg, class_name)
        if path.is_file() and read_json(path).get("config_hash") == cfg.hash:
            rows.append(read_json(path))
            continue
        LOGGER.info("Exchangeability %s", class_name)
        split = load_split(cfg, class_name, 0)
        distance = load_distances(cfg, class_name)["D"]
        summary = exchange_summary(cfg, class_name, exchange_scores(distance, split))
        del distance
        write_json(path, summary)
        rows.append(summary)
    return finalize(rows)


def finalize(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).sort_values("class_name").reset_index(drop=True)
    frame["holm_p"] = holm(frame["mannwhitney_p"].tolist())
    frame["shift_detected"] = frame["holm_p"] < 0.05
    return frame


def load_exchange(cfg: VisaConfig) -> pd.DataFrame | None:
    rows = []
    for class_name in cfg.classes:
        path = part_path(cfg, class_name)
        if not path.is_file():
            return None
        payload = read_json(path)
        if payload.get("config_hash") != cfg.hash:
            raise RuntimeError(f"{path} was written for another config hash.")
        rows.append(payload)
    return finalize(rows)
