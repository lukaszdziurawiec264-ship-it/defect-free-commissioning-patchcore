"""Unit tests for the VisA replication. They need neither VisA nor a GPU."""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from commissioning.config import load_config
from commissioning.distances import distance_tensor
from commissioning.io_utils import write_json
from commissioning.patchcore import scores_for_bank
from commissioning.visa import data as visa_data
from commissioning.visa.config import VisaConfig, load_visa_config


def toy_config(tmp_path, classes=("candle",), **overrides) -> VisaConfig:
    base = load_visa_config()
    payload = copy.deepcopy(base.payload)
    payload.update(
        {
            "classes": list(classes),
            "universe_size": 40,
            "n_calibration_small": 4,
            "n_calibration_large": 8,
            "n_repeats": 2,
            "n_grid": [1, 2, 4, 8, 16],
            "budgets": [20, 30, 40],
            "budget_grid": [1, 2, 4, 8, 16],
            "budget_min_calibration": 4,
            "bootstrap_replicates": 200,
            "n_star_min_repeats": 1,
        }
    )
    payload.update(overrides)
    root = tmp_path if tmp_path is not None else Path("unused")
    return VisaConfig(payload, base.config_path, root / "data", root / "work", root / "cache")


def toy_checked(n_train: int = 55, n_good: int = 6, n_defect: int = 5, name: str = "candle") -> dict:
    return {
        "train": [f"{name}/train/good/{i:03d}.JPG" for i in range(n_train)],
        "test_good": [f"{name}/test/good/{i:03d}.JPG" for i in range(n_good)],
        "test_defect": [f"{name}/test/bad/{i:03d}.JPG" for i in range(n_defect)],
    }


def write_toy_splits(cfg: VisaConfig, class_name: str, checked: dict) -> list[dict]:
    splits = []
    for rep in range(cfg.n_repeats):
        split = visa_data.build_split(cfg, class_name, rep, checked)
        write_json(visa_data.split_path(cfg, class_name, rep), split)
        splits.append(split)
    return splits


def toy_distance(split: dict, seed: int = 0) -> np.ndarray:
    rows = len(visa_data.image_order(split))
    patches = np.random.default_rng(seed).normal(size=(rows, 6, 3)).astype(np.float32)
    defect = visa_data.row_blocks(split)["test_defect"]
    patches[defect] += 1.5
    return distance_tensor(patches, n_train=len(split["universe"]), query_batch=8, train_batch=8, device="cpu")


def test_visa_config_and_v1_hash_unchanged():
    cfg = load_visa_config()
    assert cfg.hash == load_visa_config().hash
    assert len(cfg.hash) == 16
    assert cfg.pool_size == 240
    assert cfg.n_grid == [1, 2, 4, 8, 16, 32, 64, 128, 240]
    assert cfg.seed_key("candle") == 1000 and cfg.seed_key("pipe_fryum") == 1011
    assert load_config().hash == "ec70992336b4c4a4"


def test_scoring_stages_need_a_frozen_preregistration(tmp_path, monkeypatch):
    import commissioning.config
    from commissioning.visa.__main__ import require_frozen_preregistration

    monkeypatch.setattr(commissioning.config, "project_root", lambda: tmp_path)
    cfg = toy_config(tmp_path)
    cfg.work_root.mkdir(parents=True)
    prereg = tmp_path / "preregistration_visa.md"
    prereg.write_text("Status: **DRAFT**.\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not frozen"):
        require_frozen_preregistration(cfg, exploratory=False)
    assert require_frozen_preregistration(cfg, exploratory=True) is True
    prereg.write_text("Status: **FROZEN** on 2026-11-21.\n", encoding="utf-8")
    assert require_frozen_preregistration(cfg, exploratory=False) is False
    assert require_frozen_preregistration(cfg, exploratory=False) is False
    prereg.write_text("Status: **FROZEN** on 2026-11-21. Edited later.\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        require_frozen_preregistration(cfg, exploratory=False)


def test_convert_check_and_prepare(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    rows = []
    for i in range(50):
        path = raw / "candle" / "Data" / "Images" / "Normal" / f"{i:04d}.JPG"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (i, 255 - i, (3 * i) % 256)).save(path, format="PNG")
        rows.append(("candle", "train" if i < 44 else "test", "normal", f"candle/Data/Images/Normal/{i:04d}.JPG"))
    (raw / "candle" / "Data" / "Images" / "Normal" / "0001.JPG").write_bytes(
        (raw / "candle" / "Data" / "Images" / "Normal" / "0000.JPG").read_bytes()
    )
    for i in range(3):
        path = raw / "candle" / "Data" / "Images" / "Anomaly" / f"{i:03d}.JPG"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (200, i, 7)).save(path, format="PNG")
        rows.append(("candle", "test", "anomaly", f"candle/Data/Images/Anomaly/{i:03d}.JPG"))
    (raw / "split_csv").mkdir()
    with (raw / "split_csv" / "1cls.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["object", "split", "label", "image", "mask"])
        writer.writerows([(*row, "") for row in rows])
    cfg = toy_config(tmp_path, n_repeats=3)
    counts = visa_data.convert(raw, cfg.data_root)
    assert counts["candle"] == {"train": 44, "test_good": 6, "test_defect": 3}
    monkeypatch.setitem(visa_data.EXPECTED_COUNTS, "candle", counts["candle"])
    visa_data.prepare(cfg, ["candle"])
    checked = visa_data.checked_class(cfg, "candle")
    assert checked["train_unique"] == 43
    assert checked["train_dropped"][0]["duplicate_of"].endswith("0000.JPG")
    first = visa_data.load_split(cfg, "candle", 0)
    second = visa_data.load_split(cfg, "candle", 1)
    assert first["universe"] == second["universe"] and first["holdout"] == second["holdout"]
    assert first["calibration"] != second["calibration"]
    assert len(first["holdout"]) == 3 and len(first["pool"]) == 32
    assert visa_data.build_split(cfg, "candle", 0, checked) == first
    visa_data.prepare(cfg, ["candle"])


def test_rule_metrics_are_strict_and_r60_is_not_lower():
    from commissioning.visa.run import evaluate_grid_point

    cfg = toy_config(None)
    calibration = np.arange(8, dtype=float)
    labels = np.array([0, 0, 1, 1])
    test = np.array([3.0, 3.5, 7.0, 7.5])
    holdout = np.array([3.0, 4.0, 7.0])
    rows = {row["rule"]: row for row in evaluate_grid_point(cfg, calibration, holdout, test, labels)}
    assert rows["R20"]["threshold"] == 3.0 and rows["R60"]["threshold"] == 7.0
    assert rows["R20"]["n_fp_off"] == 1 and rows["R20"]["n_fp_h"] == 2
    assert rows["R60"]["n_tp"] == 1 and rows["R60"]["n_fp_h"] == 0
    with pytest.raises(RuntimeError):
        evaluate_grid_point(cfg, calibration, np.array([np.nan]), test, labels)


def test_trajectory_matches_direct_scores(tmp_path):
    from commissioning.visa.run import run_one, trajectory_paths

    cfg = toy_config(tmp_path)
    split = write_toy_splits(cfg, "candle", toy_checked())[0]
    distance = toy_distance(split)
    paths = visa_data.image_order(split)
    gap = np.random.default_rng(1).normal(size=(len(paths), 5))
    for strategy in ("random", "kcenter"):
        run_one(cfg, "candle", 0, strategy, True, paths, distance, gap)
    frame = pd.read_parquet(trajectory_paths(cfg, "candle", 0, "random")[0])
    assert sorted(frame["N"].unique()) == [1, 2, 4, 8, 16, 32]
    saved = np.load(trajectory_paths(cfg, "candle", 0, "random")[1])
    index = {p: i for i, p in enumerate(paths)}
    order = saved["order"]
    blocks = visa_data.row_blocks(split)
    for n in (1, 8, 32):
        bank = order[:n]
        c60 = scores_for_bank(distance, [index[p] for p in split["calibration"]], bank)
        holdout = scores_for_bank(distance, blocks["holdout"], bank)
        good = scores_for_bank(distance, blocks["test_good"], bank)
        for rule, threshold in (("R20", c60[:4].max()), ("R60", c60.max())):
            row = frame[(frame["N"] == n) & (frame["rule"] == rule)].iloc[0]
            assert row["threshold"] == pytest.approx(threshold)
            assert row["n_fp_h"] == int(np.sum(holdout > threshold))
            assert row["n_fp_off"] == int(np.sum(good > threshold))
    wide = frame.pivot_table(index="N", columns="rule", values="threshold")
    assert (wide["R60"] >= wide["R20"]).all()


def test_budget_split_matches_direct_scores(tmp_path):
    from commissioning.visa.budget import budget_class, rule_rank

    cfg = toy_config(tmp_path)
    split = visa_data.build_split(cfg, "candle", 0, toy_checked())
    distance = toy_distance(split)
    frame = budget_class(cfg, "candle", distance, split)
    assert (frame["N"] + frame["n_calibration"]).isin(cfg.budgets).all()
    blocks = visa_data.row_blocks(split)
    sequence = np.random.SeedSequence([cfg.seed, cfg.seed_key("candle"), 0, cfg.budget_seed_tag])
    permutation = blocks["universe"][np.random.default_rng(sequence).permutation(40)]
    for n, budget in ((1, 20), (8, 40)):
        bank = permutation[:n]
        calibration = np.sort(scores_for_bank(distance, permutation[n:budget], bank))
        holdout = scores_for_bank(distance, blocks["holdout"], bank)
        for rule in ("conformal", "pac"):
            k, _feasible = rule_rank(cfg, rule, budget - n)
            row = frame[(frame["rep"] == 0) & (frame["N"] == n) & (frame["budget"] == budget) & (frame["rule"] == rule)].iloc[0]
            assert row["n_fp_h"] == int(np.sum(holdout > calibration[k - 1]))


def test_exchange_summary_on_shifted_scores():
    from commissioning.visa.exchange import exchange_summary, finalize

    cfg = toy_config(None)
    rng = np.random.default_rng(0)
    shifted = exchange_summary(cfg, "candle", {
        "holdout": rng.normal(size=200), "test_good": rng.normal(1.5, 1, size=50), "test_defect": rng.normal(4, 1, size=50)})
    same = exchange_summary(cfg, "capsules", {
        "holdout": rng.normal(size=200), "test_good": rng.normal(size=50), "test_defect": rng.normal(4, 1, size=50)})
    frame = finalize([shifted, same])
    assert frame.set_index("class_name").loc["candle", "shift_detected"]
    assert frame.set_index("class_name").loc["candle", "good_shift_auroc"] > 0.75


def test_analysis_and_report_end_to_end(tmp_path):
    from commissioning.visa.analysis import analyze, load_analysis
    from commissioning.visa.budget import run_budget
    from commissioning.visa.exchange import run_exchange
    from commissioning.visa.report import build_report
    from commissioning.visa.run import run_one
    from commissioning.visa import caches

    classes = ("candle", "capsules")
    cfg = toy_config(tmp_path, classes=classes)
    for seed, class_name in enumerate(classes):
        splits = write_toy_splits(cfg, class_name, toy_checked(name=class_name))
        distance = toy_distance(splits[0], seed=seed)
        paths = visa_data.image_order(splits[0])
        caches.atomic_npy(caches.distance_path(cfg, class_name), distance)
        write_json(caches.distance_path(cfg, class_name).with_suffix(".json"),
                   {"config_hash": cfg.hash, "shape": list(distance.shape), "paths": paths})
        gap = np.random.default_rng(seed).normal(size=(len(paths), 5))
        for rep in range(cfg.n_repeats):
            for strategy in cfg.strategies:
                run_one(cfg, class_name, rep, strategy, True, paths, distance, gap)
    run_exchange(cfg, list(classes))
    run_budget(cfg, list(classes))
    result = analyze(cfg)
    statistics = result["statistics"]
    assert set(statistics["primary"]["H1_H2_R20_official"]) >= {"aulc_tpr", "aulc_fpr"}
    assert all(value >= 0 for value in statistics["H3"]["per_class"].values())
    assert statistics["P3"]["available"]
    assert "budget" in statistics
    report = build_report(cfg, load_analysis(cfg.work_root / "analysis"), None)
    assert report.is_file()
    assert (cfg.work_root / "report" / "figures" / "figV2_fpr.png").is_file()
