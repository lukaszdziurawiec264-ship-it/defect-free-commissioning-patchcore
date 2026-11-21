"""Unit tests for the commissioning protocol. They do not need MVTec or a GPU."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from commissioning.analysis import aulc, attach_stopping, fpr_aggregate, n_star, summarize_classes, stopping_table
from commissioning.calibration import (
    clopper_pearson_upper,
    evaluate_scores,
    false_reject_budget,
    order_statistic_rank,
    threshold_from_scores,
)
from commissioning.config import EXPECTED_COUNTS, FALSE_REJECT_BUDGET, load_config
from commissioning.data import _assert_no_duplicate_files, prepare, split_seeds, split_train_indices
from commissioning.distances import distance_tensor
from commissioning.integrity import assert_methods_agree, check_corrupted_input
from commissioning.patchcore import scores_along_order, scores_for_bank
from commissioning.selection import grid_for_pool, kcenter_order, random_order
from commissioning.stats import bootstrap_median_interval, holm, signflip_pvalue, wilcoxon_pvalue
from commissioning.stopping import n_stop, relative_deltas


def test_config_and_budget_table():
    cfg = load_config()
    assert cfg.hash == load_config().hash
    assert len(cfg.hash) == 16
    assert len(cfg.classes) == 15
    assert cfg.class_index("bottle") == 0
    assert cfg.class_index("carpet") == 10
    assert cfg.group_of("toothbrush") == "object"
    assert cfg.group_of("wood") == "texture"
    for name, counts in EXPECTED_COUNTS.items():
        assert false_reject_budget(counts["test_good"]) == FALSE_REJECT_BUDGET[name]


def test_clopper_pearson_zero_errors():
    assert round(clopper_pearson_upper(0, 58), 4) == 0.0503
    assert round(clopper_pearson_upper(0, 60), 4) == 0.0487
    assert clopper_pearson_upper(0, 60) < 0.05 < clopper_pearson_upper(0, 58)


def test_threshold_is_the_maximum_of_twenty_scores():
    scores = np.linspace(0.1, 2.0, 20)
    threshold, rank = threshold_from_scores(scores, 0.05)
    assert order_statistic_rank(20, 0.05) == 20
    assert rank == 20
    assert threshold == pytest.approx(scores.max())
    labels = np.array([0, 0, 1, 1])
    metrics = evaluate_scores(
        test_scores=np.array([threshold, threshold + 1.0, threshold, threshold + 1.0]),
        labels=labels,
        calibration_scores=scores,
        alpha=0.05,
        practical_tpr=0.95,
        practical_fpr=0.05,
    )
    assert metrics["n_fp"] == 1
    assert metrics["n_tp"] == 1
    with pytest.raises(RuntimeError):
        evaluate_scores(np.array([0.0, np.inf]), labels[:2], scores, 0.05, 0.95, 0.05)


def test_stopping_rule_uses_the_previous_grid_point():
    grid = [1, 2, 4, 8, 16, 32]
    assert n_stop(grid, [100, 80, 50, 40, 39, 38], 0.05) == 32
    assert n_stop([32, 64, 100], [10.0, 9.6, 9.5], 0.05) == 100
    assert n_stop([1, 2, 4, 8, 16, 32], [10, 9, 8, 4, 3.9, 3.8], 0.05) == 32
    assert relative_deltas([0.0, 0.0, 1.0])[1:] == [0.0, np.inf]


def test_aulc_and_n_star():
    grid = [1, 2, 4, 8, 16, 32, 64]
    assert aulc(grid, [1, 1, 1, 1, 1, 1, 0], 32) == pytest.approx(1.0)
    assert aulc(grid, [0, 0, 0, 0, 0, 1, 1], 32) == pytest.approx(0.1)
    assert n_star([1, 2, 4, 8], [0, 16, 16, 20], 16) == {"n_star": 2, "censored": False}
    assert n_star([1, 2, 4], [20, 10, 20], 16) == {"n_star": 4, "censored": False}
    assert n_star([1, 2], [10, 15], 16)["censored"] is True


def test_grid_and_nested_orders():
    assert grid_for_pool(40, [1, 2, 4, 8, 16, 32, 64]) == [1, 2, 4, 8, 16, 32, 40]
    assert grid_for_pool(193, [1, 2, 4, 8, 16, 32, 64]) == [1, 2, 4, 8, 16, 32, 64, 193]
    assert grid_for_pool(32, [1, 2, 4, 8, 16, 32, 64]) == [1, 2, 4, 8, 16, 32]
    features = np.random.default_rng(0).normal(size=(6, 4))
    pool = [1, 3, 5]
    first = kcenter_order(features, pool, np.random.default_rng(2))
    second = kcenter_order(features, pool, np.random.default_rng(2))
    assert np.array_equal(first, second)
    assert set(first.tolist()) == set(pool)
    order = random_order(pool, np.random.default_rng(3))
    assert set(order.tolist()) == set(pool)


def test_splits_are_paired_and_repeatable():
    cfg = load_config()

    def calibration(rep: int) -> np.ndarray:
        rng, _random_rng, _kcenter_rng = split_seeds(cfg, "transistor", rep)
        chosen, pool = split_train_indices(213, 20, rng)
        assert len(chosen) == 20
        assert len(pool) == 193
        assert len(np.intersect1d(chosen, pool)) == 0
        return chosen

    assert np.array_equal(calibration(0), calibration(0))
    assert not np.array_equal(calibration(0), calibration(1))


def test_duplicate_and_corrupt_images(tmp_path):
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    Image.new("RGB", (8, 8), (3, 2, 1)).save(first)
    Image.new("RGB", (8, 8), (3, 2, 1)).save(second)
    with pytest.raises(RuntimeError, match="duplicate"):
        _assert_no_duplicate_files("toy", {"train": [first, second], "test_good": [], "test_defect": []})
    check_corrupted_input("toy")


def test_missing_class_is_reported(tmp_path):
    cfg = load_config(data_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="bottle"):
        prepare(cfg, ["bottle"])


def test_distance_tensor_matches_direct_knn_and_nested_banks():
    patches = np.random.default_rng(7).normal(size=(6, 5, 3)).astype(np.float32)
    distance = distance_tensor(patches, n_train=4, query_batch=2, train_batch=2, device="cpu")
    assert np.isinf(distance[1, :, 1]).all()
    assert np.isfinite(distance[4]).all()
    queries = [4, 5]
    order = [0, 2, 1, 3]
    grid = [1, 2, 4]
    stacked = scores_along_order(distance, queries, order, grid)
    for index, n in enumerate(grid):
        direct = scores_for_bank(distance, queries, order[:n])
        np.testing.assert_allclose(stacked[index], direct)
    assert np.all(np.diff(stacked, axis=0) <= 1e-5)
    assert_methods_agree(patches, distance, queries, order[:2], tolerance=1e-3)
    with pytest.raises(RuntimeError):
        scores_for_bank(distance, queries, [])


def test_primary_statistics():
    assert holm([0.04, 0.01]) == pytest.approx([0.04, 0.02])
    ones = np.ones(15)
    assert signflip_pvalue(ones) == pytest.approx(2 / 32768)
    assert wilcoxon_pvalue(ones) < 0.001
    assert wilcoxon_pvalue(np.zeros(15)) == 1.0
    first = bootstrap_median_interval(np.arange(15, dtype=float), 1000, 42)
    second = bootstrap_median_interval(np.arange(15, dtype=float), 1000, 42)
    assert first == second
    assert first["ci95"][0] <= first["median"] <= first["ci95"][1]


def _synthetic_metrics() -> pd.DataFrame:
    rows = []
    for class_name, group in (("bottle", "object"), ("carpet", "texture")):
        for strategy in ("random", "kcenter"):
            for rep in (0, 1):
                for n in (1, 2, 4, 8, 16, 32, 40):
                    rows.append(
                        {
                            "class_name": class_name,
                            "group": group,
                            "strategy": strategy,
                            "rep": rep,
                            "N": n,
                            "pool_size": 40,
                            "tpr": 1.0,
                            "fpr": 0.0,
                            "oracle_fpr": 0.0,
                            "criterion": True,
                            "median_calibration": 1.0,
                            "exploratory": False,
                        }
                    )
    return pd.DataFrame(rows)


def test_summary_stopping_and_report(tmp_path):
    from commissioning.analysis import budget_table
    from commissioning.report import build_report
    from commissioning.stats import compare_strategies

    metrics = _synthetic_metrics()
    summary = summarize_classes(metrics, 32, min_passes=1)
    assert summary["aulc_tpr"].eq(1).all()
    assert summary["aulc_fpr"].eq(0).all()
    assert not summary["censored"].any()
    stopping = stopping_table(metrics, [0.025, 0.05, 0.10])
    summary = attach_stopping(summary, stopping, 0.05)
    assert summary["n_stop_median"].eq(4).all()
    aggregate = fpr_aggregate(metrics)
    assert "fraction_fpr_above_05" in aggregate.columns
    assert aggregate["fraction_fpr_above_05"].eq(0).all()
    statistics = compare_strategies(summary, margin=0.01, n_bootstrap=200, seed=42)
    frames = {
        "metrics": metrics,
        "summary": summary,
        "fpr_aggregate": aggregate,
        "stopping": stopping,
        "statistics": statistics,
        "budget": budget_table(),
    }
    report = build_report(frames, tmp_path / "report")
    assert report.is_file()
    assert (tmp_path / "report" / "figures" / "fig2_tpr.png").is_file()
    assert (tmp_path / "report" / "figures" / "fig2_tpr.pdf").is_file()
    assert (tmp_path / "report" / "tables" / "table1_summary.tex").is_file()
    assert "1/21" in report.read_text(encoding="utf-8")


def test_pooled_embedding_shape():
    torch = pytest.importorskip("torch")
    from commissioning.features import pooled_embedding

    pool = torch.nn.AvgPool2d(3, 1, 1)
    features = {"layer2": torch.randn(2, 512, 28, 28), "layer3": torch.randn(2, 1024, 14, 14)}
    embedding = pooled_embedding(features, ["layer2", "layer3"], pool, "bilinear")
    assert tuple(embedding.shape) == (2, 1536, 28, 28)


def test_exploratory_ceiling_formulas():
    from scipy.stats import betabinom

    from commissioning.exploratory import exceedance_probability, ideal_pass_probability, pac_rank

    assert exceedance_probability(20, 20) == pytest.approx(0.95**20)
    assert pac_rank(20) is None
    assert pac_rank(44) is None
    assert pac_rank(45) == 45
    assert pac_rank(100) < 100
    assert ideal_pass_probability(20, 20, 20) == pytest.approx(betabinom.cdf(1, 20, 1, 20))


def _synthetic_distance(n_train: int, n_test: int) -> np.ndarray:
    patches = np.random.default_rng(3).normal(size=(n_train + n_test, 4, 3)).astype(np.float32)
    return distance_tensor(patches, n_train=n_train, query_batch=8, train_batch=8, device="cpu")


def test_exploratory_budget_split_matches_direct_scores(monkeypatch):
    import commissioning.exploratory as exploratory

    cfg = load_config()
    n_train, n_good, n_defect = 45, 4, 3
    distance = _synthetic_distance(n_train, n_good + n_defect)
    good = np.arange(n_train, n_train + n_good)
    defect = np.arange(n_train + n_good, n_train + n_good + n_defect)
    monkeypatch.setattr(exploratory, "_test_indices", lambda *_args: (good, defect, [f"g{i}" for i in good]))
    frame = exploratory.budget_split_class(cfg, "bottle", distance, [], n_train)
    assert sorted(frame["budget"].unique()) == [40]
    assert sorted(frame["N"].unique()) == [1, 2, 4, 8, 16]
    assert (frame["n_calibration"] + frame["N"]).eq(40).all()
    sequence = np.random.SeedSequence([cfg.seed, cfg.class_index("bottle"), 0, exploratory.SEED_TAG])
    permutation = np.random.default_rng(sequence).permutation(n_train)
    for n in (1, 8):
        bank = permutation[:n]
        calibration = np.sort(scores_for_bank(distance, permutation[n:40], bank))
        test = scores_for_bank(distance, good, bank)
        for rule in exploratory.RULES:
            row = frame[(frame["rep"] == 0) & (frame["N"] == n) & (frame["rule"] == rule)].iloc[0]
            threshold = calibration[int(row["rank"]) - 1]
            assert int(row["n_fp"]) == int(np.sum(test > threshold))


def test_exploratory_full_bank_scores_are_leave_one_out():
    from commissioning.exploratory import _full_bank_scores

    n_train = 6
    distance = _synthetic_distance(n_train, 2)
    scores = _full_bank_scores(distance, np.arange(n_train + 2), chunk=3)
    for index in range(n_train):
        others = [j for j in range(n_train) if j != index]
        assert scores[index] == pytest.approx(scores_for_bank(distance, [index], others)[0])
    assert scores[n_train] == pytest.approx(scores_for_bank(distance, [n_train], list(range(n_train)))[0])
