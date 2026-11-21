"""E0 gate for VisA: Anomalib reference on U (E0a), own cache versus Anomalib with bank U (E0b), integrity."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from commissioning.e0 import (
    _auroc,
    _batches,
    _fit_anomalib,
    _minmax,
    _passed_e0b,
    _prediction_scores,
    _release,
    check_anomalib_upsampling,
    record_sampler_source,
)
from commissioning.integrity import assert_methods_agree, check_corrupted_input
from commissioning.io_utils import read_json, write_json
from commissioning.patchcore import scores_for_bank
from commissioning.selection import random_order
from commissioning.visa.caches import load_distances, load_features
from commissioning.visa.config import VisaConfig
from commissioning.visa.data import image_order, load_split, row_blocks, split_seeds

LOGGER = logging.getLogger(__name__)


def gate_path(cfg: VisaConfig) -> Path:
    return cfg.work_root / "e0" / "gate.json"


def _class_path(cfg: VisaConfig, class_name: str) -> Path:
    return cfg.work_root / "e0" / f"{class_name}.json"


def _test(split: dict) -> tuple[list[str], np.ndarray]:
    paths = list(split["test_good"]) + list(split["test_defect"])
    labels = np.array([0] * len(split["test_good"]) + [1] * len(split["test_defect"]), dtype=int)
    return paths, labels


def _score_with_anomalib(cfg: VisaConfig, class_name: str, model, device, paths: list[str]) -> np.ndarray:
    import torch

    scores = []
    model.eval()
    with torch.no_grad():
        for batch in _batches(cfg, class_name, paths, batch_size=2):
            scores.append(_prediction_scores(model(batch.to(device))))
    stacked = np.concatenate(scores)
    if stacked.shape[0] != len(paths):
        raise RuntimeError(f"{class_name}: Anomalib scored {stacked.shape[0]} of {len(paths)} test images.")
    return stacked


def integrity_class(cfg: VisaConfig, class_name: str, features: dict, distance: np.ndarray, split: dict) -> dict:
    if features["paths"] != image_order(split):
        raise RuntimeError(f"{class_name}: feature rows do not follow U, H, test good, test anomaly.")
    if len(features["paths"]) != len(set(features["paths"])):
        raise RuntimeError(f"{class_name}: duplicate paths in the feature cache.")
    blocks = row_blocks(split)
    if distance.shape != (len(features["paths"]), 784, len(split["universe"])):
        raise RuntimeError(f"{class_name}: distance tensor columns must be exactly U, got {distance.shape}.")
    if not np.isinf(distance[blocks["universe"], :, blocks["universe"]]).all():
        raise RuntimeError(f"{class_name}: universe self-distances must be +inf.")
    for name in ("holdout", "test_good", "test_defect"):
        if not np.isfinite(distance[blocks[name]]).all():
            raise RuntimeError(f"{class_name}: non-finite distances in the {name} rows.")
    index = {path: position for position, path in enumerate(features["paths"])}
    pool = [index[path] for path in split["pool"]]
    _split_rng, random_rng, _kcenter_rng = split_seeds(cfg, class_name, 0)
    order = random_order(pool, random_rng)
    queries = (
        [index[p] for p in split["calibration"][:2]]
        + [index[p] for p in split["holdout"][:2]]
        + [index[p] for p in split["test_good"][:2]]
    )
    gaps = {}
    for n in (1, 8, len(order)):
        gaps[str(n)] = assert_methods_agree(features["patches"], distance, queries, order[:n])
    return {"distance_vs_knn": gaps}


def run_class(cfg: VisaConfig, class_name: str, part: str) -> dict:
    path = _class_path(cfg, class_name)
    existing = read_json(path) if path.is_file() else {}
    if existing.get("config_hash") != cfg.hash:
        existing = {}
    split = load_split(cfg, class_name, 0)
    test_paths, labels = _test(split)
    universe = list(split["universe"])
    if "integrity" not in existing:
        features = load_features(cfg, class_name)
        distance = load_distances(cfg, class_name)["D"]
        existing["integrity"] = integrity_class(cfg, class_name, features, distance, split)
        del features, distance
        write_json(path, {**existing, "config_hash": cfg.hash, "class_name": class_name})
    if part in {"a", "all"} and "e0a" not in existing:
        LOGGER.info("E0a %s", class_name)
        model, device = _fit_anomalib(cfg, class_name, universe, int(cfg.e0["num_neighbors_ref"]), full_bank=False)
        scores = _score_with_anomalib(cfg, class_name, model, device, test_paths)
        _release(model)
        existing["e0a"] = {"auroc": _auroc(labels, scores), "n_test": int(labels.size), "n_bank_images": len(universe)}
        write_json(path, {**existing, "config_hash": cfg.hash, "class_name": class_name})
    if part in {"b", "all"} and "e0b" not in existing:
        LOGGER.info("E0b %s", class_name)
        model, device = _fit_anomalib(cfg, class_name, universe, int(cfg.e0["num_neighbors_rq"]), full_bank=True)
        anomalib_scores = _score_with_anomalib(cfg, class_name, model, device, test_paths)
        _release(model)
        distance = load_distances(cfg, class_name)["D"]
        blocks = row_blocks(split)
        rows = np.concatenate([blocks["test_good"], blocks["test_defect"]])
        own_scores = scores_for_bank(distance, rows, blocks["universe"])
        del distance
        existing["e0b"] = {
            "spearman": float(spearmanr(own_scores, anomalib_scores).statistic),
            "auroc_own": _auroc(labels, own_scores),
            "auroc_anomalib": _auroc(labels, anomalib_scores),
            "max_abs_score_gap": float(np.max(np.abs(own_scores - anomalib_scores))),
            "max_abs_gap_after_class_minmax": float(np.max(np.abs(_minmax(own_scores) - _minmax(anomalib_scores)))),
        }
    existing["config_hash"] = cfg.hash
    existing["class_name"] = class_name
    write_json(path, existing)
    return existing


def run_e0(cfg: VisaConfig, classes: list[str], part: str = "all") -> dict:
    version = record_sampler_source(cfg)
    upsampling = check_anomalib_upsampling(cfg)
    check_corrupted_input()
    rows = [run_class(cfg, class_name, part) for class_name in classes]
    e0a_ok, e0a_summary = True, None
    if part in {"a", "all"}:
        per_class = {row["class_name"]: float(row["e0a"]["auroc"]) for row in rows}
        e0a_summary = {
            "mean_auroc": float(np.mean(list(per_class.values()))),
            "min_auroc": float(np.min(list(per_class.values()))),
            "per_class": per_class,
        }
        e0a_ok = e0a_summary["mean_auroc"] >= float(cfg.e0["mean_auroc_floor"])
    e0b_ok, e0b_summary = True, None
    if part in {"b", "all"}:
        e0b_ok, e0b_summary = _passed_e0b(cfg, rows)
    complete = part == "all" and set(classes) == set(cfg.classes)
    status = "PASS" if complete and e0a_ok and e0b_ok else ("PARTIAL" if e0a_ok and e0b_ok else "FAIL")
    payload = {
        "status": status,
        "config_hash": cfg.hash,
        "anomalib_version": version,
        "anomalib_upsampling": upsampling,
        "part": part,
        "classes": classes,
        "e0a": e0a_summary,
        "e0b": e0b_summary,
        "per_class": rows,
        "reference_mean_auroc": float(cfg.e0["reference_mean_auroc"]),
        "mean_auroc_floor": float(cfg.e0["mean_auroc_floor"]),
    }
    write_json(gate_path(cfg), payload)
    LOGGER.info("E0 gate %s", status)
    return payload


def gate_is_exploratory(cfg: VisaConfig, classes: list[str], exploratory: bool) -> bool:
    path = gate_path(cfg)
    if not path.is_file():
        if exploratory:
            LOGGER.warning("E0 gate is missing. This run is exploratory.")
            return True
        raise RuntimeError("E0 gate is missing. Trajectories stay blocked until e0/gate.json has status PASS.")
    gate = read_json(path)
    if gate.get("config_hash") != cfg.hash:
        raise RuntimeError("E0 gate was written for a different config hash.")
    missing = [name for name in classes if name not in set(gate.get("classes", []))]
    if gate.get("status") == "PASS" and gate.get("part") == "all" and not missing:
        return False
    if exploratory:
        LOGGER.warning("E0 gate does not cover this run (%s). Marking it exploratory.", gate.get("status"))
        return True
    raise RuntimeError(f"E0 gate status {gate.get('status')}, missing {missing}. Trajectories stay blocked.")
