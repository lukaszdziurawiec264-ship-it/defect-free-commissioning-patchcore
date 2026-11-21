"""VisA conversion to the MVTec layout, the data check, the universe U, the held-out set H, and C/P splits."""

from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path

import numpy as np

from commissioning.data import iter_images, relative_path, sha1_file
from commissioning.io_utils import read_json, write_json
from commissioning.visa.config import EXPECTED_COUNTS, VisaConfig

LOGGER = logging.getLogger(__name__)

SPLIT_FOLDERS = {("train", "normal"): ("train", "good"), ("test", "normal"): ("test", "good"), ("test", "anomaly"): ("test", "bad")}


# Conversion ------------------------------------------------------------------------


def find_split_csv(raw_root: Path) -> Path:
    candidates = [raw_root / "split_csv" / "1cls.csv", raw_root.parent / "split_csv" / "1cls.csv"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"split_csv/1cls.csv not found next to {raw_root}. It ships inside VisA_20220922.tar and in "
        "github.com/amazon-science/spot-diff/split_csv."
    )


def convert(raw_root: Path, out_root: Path, split_csv: Path | None = None) -> dict[str, dict[str, int]]:
    """Copy the files listed in 1cls.csv into {class}/train/good, test/good, test/bad. No re-encoding."""
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    split_csv = Path(split_csv) if split_csv else find_split_csv(raw_root)
    counts: dict[str, dict[str, int]] = {}
    targets: dict[Path, Path] = {}
    with split_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["split"].strip(), row["label"].strip())
            if key not in SPLIT_FOLDERS:
                raise RuntimeError(f"Unexpected split/label in {split_csv}: {key}")
            source = raw_root / row["image"].strip()
            if not source.is_file():
                raise FileNotFoundError(f"{row['object']}: listed image is missing: {source}")
            first, second = SPLIT_FOLDERS[key]
            destination = out_root / row["object"].strip() / first / second / source.name
            if destination in targets and targets[destination] != source:
                raise RuntimeError(f"Two VisA files map to {destination}: {targets[destination]} and {source}")
            targets[destination] = source
            name = "test_defect" if second == "bad" else ("train" if first == "train" else "test_good")
            counts.setdefault(row["object"].strip(), {"train": 0, "test_good": 0, "test_defect": 0})[name] += 1
    for destination, source in targets.items():
        if destination.is_file() and destination.stat().st_size == source.stat().st_size:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    LOGGER.info("Converted %d VisA files into %s", len(targets), out_root)
    return counts


# Data check ------------------------------------------------------------------------


def collect_class(data_root: Path, class_name: str) -> dict[str, list[Path]]:
    root = Path(data_root) / class_name
    if not root.is_dir():
        raise FileNotFoundError(f"{class_name}: converted VisA directory not found: {root}")
    return {
        "train": iter_images(root / "train" / "good"),
        "test_good": iter_images(root / "test" / "good"),
        "test_defect": iter_images(root / "test" / "bad"),
    }


def check_class(cfg: VisaConfig, class_name: str) -> dict:
    """Official counts before deduplication, then the duplicate policy of preregistration_visa.md section 2."""
    collected = collect_class(cfg.data_root, class_name)
    found = {name: len(paths) for name, paths in collected.items()}
    if found != EXPECTED_COUNTS[class_name]:
        raise RuntimeError(f"{class_name}: expected official counts {EXPECTED_COUNTS[class_name]}, found {found}")
    digests = {name: [sha1_file(path) for path in paths] for name, paths in collected.items()}
    train_set = set(digests["train"])
    good_set = set(digests["test_good"])
    defect_set = set(digests["test_defect"])
    leaks = sorted((train_set & (good_set | defect_set)) | (good_set & defect_set))
    if leaks:
        raise RuntimeError(f"{class_name}: identical files across train/test or test good/anomaly: {leaks[:5]}")
    kept: list[str] = []
    dropped: list[dict[str, str]] = []
    first_seen: dict[str, str] = {}
    for path, digest in zip(collected["train"], digests["train"]):
        relative = relative_path(path, cfg.data_root)
        if digest in first_seen:
            dropped.append({"path": relative, "duplicate_of": first_seen[digest]})
            continue
        first_seen[digest] = relative
        kept.append(relative)
    test_duplicates = {
        name: len(digests[name]) - len(set(digests[name])) for name in ("test_good", "test_defect")
    }
    if len(kept) <= cfg.universe_size:
        raise RuntimeError(f"{class_name}: {len(kept)} unique train images leave no held-out set H.")
    return {
        "class_name": class_name,
        "official_counts": found,
        "train_unique": len(kept),
        "train_dropped": dropped,
        "test_duplicates_kept": test_duplicates,
        "holdout_size": len(kept) - cfg.universe_size,
        "train": kept,
        "test_good": [relative_path(path, cfg.data_root) for path in collected["test_good"]],
        "test_defect": [relative_path(path, cfg.data_root) for path in collected["test_defect"]],
    }


def data_check_path(cfg: VisaConfig) -> Path:
    return cfg.work_root / "data_check.json"


def run_data_check(cfg: VisaConfig, classes: list[str]) -> dict:
    path = data_check_path(cfg)
    existing = read_json(path) if path.is_file() else {}
    if existing and existing.get("config_hash") != cfg.hash:
        raise RuntimeError(f"{path} was written for another config hash.")
    per_class = dict(existing.get("classes", {}))
    for class_name in classes:
        if class_name in per_class:
            continue
        LOGGER.info("Data check %s", class_name)
        per_class[class_name] = check_class(cfg, class_name)
    payload = {"config_hash": cfg.hash, "classes": per_class}
    write_json(path, payload)
    return payload


def checked_class(cfg: VisaConfig, class_name: str) -> dict:
    path = data_check_path(cfg)
    if not path.is_file():
        raise FileNotFoundError("data_check.json is missing. Run the prepare stage first.")
    payload = read_json(path)
    if payload.get("config_hash") != cfg.hash or class_name not in payload.get("classes", {}):
        raise RuntimeError(f"data_check.json does not cover {class_name} for this config hash.")
    return payload["classes"][class_name]


# Universe and splits ----------------------------------------------------------------


def universe_indices(cfg: VisaConfig, class_name: str, n_train: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, cfg.seed_key(class_name)]))
    return np.sort(rng.choice(n_train, size=cfg.universe_size, replace=False).astype(int))


def split_seeds(cfg: VisaConfig, class_name: str, rep: int) -> tuple[np.random.Generator, ...]:
    sequence = np.random.SeedSequence([cfg.seed, cfg.seed_key(class_name), int(rep)])
    return tuple(np.random.default_rng(child) for child in sequence.spawn(3))


def build_split(cfg: VisaConfig, class_name: str, rep: int, checked: dict) -> dict:
    train = list(checked["train"])
    chosen = universe_indices(cfg, class_name, len(train))
    universe = [train[int(i)] for i in chosen]
    in_universe = set(universe)
    holdout = [path for path in train if path not in in_universe]
    split_rng, _random_rng, _kcenter_rng = split_seeds(cfg, class_name, rep)
    positions = split_rng.choice(len(universe), size=cfg.n_calibration_large, replace=False)
    calibration = [universe[int(i)] for i in positions]
    taken = set(calibration)
    pool = [path for path in universe if path not in taken]
    payload = {
        "config_hash": cfg.hash,
        "class_name": class_name,
        "class_index": cfg.class_index(class_name),
        "rep": int(rep),
        "seed": cfg.seed,
        "universe": universe,
        "holdout": holdout,
        "calibration": calibration,
        "pool": pool,
        "test_good": list(checked["test_good"]),
        "test_defect": list(checked["test_defect"]),
    }
    assert_split(cfg, payload)
    return payload


def assert_split(cfg: VisaConfig, split: dict) -> None:
    name = split["class_name"]
    universe, holdout = split["universe"], split["holdout"]
    calibration, pool = split["calibration"], split["pool"]
    test = list(split["test_good"]) + list(split["test_defect"])
    if len(universe) != cfg.universe_size or len(set(universe)) != len(universe):
        raise RuntimeError(f"{name}: |U| must be {cfg.universe_size} without duplicates.")
    if len(calibration) != cfg.n_calibration_large or len(set(calibration)) != len(calibration):
        raise RuntimeError(f"{name}: |C60| must be {cfg.n_calibration_large} without duplicates.")
    if len(pool) != cfg.pool_size or set(pool) | set(calibration) != set(universe) or set(pool) & set(calibration):
        raise RuntimeError(f"{name}: C60 and P must partition U with |P| = {cfg.pool_size}.")
    if not set(calibration[: cfg.n_calibration_small]) <= set(calibration):
        raise RuntimeError(f"{name}: C20 must be a subset of C60.")
    if not holdout or set(holdout) & set(universe) or set(holdout) & set(test) or set(universe) & set(test):
        raise RuntimeError(f"{name}: U, H, and the official test set must be non-empty and pairwise disjoint.")
    if len(test) != len(set(test)):
        raise RuntimeError(f"{name}: duplicate test paths.")


def split_path(cfg: VisaConfig, class_name: str, rep: int) -> Path:
    return cfg.work_root / "splits" / class_name / f"rep_{rep:02d}.json"


def prepare(cfg: VisaConfig, classes: list[str]) -> None:
    report = run_data_check(cfg, classes)
    for class_name in classes:
        checked = report["classes"][class_name]
        for rep in range(cfg.n_repeats):
            path = split_path(cfg, class_name, rep)
            payload = build_split(cfg, class_name, rep, checked)
            if path.is_file():
                if read_json(path) != payload:
                    raise RuntimeError(f"Refusing to overwrite an existing split that does not match: {path}")
                continue
            write_json(path, payload)


def load_split(cfg: VisaConfig, class_name: str, rep: int) -> dict:
    path = split_path(cfg, class_name, rep)
    if not path.is_file():
        raise FileNotFoundError(f"Missing split {path}. Run prepare first.")
    payload = read_json(path)
    if payload.get("config_hash") != cfg.hash:
        raise RuntimeError(f"Split hash mismatch for {class_name} rep {rep}.")
    assert_split(cfg, payload)
    return payload


def image_order(split: dict) -> list[str]:
    """Row order of every cache: U, H, official good test, official anomalies. Columns are U."""
    return list(split["universe"]) + list(split["holdout"]) + list(split["test_good"]) + list(split["test_defect"])


def row_blocks(split: dict) -> dict[str, np.ndarray]:
    n_u, n_h = len(split["universe"]), len(split["holdout"])
    n_g, n_d = len(split["test_good"]), len(split["test_defect"])
    return {
        "universe": np.arange(0, n_u),
        "holdout": np.arange(n_u, n_u + n_h),
        "test_good": np.arange(n_u + n_h, n_u + n_h + n_g),
        "test_defect": np.arange(n_u + n_h + n_g, n_u + n_h + n_g + n_d),
    }
