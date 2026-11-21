"""MVTec AD discovery, official-count checks, and permanent C/P splits."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from commissioning.config import EXPECTED_COUNTS, Config
from commissioning.io_utils import read_json, write_json

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def iter_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    files = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(files, key=lambda path: path.name.lower())


def collect_class(data_root: Path, class_name: str) -> dict[str, list]:
    root = data_root / class_name
    if not root.is_dir():
        raise FileNotFoundError(f"{class_name}: dataset directory not found: {root}")
    train = iter_images(root / "train" / "good")
    test_good = iter_images(root / "test" / "good")
    test_root = root / "test"
    defects: list[tuple[str, Path]] = []
    if test_root.is_dir():
        for sub in sorted((path for path in test_root.iterdir() if path.is_dir()), key=lambda path: path.name.lower()):
            if sub.name.lower() == "good":
                continue
            defects.extend((sub.name, image) for image in iter_images(sub))
    return {"train": train, "test_good": test_good, "test_defect": defects}


def assert_official_counts(class_name: str, collected: dict[str, list]) -> None:
    expected = EXPECTED_COUNTS[class_name]
    found = {"train": len(collected["train"]), "test_good": len(collected["test_good"])}
    if found != expected:
        raise RuntimeError(f"{class_name}: expected counts {expected}, found {found}")
    if len(collected["test_defect"]) < 1:
        raise RuntimeError(f"{class_name}: no defect images under test/")
    _assert_no_duplicate_files(class_name, collected)


def _assert_no_duplicate_files(class_name: str, collected: dict[str, list]) -> None:
    seen: dict[str, str] = {}
    paths: list[Path] = list(collected["train"]) + list(collected["test_good"])
    paths.extend(path for _, path in collected["test_defect"])
    for path in paths:
        digest = sha1_file(path)
        previous = seen.get(digest)
        if previous is not None:
            raise RuntimeError(f"{class_name}: duplicate image content {previous} and {path}")
        seen[digest] = str(path)


def sha1_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path, data_root: Path) -> str:
    return path.resolve().relative_to(data_root.resolve()).as_posix()


def absolute_path(data_root: Path, relative: str) -> Path:
    return data_root / Path(relative)


def split_seeds(cfg: Config, class_name: str, rep: int) -> tuple[np.random.Generator, np.random.Generator, np.random.Generator]:
    """Independent streams so C/P does not depend on which strategy runs first."""
    sequence = np.random.SeedSequence([cfg.seed, cfg.class_index(class_name), int(rep)])
    split_seed, random_seed, kcenter_seed = sequence.spawn(3)
    return (
        np.random.default_rng(split_seed),
        np.random.default_rng(random_seed),
        np.random.default_rng(kcenter_seed),
    )


def split_train_indices(n_train: int, n_calibration: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if n_train <= n_calibration:
        raise RuntimeError(f"Need more than {n_calibration} training images, found {n_train}.")
    calibration = np.sort(rng.choice(n_train, size=n_calibration, replace=False).astype(int))
    pool_mask = np.ones(n_train, dtype=bool)
    pool_mask[calibration] = False
    pool = np.flatnonzero(pool_mask).astype(int)
    _assert_split_indices(calibration, pool, n_train, n_calibration)
    return calibration, pool


def _assert_split_indices(calibration: np.ndarray, pool: np.ndarray, n_train: int, n_calibration: int) -> None:
    if len(calibration) != n_calibration:
        raise RuntimeError(f"|C| = {len(calibration)}, expected {n_calibration}")
    if len(calibration) + len(pool) != n_train:
        raise RuntimeError("|C| + |P| must equal the training set.")
    if len(np.intersect1d(calibration, pool)):
        raise RuntimeError("C and P overlap.")
    if len(np.unique(calibration)) != len(calibration) or len(np.unique(pool)) != len(pool):
        raise RuntimeError("Duplicate indices inside C or P.")


def split_path(cfg: Config, class_name: str, rep: int) -> Path:
    return cfg.work_root / "splits" / class_name / f"rep_{rep:02d}.json"


def build_split_payload(cfg: Config, class_name: str, rep: int, collected: dict[str, list]) -> dict:
    rng, _, _ = split_seeds(cfg, class_name, rep)
    calibration_idx, pool_idx = split_train_indices(len(collected["train"]), cfg.n_calibration, rng)
    train_rel = [relative_path(path, cfg.data_root) for path in collected["train"]]
    test_good_rel = [relative_path(path, cfg.data_root) for path in collected["test_good"]]
    test_defect = [
        {"path": relative_path(path, cfg.data_root), "defect_type": defect_type}
        for defect_type, path in collected["test_defect"]
    ]
    calibration = [train_rel[int(i)] for i in calibration_idx]
    pool = [train_rel[int(i)] for i in pool_idx]
    test_paths = test_good_rel + [item["path"] for item in test_defect]
    _assert_split_paths(class_name, calibration, pool, test_paths)
    return {
        "config_hash": cfg.hash,
        "class_name": class_name,
        "class_index": cfg.class_index(class_name),
        "rep": int(rep),
        "seed": cfg.seed,
        "calibration": calibration,
        "pool": pool,
        "test_good": test_good_rel,
        "test_defect": test_defect,
        "n_train": len(train_rel),
        "n_test_good": len(test_good_rel),
        "n_test_defect": len(test_defect),
    }


def _assert_split_paths(class_name: str, calibration: list[str], pool: list[str], test_paths: list[str]) -> None:
    if len(calibration) != len(set(calibration)) or len(pool) != len(set(pool)):
        raise RuntimeError(f"{class_name}: duplicate paths inside C or P.")
    if set(calibration) & set(pool):
        raise RuntimeError(f"{class_name}: C overlaps P.")
    if set(calibration) & set(test_paths) or set(pool) & set(test_paths):
        raise RuntimeError(f"{class_name}: train split overlaps the official test set.")
    if len(test_paths) != len(set(test_paths)):
        raise RuntimeError(f"{class_name}: duplicate test paths.")


def prepare_class(cfg: Config, class_name: str, reps: range | None = None) -> None:
    collected = collect_class(cfg.data_root, class_name)
    assert_official_counts(class_name, collected)
    rep_range = range(cfg.n_repeats) if reps is None else reps
    for rep in rep_range:
        path = split_path(cfg, class_name, rep)
        payload = build_split_payload(cfg, class_name, rep, collected)
        if path.is_file():
            existing = read_json(path)
            if existing != payload:
                raise RuntimeError(
                    f"Refusing to overwrite an existing split that does not match this config: {path}"
                )
            continue
        write_json(path, payload)


def prepare(cfg: Config, classes: list[str]) -> None:
    for class_name in classes:
        prepare_class(cfg, class_name)


def load_split(cfg: Config, class_name: str, rep: int) -> dict:
    path = split_path(cfg, class_name, rep)
    if not path.is_file():
        raise FileNotFoundError(f"Missing split {path}. Run prepare first.")
    payload = read_json(path)
    if payload.get("config_hash") != cfg.hash:
        raise RuntimeError(f"Split hash mismatch for {class_name} rep {rep}. The split was built from another config.")
    _assert_split_paths(
        class_name,
        payload["calibration"],
        payload["pool"],
        list(payload["test_good"]) + [item["path"] for item in payload["test_defect"]],
    )
    return payload
