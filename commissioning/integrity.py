"""Checks that must pass before any RQ trajectory is trusted."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from commissioning.config import Config
from commissioning.data import load_split
from commissioning.features import image_order, load_feature_cache, read_rgb
from commissioning.patchcore import knn_scores, scores_for_bank


def check_corrupted_input(class_name: str = "integrity") -> None:
    handle, name = tempfile.mkstemp(suffix=".png")
    os.write(handle, b"this is not a png")
    os.close(handle)
    path = Path(name)
    try:
        try:
            read_rgb(path, class_name)
        except RuntimeError as exc:
            text = str(exc)
            if class_name not in text or str(path) not in text:
                raise RuntimeError(f"Corrupted-image error must name the class and path, got: {text}") from exc
            return
        raise RuntimeError("A corrupted image was read without RuntimeError.")
    finally:
        path.unlink(missing_ok=True)


def check_feature_alignment(cfg: Config, class_name: str, cache: dict | None = None) -> None:
    if cache is None:
        cache = load_feature_cache(cfg, class_name)
    expected = image_order(cfg, class_name)
    if cache["paths"] != expected:
        raise RuntimeError(f"{class_name}: feature-cache order does not match the sorted dataset order.")
    if cache["patches"].shape[1:] != (784, 1536):
        raise RuntimeError(f"{class_name}: patch tensor shape is {cache['patches'].shape}")
    if len(cache["paths"]) != len(set(cache["paths"])):
        raise RuntimeError(f"{class_name}: duplicate paths in the feature cache.")
    split = load_split(cfg, class_name, 0)
    n_train = int(split["n_train"])
    train_paths = cache["paths"][:n_train]
    test_paths = cache["paths"][n_train:]
    if set(split["calibration"]) | set(split["pool"]) != set(train_paths):
        raise RuntimeError(f"{class_name}: C and P are not a partition of the cached training images.")
    official_test = list(split["test_good"]) + [item["path"] for item in split["test_defect"]]
    if test_paths != official_test:
        raise RuntimeError(f"{class_name}: cached test order does not match the split.")


def assert_methods_agree(
    patches: np.ndarray,
    distance: np.ndarray,
    query_indices: np.ndarray | list[int],
    bank_indices: np.ndarray | list[int],
    tolerance: float = 1e-4,
) -> float:
    """The distance-tensor score and a direct 1-NN over the same patches must agree.

    Both go through float32 GEMM-based distances with different block shapes, so the check is relative.
    """
    via_distance = scores_for_bank(distance, query_indices, bank_indices)
    via_knn = knn_scores(patches[np.asarray(query_indices)], patches[np.asarray(bank_indices)])
    gap = float(np.max(np.abs(via_distance - via_knn) / np.maximum(1.0, np.abs(via_knn))))
    if gap > tolerance:
        raise RuntimeError(f"Distance tensor and direct kNN differ by a relative {gap} (tolerance {tolerance}).")
    return gap
