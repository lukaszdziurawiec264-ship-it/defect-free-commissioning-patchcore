"""Patch features and the distance tensor for VisA.

Large tensors live under cache_root (local disk). The GAP table, which fixes the k-center order,
is written once under work_root so that a recomputed cache cannot change a finished trajectory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from commissioning.distances import distance_tensor
from commissioning.features import (
    build_feature_extractor,
    build_transform,
    patches_and_gap,
    pooled_embedding,
    read_rgb,
)
from commissioning.io_utils import atomic_npy, atomic_savez, read_json, write_json
from commissioning.visa.config import VisaConfig
from commissioning.visa.data import image_order, load_split

LOGGER = logging.getLogger(__name__)


def feature_path(cfg: VisaConfig, class_name: str) -> Path:
    return cfg.cache_root / "features" / f"{class_name}.npz"


def distance_path(cfg: VisaConfig, class_name: str) -> Path:
    return cfg.cache_root / "dist" / f"{class_name}.npy"


def gap_path(cfg: VisaConfig, class_name: str) -> Path:
    return cfg.work_root / "gap" / f"{class_name}.npz"


def _meta_path(path: Path) -> Path:
    return path.with_suffix(".json")


def _current(cfg: VisaConfig, path: Path) -> bool:
    meta = _meta_path(path)
    return path.is_file() and meta.is_file() and read_json(meta).get("config_hash") == cfg.hash


def extract_class(cfg: VisaConfig, class_name: str, device: str | None = None) -> None:
    import torch
    import torch.nn as nn

    out = feature_path(cfg, class_name)
    if _current(cfg, out):
        LOGGER.info("Feature cache exists for %s", class_name)
        return
    split = load_split(cfg, class_name, 0)
    paths = image_order(split)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    extractor, source = build_feature_extractor(cfg)
    extractor = extractor.eval().to(device_t)
    transform = build_transform(cfg)
    pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1).to(device_t)
    patches = np.empty((len(paths), 784, 1536), dtype=np.float16)
    gap = np.empty((len(paths), 1536), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(paths), cfg.feature_batch_size):
            chunk = paths[start : start + cfg.feature_batch_size]
            batch = torch.stack([transform(read_rgb(cfg.data_root / p, class_name)) for p in chunk]).to(device_t)
            features = extractor(batch)
            if not isinstance(features, dict):
                features = dict(zip(cfg.layers, features))
            embedding = pooled_embedding(features, cfg.layers, pool, cfg.upsample_mode)
            if embedding.shape[1] != 1536 or embedding.shape[-2:] != (28, 28):
                raise RuntimeError(f"{class_name}: unexpected embedding shape {tuple(embedding.shape)}")
            block_patches, block_gap = patches_and_gap(embedding)
            block_patches = block_patches.cpu().numpy()
            block_gap = block_gap.cpu().numpy()
            if not np.isfinite(block_patches).all() or not np.isfinite(block_gap).all():
                raise RuntimeError(f"{class_name}: non-finite features in batch starting at {chunk[0]}")
            patches[start : start + len(chunk)] = block_patches.astype(np.float16)
            gap[start : start + len(chunk)] = block_gap
    atomic_savez(out, patches=patches, paths=np.array(paths))
    write_json(
        _meta_path(out),
        {
            "config_hash": cfg.hash,
            "class_name": class_name,
            "n_images": len(paths),
            "n_universe": len(split["universe"]),
            "n_holdout": len(split["holdout"]),
            "n_test_good": len(split["test_good"]),
            "n_test_defect": len(split["test_defect"]),
            "extractor": source,
        },
    )
    gap_file = gap_path(cfg, class_name)
    if not _current(cfg, gap_file):
        atomic_savez(gap_file, gap=gap, paths=np.array(paths))
        write_json(_meta_path(gap_file), {"config_hash": cfg.hash, "class_name": class_name, "extractor": source})
    LOGGER.info("%s features: %d images, %.2f GB float16", class_name, len(paths), patches.nbytes / 1024**3)


def load_features(cfg: VisaConfig, class_name: str) -> dict:
    path = feature_path(cfg, class_name)
    if not _current(cfg, path):
        raise FileNotFoundError(f"Missing or stale feature cache for {class_name}. Run features first.")
    with np.load(path, allow_pickle=False) as data:
        patches = np.array(data["patches"])
        paths = [str(p) for p in data["paths"].tolist()]
    if patches.dtype != np.float16 or patches.shape[1:] != (784, 1536):
        raise RuntimeError(f"{class_name}: bad patch cache {patches.shape} {patches.dtype}")
    return {"patches": patches, "paths": paths, "meta": read_json(_meta_path(path))}


def load_gap(cfg: VisaConfig, class_name: str) -> dict:
    path = gap_path(cfg, class_name)
    if not _current(cfg, path):
        raise FileNotFoundError(f"Missing GAP table for {class_name}. Run features first.")
    with np.load(path, allow_pickle=False) as data:
        gap = np.array(data["gap"])
        paths = [str(p) for p in data["paths"].tolist()]
    if not np.isfinite(gap).all():
        raise RuntimeError(f"{class_name}: non-finite GAP features")
    return {"gap": gap, "paths": paths}


def compute_class_distances(cfg: VisaConfig, class_name: str, device: str | None = None) -> None:
    out = distance_path(cfg, class_name)
    if _current(cfg, out):
        LOGGER.info("Distance cache exists for %s", class_name)
        return
    cache = load_features(cfg, class_name)
    split = load_split(cfg, class_name, 0)
    if cache["paths"] != image_order(split):
        raise RuntimeError(f"{class_name}: feature rows do not follow U, H, test good, test anomaly.")
    n_universe = len(split["universe"])
    LOGGER.info("Distance tensor %s: %d rows x %d universe columns", class_name, len(cache["paths"]), n_universe)
    tensor = distance_tensor(
        cache["patches"],
        n_train=n_universe,
        query_batch=cfg.distance_query_batch,
        train_batch=cfg.distance_train_batch,
        device=device,
    )
    atomic_npy(out, tensor)
    write_json(
        _meta_path(out),
        {
            "config_hash": cfg.hash,
            "class_name": class_name,
            "shape": list(tensor.shape),
            "n_universe": n_universe,
            "paths": cache["paths"],
            "dtype": "float32",
        },
    )
    LOGGER.info("%s distances: %s, %.2f GB", class_name, tensor.shape, tensor.nbytes / 1024**3)


def load_distances(cfg: VisaConfig, class_name: str) -> dict:
    path = distance_path(cfg, class_name)
    if not _current(cfg, path):
        raise FileNotFoundError(f"Missing or stale distance cache for {class_name}. Run distances first.")
    meta = read_json(_meta_path(path))
    tensor = np.load(path)
    if list(tensor.shape) != list(meta["shape"]):
        raise RuntimeError(f"{class_name}: distance tensor {tensor.shape} != meta {meta['shape']}")
    return {"D": tensor, "meta": meta}


def extract(cfg: VisaConfig, classes: list[str], device: str | None = None) -> None:
    for class_name in classes:
        extract_class(cfg, class_name, device=device)


def compute_distances(cfg: VisaConfig, classes: list[str], device: str | None = None) -> None:
    for class_name in classes:
        compute_class_distances(cfg, class_name, device=device)
