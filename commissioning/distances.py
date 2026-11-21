"""Per-class distance tensor D[query, patch, train image].

D[q, p, j] is the Euclidean distance from patch p of query q to its nearest
patch in training image j. The image score for a bank S is

    max over p of min over j in S of D[q, p, j]

which is the same 1-NN maximum used by PatchCore without a coreset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from commissioning.config import Config
from commissioning.features import load_feature_cache
from commissioning.io_utils import atomic_npy, read_json, write_json

LOGGER = logging.getLogger(__name__)


def distance_cache_path(cfg: Config, class_name: str) -> Path:
    return cfg.work_root / "cache" / "dist" / f"{class_name}.npy"


def distance_meta_path(cfg: Config, class_name: str) -> Path:
    return cfg.work_root / "cache" / "dist" / f"{class_name}.json"


def min_patch_distance_block(query: np.ndarray, train: np.ndarray) -> np.ndarray:
    """query (Bq, P, C), train (Bt, P, C) -> (Bq, P, Bt) minimum over train patches.

    NumPy reference used by tests and by the CPU fallback. Production runs use torch.
    """
    query = np.ascontiguousarray(query, dtype=np.float32)
    train = np.ascontiguousarray(train, dtype=np.float32)
    bq, patches, channels = query.shape
    bt = train.shape[0]
    if train.shape[1:] != (patches, channels):
        raise RuntimeError(f"Patch shape mismatch: query {query.shape}, train {train.shape}")
    distances = np.empty((bq, patches, bt), dtype=np.float32)
    query_flat = query.reshape(bq * patches, channels)
    for train_index in range(bt):
        delta = query_flat[:, None, :] - train[train_index][None, :, :]
        nearest = np.sqrt(np.sum(delta * delta, axis=-1)).min(axis=1)
        distances[:, :, train_index] = nearest.reshape(bq, patches)
    return distances


def _distance_tensor_torch(
    patches: np.ndarray,
    n_train: int,
    query_batch: int,
    train_batch: int,
    device: str,
) -> np.ndarray:
    """Keeps the float16 cache on the device once; each block is cast to float32 before cdist."""
    import torch

    n_images, n_patches, channels = patches.shape
    resident = torch.from_numpy(np.ascontiguousarray(patches)).to(device)
    output = np.empty((n_images, n_patches, n_train), dtype=np.float32)
    with torch.no_grad():
        for q0 in range(0, n_images, query_batch):
            q1 = min(q0 + query_batch, n_images)
            if q0 == 0 or q1 == n_images or (q0 // query_batch) % 10 == 0:
                LOGGER.info("Distance queries %d/%d", q1, n_images)
            query = resident[q0:q1].float().reshape(-1, channels)
            row = torch.empty((q1 - q0, n_patches, n_train), dtype=torch.float32, device=device)
            for t0 in range(0, n_train, train_batch):
                t1 = min(t0 + train_batch, n_train)
                train = resident[t0:t1].float().reshape(-1, channels)
                block = torch.cdist(query, train).reshape(q1 - q0, n_patches, t1 - t0, n_patches)
                row[:, :, t0:t1] = block.min(dim=-1).values
            if not bool(torch.isfinite(row).all()):
                raise RuntimeError(f"Non-finite distances for queries {q0}:{q1}")
            output[q0:q1] = row.cpu().numpy()
    del resident
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return output


def distance_tensor(
    patches: np.ndarray,
    n_train: int,
    query_batch: int = 8,
    train_batch: int = 16,
    device: str | None = None,
) -> np.ndarray:
    """patches row order is train images first. Diagonal train-train entries are +inf."""
    if patches.ndim != 3:
        raise RuntimeError(f"Expected (N, P, C) patches, got {patches.shape}")
    n_images, n_patches, _channels = patches.shape
    if not 0 < n_train <= n_images:
        raise RuntimeError(f"n_train={n_train} is outside 1..{n_images}")
    torch_device = None
    if device != "cpu":
        try:
            import torch

            torch_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            torch_device = None
    if torch_device is not None:
        output = _distance_tensor_torch(patches, n_train, query_batch, train_batch, torch_device)
    else:
        output = np.empty((n_images, n_patches, n_train), dtype=np.float32)
        for q0 in range(0, n_images, query_batch):
            q1 = min(q0 + query_batch, n_images)
            for t0 in range(0, n_train, train_batch):
                t1 = min(t0 + train_batch, n_train)
                block = min_patch_distance_block(patches[q0:q1], patches[t0:t1])
                if not np.isfinite(block).all():
                    raise RuntimeError(f"Non-finite distances for queries {q0}:{q1}, train {t0}:{t1}")
                output[q0:q1, :, t0:t1] = block
    for index in range(n_train):
        output[index, :, index] = np.inf
    return output


def compute_class_distances(cfg: Config, class_name: str, device: str | None = None) -> None:
    out_path = distance_cache_path(cfg, class_name)
    meta_path = distance_meta_path(cfg, class_name)
    if out_path.is_file() and meta_path.is_file():
        meta = read_json(meta_path)
        if meta.get("config_hash") == cfg.hash:
            LOGGER.info("Distance cache exists for %s", class_name)
            return
    cache = load_feature_cache(cfg, class_name)
    n_train = int(cache["meta"]["n_train"])
    LOGGER.info("Computing distance tensor for %s (%d train images)", class_name, n_train)
    tensor = distance_tensor(
        cache["patches"],
        n_train=n_train,
        query_batch=cfg.distance_query_batch,
        train_batch=cfg.distance_train_batch,
        device=device,
    )
    atomic_npy(out_path, tensor)
    write_json(
        meta_path,
        {
            "config_hash": cfg.hash,
            "class_name": class_name,
            "shape": list(tensor.shape),
            "n_train": n_train,
            "n_test_good": int(cache["meta"]["n_test_good"]),
            "n_test_defect": int(cache["meta"]["n_test_defect"]),
            "paths": cache["paths"],
            "dtype": "float32",
            "bytes": int(tensor.nbytes),
        },
    )
    LOGGER.info("%s distances: shape %s, %.2f GB", class_name, tensor.shape, tensor.nbytes / (1024**3))


def load_distance_cache(cfg: Config, class_name: str) -> dict:
    path = distance_cache_path(cfg, class_name)
    meta_path = distance_meta_path(cfg, class_name)
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Missing distance cache for {class_name}. Run distances first.")
    meta = read_json(meta_path)
    if meta.get("config_hash") != cfg.hash:
        raise RuntimeError(f"Distance cache hash mismatch for {class_name}.")
    tensor = np.load(path)
    if tuple(tensor.shape) != tuple(meta["shape"]):
        raise RuntimeError(f"{class_name}: distance tensor shape {tensor.shape} != meta {meta['shape']}")
    return {"D": tensor, "meta": meta}


def compute_distances(cfg: Config, classes: list[str], device: str | None = None) -> None:
    for class_name in classes:
        compute_class_distances(cfg, class_name, device=device)
