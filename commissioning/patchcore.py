"""Image scores from a memory bank.

The operational score is the maximum, over patches of the query, of the distance
to the nearest patch in the bank. Banks are prefixes of one trajectory, so the
score never increases as N grows.
"""

from __future__ import annotations

import numpy as np


def _as_index(indices: np.ndarray | list[int]) -> np.ndarray:
    array = np.asarray(indices, dtype=int)
    if array.ndim != 1:
        raise RuntimeError(f"Expected a 1-D index, got shape {array.shape}")
    return array


def scores_for_bank(distance: np.ndarray, query_indices: np.ndarray | list[int], bank_indices: np.ndarray | list[int]) -> np.ndarray:
    queries = _as_index(query_indices)
    bank = _as_index(bank_indices)
    if bank.size == 0:
        raise RuntimeError("Empty memory bank")
    if len(np.unique(bank)) != len(bank):
        raise RuntimeError("Memory bank contains duplicate images")
    block = distance[queries][:, :, bank]
    scores = block.min(axis=-1).max(axis=-1)
    if not np.isfinite(scores).all():
        raise RuntimeError("Non-finite image score")
    return scores.astype(np.float64)


def scores_along_order(
    distance: np.ndarray,
    query_indices: np.ndarray | list[int],
    order: np.ndarray | list[int],
    grid: list[int],
) -> np.ndarray:
    """Scores at each nested prefix of ``order``. Returns (len(grid), n_queries)."""
    queries = _as_index(query_indices)
    order = _as_index(order)
    if len(np.unique(order)) != len(order):
        raise RuntimeError("Trajectory order contains duplicate images")
    grid = [int(n) for n in grid]
    if grid != sorted(grid) or any(n < 1 or n > len(order) for n in grid):
        raise RuntimeError(f"Grid {grid} is not a valid increasing subset of 1..{len(order)}")
    selected = distance[queries]
    running = None
    previous = 0
    rows = []
    for n in grid:
        added = order[previous:n]
        block_min = selected[:, :, added].min(axis=-1)
        running = block_min if running is None else np.minimum(running, block_min)
        scores = running.max(axis=-1)
        if not np.isfinite(scores).all():
            raise RuntimeError("Non-finite image score")
        rows.append(scores)
        previous = n
    stacked = np.stack(rows, axis=0).astype(np.float64)
    if stacked.shape[0] > 1 and np.any(np.diff(stacked, axis=0) > 1e-5):
        raise RuntimeError("Image score increased as the nested bank grew")
    return stacked


def knn_scores(query_patches: np.ndarray, bank_patches: np.ndarray, query_chunk: int = 8) -> np.ndarray:
    """Reference 1-NN maximum. ``bank_patches`` is (N_bank, P, C) or (N_patches, C)."""
    if bank_patches.ndim == 3:
        bank = np.ascontiguousarray(bank_patches.reshape(-1, bank_patches.shape[-1]), dtype=np.float32)
    elif bank_patches.ndim == 2:
        bank = np.ascontiguousarray(bank_patches, dtype=np.float32)
    else:
        raise RuntimeError(f"Bad bank shape {bank_patches.shape}")
    try:
        import torch
    except ImportError:
        torch = None
    if torch is None and bank.shape[0] > 8192:
        raise RuntimeError("Direct kNN on a bank of this size needs torch.")
    n_query, n_patches, channels = query_patches.shape
    scores = np.empty(n_query, dtype=np.float64)
    device = "cpu"
    bank_t = None
    if torch is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bank_t = torch.from_numpy(bank).to(device)
    for start in range(0, n_query, query_chunk):
        stop = min(start + query_chunk, n_query)
        flat = np.ascontiguousarray(query_patches[start:stop].reshape(-1, channels), dtype=np.float32)
        if bank_t is not None:
            query_t = torch.from_numpy(flat).to(device)
            nearest = []
            step = 256
            for p0 in range(0, flat.shape[0], step):
                distances = torch.cdist(query_t[p0 : p0 + step], bank_t)
                nearest.append(distances.min(dim=1).values.detach().cpu().numpy())
            patch_scores = np.concatenate(nearest).reshape(stop - start, n_patches)
        else:
            nearest = np.empty(flat.shape[0], dtype=np.float32)
            for p0 in range(0, flat.shape[0], 32):
                p1 = min(p0 + 32, flat.shape[0])
                delta = flat[p0:p1, None, :] - bank[None, :, :]
                nearest[p0:p1] = np.sqrt(np.sum(delta * delta, axis=-1)).min(axis=1)
            patch_scores = nearest.reshape(stop - start, n_patches)
        scores[start:stop] = patch_scores.max(axis=1)
    if not np.isfinite(scores).all():
        raise RuntimeError("Non-finite reference kNN score")
    return scores


def anomaly_map(patch_scores: np.ndarray, image_size: int = 224, sigma: float = 4.0) -> np.ndarray:
    """Upsample a 28x28 patch-score grid and blur it for a figure overlay."""
    from scipy.ndimage import gaussian_filter

    grid = np.asarray(patch_scores, dtype=np.float64).reshape(28, 28)
    zoom = image_size // 28
    if image_size % 28 != 0:
        raise RuntimeError("image_size must be a multiple of 28")
    upsampled = np.repeat(np.repeat(grid, zoom, axis=0), zoom, axis=1)
    return gaussian_filter(upsampled, sigma=sigma)


def patch_scores_for_bank(distance_row: np.ndarray, bank_indices: np.ndarray | list[int]) -> np.ndarray:
    """distance_row: (784, n_train) for one query."""
    bank = _as_index(bank_indices)
    return distance_row[:, bank].min(axis=1)
