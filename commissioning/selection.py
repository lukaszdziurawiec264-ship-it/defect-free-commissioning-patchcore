"""Nested memory-bank trajectories. Both strategies return a permutation of P."""

from __future__ import annotations

import numpy as np


def grid_for_pool(pool_size: int, base_grid: list[int]) -> list[int]:
    if pool_size < 1:
        raise RuntimeError("Pool is empty.")
    grid = [int(n) for n in base_grid if 1 <= int(n) <= pool_size]
    if pool_size not in grid:
        grid.append(int(pool_size))
    if grid != sorted(set(grid)):
        raise RuntimeError(f"Grid is not strictly increasing: {grid}")
    return grid


def random_order(pool: np.ndarray | list[int], rng: np.random.Generator) -> np.ndarray:
    order = np.array(pool, dtype=int, copy=True)
    if order.size == 0 or len(np.unique(order)) != len(order):
        raise RuntimeError("Random trajectory expects a non-empty pool without duplicates.")
    rng.shuffle(order)
    return order


def kcenter_order(features: np.ndarray, pool: np.ndarray | list[int], rng: np.random.Generator) -> np.ndarray:
    """Farthest-first traversal on GAP features. The first image is random.

    This is image-level diversity selection. It is not coreset selection on PatchCore patches.
    """
    pool = np.array(pool, dtype=int, copy=True)
    if pool.size == 0 or len(np.unique(pool)) != len(pool):
        raise RuntimeError("k-center trajectory expects a non-empty pool without duplicates.")
    if np.any(pool < 0) or np.any(pool >= len(features)):
        raise RuntimeError("Pool index is outside the feature table.")
    first = int(rng.integers(0, len(pool)))
    remaining = np.ones(len(pool), dtype=bool)
    remaining[first] = False
    selected = [first]
    chosen = features[pool[first]]
    min_dist = np.linalg.norm(features[pool] - chosen, axis=1)
    while bool(remaining.any()):
        masked = np.where(remaining, min_dist, -np.inf)
        nxt = int(np.argmax(masked))
        remaining[nxt] = False
        selected.append(nxt)
        updated = np.linalg.norm(features[pool] - features[pool[nxt]], axis=1)
        min_dist = np.minimum(min_dist, updated)
    order = pool[np.array(selected, dtype=int)]
    if len(np.unique(order)) != len(pool) or set(order.tolist()) != set(pool.tolist()):
        raise RuntimeError("k-center did not return a permutation of the pool.")
    return order
