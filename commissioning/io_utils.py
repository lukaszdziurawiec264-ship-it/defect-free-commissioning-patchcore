"""Atomic writes and small file helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_savez(path: Path, **arrays: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    partial = path.parent / f"{path.name}.partial.npz"
    np.savez(partial, **arrays)
    os.replace(partial, path)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    partial = path.parent / f"{path.name}.partial.npy"
    np.save(partial, array)
    os.replace(partial, path)


def atomic_parquet(path: Path, frame: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    partial = path.parent / f"{path.name}.partial.parquet"
    frame.to_parquet(partial, index=False)
    os.replace(partial, path)
