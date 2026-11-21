"""Load the frozen experiment configuration and resolve paths."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Seed indices follow this order. It is part of the hashed payload via config.yaml.
OBJECTS = [
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
]
TEXTURES = ["carpet", "grid", "leather", "tile", "wood"]

# Official MVTec AD counts. A mismatch stops the run instead of silently using the wrong split.
EXPECTED_COUNTS: dict[str, dict[str, int]] = {
    "bottle": {"train": 209, "test_good": 20},
    "cable": {"train": 224, "test_good": 58},
    "capsule": {"train": 219, "test_good": 23},
    "carpet": {"train": 280, "test_good": 28},
    "grid": {"train": 264, "test_good": 21},
    "hazelnut": {"train": 391, "test_good": 40},
    "leather": {"train": 245, "test_good": 32},
    "metal_nut": {"train": 220, "test_good": 22},
    "pill": {"train": 267, "test_good": 26},
    "screw": {"train": 320, "test_good": 41},
    "tile": {"train": 230, "test_good": 33},
    "toothbrush": {"train": 60, "test_good": 12},
    "transistor": {"train": 213, "test_good": 60},
    "wood": {"train": 247, "test_good": 19},
    "zipper": {"train": 240, "test_good": 32},
}

# Largest k with k / n_good <= 0.05 on the official good test images.
FALSE_REJECT_BUDGET: dict[str, int] = {
    "toothbrush": 0,
    "wood": 0,
    "bottle": 1,
    "grid": 1,
    "metal_nut": 1,
    "capsule": 1,
    "pill": 1,
    "carpet": 1,
    "leather": 1,
    "zipper": 1,
    "tile": 1,
    "hazelnut": 2,
    "screw": 2,
    "cable": 2,
    "transistor": 3,
}


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


@dataclass
class Config:
    payload: dict[str, Any]
    config_path: Path
    data_root: Path
    work_root: Path

    @property
    def protocol_version(self) -> int:
        return int(self.payload["protocol_version"])

    @property
    def seed(self) -> int:
        return int(self.payload["seed"])

    @property
    def n_calibration(self) -> int:
        return int(self.payload["n_calibration"])

    @property
    def alpha(self) -> float:
        return float(self.payload["alpha"])

    @property
    def n_repeats(self) -> int:
        return int(self.payload["n_repeats"])

    @property
    def n_grid(self) -> list[int]:
        return [int(n) for n in self.payload["n_grid"]]

    @property
    def strategies(self) -> list[str]:
        return [str(s) for s in self.payload["strategies"]]

    @property
    def aulc_n_max(self) -> int:
        return int(self.payload["aulc_n_max"])

    @property
    def practical_tpr(self) -> float:
        return float(self.payload["practical_tpr"])

    @property
    def practical_fpr(self) -> float:
        return float(self.payload["practical_fpr"])

    @property
    def n_star_min_repeats(self) -> int:
        return int(self.payload["n_star_min_repeats"])

    @property
    def fpr_aulc_margin(self) -> float:
        return float(self.payload["fpr_aulc_margin"])

    @property
    def stopping_deltas(self) -> list[float]:
        return [float(v) for v in self.payload["stopping_deltas"]]

    @property
    def stopping_delta_main(self) -> float:
        return float(self.payload["stopping_delta_main"])

    @property
    def bootstrap_replicates(self) -> int:
        return int(self.payload["bootstrap_replicates"])

    @property
    def bootstrap_seed(self) -> int:
        return int(self.payload["bootstrap_seed"])

    @property
    def backbone(self) -> str:
        return str(self.payload["backbone"])

    @property
    def layers(self) -> list[str]:
        return [str(v) for v in self.payload["layers"]]

    @property
    def resize(self) -> int:
        return int(self.payload["resize"])

    @property
    def center_crop(self) -> int:
        return int(self.payload["center_crop"])

    @property
    def upsample_mode(self) -> str:
        return str(self.payload["upsample_mode"])

    @property
    def imagenet_mean(self) -> list[float]:
        return [float(v) for v in self.payload["imagenet_mean"]]

    @property
    def imagenet_std(self) -> list[float]:
        return [float(v) for v in self.payload["imagenet_std"]]

    @property
    def feature_batch_size(self) -> int:
        return int(self.payload["feature_batch_size"])

    @property
    def distance_query_batch(self) -> int:
        return int(self.payload["distance_query_batch"])

    @property
    def distance_train_batch(self) -> int:
        return int(self.payload["distance_train_batch"])

    @property
    def e0(self) -> dict[str, Any]:
        return dict(self.payload["e0"])

    @property
    def objects(self) -> list[str]:
        return [str(v) for v in self.payload["objects"]]

    @property
    def textures(self) -> list[str]:
        return [str(v) for v in self.payload["textures"]]

    @property
    def classes(self) -> list[str]:
        return self.objects + self.textures

    def class_index(self, class_name: str) -> int:
        try:
            return self.classes.index(class_name)
        except ValueError as exc:
            raise KeyError(f"Unknown class: {class_name}") from exc

    def group_of(self, class_name: str) -> str:
        if class_name in self.objects:
            return "object"
        if class_name in self.textures:
            return "texture"
        raise KeyError(f"Unknown class: {class_name}")

    def select_classes(self, requested: list[str] | None) -> list[str]:
        if not requested:
            return list(self.classes)
        unknown = [name for name in requested if name not in self.classes]
        if unknown:
            raise KeyError(f"Unknown classes: {unknown}")
        return list(requested)

    @property
    def hash(self) -> str:
        scientific = {key: value for key, value in self.payload.items() if key != "paths"}
        blob = {
            "payload": scientific,
            "expected_counts": EXPECTED_COUNTS,
            "false_reject_budget": FALSE_REJECT_BUDGET,
        }
        raw = json.dumps(blob, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def work(self, *parts: str) -> Path:
        path = self.work_root.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path


def load_config(
    config_path: str | Path | None = None,
    data_root: str | Path | None = None,
    work_root: str | Path | None = None,
) -> Config:
    root = project_root()
    path = Path(config_path) if config_path else root / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Config must be a mapping: {path}")

    _validate_payload(payload)

    # DATA_ROOT is the public name. COMMISSIONING_DATA remains accepted.
    # An explicit data_root argument still wins. Paths are not part of the config hash.
    data_value = (
        data_root
        or os.environ.get("DATA_ROOT")
        or os.environ.get("COMMISSIONING_DATA")
        or payload["paths"]["data_root"]
    )
    work_value = work_root or os.environ.get("COMMISSIONING_WORK") or payload["paths"]["work_root"]
    return Config(
        payload=payload,
        config_path=path.resolve(),
        data_root=_resolve_path(data_value, root),
        work_root=_resolve_path(work_value, root),
    )


def _validate_payload(payload: dict[str, Any]) -> None:
    if list(payload.get("objects", [])) != OBJECTS or list(payload.get("textures", [])) != TEXTURES:
        raise RuntimeError("objects/textures in config.yaml must match the canonical lists in config.py.")
    classes = OBJECTS + TEXTURES
    if set(EXPECTED_COUNTS) != set(classes) or set(FALSE_REJECT_BUDGET) != set(classes):
        raise RuntimeError("EXPECTED_COUNTS and FALSE_REJECT_BUDGET must cover every MVTec class.")
    if int(payload["n_calibration"]) != 20 or float(payload["alpha"]) != 0.05:
        raise RuntimeError("This protocol fixes n_calibration=20 and alpha=0.05, so the threshold is max(C).")
    if "random" not in payload["strategies"] or "kcenter" not in payload["strategies"]:
        raise RuntimeError("strategies must include random and kcenter.")
    if int(payload["aulc_n_max"]) != 32:
        raise RuntimeError("Main AULC is fixed on N <= 32.")
    grid = [int(n) for n in payload["n_grid"]]
    if grid != sorted(grid) or len(grid) != len(set(grid)):
        raise RuntimeError(f"n_grid must be strictly increasing: {grid}")
    if 1 not in grid or 32 not in grid:
        raise RuntimeError("n_grid must contain 1 and 32.")
