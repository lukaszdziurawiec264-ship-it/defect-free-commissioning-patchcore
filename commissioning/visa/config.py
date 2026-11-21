"""Frozen VisA configuration, official counts, and paths."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from commissioning.config import project_root

# Seed indices follow this order (class_seed_offset + index).
CLASSES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]

GROUPS: dict[str, str] = {
    "candle": "multiple",
    "capsules": "multiple",
    "macaroni1": "multiple",
    "macaroni2": "multiple",
    "cashew": "single",
    "chewinggum": "single",
    "fryum": "single",
    "pipe_fryum": "single",
    "pcb1": "complex",
    "pcb2": "complex",
    "pcb3": "complex",
    "pcb4": "complex",
}

# Official one-class split, split_csv/1cls.csv. Checked before duplicate removal.
EXPECTED_COUNTS: dict[str, dict[str, int]] = {
    "candle": {"train": 900, "test_good": 100, "test_defect": 100},
    "capsules": {"train": 542, "test_good": 60, "test_defect": 100},
    "cashew": {"train": 450, "test_good": 50, "test_defect": 100},
    "chewinggum": {"train": 453, "test_good": 50, "test_defect": 100},
    "fryum": {"train": 450, "test_good": 50, "test_defect": 100},
    "macaroni1": {"train": 900, "test_good": 100, "test_defect": 100},
    "macaroni2": {"train": 900, "test_good": 100, "test_defect": 100},
    "pcb1": {"train": 904, "test_good": 100, "test_defect": 100},
    "pcb2": {"train": 901, "test_good": 100, "test_defect": 100},
    "pcb3": {"train": 905, "test_good": 101, "test_defect": 100},
    "pcb4": {"train": 904, "test_good": 101, "test_defect": 100},
    "pipe_fryum": {"train": 450, "test_good": 50, "test_defect": 100},
}

RULES = ("R20", "R60")
FPR_SOURCES = ("off", "H")


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


@dataclass
class VisaConfig:
    payload: dict[str, Any]
    config_path: Path
    data_root: Path
    work_root: Path
    cache_root: Path

    def _get(self, key: str) -> Any:
        return self.payload[key]

    seed = property(lambda self: int(self._get("seed")))
    class_seed_offset = property(lambda self: int(self._get("class_seed_offset")))
    universe_size = property(lambda self: int(self._get("universe_size")))
    n_calibration_small = property(lambda self: int(self._get("n_calibration_small")))
    n_calibration_large = property(lambda self: int(self._get("n_calibration_large")))
    alpha = property(lambda self: float(self._get("alpha")))
    n_repeats = property(lambda self: int(self._get("n_repeats")))
    strategies = property(lambda self: [str(s) for s in self._get("strategies")])
    aulc_n_max = property(lambda self: int(self._get("aulc_n_max")))
    practical_tpr = property(lambda self: float(self._get("practical_tpr")))
    practical_fpr = property(lambda self: float(self._get("practical_fpr")))
    n_star_min_repeats = property(lambda self: int(self._get("n_star_min_repeats")))
    fpr_aulc_margin = property(lambda self: float(self._get("fpr_aulc_margin")))
    tpr_cost_margin = property(lambda self: float(self._get("tpr_cost_margin")))
    stopping_deltas = property(lambda self: [float(v) for v in self._get("stopping_deltas")])
    stopping_delta_main = property(lambda self: float(self._get("stopping_delta_main")))
    bootstrap_replicates = property(lambda self: int(self._get("bootstrap_replicates")))
    bootstrap_seed = property(lambda self: int(self._get("bootstrap_seed")))
    budget_seed_tag = property(lambda self: int(self._get("budget_seed_tag")))
    budgets = property(lambda self: [int(v) for v in self._get("budgets")])
    budget_grid = property(lambda self: [int(v) for v in self._get("budget_grid")])
    budget_min_calibration = property(lambda self: int(self._get("budget_min_calibration")))
    pac_epsilon = property(lambda self: float(self._get("pac_epsilon")))
    pac_delta = property(lambda self: float(self._get("pac_delta")))
    backbone = property(lambda self: str(self._get("backbone")))
    layers = property(lambda self: [str(v) for v in self._get("layers")])
    resize = property(lambda self: int(self._get("resize")))
    center_crop = property(lambda self: int(self._get("center_crop")))
    upsample_mode = property(lambda self: str(self._get("upsample_mode")))
    imagenet_mean = property(lambda self: [float(v) for v in self._get("imagenet_mean")])
    imagenet_std = property(lambda self: [float(v) for v in self._get("imagenet_std")])
    feature_batch_size = property(lambda self: int(self._get("feature_batch_size")))
    distance_query_batch = property(lambda self: int(self._get("distance_query_batch")))
    distance_train_batch = property(lambda self: int(self._get("distance_train_batch")))
    e0 = property(lambda self: dict(self._get("e0")))
    classes = property(lambda self: [str(v) for v in self._get("classes")])

    @property
    def pool_size(self) -> int:
        return self.universe_size - self.n_calibration_large

    @property
    def n_grid(self) -> list[int]:
        """Base grid plus the full pool |P|."""
        grid = [int(n) for n in self._get("n_grid") if int(n) < self.pool_size]
        return grid + [self.pool_size]

    def class_index(self, class_name: str) -> int:
        try:
            return self.classes.index(class_name)
        except ValueError as exc:
            raise KeyError(f"Unknown VisA class: {class_name}") from exc

    def seed_key(self, class_name: str) -> int:
        return self.class_seed_offset + self.class_index(class_name)

    def group_of(self, class_name: str) -> str:
        return GROUPS[class_name]

    def select_classes(self, requested: list[str] | None) -> list[str]:
        if not requested:
            return list(self.classes)
        unknown = [name for name in requested if name not in self.classes]
        if unknown:
            raise KeyError(f"Unknown VisA classes: {unknown}")
        return list(requested)

    @property
    def hash(self) -> str:
        scientific = {key: value for key, value in self.payload.items() if key != "paths"}
        blob = {"payload": scientific, "expected_counts": EXPECTED_COUNTS, "groups": GROUPS}
        raw = json.dumps(blob, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_visa_config(
    config_path: str | Path | None = None,
    data_root: str | Path | None = None,
    work_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> VisaConfig:
    root = project_root()
    path = Path(config_path) if config_path else root / "config_visa.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Config must be a mapping: {path}")
    _validate(payload)
    paths = payload["paths"]
    return VisaConfig(
        payload=payload,
        config_path=path.resolve(),
        data_root=_resolve(data_root or os.environ.get("VISA_DATA") or paths["data_root"], root),
        work_root=_resolve(work_root or os.environ.get("VISA_WORK") or paths["work_root"], root),
        cache_root=_resolve(cache_root or os.environ.get("VISA_CACHE") or paths["cache_root"], root),
    )


def _validate(payload: dict[str, Any]) -> None:
    if payload.get("dataset") != "visa" or int(payload.get("protocol_version", 0)) != 2:
        raise RuntimeError("config_visa.yaml must declare dataset: visa and protocol_version: 2.")
    if list(payload.get("classes", [])) != CLASSES:
        raise RuntimeError("classes in config_visa.yaml must match the canonical VisA order.")
    if int(payload["n_calibration_small"]) != 20 or int(payload["n_calibration_large"]) != 60:
        raise RuntimeError("The protocol fixes |C20| = 20 and |C60| = 60.")
    if int(payload["universe_size"]) != 300 or float(payload["alpha"]) != 0.05:
        raise RuntimeError("The protocol fixes |U| = 300 and alpha = 0.05.")
    if int(payload["aulc_n_max"]) != 32:
        raise RuntimeError("Main AULC is fixed on N <= 32.")
    grid = [int(n) for n in payload["n_grid"]]
    if grid != sorted(set(grid)) or 1 not in grid or 32 not in grid:
        raise RuntimeError(f"n_grid must be strictly increasing and contain 1 and 32: {grid}")
    if "random" not in payload["strategies"] or "kcenter" not in payload["strategies"]:
        raise RuntimeError("strategies must include random and kcenter.")
    smallest_train = min(counts["train"] for counts in EXPECTED_COUNTS.values())
    if int(payload["universe_size"]) >= smallest_train:
        raise RuntimeError("The universe must leave a non-empty held-out set H in every class.")
