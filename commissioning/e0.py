"""E0a reference sanity check and E0b own-cache versus Anomalib.

RQ trajectories refuse to start unless e0/gate.json says PASS for this config hash.
"""

from __future__ import annotations

import inspect
import logging

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from commissioning.config import Config
from commissioning.data import absolute_path, load_split, split_seeds
from commissioning.distances import load_distance_cache
from commissioning.features import build_transform, load_feature_cache, read_rgb
from commissioning.integrity import assert_methods_agree, check_corrupted_input, check_feature_alignment
from commissioning.io_utils import read_json, write_json
from commissioning.patchcore import scores_for_bank
from commissioning.selection import random_order

LOGGER = logging.getLogger(__name__)


def gate_path(cfg: Config):
    return cfg.work_root / "e0" / "gate.json"


def _class_path(cfg: Config, class_name: str):
    return cfg.work_root / "e0" / f"{class_name}.json"


def _kcenter_greedy_class():
    try:
        from anomalib.models.components import KCenterGreedy
    except ImportError:
        from anomalib.models.components.sampling import KCenterGreedy
    return KCenterGreedy


def check_anomalib_upsampling(cfg: Config) -> str:
    """The cache and Anomalib must resize layer3 the same way, otherwise E0b compares two pipelines."""
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    source = inspect.getsource(PatchcoreModel.generate_embedding)
    modes = [mode for mode in ("bilinear", "nearest", "bicubic") if f'mode="{mode}"' in source or f"mode='{mode}'" in source]
    if len(modes) != 1:
        raise RuntimeError(f"Cannot read the interpolation mode of the installed Anomalib PatchCore: {modes}")
    if modes[0] != cfg.upsample_mode:
        raise RuntimeError(
            f"Installed Anomalib upsamples layer3 with {modes[0]!r}, config.yaml uses {cfg.upsample_mode!r}. "
            "Set upsample_mode in config.yaml to match before computing features."
        )
    return modes[0]


def record_sampler_source(cfg: Config) -> str:
    import anomalib

    chunks = [f"anomalib {getattr(anomalib, '__version__', 'unknown')}"]
    try:
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel

        chunks.append(inspect.getsource(PatchcoreModel.subsample_embedding))
        chunks.append(inspect.getsource(PatchcoreModel.generate_embedding))
    except Exception as exc:
        chunks.append(f"Could not read PatchcoreModel source: {exc}")
    try:
        from anomalib.models.components.sampling import KCenterGreedy
    except ImportError:
        try:
            from anomalib.models.components import KCenterGreedy
        except ImportError as exc:
            KCenterGreedy = None
            chunks.append(f"Could not import KCenterGreedy: {exc}")
    if KCenterGreedy is not None:
        chunks.append(inspect.getsource(KCenterGreedy.sample_coreset))
    path = cfg.work_root / "e0" / "sampler_source.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(chunks), encoding="utf-8")
    return str(getattr(anomalib, "__version__", "unknown"))


def _batches(cfg: Config, class_name: str, relative_paths: list[str], batch_size: int):
    import torch

    transform = build_transform(cfg)
    for start in range(0, len(relative_paths), batch_size):
        chunk = relative_paths[start : start + batch_size]
        images = [transform(read_rgb(absolute_path(cfg.data_root, path), class_name)) for path in chunk]
        yield torch.stack(images, dim=0)


def _prediction_scores(output) -> "np.ndarray":
    import torch

    if hasattr(output, "pred_score"):
        scores = output.pred_score
    elif isinstance(output, tuple) and len(output) >= 2:
        scores = output[1]
    else:
        raise RuntimeError(f"Anomalib output has no image score: {type(output)}")
    array = scores.detach().cpu().numpy().reshape(-1)
    if not np.isfinite(array).all():
        raise RuntimeError("Anomalib returned a non-finite image score.")
    return array


def _fit_anomalib(cfg: Config, class_name: str, train_paths: list[str], num_neighbors: int, full_bank: bool):
    """Training-mode forward returns the patch embedding in every Anomalib 2.x release.

    Embeddings are collected here instead of through ``embedding_store`` or ``subsample_embedding``,
    whose signatures changed between releases.
    """
    import torch
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchcoreModel(
        layers=list(cfg.layers),
        backbone=cfg.backbone,
        pre_trained=True,
        num_neighbors=num_neighbors,
    ).to(device)
    model.train()
    collected = []
    with torch.no_grad():
        for batch in _batches(cfg, class_name, train_paths, batch_size=4):
            embedding = model(batch.to(device))
            if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2:
                raise RuntimeError(f"{class_name}: unexpected Anomalib training output {type(embedding)}")
            collected.append(embedding)
            store = getattr(model, "embedding_store", None)
            if isinstance(store, list):
                store.clear()
    model.eval()
    stored = torch.vstack(collected)
    del collected
    expected_rows = len(train_paths) * 784
    if int(stored.shape[0]) != expected_rows or int(stored.shape[1]) != 1536:
        raise RuntimeError(f"{class_name}: Anomalib embeddings {tuple(stored.shape)}, expected ({expected_rows}, 1536)")
    if full_bank:
        # sampling_ratio=1 would still run KCenterGreedy over every patch. The sampler is skipped.
        model.memory_bank = stored
    else:
        sampler = _kcenter_greedy_class()(embedding=stored, sampling_ratio=float(cfg.e0["coreset_ratio"]))
        model.memory_bank = sampler.sample_coreset()
        del stored
    if int(model.memory_bank.shape[0]) == 0:
        raise RuntimeError(f"{class_name}: memory bank is empty.")
    return model, device


def _release(model) -> None:
    import gc

    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _score_test(cfg: Config, class_name: str, model, device) -> tuple[np.ndarray, np.ndarray]:
    import torch

    split = load_split(cfg, class_name, 0)
    test_paths = list(split["test_good"]) + [item["path"] for item in split["test_defect"]]
    labels = np.array([0] * len(split["test_good"]) + [1] * len(split["test_defect"]), dtype=int)
    scores = []
    model.eval()
    with torch.no_grad():
        for batch in _batches(cfg, class_name, test_paths, batch_size=2):
            output = model(batch.to(device))
            scores.append(_prediction_scores(output))
    stacked = np.concatenate(scores)
    if stacked.shape != labels.shape:
        raise RuntimeError(f"{class_name}: scored {stacked.shape[0]} test images, expected {labels.shape[0]}.")
    return stacked, labels


def own_full_bank_scores(split: dict, distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_train = int(split["n_train"])
    n_test = int(split["n_test_good"]) + int(split["n_test_defect"])
    queries = np.arange(n_train, n_train + n_test)
    bank = np.arange(n_train)
    scores = scores_for_bank(distance, queries, bank)
    labels = np.array([0] * int(split["n_test_good"]) + [1] * int(split["n_test_defect"]), dtype=int)
    return scores, labels


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def _minmax(values: np.ndarray) -> np.ndarray:
    span = float(values.max() - values.min())
    if span <= 0.0:
        return np.zeros_like(values, dtype=float)
    return (values - values.min()) / span


def run_class(cfg: Config, class_name: str, part: str, train_paths: list[str], distance: np.ndarray) -> dict:
    path = _class_path(cfg, class_name)
    existing = read_json(path) if path.is_file() else {}
    if existing.get("config_hash") != cfg.hash:
        existing = {}
    split = load_split(cfg, class_name, 0)
    if part in {"a", "all"} and "e0a" not in existing:
        LOGGER.info("E0a %s", class_name)
        model, device = _fit_anomalib(cfg, class_name, train_paths, int(cfg.e0["num_neighbors_ref"]), full_bank=False)
        scores, labels = _score_test(cfg, class_name, model, device)
        existing["e0a"] = {"auroc": _auroc(labels, scores), "n_test": int(labels.size)}
        _release(model)
        write_json(path, {**existing, "config_hash": cfg.hash, "class_name": class_name})
    if part in {"b", "all"} and "e0b" not in existing:
        LOGGER.info("E0b %s", class_name)
        model, device = _fit_anomalib(cfg, class_name, train_paths, int(cfg.e0["num_neighbors_rq"]), full_bank=True)
        anomalib_scores, labels = _score_test(cfg, class_name, model, device)
        _release(model)
        own_scores, own_labels = own_full_bank_scores(split, distance)
        if not np.array_equal(labels, own_labels):
            raise RuntimeError(f"{class_name}: E0b label order differs between the two implementations.")
        rho = float(spearmanr(own_scores, anomalib_scores).statistic)
        existing["e0b"] = {
            "spearman": rho,
            "auroc_own": _auroc(labels, own_scores),
            "auroc_anomalib": _auroc(labels, anomalib_scores),
            "max_abs_score_gap": float(np.max(np.abs(own_scores - anomalib_scores))),
            "max_abs_gap_after_class_minmax": float(np.max(np.abs(_minmax(own_scores) - _minmax(anomalib_scores)))),
        }
    existing["config_hash"] = cfg.hash
    existing["class_name"] = class_name
    write_json(path, existing)
    return existing


def distance_agreement(cfg: Config, class_name: str, features: dict, distance: np.ndarray, paths: list[str]) -> dict[str, float]:
    split = load_split(cfg, class_name, 0)
    index = {path: position for position, path in enumerate(paths)}
    pool = [index[path] for path in split["pool"]]
    _split_rng, random_rng, _kcenter_rng = split_seeds(cfg, class_name, 0)
    order = random_order(pool, random_rng)
    queries = [index[path] for path in list(split["calibration"])[:2] + list(split["test_good"])[:2]]
    gaps = {}
    for n in (1, 8, len(order)):
        if n > len(order):
            continue
        gaps[str(n)] = assert_methods_agree(features["patches"], distance, queries, order[:n])
    return gaps


def _passed_e0b(cfg: Config, rows: list[dict]) -> tuple[bool, dict]:
    rhos = [float(row["e0b"]["spearman"]) for row in rows]
    deltas = [abs(float(row["e0b"]["auroc_own"]) - float(row["e0b"]["auroc_anomalib"])) for row in rows]
    own_mean = float(np.mean([float(row["e0b"]["auroc_own"]) for row in rows]))
    anomalib_mean = float(np.mean([float(row["e0b"]["auroc_anomalib"]) for row in rows]))
    mean_delta = abs(own_mean - anomalib_mean)
    summary = {
        "spearman_min": float(np.min(rhos)),
        "spearman_median": float(np.median(rhos)),
        "mean_auroc_own": own_mean,
        "mean_auroc_anomalib": anomalib_mean,
        "abs_delta_of_mean_auroc": mean_delta,
        "mean_of_per_class_abs_auroc_delta": float(np.mean(deltas)),
        "failed_classes": [
            row["class_name"]
            for row in rows
            if float(row["e0b"]["spearman"]) < float(cfg.e0["spearman_min"])
        ],
    }
    ok = summary["spearman_min"] >= float(cfg.e0["spearman_min"]) and mean_delta <= float(cfg.e0["auroc_mean_tolerance"])
    return ok, summary


def run_e0(cfg: Config, classes: list[str], part: str = "all") -> dict:
    version = record_sampler_source(cfg)
    upsampling = check_anomalib_upsampling(cfg)
    check_corrupted_input()
    rows = []
    integrity = {"corrupted_input": "pass", "anomalib_upsampling": upsampling, "classes": {}}
    for class_name in classes:
        features = load_feature_cache(cfg, class_name)
        check_feature_alignment(cfg, class_name, features)
        distance_cache = load_distance_cache(cfg, class_name)
        if list(distance_cache["meta"]["paths"]) != features["paths"]:
            raise RuntimeError(f"{class_name}: feature and distance caches are not aligned.")
        agreement = distance_agreement(cfg, class_name, features, distance_cache["D"], features["paths"])
        integrity["classes"][class_name] = {"distance_vs_knn": agreement}
        train_paths = features["paths"][: int(features["meta"]["n_train"])]
        del features
        rows.append(run_class(cfg, class_name, part, train_paths, distance_cache["D"]))
        del distance_cache
    e0a_ok = True
    e0a_summary = None
    if part in {"a", "all"}:
        per_class = {row["class_name"]: float(row["e0a"]["auroc"]) for row in rows}
        e0a_summary = {
            "mean_auroc": float(np.mean(list(per_class.values()))),
            "min_auroc": float(np.min(list(per_class.values()))),
            "per_class": per_class,
        }
        e0a_ok = e0a_summary["mean_auroc"] >= float(cfg.e0["mean_auroc_floor"])
    e0b_ok = True
    e0b_summary = None
    if part in {"b", "all"}:
        e0b_ok, e0b_summary = _passed_e0b(cfg, rows)
    complete = part == "all" and set(classes) == set(cfg.classes)
    status = "PASS" if complete and e0a_ok and e0b_ok else "FAIL"
    if e0a_ok and e0b_ok and not complete:
        status = "PARTIAL"
    payload = {
        "status": status,
        "config_hash": cfg.hash,
        "anomalib_version": version,
        "part": part,
        "classes": classes,
        "e0a": e0a_summary,
        "e0b": e0b_summary,
        "per_class": rows,
        "integrity": integrity,
        "reference_mean_auroc": float(cfg.e0["reference_mean_auroc"]),
    }
    write_json(gate_path(cfg), payload)
    LOGGER.info("E0 gate %s", status)
    return payload
