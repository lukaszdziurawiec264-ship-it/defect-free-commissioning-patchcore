"""Wide-ResNet50-2 patch features: layer2 + layer3, 3x3 pooling, 784 x 1536."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from commissioning.config import Config
from commissioning.data import absolute_path, collect_class, load_split, relative_path
from commissioning.io_utils import atomic_savez, read_json, write_json

LOGGER = logging.getLogger(__name__)


def read_rgb(path: Path, class_name: str) -> Image.Image:
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            rgb.load()
        if rgb.size[0] < 1 or rgb.size[1] < 1:
            raise ValueError("empty image")
        return rgb
    except Exception as exc:
        raise RuntimeError(f"{class_name}: {path}") from exc


def build_transform(cfg: Config):
    """Match Anomalib's Patchcore.configure_pre_processor(256, center crop 224)."""
    from torchvision.transforms import v2

    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch_float32(), scale=True),
            v2.Resize((cfg.resize, cfg.resize), antialias=True),
            v2.CenterCrop((cfg.center_crop, cfg.center_crop)),
            v2.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
        ]
    )


def torch_float32():
    import torch

    return torch.float32


def build_feature_extractor(cfg: Config):
    """Prefer Anomalib's extractor so the cache and E0b share weights and hooks."""
    import torch.nn as nn

    try:
        from anomalib.models.components import TimmFeatureExtractor
    except ImportError as exc:
        LOGGER.warning("anomalib is not installed (%s); using a timm hook extractor.", exc)
        return _HookExtractor(cfg.backbone, cfg.layers), f"timm.hooks ({exc})"

    model = TimmFeatureExtractor(backbone=cfg.backbone, layers=list(cfg.layers), pre_trained=True)
    if not isinstance(model, nn.Module):
        raise RuntimeError("TimmFeatureExtractor did not return a torch module.")
    return model, f"anomalib.TimmFeatureExtractor weights={_weight_tag(model)}"


def _weight_tag(model) -> str:
    """timm names its weights, e.g. wide_resnet50_2.racm_in1k vs .tv_in1k. The paper must state which."""
    for module in model.modules():
        config = getattr(module, "pretrained_cfg", None)
        if isinstance(config, dict) and (config.get("tag") or config.get("hf_hub_id")):
            architecture = config.get("architecture", "")
            tag = config.get("tag", "")
            return f"{architecture}.{tag}" if tag else str(config.get("hf_hub_id"))
    return "unknown"


class _HookExtractor:
    def __init__(self, backbone: str, layers: list[str]) -> None:
        import timm
        import torch.nn as nn

        self.layers = list(layers)
        self.backbone = timm.create_model(backbone, pretrained=True)
        self.backbone.eval()
        named = dict(self.backbone.named_modules())
        missing = [name for name in self.layers if name not in named]
        if missing:
            raise RuntimeError(f"Backbone {backbone} has no layers {missing}")
        self._features: dict = {}
        for name in self.layers:
            named[name].register_forward_hook(self._make_hook(name))
        self._nn = nn

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output) -> None:
            self._features[name] = output

        return hook

    def eval(self):
        self.backbone.eval()
        return self

    def to(self, device):
        self.backbone.to(device)
        return self

    def __call__(self, batch):
        self._features = {}
        _ = self.backbone(batch)
        missing = [name for name in self.layers if name not in self._features]
        if missing:
            raise RuntimeError(f"Feature hooks did not fire for {missing}")
        return {name: self._features[name] for name in self.layers}


def pooled_embedding(features: dict, layers: list[str], pool, upsample_mode: str):
    """Average-pool each map, upsample later maps to the first map, then concatenate."""
    import torch
    import torch.nn.functional as F

    pooled = [pool(features[name]) for name in layers]
    embedding = pooled[0]
    for layer in pooled[1:]:
        kwargs = {"align_corners": False} if upsample_mode in {"bilinear", "bicubic"} else {}
        layer = F.interpolate(layer, size=embedding.shape[-2:], mode=upsample_mode, **kwargs)
        embedding = torch.cat((embedding, layer), dim=1)
    return embedding


def patches_and_gap(embedding):
    """embedding: (B, C, H, W) -> patches (B, H*W, C), gap (B, C)."""
    batch, channels, _height, _width = embedding.shape
    patches = embedding.permute(0, 2, 3, 1).reshape(batch, -1, channels)
    gap = patches.mean(dim=1)
    return patches, gap


def feature_cache_path(cfg: Config, class_name: str) -> Path:
    return cfg.work_root / "cache" / "features" / f"{class_name}.npz"


def feature_meta_path(cfg: Config, class_name: str) -> Path:
    return cfg.work_root / "cache" / "features" / f"{class_name}.json"


def image_order(cfg: Config, class_name: str) -> list[str]:
    """Train, then official good test, then defects. This order is the row order of the cache."""
    collected = collect_class(cfg.data_root, class_name)
    paths = [relative_path(path, cfg.data_root) for path in collected["train"]]
    paths.extend(relative_path(path, cfg.data_root) for path in collected["test_good"])
    paths.extend(relative_path(path, cfg.data_root) for _, path in collected["test_defect"])
    return paths


def extract_class(cfg: Config, class_name: str, device: str | None = None) -> None:
    import torch
    import torch.nn as nn

    out_path = feature_cache_path(cfg, class_name)
    meta_path = feature_meta_path(cfg, class_name)
    if out_path.is_file() and meta_path.is_file():
        meta = read_json(meta_path)
        if meta.get("config_hash") == cfg.hash and int(meta.get("n_images", 0)) > 0:
            LOGGER.info("Feature cache exists for %s", class_name)
            return

    paths = image_order(cfg, class_name)
    if not paths:
        raise RuntimeError(f"{class_name}: no images to encode")
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    extractor, source = build_feature_extractor(cfg)
    extractor = extractor.eval().to(device_t)
    transform = build_transform(cfg)
    pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1).to(device_t)

    patch_rows: list[np.ndarray] = []
    gap_rows: list[np.ndarray] = []
    batch_size = cfg.feature_batch_size
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = []
            for relative in batch_paths:
                rgb = read_rgb(absolute_path(cfg.data_root, relative), class_name)
                images.append(transform(rgb))
            batch = torch.stack(images, dim=0).to(device_t)
            features = extractor(batch)
            if not isinstance(features, dict):
                features = {name: tensor for name, tensor in zip(cfg.layers, features)}
            embedding = pooled_embedding(features, cfg.layers, pool, cfg.upsample_mode)
            if embedding.shape[1] != 1536 or embedding.shape[-2:] != (28, 28):
                raise RuntimeError(f"{class_name}: unexpected embedding shape {tuple(embedding.shape)}")
            patches, gap = patches_and_gap(embedding)
            patches_np = patches.detach().cpu().numpy()
            gap_np = gap.detach().cpu().numpy()
            if not np.isfinite(patches_np).all() or not np.isfinite(gap_np).all():
                raise RuntimeError(f"{class_name}: non-finite features in batch starting at {batch_paths[0]}")
            patch_rows.append(patches_np.astype(np.float16))
            gap_rows.append(gap_np.astype(np.float32))

    patches_all = np.concatenate(patch_rows, axis=0)
    gap_all = np.concatenate(gap_rows, axis=0)
    if patches_all.shape[1:] != (784, 1536):
        raise RuntimeError(f"{class_name}: expected patches (N, 784, 1536), got {patches_all.shape}")
    split = load_split(cfg, class_name, 0)
    n_train = int(split["n_train"])
    n_test_good = int(split["n_test_good"])
    n_test_defect = int(split["n_test_defect"])
    if patches_all.shape[0] != n_train + n_test_good + n_test_defect:
        raise RuntimeError(f"{class_name}: feature rows do not match split counts")

    atomic_savez(
        out_path,
        patches=patches_all,
        gap=gap_all,
        paths=np.array(paths),
    )
    write_json(
        meta_path,
        {
            "config_hash": cfg.hash,
            "class_name": class_name,
            "n_images": int(patches_all.shape[0]),
            "n_train": n_train,
            "n_test_good": n_test_good,
            "n_test_defect": n_test_defect,
            "patch_shape": [784, 1536],
            "extractor": source,
            "upsample_mode": cfg.upsample_mode,
            "resize": cfg.resize,
            "center_crop": cfg.center_crop,
            "bytes_patches": int(patches_all.nbytes),
        },
    )
    LOGGER.info(
        "%s features: %s images, %.2f GB float16",
        class_name,
        patches_all.shape[0],
        patches_all.nbytes / (1024**3),
    )


def load_gap_table(cfg: Config, class_name: str) -> dict:
    """GAP vectors only. Trajectory selection must not load the patch tensor."""
    path = feature_cache_path(cfg, class_name)
    meta_path = feature_meta_path(cfg, class_name)
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Missing feature cache for {class_name}. Run features first.")
    meta = read_json(meta_path)
    if meta.get("config_hash") != cfg.hash:
        raise RuntimeError(f"Feature cache hash mismatch for {class_name}.")
    with np.load(path, allow_pickle=False) as data:
        gap = np.array(data["gap"])
        paths = [str(item) for item in data["paths"].tolist()]
    if not np.isfinite(gap).all():
        raise RuntimeError(f"{class_name}: non-finite GAP features")
    return {"gap": gap, "paths": paths, "meta": meta}


def load_feature_cache(cfg: Config, class_name: str) -> dict:
    path = feature_cache_path(cfg, class_name)
    meta_path = feature_meta_path(cfg, class_name)
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Missing feature cache for {class_name}. Run features first.")
    meta = read_json(meta_path)
    if meta.get("config_hash") != cfg.hash:
        raise RuntimeError(f"Feature cache hash mismatch for {class_name}.")
    with np.load(path, allow_pickle=False) as data:
        patches = np.array(data["patches"])
        gap = np.array(data["gap"])
        paths = [str(item) for item in data["paths"].tolist()]
    if patches.dtype != np.float16 or patches.shape[1:] != (784, 1536):
        raise RuntimeError(f"{class_name}: bad patch cache shape {patches.shape} {patches.dtype}")
    if not np.isfinite(patches.astype(np.float32)).all() or not np.isfinite(gap).all():
        raise RuntimeError(f"{class_name}: non-finite values in the feature cache")
    return {"patches": patches, "gap": gap, "paths": paths, "meta": meta}


def extract(cfg: Config, classes: list[str], device: str | None = None) -> None:
    for class_name in classes:
        extract_class(cfg, class_name, device=device)
