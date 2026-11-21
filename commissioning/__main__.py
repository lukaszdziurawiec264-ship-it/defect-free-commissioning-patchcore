"""Command line: python -m commissioning {prepare|features|distances|e0|run|analyze|report|all|explore}

explore is not part of all. It runs the post hoc analyses from exploratory_addendum.md.
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys

from commissioning import __version__
from commissioning.analysis import analyze
from commissioning.config import load_config
from commissioning.data import prepare
from commissioning.distances import compute_distances
from commissioning.e0 import gate_path, run_e0
from commissioning.features import extract
from commissioning.io_utils import read_json, write_json
from commissioning.report import build_report
from commissioning.run import run


def write_environment(cfg) -> None:
    info = {
        "commissioning": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "config_hash": cfg.hash,
        "data_root": str(cfg.data_root),
        "work_root": str(cfg.work_root),
    }
    for name in ("numpy", "scipy", "pandas", "sklearn", "matplotlib", "PIL"):
        try:
            module = __import__(name if name != "sklearn" else "sklearn")
            if name == "PIL":
                module = __import__("PIL")
            info[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            info[name] = None
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["torch"] = None
    try:
        import anomalib

        info["anomalib"] = getattr(anomalib, "__version__", "unknown")
    except ImportError:
        info["anomalib"] = None
    write_json(cfg.work_root / "environment.json", info)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="commissioning")
    parser.add_argument(
        "stage", choices=["prepare", "features", "distances", "e0", "run", "analyze", "report", "all", "explore"]
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--work-root", default=None)
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--exploratory", action="store_true")
    parser.add_argument("--e0-part", choices=["a", "b", "all"], default="all")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    cfg = load_config(args.config, data_root=args.data_root, work_root=args.work_root)
    classes = cfg.select_classes(args.classes)
    write_environment(cfg)
    logging.info("Config hash %s", cfg.hash)
    if args.stage in {"prepare", "all"}:
        prepare(cfg, classes)
    if args.stage in {"features", "all"}:
        extract(cfg, classes, device=args.device)
    if args.stage in {"distances", "all"}:
        compute_distances(cfg, classes, device=args.device)
    if args.stage in {"e0", "all"}:
        gate = run_e0(cfg, classes, part=args.e0_part if args.stage == "e0" else "all")
        if args.stage == "all" and gate["status"] != "PASS" and not args.exploratory:
            raise RuntimeError("E0 did not pass. The remaining stages were not started.")
    if args.stage in {"run", "all"}:
        run(cfg, classes, exploratory=args.exploratory)
    if args.stage in {"analyze", "all"}:
        analyze(cfg)
    if args.stage in {"report", "all"}:
        from commissioning.analysis import load_analysis

        frames = load_analysis(cfg.work_root / "analysis")
        gate_file = gate_path(cfg)
        gate = read_json(gate_file) if gate_file.is_file() else None
        path = build_report(
            frames,
            cfg.work_root / "report",
            gate=gate,
            data_root=cfg.data_root,
            work_root=cfg.work_root,
            resize=cfg.resize,
            crop=cfg.center_crop,
        )
        logging.info("Report written to %s", path)
    if args.stage == "explore":
        from commissioning.exploratory import explore

        explore(cfg, classes)


if __name__ == "__main__":
    main()
