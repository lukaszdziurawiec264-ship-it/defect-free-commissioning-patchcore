"""Command line: python -m commissioning.visa {convert|prepare|features|distances|e0|run|exchange|budget|analyze|report|all}"""

from __future__ import annotations

import argparse
import logging
import platform
import sys

from commissioning import __version__
from commissioning.io_utils import write_json
from commissioning.visa.config import load_visa_config

STAGES = ["convert", "prepare", "features", "distances", "e0", "run", "exchange", "budget", "analyze", "report", "all"]
SCORING_STAGES = {"distances", "e0", "run", "exchange", "budget", "all"}
FROZEN_MARK = "Status: **FROZEN**"


def require_frozen_preregistration(cfg, exploratory: bool) -> bool:
    """No VisA image is scored before preregistration_visa.md is frozen; afterwards it must not change.

    Returns True when the run must be marked exploratory because the preregistration is not frozen.
    """
    import hashlib

    from commissioning.config import project_root

    path = project_root() / "preregistration_visa.md"
    text = path.read_text(encoding="utf-8")
    if FROZEN_MARK not in text:
        if exploratory:
            logging.warning("preregistration_visa.md is not frozen. This run is exploratory.")
            return True
        raise RuntimeError(f"{path.name} is not frozen. Replace the DRAFT status line with '{FROZEN_MARK} on <date>'.")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record = cfg.work_root / "preregistration_visa.sha256"
    if record.is_file():
        if record.read_text(encoding="utf-8").strip() != digest:
            raise RuntimeError("preregistration_visa.md changed after the first scoring stage. It must stay frozen.")
    else:
        record.write_text(digest + "\n", encoding="utf-8")
    return False


def write_environment(cfg) -> None:
    info = {
        "commissioning": __version__,
        "protocol": "visa v2",
        "python": sys.version,
        "platform": platform.platform(),
        "config_hash": cfg.hash,
        "data_root": str(cfg.data_root),
        "work_root": str(cfg.work_root),
        "cache_root": str(cfg.cache_root),
    }
    for name in ("numpy", "scipy", "pandas", "sklearn", "matplotlib", "PIL", "anomalib", "timm"):
        try:
            info[name] = getattr(__import__(name), "__version__", "unknown")
        except ImportError:
            info[name] = None
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["torch"] = None
    write_json(cfg.work_root / "environment.json", info)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="commissioning.visa")
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--work-root", default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--raw-root", default=None, help="Extracted VisA_20220922.tar (convert only).")
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--exploratory", action="store_true")
    parser.add_argument("--e0-part", choices=["a", "b", "all"], default="all")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    cfg = load_visa_config(args.config, args.data_root, args.work_root, args.cache_root)
    classes = cfg.select_classes(args.classes)
    cfg.work_root.mkdir(parents=True, exist_ok=True)
    write_environment(cfg)
    logging.info("VisA config hash %s", cfg.hash)
    stage = args.stage
    not_frozen = require_frozen_preregistration(cfg, args.exploratory) if stage in SCORING_STAGES else False
    if stage == "convert":
        from commissioning.visa.data import convert

        if not args.raw_root:
            raise SystemExit("convert needs --raw-root pointing at the extracted VisA archive.")
        counts = convert(args.raw_root, cfg.data_root)
        logging.info("Converted counts: %s", counts)
        return
    if stage in {"prepare", "all"}:
        from commissioning.visa.data import prepare

        prepare(cfg, classes)
    if stage in {"features", "all"}:
        from commissioning.visa.caches import extract

        extract(cfg, classes, device=args.device)
    if stage in {"distances", "all"}:
        from commissioning.visa.caches import compute_distances

        compute_distances(cfg, classes, device=args.device)
    if stage in {"e0", "all"}:
        from commissioning.visa.e0 import run_e0

        gate = run_e0(cfg, classes, part=args.e0_part if stage == "e0" else "all")
        if stage == "all" and gate["status"] != "PASS" and not args.exploratory:
            raise RuntimeError("E0 did not pass. The remaining stages were not started.")
    if stage in {"run", "all"}:
        from commissioning.visa.run import run

        run(cfg, classes, exploratory=args.exploratory, force_exploratory=not_frozen)
    if stage in {"exchange", "all"}:
        from commissioning.visa.exchange import run_exchange

        run_exchange(cfg, classes)
    if stage in {"budget", "all"}:
        from commissioning.visa.budget import run_budget

        run_budget(cfg, classes)
    if stage in {"analyze", "all"}:
        from commissioning.visa.analysis import analyze

        analyze(cfg)
    if stage in {"report", "all"}:
        from commissioning.visa.analysis import load_analysis
        from commissioning.visa.report import build_report
        from commissioning.visa.run import read_gate

        frames = load_analysis(cfg.work_root / "analysis")
        path = build_report(cfg, frames, read_gate(cfg))
        logging.info("Report written to %s", path)


if __name__ == "__main__":
    main()
