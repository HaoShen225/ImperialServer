"""Run SAR for patient-volume BS=8 and random-slice BS=4 at LR=1e-3."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from run_tta import run_experiment
from utils import get_device, load_config


METHOD = "sar"
STREAM_MODES = ("patient_volume", "slice_random")
SOURCE_SEEDS = (2022, 2023, 2024, 2025, 2026)
BATCH_SIZES = {"patient_volume": 8, "slice_random": 4}
LEARNING_RATE = 1e-3
PROFILE_KIND = "sar_bs8_bs4_lr_1e-3_comparison"
DEFAULT_RESULTS_ROOT = Path(
    "results/Stochastic_Ini_ForegroundOnly/"
    "dual_protocol_patient_bs8_slice_bs4/adaptive_lr_1e-3"
)


def comparison_cells() -> tuple[tuple[int, str], ...]:
    """Return PBS cell order as (seed, stream_mode) tuples."""
    return tuple(
        (seed, stream_mode)
        for stream_mode in STREAM_MODES
        for seed in SOURCE_SEEDS
    )


def resolve_array_index(index: int) -> tuple[int, str]:
    cells = comparison_cells()
    if not 0 <= index < len(cells):
        raise ValueError(f"SAR array index must be in [0, {len(cells) - 1}]")
    return cells[index]


def configure_sar_comparison(
    cfg: dict[str, Any],
    stream_mode: str,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
) -> dict[str, Any]:
    """Return an isolated, fully resolved configuration for one SAR cell."""
    if stream_mode not in STREAM_MODES:
        raise ValueError(f"Unsupported SAR stream mode: {stream_mode}")
    resolved = deepcopy(cfg)
    resolved["tta"]["stream_mode"] = stream_mode
    resolved["tta"]["batch_size"] = BATCH_SIZES[stream_mode]
    resolved["tta"]["timing"] = "adapt_then_predict"
    resolved["tta"]["reset"] = "vendor"
    resolved["tta"]["results_dir"] = str(Path(results_root))
    resolved["methods"][METHOD]["lr"] = LEARNING_RATE
    resolved["methods"][METHOD]["profile_kind"] = PROFILE_KIND
    return resolved


def run_root(results_root: str | Path, seed: int, stream_mode: str) -> Path:
    suffix = (
        "adapt_then_predict_vendor"
        if stream_mode == "patient_volume"
        else "slice_random_adapt_then_predict_vendor"
    )
    return Path(results_root) / METHOD / f"seed{seed}" / suffix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source-seed", required=True, type=int, choices=SOURCE_SEEDS)
    parser.add_argument("--stream-mode", required=True, choices=STREAM_MODES)
    parser.add_argument("--vendors", nargs="+", choices=["B", "C", "D"], default=["B", "C", "D"])
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = configure_sar_comparison(
        load_config(args.config), args.stream_mode, args.results_root
    )
    manifest = run_experiment(
        cfg, METHOD, int(args.source_seed), list(args.vendors), get_device(args.device)
    )
    print(json.dumps({
        "method": manifest["method"],
        "source_seed": manifest["source_seed"],
        "stream_mode": manifest["stream_mode"],
        "batch_size": manifest["resolved_config"]["tta"]["batch_size"],
        "learning_rate": manifest["resolved_method_config"]["lr"],
        "results_root": str(args.results_root),
        "summaries": manifest["summaries"],
    }, indent=2))


if __name__ == "__main__":
    main()
