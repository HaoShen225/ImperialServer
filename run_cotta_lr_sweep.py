"""Run one locked CoTTA learning-rate sweep combination."""

from __future__ import annotations

import argparse
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

from run_tta import run_experiment
from utils import get_device, load_config


SWEEP_LEARNING_RATES = {
    Decimal("1e-5"): "1e-5",
    Decimal("1e-4"): "1e-4",
    Decimal("1e-3"): "1e-3",
    Decimal("1e-2"): "1e-2",
    Decimal("1e-1"): "1e-1",
    Decimal("1"): "1",
}


def resolve_learning_rate(value: str) -> tuple[float, str]:
    try:
        decimal_value = Decimal(value)
    except Exception as error:
        raise ValueError(f"Invalid CoTTA learning rate: {value!r}") from error
    try:
        tag = SWEEP_LEARNING_RATES[decimal_value]
    except KeyError as error:
        allowed = ", ".join(SWEEP_LEARNING_RATES.values())
        raise ValueError(
            f"CoTTA sweep learning rate must be one of: {allowed}"
        ) from error
    return float(decimal_value), tag


def configure_sweep(
    cfg: dict[str, Any],
    learning_rate: float,
    learning_rate_tag: str,
    stream_mode: str,
    results_root: str | Path,
) -> dict[str, Any]:
    resolved = deepcopy(cfg)
    resolved["methods"]["cotta"]["lr"] = learning_rate
    resolved["methods"]["cotta"]["profile_kind"] = "lr_sweep"
    resolved["tta"]["stream_mode"] = stream_mode
    resolved["tta"]["batch_size"] = 4 if stream_mode == "patient_volume" else 8
    resolved["tta"]["results_dir"] = str(
        Path(results_root) / f"lr_{learning_rate_tag}"
    )
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source-seed", type=int, required=True)
    parser.add_argument("--learning-rate", required=True)
    parser.add_argument(
        "--stream-mode",
        choices=["patient_volume", "slice_random"],
        required=True,
    )
    parser.add_argument(
        "--vendors", nargs="+", choices=["B", "C", "D"], default=["B", "C", "D"]
    )
    parser.add_argument(
        "--results-root",
        default="results/Stochastic_Ini_ForegroundOnly/cotta_lr_sweep",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    learning_rate, tag = resolve_learning_rate(args.learning_rate)
    cfg = configure_sweep(
        load_config(args.config),
        learning_rate,
        tag,
        args.stream_mode,
        args.results_root,
    )
    if int(args.source_seed) not in {
        int(seed) for seed in cfg["experiment"]["source_seeds"]
    }:
        raise ValueError(f"Unknown source seed: {args.source_seed}")
    manifest = run_experiment(
        cfg,
        "cotta",
        int(args.source_seed),
        list(args.vendors),
        get_device(args.device),
    )
    print(json.dumps({
        "method": manifest["method"],
        "source_seed": manifest["source_seed"],
        "stream_mode": manifest["stream_mode"],
        "learning_rate": learning_rate,
        "learning_rate_tag": tag,
        "summaries": manifest["summaries"],
    }, indent=2))


if __name__ == "__main__":
    main()
