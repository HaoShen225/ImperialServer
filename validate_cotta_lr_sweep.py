"""Validate one completed CoTTA learning-rate sweep combination."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from run_cotta_lr_sweep import resolve_learning_rate
from utils import load_config
from validate_cotta_results import LOCKED_COTTA, validate_run


def validate(
    config_path: str,
    results_root: Path,
    seed: int,
    learning_rate_text: str,
    stream_mode: str,
) -> None:
    learning_rate, tag = resolve_learning_rate(learning_rate_text)
    cfg = load_config(config_path)
    cfg["tta"]["results_dir"] = str(results_root / f"lr_{tag}")
    expected_method_cfg = deepcopy(LOCKED_COTTA)
    expected_method_cfg["profile_kind"] = "lr_sweep"
    expected_method_cfg["lr"] = learning_rate
    result = validate_run(
        seed,
        stream_mode,
        cfg,
        expected_method_cfg=expected_method_cfg,
    )
    method_cfg = result["manifest"]["resolved_method_config"]
    if method_cfg != expected_method_cfg:
        raise RuntimeError("CoTTA sweep changed a non-learning-rate method setting")
    print(
        f"[VALIDATED] method=cotta lr={tag} seed={seed} "
        f"stream={stream_mode} results_root={results_root}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/Stochastic_Ini_ForegroundOnly/cotta_lr_sweep"),
    )
    parser.add_argument("--source-seed", type=int, required=True)
    parser.add_argument("--learning-rate", required=True)
    parser.add_argument(
        "--stream-mode",
        choices=["patient_volume", "slice_random"],
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate(
        args.config,
        args.results_root,
        args.source_seed,
        args.learning_rate,
        args.stream_mode,
    )


if __name__ == "__main__":
    main()
