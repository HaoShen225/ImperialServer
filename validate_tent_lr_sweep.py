"""Validate one completed TENT learning-rate sweep combination."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from run_tent_lr_sweep import resolve_learning_rate
from target_order_validation import validate_target_order, validate_target_slice_order
from utils import file_sha256, load_config


PATIENT_COUNTS = {
    "B": {"patients": 125, "volumes": 250},
    "C": {"patients": 50, "volumes": 100},
    "D": {"patients": 50, "volumes": 100},
}
SLICE_COUNTS = {
    "B": {"patients": 125, "slices": 2049, "batches": 257},
    "C": {"patients": 50, "slices": 806, "batches": 101},
    "D": {"patients": 50, "slices": 835, "batches": 105},
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _result_root(
    results_root: Path,
    tag: str,
    seed: int,
    stream_mode: str,
) -> Path:
    protocol = (
        "adapt_then_predict_vendor"
        if stream_mode == "patient_volume"
        else "slice_random_adapt_then_predict_vendor"
    )
    return results_root / f"lr_{tag}" / "tent" / f"seed{seed}" / protocol


def _validate_adaptation(adaptation: dict[str, Any], maximum_batch_size: int) -> None:
    if not adaptation["updated"]:
        raise RuntimeError("TENT sweep skipped an adaptation update")
    if int(adaptation["n_seen"]) != int(adaptation["n_selected"]):
        raise RuntimeError("TENT sweep did not select the complete arrival batch")
    if not 1 <= int(adaptation["n_seen"]) <= maximum_batch_size:
        raise RuntimeError("TENT sweep recorded an invalid arrival batch size")
    for label, value in (
        ("entropy", adaptation["loss"]),
        ("parameter drift", adaptation["extras"]["parameter_drift"]),
    ):
        if value is None or not math.isfinite(float(value)):
            raise RuntimeError(f"TENT sweep produced non-finite {label}: {value}")


def validate(
    config_path: str,
    results_root: Path,
    seed: int,
    learning_rate_text: str,
    stream_mode: str,
) -> None:
    learning_rate, tag = resolve_learning_rate(learning_rate_text)
    cfg = load_config(config_path)
    root = _result_root(results_root, tag, seed, stream_mode)
    manifest = _read_json(root / "run_manifest.json")
    expected_batch_size = 4 if stream_mode == "patient_volume" else 8
    method_cfg = manifest["resolved_method_config"]
    if (
        manifest.get("method") != "tent"
        or int(manifest.get("source_seed")) != seed
        or manifest.get("stream_mode") != stream_mode
        or manifest.get("vendors") != ["B", "C", "D"]
    ):
        raise RuntimeError("TENT sweep manifest identifies the wrong experiment")
    if (
        method_cfg.get("profile_kind") != "lr_sweep"
        or float(method_cfg.get("lr")) != learning_rate
        or method_cfg.get("optimizer") != "sgd"
        or float(method_cfg.get("momentum")) != 0.9
        or float(method_cfg.get("weight_decay")) != 0.0
    ):
        raise RuntimeError("TENT sweep manifest contains the wrong optimizer profile")
    if int(manifest["resolved_config"]["tta"]["batch_size"]) != expected_batch_size:
        raise RuntimeError("TENT sweep manifest contains the wrong batch size")
    checkpoint = Path(cfg["source"]["checkpoint_dir"]) / f"seed{seed}_best.pt"
    if manifest["source_checkpoint_sha256"] != file_sha256(checkpoint):
        raise RuntimeError("TENT sweep used the wrong source checkpoint")
    trainable_parameters = manifest.get("trainable_parameters", [])
    if not trainable_parameters:
        raise RuntimeError("TENT sweep exposes no BN affine parameters")
    if any(
        not name.endswith((".weight", ".bias"))
        for name in trainable_parameters
    ):
        raise RuntimeError("TENT sweep exposes parameters outside BN affine tensors")

    for vendor in ("B", "C", "D"):
        records = _read_jsonl(root / f"vendor_{vendor}.jsonl")
        if stream_mode == "patient_volume":
            expected = PATIENT_COUNTS[vendor]
            validate_target_order(manifest, records, cfg, vendor, seed)
            if (
                len(records) != expected["volumes"]
                or len({record["patient_id"] for record in records})
                != expected["patients"]
            ):
                raise RuntimeError(f"Vendor {vendor} has incorrect volume coverage")
            adaptations = [
                adaptation
                for record in records
                for adaptation in record["adaptation"]
            ]
        else:
            expected = SLICE_COUNTS[vendor]
            validate_target_slice_order(manifest, records, cfg, vendor, seed)
            batches = _read_jsonl(root / f"vendor_{vendor}_batches.jsonl")
            if (
                len(records) != expected["slices"]
                or len(batches) != expected["batches"]
                or len({record["patient_id"] for record in records})
                != expected["patients"]
            ):
                raise RuntimeError(f"Vendor {vendor} has incorrect slice coverage")
            adaptations = [batch["adaptation"] for batch in batches]
        if not all(
            math.isfinite(float(value))
            for record in records
            for value in record["metrics"].values()
        ):
            raise RuntimeError(f"Vendor {vendor} contains non-finite metrics")
        for adaptation in adaptations:
            _validate_adaptation(adaptation, expected_batch_size)
        _read_json(root / f"vendor_{vendor}_summary.json")
    print(
        f"[VALIDATED] method=tent lr={tag} seed={seed} "
        f"stream={stream_mode} root={root}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/Stochastic_Ini_ForegroundOnly/tent_lr_sweep"),
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
