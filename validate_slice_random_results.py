"""Validate one complete Source/TENT/SAR random-slice result directory."""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn

from metrics import SLICE_METRIC_POLICY
from model import build_model
from target_order_validation import validate_target_slice_order
from tta_methods import build_method
from tta_methods.sar.sam import SAM
from utils import file_sha256, load_config


SEEDS = (2022, 2023, 2024, 2025, 2026)
VENDORS = ("B", "C", "D")
EXPECTED = {
    "B": {"slices": 2049, "patients": 125, "batches": 257, "last_batch": 1},
    "C": {"slices": 806, "patients": 50, "batches": 101, "last_batch": 6},
    "D": {"slices": 835, "patients": 50, "batches": 105, "last_batch": 3},
}
EXPECTED_METRICS = {
    "dice_rv", "dice_myo", "dice_lv", "dice_macro",
    "hd95_2d_px_rv", "hd95_2d_px_myo", "hd95_2d_px_lv", "hd95_2d_px_macro",
}
LOCKED_METHOD_CONFIGS = {
    "tent": {
        "optimizer": "sgd", "lr": 6.25e-5, "momentum": 0.9,
        "weight_decay": 0.0, "steps": 1,
    },
    "sar": {
        "optimizer": "sgd_sam", "lr": 6.25e-5, "momentum": 0.9,
        "weight_decay": 0.0, "rho": 0.05, "steps": 1,
        "entropy_margin_factor": 0.4, "recovery_ema": 0.9,
        "recovery_threshold": 0.2,
    },
}
PROBE_COUNT_KEYS = (
    "seen_slices", "selected_slices", "selected_pixels", "correct_pixels",
    "gt_foreground_pixels", "correct_gt_foreground_pixels",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _finite(value: Any, label: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise RuntimeError(f"Non-finite {label}: {value}")
    return resolved


def _validate_method_setup(
    method_name: str, manifest: dict[str, Any], cfg: dict[str, Any]
) -> None:
    method_cfg = manifest["resolved_method_config"]
    for key, expected in LOCKED_METHOD_CONFIGS.get(method_name, {}).items():
        if method_cfg.get(key) != expected:
            raise RuntimeError(
                f"Unexpected {method_name} setting {key}: {method_cfg.get(key)!r} != {expected!r}"
            )
    if method_name == "sar" and method_cfg.get("profile_kind") != "official_mechanism_custom_lr":
        raise RuntimeError("SAR profile label is not locked")

    model = build_model(cfg, pretrained_override=False)
    method = build_method(
        method_name, model, deepcopy(method_cfg), manifest["resolved_config"]["tta"],
        torch.device("cpu"),
    )
    if method_name == "source":
        expected_names: set[str] = set()
    elif method_name == "tent":
        if not isinstance(method.optimizer, torch.optim.SGD):
            raise RuntimeError("Resolved TENT optimizer is not SGD")
        expected_names = {
            f"{module_name}.{parameter_name}"
            for module_name, module in method.model.named_modules()
            if isinstance(module, nn.BatchNorm2d)
            for parameter_name in ("weight", "bias")
        }
    else:
        if not isinstance(method.optimizer, SAM) or not isinstance(
            method.optimizer.base_optimizer, torch.optim.SGD
        ):
            raise RuntimeError("Resolved SAR optimizer is not SAM with SGD")
        expected_names = {
            f"{module_name}.{parameter_name}"
            for module_name, module in method.model.named_modules()
            if isinstance(module, nn.BatchNorm2d)
            and not module_name.startswith("encoder.layer4")
            for parameter_name in ("weight", "bias")
        }
    if set(manifest["trainable_parameters"]) != expected_names:
        raise RuntimeError(f"{method_name} manifest has the wrong trainable scope")


def _validate_probe(probe: dict[str, Any], arrival_size: int) -> None:
    first, second = probe["first_filter"], probe["second_filter"]
    if int(first["seen_slices"]) != arrival_size or int(second["seen_slices"]) != arrival_size:
        raise RuntimeError("SAR probe seen-slice count differs from the batch size")
    if int(second["selected_slices"]) > int(first["selected_slices"]):
        raise RuntimeError("SAR second filter is not a subset of the first")
    for stage in (first, second):
        for key in PROBE_COUNT_KEYS:
            if int(stage[key]) < 0:
                raise RuntimeError(f"Negative SAR probe count {key}")
        for key in ("selection_coverage", "pixel_accuracy", "foreground_pixel_accuracy"):
            value = stage[key]
            if value is not None and not (0.0 <= float(value) <= 1.0):
                raise RuntimeError(f"Invalid SAR probe metric {key}={value}")


def _validate_summary(summary: dict[str, Any], expected: dict[str, int]) -> None:
    if summary.get("aggregation_unit") != "slice":
        raise RuntimeError("Summary aggregation unit is not slice")
    if summary.get("metric_policy") != SLICE_METRIC_POLICY:
        raise RuntimeError("Summary slice metric policy differs from the locked policy")
    if set(summary.get("all_slices", {})) != EXPECTED_METRICS:
        raise RuntimeError("all_slices summary has an unexpected metric set")
    if set(summary.get("foreground_present", {})) != EXPECTED_METRICS:
        raise RuntimeError("foreground_present summary has an unexpected metric set")
    for metric, item in summary["all_slices"].items():
        for key in ("mean", "ci95_low", "ci95_high"):
            _finite(item[key], f"all_slices/{metric}/{key}")
        if int(item["n_slices"]) != expected["slices"]:
            raise RuntimeError(f"all_slices/{metric} has the wrong slice count")
        if int(item["n_patients"]) != expected["patients"]:
            raise RuntimeError(f"all_slices/{metric} has the wrong patient count")
    for metric, item in summary["foreground_present"].items():
        for key in ("mean", "ci95_low", "ci95_high"):
            _finite(item[key], f"foreground_present/{metric}/{key}")
        if int(item["n_patients"]) != expected["patients"]:
            raise RuntimeError(f"foreground_present/{metric} has the wrong patient count")


def validate_run(
    method_name: str,
    seed: int,
    cfg: dict[str, Any],
    *,
    print_hashes: bool = True,
) -> dict[str, Any]:
    if method_name not in {"source", "tent", "sar"}:
        raise ValueError(f"Unsupported method: {method_name}")
    if seed not in SEEDS:
        raise ValueError(f"Unexpected source seed: {seed}")
    root = (
        Path(cfg["tta"]["results_dir"]) / method_name / f"seed{seed}"
        / "slice_random_adapt_then_predict_vendor"
    )
    manifest = _read_json(root / "run_manifest.json")
    metadata_path = Path(cfg["source"]["checkpoint_dir"]) / f"seed{seed}_best.json"
    checkpoint_path = Path(cfg["source"]["checkpoint_dir"]) / f"seed{seed}_best.pt"
    metadata = _read_json(metadata_path)
    checkpoint_hash = file_sha256(checkpoint_path)

    if manifest.get("method") != method_name or int(manifest.get("source_seed", -1)) != seed:
        raise RuntimeError("Manifest method or source seed is incorrect")
    if manifest.get("stream_mode") != "slice_random" or manifest.get("vendors") != list(VENDORS):
        raise RuntimeError("Manifest does not describe the B/C/D random-slice protocol")
    if manifest.get("slice_filter") != "manifest_has_fg_equals_1":
        raise RuntimeError("Manifest does not lock the foreground-only slice filter")
    if manifest.get("initialization_profile") != "stochastic":
        raise RuntimeError("Manifest initialization profile is not stochastic")
    resolved_tta = manifest["resolved_config"]["tta"]
    if (
        int(resolved_tta["batch_size"]) != 8
        or resolved_tta["timing"] != "adapt_then_predict"
        or resolved_tta["reset"] != "vendor"
        or resolved_tta["stream_mode"] != "slice_random"
    ):
        raise RuntimeError("Manifest has the wrong BS=8 random-slice TTA protocol")
    if checkpoint_hash != metadata["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint bytes differ from source training metadata")
    if manifest.get("source_checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Manifest checkpoint hash is incorrect")
    if manifest.get("protocol_sha256") != file_sha256(cfg["data"]["protocol_file"]):
        raise RuntimeError("Manifest protocol hash is stale")
    if manifest.get("target_stream_sha256") != file_sha256(cfg["data"]["slice_stream_file"]):
        raise RuntimeError("Manifest random-slice stream hash is stale")
    if manifest.get("slice_metric_policy") != SLICE_METRIC_POLICY:
        raise RuntimeError("Manifest slice metric policy differs from the locked policy")
    _validate_method_setup(method_name, manifest, cfg)

    prediction_source = "source_model" if method_name == "source" else "student"
    any_sar_update = False
    for vendor in VENDORS:
        expected = EXPECTED[vendor]
        records = _read_jsonl(root / f"vendor_{vendor}.jsonl")
        batches = _read_jsonl(root / f"vendor_{vendor}_batches.jsonl")
        order_hash = validate_target_slice_order(manifest, records, cfg, vendor, seed)
        if len(records) != expected["slices"] or len(batches) != expected["batches"]:
            raise RuntimeError(f"Vendor {vendor} has incorrect slice or batch counts")
        if len({record["patient_id"] for record in records}) != expected["patients"]:
            raise RuntimeError(f"Vendor {vendor} has an incorrect patient count")
        if len({record["slice_id"] for record in records}) != expected["slices"]:
            raise RuntimeError(f"Vendor {vendor} contains duplicate slice IDs")

        for record in records:
            if record.get("method") != method_name or record.get("prediction_source") != prediction_source:
                raise RuntimeError(f"Vendor {vendor} contains a record for the wrong method")
            if record.get("source_checkpoint_sha256") != checkpoint_hash:
                raise RuntimeError(f"Vendor {vendor} contains a stale checkpoint hash")
            if record.get("slice_filter") != "manifest_has_fg_equals_1":
                raise RuntimeError(f"Vendor {vendor} contains an unfiltered slice record")
            if int(record["arrival_batch_size"]) > 8:
                raise RuntimeError(f"Vendor {vendor} contains a batch larger than 8")
            if set(record["metrics"]) != EXPECTED_METRICS:
                raise RuntimeError(f"Vendor {vendor} slice metric schema is invalid")
            if set(record["gt_present"]) != {"rv", "myo", "lv"}:
                raise RuntimeError(f"Vendor {vendor} GT-presence schema is invalid")
            for metric, value in record["metrics"].items():
                _finite(value, f"{vendor}/{record['slice_id']}/{metric}")

        flattened_ids: list[str] = []
        for batch_index, batch in enumerate(batches):
            arrival_size = int(batch["arrival_batch_size"])
            expected_size = expected["last_batch"] if batch_index == len(batches) - 1 else 8
            if arrival_size != expected_size:
                raise RuntimeError(
                    f"Vendor {vendor} batch {batch_index} has size {arrival_size}; expected {expected_size}"
                )
            if (
                batch.get("method") != method_name
                or int(batch.get("source_seed", -1)) != seed
                or batch.get("vendor") != vendor
                or int(batch.get("batch_arrival_index", -1)) != batch_index
                or int(batch.get("target_order_seed", -1)) != seed
                or batch.get("slice_order_sha256") != order_hash
                or batch.get("slice_filter") != "manifest_has_fg_equals_1"
            ):
                raise RuntimeError(f"Vendor {vendor} batch {batch_index} metadata is invalid")
            if len(batch["slice_ids"]) != arrival_size:
                raise RuntimeError(f"Vendor {vendor} batch {batch_index} slice count is invalid")
            flattened_ids.extend(batch["slice_ids"])
            adaptation = batch["adaptation"]
            if int(adaptation["n_seen"]) != arrival_size:
                raise RuntimeError(f"Vendor {vendor} batch {batch_index} n_seen is invalid")
            for value in adaptation["predicted_foreground_area"].values():
                _finite(value, "predicted foreground area")
            drift = _finite(adaptation["extras"]["parameter_drift"], "parameter drift")
            if method_name == "source":
                if adaptation["updated"] or int(adaptation["n_selected"]) != 0 or drift != 0.0:
                    raise RuntimeError("Source-only changed parameters or selected samples")
            elif method_name == "tent":
                if not adaptation["updated"] or int(adaptation["n_selected"]) != arrival_size:
                    raise RuntimeError("TENT skipped part of an arrival batch")
                if adaptation["loss"] is None:
                    raise RuntimeError("TENT is missing an entropy loss")
                _finite(adaptation["loss"], "TENT entropy loss")
                if drift <= 0.0:
                    raise RuntimeError("TENT did not change BN affine parameters")
            else:
                probe = batch.get("entropy_label_probe")
                adaptation_probe = adaptation.get("entropy_label_probe")
                if probe is None or adaptation_probe is None or probe != adaptation_probe:
                    raise RuntimeError("SAR batch and adaptation probe payloads differ")
                _validate_probe(probe, arrival_size)
                selected = int(probe["second_filter"]["selected_slices"])
                if int(adaptation["n_selected"]) != selected:
                    raise RuntimeError("SAR selected count differs from the probe")
                if bool(adaptation["updated"]) != (selected > 0):
                    raise RuntimeError("SAR update flag differs from final selection")
                if adaptation["loss"] is not None:
                    _finite(adaptation["loss"], "SAR entropy loss")
                any_sar_update = any_sar_update or bool(adaptation["updated"])
        if flattened_ids != [record["slice_id"] for record in records]:
            raise RuntimeError(f"Vendor {vendor} batch boundaries do not cover the slice stream")
        _validate_summary(_read_json(root / f"vendor_{vendor}_summary.json"), expected)

    if method_name == "sar" and not any_sar_update:
        raise RuntimeError("SAR did not execute a second-pass SGD update")
    if print_hashes:
        for path in sorted(root.iterdir()):
            if path.is_file():
                print(file_sha256(path), path)
        print(f"[VALIDATED] method={method_name} seed={seed} batch_size=8 root={root}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--method", required=True, choices=["source", "tent", "sar"])
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_run(args.method, args.seed, load_config(args.config))


if __name__ == "__main__":
    main()
