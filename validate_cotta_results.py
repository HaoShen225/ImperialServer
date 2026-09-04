"""Validate one complete five-seed-ready CoTTA result for either target stream."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

from metrics import SLICE_METRIC_POLICY
from model import build_model
from target_order_validation import validate_target_order, validate_target_slice_order
from utils import file_sha256, load_config


SEEDS = (2022, 2023, 2024, 2025, 2026)
VENDORS = ("B", "C", "D")
EXPECTED_PATIENT = {
    "B": {"patients": 125, "volumes": 250},
    "C": {"patients": 50, "volumes": 100},
    "D": {"patients": 50, "volumes": 100},
}
EXPECTED_SLICE = {
    "B": {"slices": 2049, "patients": 125, "batches": 257, "last_batch": 1},
    "C": {"slices": 806, "patients": 50, "batches": 101, "last_batch": 6},
    "D": {"slices": 835, "patients": 50, "batches": 105, "last_batch": 3},
}
PATIENT_METRICS = {
    "dice_rv", "dice_myo", "dice_lv", "dice_macro",
    "hd95_px_rv", "hd95_px_myo", "hd95_px_lv", "hd95_px_macro",
}
SLICE_METRICS = {
    "dice_rv", "dice_myo", "dice_lv", "dice_macro",
    "hd95_2d_px_rv", "hd95_2d_px_myo", "hd95_2d_px_lv", "hd95_2d_px_macro",
}
LOCKED_COTTA = {
    "profile_verified": True,
    "profile_kind": "official_segmentation_mms_adapted",
    "method_seed": 4101,
    "steps": 1,
    "update_scope": "all",
    "bn_policy": "batch_no_running",
    "optimizer": "adam",
    "lr": 7.5e-6,
    "beta1": 0.9,
    "beta2": 0.999,
    "weight_decay": 0.0,
    "teacher_momentum": 0.999,
    "confidence_gate": 0.9,
    "restore_probability": 0.01,
    "augmentation_scales": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
    "augmentation_flips": [False, True],
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _finite(value: Any, label: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise RuntimeError(f"Non-finite {label}: {value}")
    return resolved


def _run_root(results_root: Path, seed: int, stream_mode: str) -> Path:
    suffix = (
        "adapt_then_predict_vendor"
        if stream_mode == "patient_volume"
        else "slice_random_adapt_then_predict_vendor"
    )
    return results_root / "cotta" / f"seed{seed}" / suffix


def _validate_setup(
    manifest: dict[str, Any],
    cfg: dict[str, Any],
    stream_mode: str,
    expected_method_cfg: dict[str, Any] | None = None,
) -> None:
    method_cfg = manifest["resolved_method_config"]
    locked = LOCKED_COTTA if expected_method_cfg is None else expected_method_cfg
    for key, expected in locked.items():
        if method_cfg.get(key) != expected:
            raise RuntimeError(
                f"Unexpected locked CoTTA setting {key}: {method_cfg.get(key)!r} != {expected!r}"
            )
    resolved_tta = manifest["resolved_config"]["tta"]
    expected_batch_size = 4 if stream_mode == "patient_volume" else 8
    if (
        resolved_tta.get("stream_mode") != stream_mode
        or int(resolved_tta["batch_size"]) != expected_batch_size
        or resolved_tta.get("timing") != "adapt_then_predict"
        or resolved_tta.get("reset") != "vendor"
    ):
        raise RuntimeError(f"CoTTA manifest has the wrong locked {stream_mode} protocol")
    expected_names = {name for name, _ in build_model(cfg, pretrained_override=False).named_parameters()}
    if set(manifest.get("trainable_parameters", [])) != expected_names:
        raise RuntimeError("CoTTA manifest does not declare the complete model trainable")


def _validate_adaptations(
    adaptations: Iterable[dict[str, Any]],
    *,
    maximum_batch_size: int,
    vendor: str,
) -> dict[str, float | int]:
    cumulative = 0
    n_batches = 0
    n_seen = 0
    n_augmented = 0
    weighted_loss = 0.0
    weighted_confidence = 0.0
    adaptation_seconds = 0.0
    prediction_seconds = 0.0
    final_parameter_drift = 0.0
    for batch_index, adaptation in enumerate(adaptations):
        n_batches += 1
        arrival_size = int(adaptation["arrival_batch_size"])
        if not 1 <= arrival_size <= maximum_batch_size:
            raise RuntimeError(f"Vendor {vendor} has invalid arrival batch size {arrival_size}")
        if (
            not adaptation["updated"]
            or int(adaptation["n_seen"]) != arrival_size
            or int(adaptation["n_selected"]) != arrival_size
        ):
            raise RuntimeError(f"Vendor {vendor} CoTTA skipped an arrival batch")
        loss = _finite(adaptation["loss"], "CoTTA consistency loss")
        extras = adaptation["extras"]
        confidence = _finite(extras["anchor_confidence_mean"], "anchor confidence")
        if not 0.0 <= confidence <= 1.0:
            raise RuntimeError(f"Vendor {vendor} has invalid anchor confidence")
        augmented = int(extras["augmentation_triggered_slices"])
        coverage = _finite(extras["augmentation_coverage"], "augmentation coverage")
        if not 0 <= augmented <= arrival_size or not math.isclose(
            coverage, augmented / arrival_size, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(f"Vendor {vendor} has inconsistent augmentation coverage")
        if int(extras["teacher_views_when_triggered"]) != 14:
            raise RuntimeError(f"Vendor {vendor} did not use the official 14 teacher views")
        restored_step = int(extras["restored_parameters_step"])
        restored_cumulative = int(extras["restored_parameters_cumulative"])
        if restored_step < 0:
            raise RuntimeError(f"Vendor {vendor} has a negative restore count")
        cumulative += restored_step
        if restored_cumulative != cumulative:
            raise RuntimeError(
                f"Vendor {vendor} restore counter is not cumulative at batch {batch_index}"
            )
        final_parameter_drift = _finite(extras["parameter_drift"], "parameter drift")
        if final_parameter_drift <= 0.0:
            raise RuntimeError(f"Vendor {vendor} CoTTA did not change student parameters")
        batch_adaptation_seconds = _finite(extras["adaptation_seconds"], "adaptation_seconds")
        batch_prediction_seconds = _finite(extras["prediction_seconds"], "prediction_seconds")
        for key, value in (
            ("adaptation_seconds", batch_adaptation_seconds),
            ("prediction_seconds", batch_prediction_seconds),
        ):
            if value < 0.0:
                raise RuntimeError(f"Vendor {vendor} has negative runtime diagnostic {key}")
        for value in adaptation["predicted_foreground_area"].values():
            _finite(value, "predicted foreground area")
        n_seen += arrival_size
        n_augmented += augmented
        weighted_loss += loss * arrival_size
        weighted_confidence += confidence * arrival_size
        adaptation_seconds += batch_adaptation_seconds
        prediction_seconds += batch_prediction_seconds
    if n_batches == 0:
        raise RuntimeError(f"Vendor {vendor} contains no CoTTA arrival batches")
    return {
        "batches": n_batches,
        "seen_slices": n_seen,
        "augmented_slices": n_augmented,
        "augmentation_coverage": n_augmented / n_seen,
        "restored_parameters": cumulative,
        "mean_loss": weighted_loss / n_seen,
        "mean_anchor_confidence": weighted_confidence / n_seen,
        "adaptation_seconds": adaptation_seconds,
        "prediction_seconds": prediction_seconds,
        "final_parameter_drift": final_parameter_drift,
    }


def _validate_patient_summary(summary: dict[str, Any], patient_count: int) -> None:
    if set(summary) != PATIENT_METRICS:
        raise RuntimeError("Patient-volume summary has an unexpected metric set")
    for metric, item in summary.items():
        for key in ("mean", "ci95_low", "ci95_high"):
            _finite(item[key], f"{metric}/{key}")
        if int(item["n_patients"]) != patient_count:
            raise RuntimeError(f"{metric} has the wrong patient count")


def _validate_slice_summary(summary: dict[str, Any], expected: dict[str, int]) -> None:
    if summary.get("aggregation_unit") != "slice":
        raise RuntimeError("Random-slice summary aggregation unit is not slice")
    if summary.get("metric_policy") != SLICE_METRIC_POLICY:
        raise RuntimeError("Random-slice summary metric policy is stale")
    for stratum in ("all_slices", "foreground_present"):
        if set(summary.get(stratum, {})) != SLICE_METRICS:
            raise RuntimeError(f"Random-slice {stratum} has an unexpected metric set")
        for metric, item in summary[stratum].items():
            for key in ("mean", "ci95_low", "ci95_high"):
                _finite(item[key], f"{stratum}/{metric}/{key}")
            if int(item["n_patients"]) != expected["patients"]:
                raise RuntimeError(f"{stratum}/{metric} has the wrong patient count")
            if stratum == "all_slices" and int(item["n_slices"]) != expected["slices"]:
                raise RuntimeError(f"{stratum}/{metric} has the wrong slice count")


def validate_run(
    seed: int,
    stream_mode: str,
    cfg: dict[str, Any],
    *,
    print_hashes: bool = True,
    expected_method_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError(f"Unexpected source seed: {seed}")
    if stream_mode not in {"patient_volume", "slice_random"}:
        raise ValueError(f"Unexpected stream mode: {stream_mode}")
    root = _run_root(Path(cfg["tta"]["results_dir"]), seed, stream_mode)
    manifest = _read_json(root / "run_manifest.json")
    checkpoint = Path(cfg["source"]["checkpoint_dir"]) / f"seed{seed}_best.pt"
    metadata = _read_json(Path(cfg["source"]["checkpoint_dir"]) / f"seed{seed}_best.json")
    checkpoint_hash = file_sha256(checkpoint)

    if (
        manifest.get("method") != "cotta"
        or int(manifest.get("source_seed", -1)) != seed
        or manifest.get("stream_mode") != stream_mode
        or manifest.get("vendors") != list(VENDORS)
        or manifest.get("initialization_profile") != "stochastic"
        or manifest.get("slice_filter") != "manifest_has_fg_equals_1"
    ):
        raise RuntimeError("Manifest does not describe the locked CoTTA B/C/D experiment")
    if checkpoint_hash != metadata["checkpoint_sha256"]:
        raise RuntimeError("Source checkpoint differs from its training metadata")
    if manifest.get("source_checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("CoTTA manifest has an incorrect source checkpoint hash")
    if manifest.get("protocol_sha256") != file_sha256(cfg["data"]["protocol_file"]):
        raise RuntimeError("CoTTA manifest has a stale protocol hash")
    stream_file = (
        cfg["data"]["stream_file"]
        if stream_mode == "patient_volume"
        else cfg["data"]["slice_stream_file"]
    )
    if manifest.get("target_stream_sha256") != file_sha256(stream_file):
        raise RuntimeError("CoTTA manifest has a stale target-stream hash")
    if stream_mode == "slice_random" and manifest.get("slice_metric_policy") != SLICE_METRIC_POLICY:
        raise RuntimeError("CoTTA manifest has a stale random-slice metric policy")
    _validate_setup(manifest, cfg, stream_mode, expected_method_cfg)

    diagnostics: dict[str, dict[str, float | int]] = {}
    for vendor in VENDORS:
        if stream_mode == "patient_volume":
            expected = EXPECTED_PATIENT[vendor]
            records = _read_jsonl(root / f"vendor_{vendor}.jsonl")
            validate_target_order(manifest, records, cfg, vendor, seed)
            if len(records) != expected["volumes"]:
                raise RuntimeError(f"Vendor {vendor} has the wrong volume count")
            if len({record["patient_id"] for record in records}) != expected["patients"]:
                raise RuntimeError(f"Vendor {vendor} has the wrong patient count")
            phases: dict[str, set[str]] = {}
            adaptations = []
            for record in records:
                phases.setdefault(record["patient_id"], set()).add(record["phase"])
                if (
                    record.get("method") != "cotta"
                    or record.get("prediction_source") != "ema_teacher"
                    or record.get("source_checkpoint_sha256") != checkpoint_hash
                    or set(record.get("trainable_parameters", []))
                    != set(manifest["trainable_parameters"])
                    or set(record.get("metrics", {})) != PATIENT_METRICS
                ):
                    raise RuntimeError(f"Vendor {vendor} contains an invalid CoTTA volume record")
                for metric, value in record["metrics"].items():
                    _finite(value, f"{vendor}/{record['volume_id']}/{metric}")
                adaptations.extend(record["adaptation"])
            if any(value != {"ED", "ES"} for value in phases.values()):
                raise RuntimeError(f"Vendor {vendor} is missing an ED or ES phase")
            diagnostics[vendor] = _validate_adaptations(
                adaptations, maximum_batch_size=4, vendor=vendor
            )
            _validate_patient_summary(
                _read_json(root / f"vendor_{vendor}_summary.json"), expected["patients"]
            )
        else:
            expected = EXPECTED_SLICE[vendor]
            records = _read_jsonl(root / f"vendor_{vendor}.jsonl")
            batches = _read_jsonl(root / f"vendor_{vendor}_batches.jsonl")
            order_hash = validate_target_slice_order(manifest, records, cfg, vendor, seed)
            if len(records) != expected["slices"] or len(batches) != expected["batches"]:
                raise RuntimeError(f"Vendor {vendor} has the wrong slice or batch count")
            if len({record["patient_id"] for record in records}) != expected["patients"]:
                raise RuntimeError(f"Vendor {vendor} has the wrong random-slice patient count")
            flattened_ids: list[str] = []
            adaptations = []
            for record in records:
                if (
                    record.get("method") != "cotta"
                    or record.get("prediction_source") != "ema_teacher"
                    or record.get("source_checkpoint_sha256") != checkpoint_hash
                    or set(record.get("metrics", {})) != SLICE_METRICS
                    or set(record.get("gt_present", {})) != {"rv", "myo", "lv"}
                ):
                    raise RuntimeError(f"Vendor {vendor} contains an invalid CoTTA slice record")
                for metric, value in record["metrics"].items():
                    _finite(value, f"{vendor}/{record['slice_id']}/{metric}")
            for batch_index, batch in enumerate(batches):
                arrival_size = int(batch["arrival_batch_size"])
                expected_size = expected["last_batch"] if batch_index == len(batches) - 1 else 8
                if (
                    arrival_size != expected_size
                    or batch.get("method") != "cotta"
                    or int(batch.get("source_seed", -1)) != seed
                    or batch.get("vendor") != vendor
                    or int(batch.get("batch_arrival_index", -1)) != batch_index
                    or batch.get("slice_order_sha256") != order_hash
                ):
                    raise RuntimeError(f"Vendor {vendor} batch {batch_index} metadata is invalid")
                flattened_ids.extend(batch["slice_ids"])
                adaptation = batch["adaptation"]
                adaptation["arrival_batch_size"] = arrival_size
                adaptations.append(adaptation)
            if flattened_ids != [record["slice_id"] for record in records]:
                raise RuntimeError(f"Vendor {vendor} batches do not cover its slice stream")
            diagnostics[vendor] = _validate_adaptations(
                adaptations, maximum_batch_size=8, vendor=vendor
            )
            _validate_slice_summary(
                _read_json(root / f"vendor_{vendor}_summary.json"), expected
            )

    if print_hashes:
        for path in sorted(root.iterdir()):
            if path.is_file():
                print(file_sha256(path), path)
        print(
            f"[VALIDATED] method=cotta seed={seed} stream={stream_mode} "
            f"batch_size={4 if stream_mode == 'patient_volume' else 8} root={root}"
        )
    return {"manifest": manifest, "diagnostics": diagnostics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--stream-mode", choices=["patient_volume", "slice_random"], required=True
    )
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_run(args.seed, args.stream_mode, load_config(args.config))


if __name__ == "__main__":
    main()
