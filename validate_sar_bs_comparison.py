"""Validate one SAR patient-BS8/random-slice-BS4 LR=1e-3 result cell."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

from torch import nn

from metrics import SLICE_METRIC_POLICY
from model import build_model
from run_sar_bs_comparison import (
    BATCH_SIZES,
    DEFAULT_RESULTS_ROOT,
    LEARNING_RATE,
    METHOD,
    PROFILE_KIND,
    SOURCE_SEEDS,
    STREAM_MODES,
    configure_sar_comparison,
    run_root,
)
from target_order_validation import validate_target_order, validate_target_slice_order
from utils import file_sha256, load_config


VENDORS = ("B", "C", "D")
EXPECTED = {
    "B": {"patients": 125, "volumes": 250, "slices": 2049},
    "C": {"patients": 50, "volumes": 100, "slices": 806},
    "D": {"patients": 50, "volumes": 100, "slices": 835},
}
PATIENT_METRICS = {
    "dice_rv", "dice_myo", "dice_lv", "dice_macro",
    "hd95_px_rv", "hd95_px_myo", "hd95_px_lv", "hd95_px_macro",
}
SLICE_METRICS = {
    "dice_rv", "dice_myo", "dice_lv", "dice_macro",
    "hd95_2d_px_rv", "hd95_2d_px_myo", "hd95_2d_px_lv", "hd95_2d_px_macro",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _json_normalize(value: Any) -> Any:
    """Normalize YAML integer keys to the representation stored in JSON manifests."""
    return json.loads(json.dumps(value, sort_keys=True))


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite {label}: {value}")
    return result


def _expected_trainable(cfg: dict[str, Any]) -> set[str]:
    model = build_model(cfg, pretrained_override=False)
    return {
        f"{module_name}.{parameter_name}"
        for module_name, module in model.named_modules()
        if isinstance(module, nn.BatchNorm2d)
        and not module_name.startswith("encoder.layer4")
        for parameter_name in ("weight", "bias")
    }


def _validate_setup(
    manifest: dict[str, Any],
    base_cfg: dict[str, Any],
    seed: int,
    stream_mode: str,
    results_root: Path,
) -> str:
    expected_cfg = configure_sar_comparison(base_cfg, stream_mode, results_root)
    if _json_normalize(manifest.get("resolved_method_config")) != _json_normalize(
        expected_cfg["methods"][METHOD]
    ):
        raise RuntimeError("Resolved SAR configuration differs from the locked profile")
    if _json_normalize(manifest.get("resolved_config")) != _json_normalize(expected_cfg):
        raise RuntimeError("Resolved run configuration differs from the locked profile")
    method_cfg = manifest["resolved_method_config"]
    locked = {
        "profile_kind": PROFILE_KIND,
        "optimizer": "sgd_sam",
        "lr": LEARNING_RATE,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "rho": 0.05,
        "steps": 1,
        "entropy_margin_factor": 0.4,
        "recovery_ema": 0.9,
        "recovery_threshold": 0.2,
        "update_scope": "bn_affine_except_encoder_layer4",
        "bn_policy": "batch_no_running",
    }
    if any(method_cfg.get(key) != value for key, value in locked.items()):
        raise RuntimeError("SAR does not use the locked BS comparison profile")
    if (
        manifest.get("method") != METHOD
        or int(manifest.get("source_seed", -1)) != seed
        or manifest.get("stream_mode") != stream_mode
        or manifest.get("vendors") != list(VENDORS)
        or manifest.get("initialization_profile") != "stochastic"
        or manifest.get("slice_filter") != "manifest_has_fg_equals_1"
    ):
        raise RuntimeError("Manifest identifies the wrong SAR comparison experiment")
    tta_cfg = manifest["resolved_config"]["tta"]
    if (
        int(tta_cfg["batch_size"]) != BATCH_SIZES[stream_mode]
        or tta_cfg.get("timing") != "adapt_then_predict"
        or tta_cfg.get("reset") != "vendor"
    ):
        raise RuntimeError("Manifest violates the locked SAR arrival protocol")
    expected_trainable = _expected_trainable(expected_cfg)
    if set(manifest.get("trainable_parameters", [])) != expected_trainable:
        raise RuntimeError("SAR has the wrong trainable parameter scope")

    checkpoint = Path(base_cfg["source"]["checkpoint_dir"]) / f"seed{seed}_best.pt"
    metadata = _read_json(
        Path(base_cfg["source"]["checkpoint_dir"]) / f"seed{seed}_best.json"
    )
    checkpoint_hash = file_sha256(checkpoint)
    if checkpoint_hash != metadata["checkpoint_sha256"]:
        raise RuntimeError("Source checkpoint differs from its training metadata")
    if manifest.get("source_checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Manifest contains the wrong source checkpoint hash")
    if manifest.get("protocol_sha256") != file_sha256(base_cfg["data"]["protocol_file"]):
        raise RuntimeError("Manifest contains a stale protocol hash")
    stream_file = (
        base_cfg["data"]["stream_file"]
        if stream_mode == "patient_volume"
        else base_cfg["data"]["slice_stream_file"]
    )
    if manifest.get("target_stream_sha256") != file_sha256(stream_file):
        raise RuntimeError("Manifest contains a stale target stream hash")
    if stream_mode == "slice_random" and manifest.get("slice_metric_policy") != SLICE_METRIC_POLICY:
        raise RuntimeError("Random-slice metric policy is stale")
    return checkpoint_hash


def _validate_probe(probe: dict[str, Any], arrival_size: int, vendor: str) -> None:
    first = probe.get("first_filter", {})
    second = probe.get("second_filter", {})
    if (
        int(first.get("seen_slices", -1)) != arrival_size
        or int(second.get("seen_slices", -1)) != arrival_size
        or int(second.get("selected_slices", -1)) > int(first.get("selected_slices", -1))
    ):
        raise RuntimeError(f"Vendor {vendor} has an invalid SAR filter probe")
    for stage in (first, second):
        selected = int(stage.get("selected_slices", -1))
        if not 0 <= selected <= arrival_size:
            raise RuntimeError(f"Vendor {vendor} has an invalid selected-slice count")
        for key in ("selection_coverage", "pixel_accuracy", "foreground_pixel_accuracy"):
            value = stage.get(key)
            if value is not None and not 0.0 <= _finite(value, key) <= 1.0:
                raise RuntimeError(f"Vendor {vendor} has invalid probe metric {key}")


def _validate_adaptation(adaptation: dict[str, Any], arrival_size: int, vendor: str) -> bool:
    if (
        int(adaptation.get("arrival_batch_size", -1)) != arrival_size
        or int(adaptation.get("n_seen", -1)) != arrival_size
    ):
        raise RuntimeError(f"Vendor {vendor} has inconsistent SAR arrival metadata")
    probe = adaptation.get("entropy_label_probe")
    if probe is None:
        raise RuntimeError(f"Vendor {vendor} is missing a SAR batch probe")
    _validate_probe(probe, arrival_size, vendor)
    selected = int(probe["second_filter"]["selected_slices"])
    updated = bool(adaptation.get("updated"))
    if int(adaptation.get("n_selected", -1)) != selected or updated != (selected > 0):
        raise RuntimeError(f"Vendor {vendor} has inconsistent SAR update metadata")
    if adaptation.get("loss") is not None:
        _finite(adaptation["loss"], "SAR entropy loss")
    extras = adaptation.get("extras", {})
    drift = _finite(extras.get("parameter_drift"), "parameter drift")
    if drift < 0.0:
        raise RuntimeError(f"Vendor {vendor} has negative parameter drift")
    for key in ("adaptation_seconds", "prediction_seconds"):
        if _finite(extras.get(key), key) < 0.0:
            raise RuntimeError(f"Vendor {vendor} has negative {key}")
    for value in adaptation.get("predicted_foreground_area", {}).values():
        _finite(value, "predicted foreground area")
    if updated:
        recovered = bool(extras.get("recovered", 0.0))
        if not recovered and drift <= 0.0:
            raise RuntimeError(f"Vendor {vendor} reports an SAR update without parameter drift")
        _finite(extras.get("ema_loss"), "SAR EMA loss")
        if int(float(extras.get("recovery_count", -1))) < 0:
            raise RuntimeError(f"Vendor {vendor} has invalid recovery metadata")
    return updated


def _validate_adaptations(
    adaptations: Iterable[dict[str, Any]],
    expected_slices: int,
    maximum_batch_size: int,
    vendor: str,
) -> bool:
    seen = 0
    batches = 0
    any_update = False
    for adaptation in adaptations:
        arrival_size = int(adaptation.get("arrival_batch_size", -1))
        if not 1 <= arrival_size <= maximum_batch_size:
            raise RuntimeError(f"Vendor {vendor} has invalid batch size {arrival_size}")
        any_update = _validate_adaptation(adaptation, arrival_size, vendor) or any_update
        seen += arrival_size
        batches += 1
    if batches == 0 or seen != expected_slices:
        raise RuntimeError(
            f"Vendor {vendor} covers {seen}/{expected_slices} slices in {batches} batches"
        )
    return any_update


def _validate_summary(path: Path, stream_mode: str, expected: dict[str, int]) -> None:
    summary = _read_json(path)
    if stream_mode == "patient_volume":
        if set(summary) != PATIENT_METRICS:
            raise RuntimeError("Patient-volume summary has the wrong metric set")
        groups = [summary]
    else:
        if summary.get("aggregation_unit") != "slice" or summary.get("metric_policy") != SLICE_METRIC_POLICY:
            raise RuntimeError("Random-slice summary has the wrong aggregation policy")
        groups = [summary.get("all_slices", {}), summary.get("foreground_present", {})]
        if any(set(group) != SLICE_METRICS for group in groups):
            raise RuntimeError("Random-slice summary has the wrong metric set")
    for index, group in enumerate(groups):
        for metric, item in group.items():
            for key in ("mean", "ci95_low", "ci95_high"):
                _finite(item[key], f"{metric}/{key}")
            if int(item["n_patients"]) != expected["patients"]:
                raise RuntimeError(f"{metric} has the wrong patient count")
            if stream_mode == "slice_random" and index == 0 and int(item["n_slices"]) != expected["slices"]:
                raise RuntimeError(f"{metric} has the wrong all-slice count")


def validate(
    config_path: str, results_root: Path, seed: int, stream_mode: str
) -> dict[str, Any]:
    if seed not in SOURCE_SEEDS or stream_mode not in STREAM_MODES:
        raise ValueError("Invalid SAR comparison cell")
    base_cfg = load_config(config_path)
    root = run_root(results_root, seed, stream_mode)
    manifest = _read_json(root / "run_manifest.json")
    checkpoint_hash = _validate_setup(manifest, base_cfg, seed, stream_mode, results_root)
    batch_size = BATCH_SIZES[stream_mode]
    any_update = False

    for vendor in VENDORS:
        expected = EXPECTED[vendor]
        records = _read_jsonl(root / f"vendor_{vendor}.jsonl")
        if stream_mode == "patient_volume":
            validate_target_order(manifest, records, base_cfg, vendor, seed)
            if len(records) != expected["volumes"] or len({r["patient_id"] for r in records}) != expected["patients"]:
                raise RuntimeError(f"Vendor {vendor} has incorrect patient/volume coverage")
            phases: dict[str, set[str]] = {}
            adaptations: list[dict[str, Any]] = []
            seen_slices = 0
            for record in records:
                phases.setdefault(record["patient_id"], set()).add(record["phase"])
                if (
                    record.get("method") != METHOD
                    or record.get("prediction_source") != "student"
                    or record.get("source_checkpoint_sha256") != checkpoint_hash
                    or set(record.get("trainable_parameters", [])) != set(manifest["trainable_parameters"])
                    or set(record.get("metrics", {})) != PATIENT_METRICS
                ):
                    raise RuntimeError(f"Vendor {vendor} has an invalid SAR volume record")
                if not all(math.isfinite(float(value)) for value in record["metrics"].values()):
                    raise RuntimeError(f"Vendor {vendor} has non-finite metrics")
                volume_slices = int(record["n_slices"])
                volume_adaptations = record["adaptation"]
                if sum(int(item["arrival_batch_size"]) for item in volume_adaptations) != volume_slices:
                    raise RuntimeError(f"Vendor {vendor} volume batching is incomplete")
                if "entropy_label_probe" not in record:
                    raise RuntimeError(f"Vendor {vendor} is missing a volume SAR probe")
                _validate_probe(record["entropy_label_probe"], volume_slices, vendor)
                seen_slices += volume_slices
                adaptations.extend(volume_adaptations)
            if any(value != {"ED", "ES"} for value in phases.values()):
                raise RuntimeError(f"Vendor {vendor} is missing ED or ES")
            if seen_slices != expected["slices"]:
                raise RuntimeError(f"Vendor {vendor} has the wrong slice count")
            any_update = _validate_adaptations(
                adaptations, expected["slices"], batch_size, vendor
            ) or any_update
        else:
            order_hash = validate_target_slice_order(manifest, records, base_cfg, vendor, seed)
            batches = _read_jsonl(root / f"vendor_{vendor}_batches.jsonl")
            expected_batches = math.ceil(expected["slices"] / batch_size)
            if (
                len(records) != expected["slices"]
                or len(batches) != expected_batches
                or len({record["patient_id"] for record in records}) != expected["patients"]
            ):
                raise RuntimeError(f"Vendor {vendor} has incorrect random-slice coverage")
            flattened_ids: list[str] = []
            adaptations = []
            for record in records:
                if (
                    record.get("method") != METHOD
                    or record.get("prediction_source") != "student"
                    or record.get("source_checkpoint_sha256") != checkpoint_hash
                    or set(record.get("metrics", {})) != SLICE_METRICS
                ):
                    raise RuntimeError(f"Vendor {vendor} has an invalid SAR slice record")
                if not all(math.isfinite(float(value)) for value in record["metrics"].values()):
                    raise RuntimeError(f"Vendor {vendor} has non-finite slice metrics")
            for batch_index, batch in enumerate(batches):
                remainder = expected["slices"] % batch_size
                expected_size = remainder if batch_index == expected_batches - 1 and remainder else batch_size
                if (
                    batch.get("method") != METHOD
                    or int(batch.get("source_seed", -1)) != seed
                    or batch.get("vendor") != vendor
                    or int(batch.get("batch_arrival_index", -1)) != batch_index
                    or int(batch.get("arrival_batch_size", -1)) != expected_size
                    or batch.get("slice_order_sha256") != order_hash
                ):
                    raise RuntimeError(f"Vendor {vendor} batch {batch_index} is invalid")
                flattened_ids.extend(batch["slice_ids"])
                adaptation = batch["adaptation"]
                adaptation["arrival_batch_size"] = expected_size
                if batch.get("entropy_label_probe") != adaptation.get("entropy_label_probe"):
                    raise RuntimeError(f"Vendor {vendor} batch probe does not match adaptation probe")
                adaptations.append(adaptation)
            if flattened_ids != [record["slice_id"] for record in records]:
                raise RuntimeError(f"Vendor {vendor} batches do not reproduce the slice stream")
            any_update = _validate_adaptations(
                adaptations, expected["slices"], batch_size, vendor
            ) or any_update
        _validate_summary(root / f"vendor_{vendor}_summary.json", stream_mode, expected)

    if not any_update:
        raise RuntimeError("SAR did not execute any second-pass SGD update")
    for path in sorted(root.iterdir()):
        if path.is_file():
            print(file_sha256(path), path)
    print(
        f"[VALIDATED] method=sar seed={seed} stream={stream_mode} "
        f"batch_size={batch_size} lr={LEARNING_RATE} root={root}"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--source-seed", required=True, type=int, choices=SOURCE_SEEDS)
    parser.add_argument("--stream-mode", required=True, choices=STREAM_MODES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate(args.config, args.results_root, args.source_seed, args.stream_mode)


if __name__ == "__main__":
    main()
