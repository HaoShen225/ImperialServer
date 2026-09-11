"""Validate one cell of the locked BS(8/4), LR=1e-3 comparison."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Iterable

from torch import nn

from metrics import SLICE_METRIC_POLICY
from model import build_model
from run_bs_lr_comparison import (
    ADAPTIVE_METHODS,
    BATCH_SIZES,
    DEFAULT_RESULTS_ROOT,
    LEARNING_RATE,
    METHODS,
    PROFILE_KIND,
    SOURCE_SEEDS,
    STREAM_MODES,
    configure_comparison,
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
PREDICTION_SOURCES = {
    "source": "source_model",
    "tbn": "tbn_model",
    "tent": "student",
    "cotta": "ema_teacher",
    "grata": "student",
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


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite {label}: {value}")
    return result


def _expected_trainable(cfg: dict[str, Any], method_name: str) -> set[str]:
    if method_name in {"source", "tbn"}:
        return set()
    model = build_model(cfg, pretrained_override=False)
    if method_name == "cotta":
        return {name for name, _ in model.named_parameters()}
    batch_norm_names = {
        f"{module_name}.{parameter_name}"
        for module_name, module in model.named_modules()
        if isinstance(module, nn.BatchNorm2d)
        for parameter_name in ("weight", "bias")
    }
    return batch_norm_names


def _validate_setup(
    manifest: dict[str, Any],
    base_cfg: dict[str, Any],
    method_name: str,
    seed: int,
    stream_mode: str,
    results_root: Path,
) -> str:
    expected_cfg = configure_comparison(base_cfg, method_name, stream_mode, results_root)
    expected_method_cfg = expected_cfg["methods"][method_name]
    if manifest.get("resolved_method_config") != expected_method_cfg:
        raise RuntimeError("Resolved method configuration differs from the comparison profile")
    if manifest.get("resolved_config") != expected_cfg:
        raise RuntimeError("Resolved run configuration differs from the comparison profile")
    if method_name in ADAPTIVE_METHODS:
        method_cfg = manifest["resolved_method_config"]
        if (
            method_cfg.get("profile_kind") != PROFILE_KIND
            or float(method_cfg.get("lr")) != LEARNING_RATE
        ):
            raise RuntimeError(f"{method_name} does not use the locked LR=1e-3 profile")
    elif "lr" in manifest["resolved_method_config"]:
        raise RuntimeError(f"{method_name} unexpectedly exposes a learning rate")

    if (
        manifest.get("method") != method_name
        or int(manifest.get("source_seed", -1)) != seed
        or manifest.get("stream_mode") != stream_mode
        or manifest.get("vendors") != list(VENDORS)
        or manifest.get("initialization_profile") != "stochastic"
        or manifest.get("slice_filter") != "manifest_has_fg_equals_1"
    ):
        raise RuntimeError("Manifest identifies the wrong comparison experiment")

    resolved_tta = manifest["resolved_config"]["tta"]
    if (
        int(resolved_tta["batch_size"]) != BATCH_SIZES[stream_mode]
        or resolved_tta.get("timing") != "adapt_then_predict"
        or resolved_tta.get("reset") != "vendor"
    ):
        raise RuntimeError("Manifest violates the locked arrival protocol")

    expected_trainable = _expected_trainable(base_cfg, method_name)
    if set(manifest.get("trainable_parameters", [])) != expected_trainable:
        raise RuntimeError(f"{method_name} has the wrong trainable parameter scope")

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
        raise RuntimeError("Manifest contains a stale source/target protocol hash")
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


def _validate_adaptation(
    adaptation: dict[str, Any], method_name: str, arrival_size: int, vendor: str
) -> None:
    if int(adaptation.get("arrival_batch_size", -1)) != arrival_size:
        raise RuntimeError(f"Vendor {vendor} has inconsistent arrival batch metadata")
    if int(adaptation.get("n_seen", -1)) != arrival_size:
        raise RuntimeError(f"Vendor {vendor} has inconsistent n_seen metadata")
    extras = adaptation.get("extras", {})
    drift = _finite(extras.get("parameter_drift"), "parameter drift")
    for key in ("adaptation_seconds", "prediction_seconds"):
        if _finite(extras.get(key), key) < 0.0:
            raise RuntimeError(f"Vendor {vendor} has negative {key}")
    for value in adaptation.get("predicted_foreground_area", {}).values():
        _finite(value, "predicted foreground area")

    if method_name in {"source", "tbn"}:
        if (
            bool(adaptation.get("updated"))
            or int(adaptation.get("n_selected", -1)) != 0
            or adaptation.get("loss") is not None
            or drift != 0.0
        ):
            raise RuntimeError(f"Vendor {vendor} {method_name} changed model parameters")
        return

    if method_name in {"tent", "cotta"}:
        if (
            not bool(adaptation.get("updated"))
            or int(adaptation.get("n_selected", -1)) != arrival_size
            or not math.isfinite(float(adaptation.get("loss")))
            or drift <= 0.0
        ):
            raise RuntimeError(f"Vendor {vendor} has an invalid {method_name} update")
        if method_name == "cotta":
            augmented = int(extras["augmentation_triggered_slices"])
            coverage = _finite(extras["augmentation_coverage"], "augmentation coverage")
            if (
                not 0 <= augmented <= arrival_size
                or not math.isclose(coverage, augmented / arrival_size, abs_tol=1e-12)
                or int(extras["teacher_views_when_triggered"]) != 14
            ):
                raise RuntimeError(f"Vendor {vendor} has invalid CoTTA diagnostics")
        return

    if method_name != "grata":
        raise AssertionError(method_name)
    loss = _finite(adaptation.get("loss"), "GraTA consistency loss")
    cosine = _finite(extras["gradient_cosine"], "gradient cosine")
    effective_lr = _finite(extras["effective_lr"], "effective learning rate")
    consistency_norm = _finite(
        extras["consistency_gradient_norm"], "consistency gradient norm"
    )
    expected_lr = LEARNING_RATE * 0.25 * (cosine + 1.0) ** 2
    should_update = effective_lr > 0.0 and consistency_norm > 0.0
    if (
        loss < 0.0
        or int(adaptation.get("n_selected", -1)) != arrival_size
        or not -1.0 <= cosine <= 1.0
        or not math.isclose(effective_lr, expected_lr, rel_tol=1e-9, abs_tol=1e-12)
        or not 0.0 <= effective_lr <= LEARNING_RATE
        or bool(adaptation.get("updated")) != should_update
        or (should_update and drift <= 0.0)
        or int(extras["weak_view_count"]) != 6
    ):
        raise RuntimeError(f"Vendor {vendor} has an invalid GraTA update")


def _validate_adaptations(
    adaptations: Iterable[dict[str, Any]],
    method_name: str,
    expected_slices: int,
    maximum_batch_size: int,
    vendor: str,
) -> None:
    seen = 0
    batches = 0
    for adaptation in adaptations:
        arrival_size = int(adaptation.get("arrival_batch_size", -1))
        if not 1 <= arrival_size <= maximum_batch_size:
            raise RuntimeError(f"Vendor {vendor} has invalid batch size {arrival_size}")
        _validate_adaptation(adaptation, method_name, arrival_size, vendor)
        seen += arrival_size
        batches += 1
    if batches == 0 or seen != expected_slices:
        raise RuntimeError(
            f"Vendor {vendor} covers {seen}/{expected_slices} slices in {batches} batches"
        )


def _validate_summary(path: Path, stream_mode: str, expected: dict[str, int]) -> None:
    summary = _read_json(path)
    if stream_mode == "patient_volume":
        if set(summary) != PATIENT_METRICS:
            raise RuntimeError("Patient-volume summary has the wrong metric set")
        metric_groups = [summary]
    else:
        if (
            summary.get("aggregation_unit") != "slice"
            or summary.get("metric_policy") != SLICE_METRIC_POLICY
        ):
            raise RuntimeError("Random-slice summary has the wrong aggregation policy")
        metric_groups = [summary.get("all_slices", {}), summary.get("foreground_present", {})]
        if any(set(group) != SLICE_METRICS for group in metric_groups):
            raise RuntimeError("Random-slice summary has the wrong metric set")
    for group_index, group in enumerate(metric_groups):
        for metric, item in group.items():
            for key in ("mean", "ci95_low", "ci95_high"):
                _finite(item[key], f"{metric}/{key}")
            if int(item["n_patients"]) != expected["patients"]:
                raise RuntimeError(f"{metric} has the wrong patient count")
            if stream_mode == "slice_random" and group_index == 0:
                if int(item["n_slices"]) != expected["slices"]:
                    raise RuntimeError(f"{metric} has the wrong all-slice count")


def validate(
    config_path: str,
    results_root: Path,
    method_name: str,
    seed: int,
    stream_mode: str,
) -> dict[str, Any]:
    if method_name not in METHODS or seed not in SOURCE_SEEDS or stream_mode not in STREAM_MODES:
        raise ValueError("Invalid comparison cell")
    base_cfg = load_config(config_path)
    root = run_root(results_root, method_name, seed, stream_mode)
    manifest = _read_json(root / "run_manifest.json")
    checkpoint_hash = _validate_setup(
        manifest, base_cfg, method_name, seed, stream_mode, results_root
    )
    batch_size = BATCH_SIZES[stream_mode]

    for vendor in VENDORS:
        expected = EXPECTED[vendor]
        records = _read_jsonl(root / f"vendor_{vendor}.jsonl")
        if stream_mode == "patient_volume":
            validate_target_order(manifest, records, base_cfg, vendor, seed)
            if (
                len(records) != expected["volumes"]
                or len({record["patient_id"] for record in records}) != expected["patients"]
            ):
                raise RuntimeError(f"Vendor {vendor} has incorrect patient/volume coverage")
            phases: dict[str, set[str]] = {}
            adaptations: list[dict[str, Any]] = []
            seen_slices = 0
            for record in records:
                phases.setdefault(record["patient_id"], set()).add(record["phase"])
                if (
                    record.get("method") != method_name
                    or record.get("prediction_source") != PREDICTION_SOURCES[method_name]
                    or record.get("source_checkpoint_sha256") != checkpoint_hash
                    or set(record.get("trainable_parameters", []))
                    != set(manifest["trainable_parameters"])
                    or set(record.get("metrics", {})) != PATIENT_METRICS
                ):
                    raise RuntimeError(f"Vendor {vendor} has an invalid volume record")
                if not all(math.isfinite(float(value)) for value in record["metrics"].values()):
                    raise RuntimeError(f"Vendor {vendor} has non-finite metrics")
                volume_adaptations = record["adaptation"]
                volume_slices = int(record["n_slices"])
                if sum(int(item["arrival_batch_size"]) for item in volume_adaptations) != volume_slices:
                    raise RuntimeError(f"Vendor {vendor} volume batching is incomplete")
                seen_slices += volume_slices
                adaptations.extend(volume_adaptations)
            if any(value != {"ED", "ES"} for value in phases.values()):
                raise RuntimeError(f"Vendor {vendor} is missing ED or ES")
            if seen_slices != expected["slices"]:
                raise RuntimeError(f"Vendor {vendor} has the wrong foreground slice count")
            _validate_adaptations(
                adaptations, method_name, expected["slices"], batch_size, vendor
            )
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
                    record.get("method") != method_name
                    or record.get("prediction_source") != PREDICTION_SOURCES[method_name]
                    or record.get("source_checkpoint_sha256") != checkpoint_hash
                    or set(record.get("metrics", {})) != SLICE_METRICS
                ):
                    raise RuntimeError(f"Vendor {vendor} has an invalid slice record")
                if not all(math.isfinite(float(value)) for value in record["metrics"].values()):
                    raise RuntimeError(f"Vendor {vendor} has non-finite slice metrics")
            for batch_index, batch in enumerate(batches):
                expected_size = (
                    expected["slices"] % batch_size
                    if batch_index == expected_batches - 1 and expected["slices"] % batch_size
                    else batch_size
                )
                if (
                    batch.get("method") != method_name
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
                adaptations.append(adaptation)
            if flattened_ids != [record["slice_id"] for record in records]:
                raise RuntimeError(f"Vendor {vendor} batches do not reproduce the slice stream")
            _validate_adaptations(
                adaptations, method_name, expected["slices"], batch_size, vendor
            )
        _validate_summary(
            root / f"vendor_{vendor}_summary.json", stream_mode, expected
        )

    for path in sorted(root.iterdir()):
        if path.is_file():
            print(file_sha256(path), path)
    print(
        f"[VALIDATED] method={method_name} seed={seed} stream={stream_mode} "
        f"batch_size={batch_size} root={root}"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--source-seed", required=True, type=int, choices=SOURCE_SEEDS)
    parser.add_argument("--stream-mode", required=True, choices=STREAM_MODES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate(
        args.config, args.results_root, args.method, args.source_seed, args.stream_mode
    )


if __name__ == "__main__":
    main()
