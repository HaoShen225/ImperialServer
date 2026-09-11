"""Validate one completed GraTA learning-rate sweep combination."""

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
from run_grata_lr_sweep import resolve_learning_rate
from target_order_validation import validate_target_order, validate_target_slice_order
from utils import file_sha256, load_config


VENDORS = ("B", "C", "D")
PATIENT_COUNTS = {
    "B": {"patients": 125, "volumes": 250, "slices": 2049},
    "C": {"patients": 50, "volumes": 100, "slices": 806},
    "D": {"patients": 50, "volumes": 100, "slices": 835},
}
SLICE_COUNTS = {
    "B": {"patients": 125, "slices": 2049, "batches": 257, "last_batch": 1},
    "C": {"patients": 50, "slices": 806, "batches": 101, "last_batch": 6},
    "D": {"patients": 50, "slices": 835, "batches": 105, "last_batch": 3},
}
PATIENT_METRICS = {
    "dice_rv", "dice_myo", "dice_lv", "dice_macro",
    "hd95_px_rv", "hd95_px_myo", "hd95_px_lv", "hd95_px_macro",
}
SLICE_METRICS = {
    "dice_rv", "dice_myo", "dice_lv", "dice_macro",
    "hd95_2d_px_rv", "hd95_2d_px_myo", "hd95_2d_px_lv", "hd95_2d_px_macro",
}
STRONG_COVERAGE_KEYS = {
    "strong_brightness_coverage",
    "strong_contrast_coverage",
    "strong_gamma_coverage",
    "strong_noise_coverage",
    "strong_blur_coverage",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _finite(value: Any, label: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise RuntimeError(f"Non-finite {label}: {value}")
    return resolved


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
    return results_root / f"lr_{tag}" / "grata" / f"seed{seed}" / protocol


def _expected_method_cfg(
    cfg: dict[str, Any], learning_rate: float
) -> dict[str, Any]:
    expected = deepcopy(cfg["methods"]["grata"])
    expected["profile_kind"] = "lr_sweep"
    expected["lr"] = learning_rate
    return expected


def _validate_setup(
    manifest: dict[str, Any],
    cfg: dict[str, Any],
    seed: int,
    stream_mode: str,
    learning_rate: float,
) -> None:
    expected_method_cfg = _expected_method_cfg(cfg, learning_rate)
    if manifest.get("resolved_method_config") != expected_method_cfg:
        raise RuntimeError("GraTA sweep changed a non-learning-rate method setting")
    if manifest["resolved_config"]["methods"]["grata"] != expected_method_cfg:
        raise RuntimeError("GraTA resolved config and method config disagree")

    expected_batch_size = 4 if stream_mode == "patient_volume" else 8
    resolved_tta = manifest["resolved_config"]["tta"]
    if (
        resolved_tta.get("stream_mode") != stream_mode
        or int(resolved_tta["batch_size"]) != expected_batch_size
        or resolved_tta.get("timing") != "adapt_then_predict"
        or resolved_tta.get("reset") != "vendor"
    ):
        raise RuntimeError(f"GraTA manifest has the wrong locked {stream_mode} protocol")

    model = build_model(cfg, pretrained_override=False)
    expected_names = {
        f"{module_name}.{parameter_name}"
        for module_name, module in model.named_modules()
        if isinstance(module, nn.BatchNorm2d)
        for parameter_name in ("weight", "bias")
    }
    if set(manifest.get("trainable_parameters", [])) != expected_names:
        raise RuntimeError("GraTA manifest does not declare exactly all BN affine parameters")

    checkpoint = Path(cfg["source"]["checkpoint_dir"]) / f"seed{seed}_best.pt"
    metadata = _read_json(
        Path(cfg["source"]["checkpoint_dir"]) / f"seed{seed}_best.json"
    )
    checkpoint_hash = file_sha256(checkpoint)
    if checkpoint_hash != metadata["checkpoint_sha256"]:
        raise RuntimeError("Source checkpoint differs from its training metadata")
    if manifest.get("source_checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("GraTA manifest has an incorrect source checkpoint hash")
    if manifest.get("protocol_sha256") != file_sha256(cfg["data"]["protocol_file"]):
        raise RuntimeError("GraTA manifest has a stale protocol hash")
    stream_file = (
        cfg["data"]["stream_file"]
        if stream_mode == "patient_volume"
        else cfg["data"]["slice_stream_file"]
    )
    if manifest.get("target_stream_sha256") != file_sha256(stream_file):
        raise RuntimeError("GraTA manifest has a stale target-stream hash")
    if (
        stream_mode == "slice_random"
        and manifest.get("slice_metric_policy") != SLICE_METRIC_POLICY
    ):
        raise RuntimeError("GraTA manifest has a stale random-slice metric policy")


def _validate_adaptations(
    adaptations: Iterable[dict[str, Any]],
    *,
    maximum_batch_size: int,
    learning_rate: float,
    expected_slices: int,
    vendor: str,
) -> None:
    seen_slices = 0
    batches = 0
    for batch_index, adaptation in enumerate(adaptations):
        batches += 1
        arrival_size = int(adaptation["arrival_batch_size"])
        if not 1 <= arrival_size <= maximum_batch_size:
            raise RuntimeError(
                f"Vendor {vendor} batch {batch_index} has invalid size {arrival_size}"
            )
        if (
            int(adaptation["n_seen"]) != arrival_size
            or int(adaptation["n_selected"]) != arrival_size
        ):
            raise RuntimeError(f"Vendor {vendor} GraTA skipped arrival samples")

        consistency_loss = _finite(adaptation["loss"], "GraTA consistency loss")
        extras = adaptation["extras"]
        entropy_loss = _finite(extras["entropy_loss"], "entropy loss")
        recorded_consistency = _finite(
            extras["consistency_loss"], "recorded consistency loss"
        )
        cosine = _finite(extras["gradient_cosine"], "gradient cosine")
        entropy_norm = _finite(extras["entropy_gradient_norm"], "entropy gradient norm")
        consistency_norm = _finite(
            extras["consistency_gradient_norm"], "consistency gradient norm"
        )
        effective_lr = _finite(extras["effective_lr"], "effective learning rate")
        if entropy_loss < 0.0 or entropy_norm < 0.0 or consistency_norm < 0.0:
            raise RuntimeError(f"Vendor {vendor} has a negative GraTA diagnostic")
        if not math.isclose(
            consistency_loss, recorded_consistency, rel_tol=1e-7, abs_tol=1e-9
        ):
            raise RuntimeError(f"Vendor {vendor} has inconsistent GraTA loss records")
        if not -1.0 <= cosine <= 1.0:
            raise RuntimeError(f"Vendor {vendor} gradient cosine is outside [-1, 1]")
        expected_lr = learning_rate * 0.25 * (cosine + 1.0) ** 2
        if not math.isclose(effective_lr, expected_lr, rel_tol=1e-9, abs_tol=1e-12):
            raise RuntimeError(f"Vendor {vendor} GraTA dynamic learning rate is incorrect")
        if not 0.0 <= effective_lr <= learning_rate:
            raise RuntimeError(f"Vendor {vendor} effective LR is outside [0, beta]")
        should_update = effective_lr > 0.0 and consistency_norm > 0.0
        if bool(adaptation["updated"]) != should_update:
            raise RuntimeError(f"Vendor {vendor} GraTA update flag is inconsistent")
        drift = _finite(extras["parameter_drift"], "parameter drift")
        if drift < 0.0 or (should_update and drift <= 0.0):
            raise RuntimeError(f"Vendor {vendor} GraTA has invalid parameter drift")
        if int(extras["weak_view_count"]) != 6:
            raise RuntimeError(f"Vendor {vendor} did not use six GraTA weak views")
        for key in STRONG_COVERAGE_KEYS:
            coverage = _finite(extras[key], key)
            if not 0.0 <= coverage <= 1.0:
                raise RuntimeError(f"Vendor {vendor} has invalid {key}")
        for key in ("adaptation_seconds", "prediction_seconds"):
            if _finite(extras[key], key) < 0.0:
                raise RuntimeError(f"Vendor {vendor} has negative {key}")
        for value in adaptation["predicted_foreground_area"].values():
            _finite(value, "predicted foreground area")
        seen_slices += arrival_size
    if batches == 0 or seen_slices != expected_slices:
        raise RuntimeError(
            f"Vendor {vendor} GraTA covered {seen_slices}/{expected_slices} slices"
        )


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
    if (
        manifest.get("method") != "grata"
        or int(manifest.get("source_seed", -1)) != seed
        or manifest.get("stream_mode") != stream_mode
        or manifest.get("vendors") != list(VENDORS)
        or manifest.get("initialization_profile") != "stochastic"
        or manifest.get("slice_filter") != "manifest_has_fg_equals_1"
    ):
        raise RuntimeError("Manifest does not describe the locked GraTA B/C/D sweep")
    _validate_setup(manifest, cfg, seed, stream_mode, learning_rate)
    maximum_batch_size = 4 if stream_mode == "patient_volume" else 8

    for vendor in VENDORS:
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
            adaptations = []
            for record in records:
                if (
                    record.get("method") != "grata"
                    or record.get("prediction_source") != "student"
                    or set(record.get("metrics", {})) != PATIENT_METRICS
                ):
                    raise RuntimeError(f"Vendor {vendor} contains an invalid volume record")
                for metric, value in record["metrics"].items():
                    _finite(value, f"{vendor}/{record['volume_id']}/{metric}")
                adaptations.extend(record["adaptation"])
            _validate_patient_summary(
                _read_json(root / f"vendor_{vendor}_summary.json"),
                expected["patients"],
            )
        else:
            expected = SLICE_COUNTS[vendor]
            order_hash = validate_target_slice_order(manifest, records, cfg, vendor, seed)
            batches = _read_jsonl(root / f"vendor_{vendor}_batches.jsonl")
            if (
                len(records) != expected["slices"]
                or len(batches) != expected["batches"]
                or len({record["patient_id"] for record in records})
                != expected["patients"]
            ):
                raise RuntimeError(f"Vendor {vendor} has incorrect slice coverage")
            for record in records:
                if (
                    record.get("method") != "grata"
                    or record.get("prediction_source") != "student"
                    or set(record.get("metrics", {})) != SLICE_METRICS
                ):
                    raise RuntimeError(f"Vendor {vendor} contains an invalid slice record")
                for metric, value in record["metrics"].items():
                    _finite(value, f"{vendor}/{record['slice_id']}/{metric}")
            adaptations = []
            flattened_ids: list[str] = []
            for batch_index, batch in enumerate(batches):
                expected_size = (
                    expected["last_batch"]
                    if batch_index == len(batches) - 1
                    else maximum_batch_size
                )
                if (
                    int(batch["arrival_batch_size"]) != expected_size
                    or batch.get("method") != "grata"
                    or int(batch.get("source_seed", -1)) != seed
                    or batch.get("vendor") != vendor
                    or int(batch.get("batch_arrival_index", -1)) != batch_index
                    or batch.get("slice_order_sha256") != order_hash
                ):
                    raise RuntimeError(f"Vendor {vendor} batch {batch_index} is invalid")
                flattened_ids.extend(batch["slice_ids"])
                adaptations.append(batch["adaptation"])
            if flattened_ids != [record["slice_id"] for record in records]:
                raise RuntimeError(f"Vendor {vendor} batches do not cover its slice stream")
            _validate_slice_summary(
                _read_json(root / f"vendor_{vendor}_summary.json"), expected
            )

        _validate_adaptations(
            adaptations,
            maximum_batch_size=maximum_batch_size,
            learning_rate=learning_rate,
            expected_slices=expected["slices"],
            vendor=vendor,
        )
    print(
        f"[VALIDATED] method=grata beta={tag} seed={seed} "
        f"stream={stream_mode} root={root}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/Stochastic_Ini_ForegroundOnly/grata_lr_sweep"),
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
