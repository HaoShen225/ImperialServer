"""Run locked image-only volume or random-slice TTA evaluation protocols."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data import (
    TARGET_ORDER_POLICY,
    TARGET_SLICE_ORDER_POLICY,
    MMSTargetSliceDataset,
    build_target_slice_loader,
    build_target_stream,
    split_volume_into_batches,
)
from metrics import (
    SLICE_METRIC_POLICY,
    aggregate_results,
    aggregate_slice_results,
    evaluate_slice,
    evaluate_volume,
)
from model import build_model, load_source_checkpoint
from tta_methods import METHODS, BaseTTA, build_method
from tta_methods.common import predicted_foreground_area
from utils import file_sha256, get_device, load_config, run_metadata, save_json, set_seed


def run_volume(
    method: BaseTTA,
    images: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Run an image tensor; labels and volume metadata cannot enter this boundary."""
    predictions, records = [], []
    for batch in split_volume_into_batches(images, batch_size):
        device_batch = batch.to(device)
        logits, info = method.process_batch(device_batch)
        predictions.append(logits.argmax(dim=1).cpu())
        record = info.to_dict()
        record["arrival_batch_size"] = int(batch.shape[0])
        record["predicted_foreground_area"] = predicted_foreground_area(logits)
        if info.probe_payload is not None:
            record["_probe_payload"] = {
                stage: {
                    key: value.detach().cpu()
                    for key, value in payload.items()
                }
                for stage, payload in info.probe_payload.items()
            }
        records.append(record)
    return torch.cat(predictions), records


def run_random_slice_batch(
    method: BaseTTA,
    images: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, dict[str, torch.Tensor]] | None]:
    """Adapt and predict one image-only random-slice batch without label access."""
    device_batch = images.to(device)
    logits, info = method.process_batch(device_batch)
    record = info.to_dict()
    record["arrival_batch_size"] = int(images.shape[0])
    record["predicted_foreground_area"] = predicted_foreground_area(logits)
    payload = None
    if info.probe_payload is not None:
        payload = {
            stage: {key: value.detach().cpu() for key, value in stage_payload.items()}
            for stage, stage_payload in info.probe_payload.items()
        }
    return logits.argmax(dim=1).cpu(), record, payload


def _probe_stage_counts(
    payload: dict[str, torch.Tensor], target: torch.Tensor
) -> dict[str, int | float | None]:
    selected = payload["selected"].to(dtype=torch.bool, device="cpu")
    labels = payload["labels"].to(device="cpu")
    if selected.ndim != 1 or selected.shape[0] != target.shape[0]:
        raise ValueError("Entropy probe selection shape does not match the target batch")
    if labels.shape != target.shape:
        raise ValueError("Entropy probe pseudo-label shape does not match the target batch")
    selected_target = target[selected]
    selected_labels = labels[selected]
    selected_pixels = int(selected_target.numel())
    correct_pixels = int((selected_labels == selected_target).sum()) if selected_pixels else 0
    foreground = selected_target > 0
    foreground_pixels = int(foreground.sum()) if selected_pixels else 0
    correct_foreground_pixels = (
        int(((selected_labels == selected_target) & foreground).sum())
        if foreground_pixels
        else 0
    )
    seen_slices = int(target.shape[0])
    selected_slices = int(selected.sum())
    return {
        "seen_slices": seen_slices,
        "selected_slices": selected_slices,
        "selection_coverage": selected_slices / seen_slices if seen_slices else None,
        "selected_pixels": selected_pixels,
        "correct_pixels": correct_pixels,
        "pixel_accuracy": correct_pixels / selected_pixels if selected_pixels else None,
        "gt_foreground_pixels": foreground_pixels,
        "correct_gt_foreground_pixels": correct_foreground_pixels,
        "foreground_pixel_accuracy": (
            correct_foreground_pixels / foreground_pixels if foreground_pixels else None
        ),
    }


def _aggregate_probe_counts(
    probes: list[dict[str, dict[str, int | float | None]]]
) -> dict[str, dict[str, int | float | None]]:
    stages = ("first_filter", "second_filter")
    aggregate: dict[str, dict[str, int | float | None]] = {}
    for stage in stages:
        counts = {
            key: sum(int(probe[stage][key]) for probe in probes)
            for key in (
                "seen_slices",
                "selected_slices",
                "selected_pixels",
                "correct_pixels",
                "gt_foreground_pixels",
                "correct_gt_foreground_pixels",
            )
        }
        aggregate[stage] = {
            **counts,
            "selection_coverage": (
                counts["selected_slices"] / counts["seen_slices"]
                if counts["seen_slices"]
                else None
            ),
            "pixel_accuracy": (
                counts["correct_pixels"] / counts["selected_pixels"]
                if counts["selected_pixels"]
                else None
            ),
            "foreground_pixel_accuracy": (
                counts["correct_gt_foreground_pixels"] / counts["gt_foreground_pixels"]
                if counts["gt_foreground_pixels"]
                else None
            ),
        }
    return aggregate


def attach_entropy_label_probe(
    adaptation_records: list[dict[str, Any]], target: torch.Tensor
) -> dict[str, dict[str, int | float | None]] | None:
    """Attach label-aware diagnostics after adaptation without leaking labels into TTA."""
    offset = 0
    probes = []
    for record in adaptation_records:
        batch_size = int(record["arrival_batch_size"])
        batch_target = target[offset : offset + batch_size].cpu()
        offset += batch_size
        payload = record.pop("_probe_payload", None)
        if payload is None:
            continue
        probe = {
            stage: _probe_stage_counts(stage_payload, batch_target)
            for stage, stage_payload in payload.items()
        }
        if probe["second_filter"]["selected_slices"] > probe["first_filter"]["selected_slices"]:
            raise RuntimeError("SAR second entropy filter is not a subset of the first filter")
        record["entropy_label_probe"] = probe
        probes.append(probe)
    if offset != int(target.shape[0]):
        raise ValueError("Adaptation batches do not cover the complete target volume")
    return _aggregate_probe_counts(probes) if probes else None


_CSL_CLASS_NAMES = {0: "bg", 1: "rv", 2: "myo", 3: "lv"}


def _finalize_csl_counts(counts: dict[str, Any]) -> dict[str, Any]:
    seen_pixels = int(counts["seen_pixels"])
    reliable_pixels = int(counts["reliable_pixels"])
    weight_sum = float(counts["weight_sum"])
    foreground_pixels = int(counts["gt_foreground_pixels"])
    reliable_foreground = int(counts["reliable_gt_foreground_pixels"])
    result = {
        **counts,
        "reliable_coverage": reliable_pixels / seen_pixels if seen_pixels else None,
        "reliable_accuracy": (
            int(counts["correct_reliable_pixels"]) / reliable_pixels
            if reliable_pixels
            else None
        ),
        "effective_weight_coverage": weight_sum / seen_pixels if seen_pixels else None,
        "weighted_accuracy": (
            float(counts["weighted_correct_pixels"]) / weight_sum
            if weight_sum > 0.0
            else None
        ),
        "reliable_foreground_coverage": (
            reliable_foreground / foreground_pixels if foreground_pixels else None
        ),
        "reliable_foreground_accuracy": (
            int(counts["correct_reliable_foreground_pixels"]) / reliable_foreground
            if reliable_foreground
            else None
        ),
        "weighted_foreground_accuracy": (
            float(counts["weighted_correct_foreground_pixels"])
            / float(counts["foreground_weight_sum"])
            if float(counts["foreground_weight_sum"]) > 0.0
            else None
        ),
    }
    per_class = {}
    for class_name, class_counts in counts["per_class"].items():
        predicted_pixels = int(class_counts["predicted_pixels"])
        class_reliable = int(class_counts["reliable_pixels"])
        class_weight_sum = float(class_counts["weight_sum"])
        per_class[class_name] = {
            **class_counts,
            "reliable_coverage": (
                class_reliable / predicted_pixels if predicted_pixels else None
            ),
            "reliable_accuracy": (
                int(class_counts["correct_reliable_pixels"]) / class_reliable
                if class_reliable
                else None
            ),
            "effective_weight_coverage": (
                class_weight_sum / predicted_pixels if predicted_pixels else None
            ),
            "weighted_accuracy": (
                float(class_counts["weighted_correct_pixels"]) / class_weight_sum
                if class_weight_sum > 0.0
                else None
            ),
        }
    result["per_class"] = per_class
    return result


def _csl_probe_counts(
    payload: dict[str, torch.Tensor], target: torch.Tensor
) -> dict[str, Any]:
    selected = payload["selected"].to(dtype=torch.bool, device="cpu")
    labels = payload["labels"].to(dtype=torch.long, device="cpu")
    weights = payload["weights"].to(dtype=torch.float64, device="cpu")
    target = target.to(dtype=torch.long, device="cpu")
    if selected.shape != target.shape or labels.shape != target.shape or weights.shape != target.shape:
        raise ValueError("CSL probe tensors must match the pixel-level target shape")
    if not torch.isfinite(weights).all() or bool((weights < 0.0).any()) or bool((weights > 1.0).any()):
        raise ValueError("CSL probe weights must be finite and lie in [0, 1]")

    correct = labels == target
    foreground = target > 0
    reliable_foreground = selected & foreground
    foreground_weights = weights * foreground
    counts: dict[str, Any] = {
        "seen_slices": int(target.shape[0]),
        "slices_with_reliable_pixels": int(selected.flatten(1).any(dim=1).sum()),
        "seen_pixels": int(target.numel()),
        "reliable_pixels": int(selected.sum()),
        "correct_reliable_pixels": int((selected & correct).sum()),
        "weight_sum": float(weights.sum()),
        "weighted_correct_pixels": float((weights * correct).sum()),
        "gt_foreground_pixels": int(foreground.sum()),
        "reliable_gt_foreground_pixels": int(reliable_foreground.sum()),
        "correct_reliable_foreground_pixels": int((reliable_foreground & correct).sum()),
        "foreground_weight_sum": float(foreground_weights.sum()),
        "weighted_correct_foreground_pixels": float((foreground_weights * correct).sum()),
        "per_class": {},
    }
    for class_id, class_name in _CSL_CLASS_NAMES.items():
        predicted = labels == class_id
        reliable = selected & predicted
        class_weights = weights * predicted
        counts["per_class"][class_name] = {
            "predicted_pixels": int(predicted.sum()),
            "reliable_pixels": int(reliable.sum()),
            "correct_reliable_pixels": int((reliable & correct).sum()),
            "weight_sum": float(class_weights.sum()),
            "weighted_correct_pixels": float((class_weights * correct).sum()),
        }
    return _finalize_csl_counts(counts)


def _aggregate_csl_probe_counts(probes: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_keys = (
        "seen_slices",
        "slices_with_reliable_pixels",
        "seen_pixels",
        "reliable_pixels",
        "correct_reliable_pixels",
        "weight_sum",
        "weighted_correct_pixels",
        "gt_foreground_pixels",
        "reliable_gt_foreground_pixels",
        "correct_reliable_foreground_pixels",
        "foreground_weight_sum",
        "weighted_correct_foreground_pixels",
    )
    counts: dict[str, Any] = {
        key: sum(probe[key] for probe in probes) for key in scalar_keys
    }
    counts["per_class"] = {
        class_name: {
            key: sum(probe["per_class"][class_name][key] for probe in probes)
            for key in (
                "predicted_pixels",
                "reliable_pixels",
                "correct_reliable_pixels",
                "weight_sum",
                "weighted_correct_pixels",
            )
        }
        for class_name in _CSL_CLASS_NAMES.values()
    }
    return _finalize_csl_counts(counts)


def attach_csl_label_probe(
    adaptation_records: list[dict[str, Any]], target: torch.Tensor
) -> dict[str, Any] | None:
    """Score CSL selections after adaptation; labels never cross the method boundary."""
    offset = 0
    probes = []
    for record in adaptation_records:
        batch_size = int(record["arrival_batch_size"])
        batch_target = target[offset : offset + batch_size].cpu()
        offset += batch_size
        payload = record.pop("_probe_payload", None)
        if payload is None:
            continue
        if set(payload) != {"csl_reliable"}:
            raise ValueError("Unexpected probe stages for a CSL adaptation record")
        probe = _csl_probe_counts(payload["csl_reliable"], batch_target)
        record["csl_label_probe"] = probe
        probes.append(probe)
    if offset != int(target.shape[0]):
        raise ValueError("Adaptation batches do not cover the complete target volume")
    return _aggregate_csl_probe_counts(probes) if probes else None


def attach_method_label_probe(
    adaptation_records: list[dict[str, Any]], target: torch.Tensor
) -> tuple[str, dict[str, Any]] | None:
    stages = {
        stage
        for record in adaptation_records
        for stage in (record.get("_probe_payload") or {})
    }
    if not stages:
        return None
    if stages == {"csl_reliable"}:
        probe = attach_csl_label_probe(adaptation_records, target)
        return ("csl_label_probe", probe) if probe is not None else None
    if stages == {"first_filter", "second_filter"}:
        probe = attach_entropy_label_probe(adaptation_records, target)
        return ("entropy_label_probe", probe) if probe is not None else None
    raise ValueError(f"Unsupported method probe stages: {sorted(stages)}")


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def _method_config(cfg: dict[str, Any], method_name: str, source_seed: int) -> dict[str, Any]:
    method_cfg = deepcopy(cfg["methods"][method_name])
    if not bool(method_cfg["profile_verified"]):
        raise ValueError(f"Final execution rejects unverified profile: {method_name}")
    if method_name == "eata" and not method_cfg.get("fisher_path"):
        method_cfg["fisher_path"] = str(Path(cfg["source"]["checkpoint_dir"]) / f"fisher_seed{source_seed}.pt")
    if method_name == "eata" and not Path(method_cfg["fisher_path"]).is_file():
        raise FileNotFoundError(f"EATA Fisher artifact is missing: {method_cfg['fisher_path']}")
    return method_cfg


def _prepare_experiment(
    cfg: dict[str, Any],
    method_name: str,
    source_seed: int,
    device: torch.device,
) -> tuple[BaseTTA, dict[str, Any], Path, str, str | None]:
    set_seed(int(cfg["experiment"]["harness_seed"]), deterministic=bool(cfg["tta"]["deterministic"]))
    checkpoint = Path(cfg["source"]["checkpoint_dir"]) / f"seed{source_seed}_best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Source checkpoint is missing: {checkpoint}")
    model = build_model(cfg, pretrained_override=False)
    checkpoint_payload = load_source_checkpoint(model, checkpoint, map_location="cpu")
    expected_initialization = str(cfg["experiment"]["initialization_profile"])
    checkpoint_initialization = checkpoint_payload.get("initialization_profile")
    if checkpoint_initialization is None:
        checkpoint_pretrained = bool(
            checkpoint_payload.get("config", {}).get("model", {}).get("pretrained_encoder")
        )
        checkpoint_initialization = "imagenet" if checkpoint_pretrained else "stochastic"
    if checkpoint_initialization != expected_initialization:
        raise RuntimeError(
            "Source checkpoint initialization does not match the active experiment profile: "
            f"{checkpoint_initialization!r} != {expected_initialization!r}"
        )
    method_cfg = _method_config(cfg, method_name, source_seed)
    method = build_method(method_name, model, method_cfg, cfg["tta"], device)
    fisher_hash = file_sha256(method_cfg["fisher_path"]) if method_name == "eata" else None
    return method, method_cfg, checkpoint, expected_initialization, fisher_hash


def _validate_evaluation_target(
    target: torch.Tensor,
    classes: list[int],
    vendor: str,
    patient_id: str,
    phase: str,
) -> None:
    if not any(bool(torch.any(target == class_id)) for class_id in classes):
        raise ValueError(
            "Evaluation target has no configured foreground labels: "
            f"{vendor}/{patient_id}/{phase}"
        )


def run_slice_experiment(
    cfg: dict[str, Any],
    method_name: str,
    source_seed: int,
    vendors: list[str],
    device: torch.device,
) -> dict[str, Any]:
    """Run Vendor-local random slice streams and emit slice-level scores."""
    if cfg["tta"]["reset"] == "patient":
        raise ValueError("slice_random does not support patient reset because batches mix patients")
    method, method_cfg, checkpoint, initialization_profile, fisher_hash = _prepare_experiment(
        cfg, method_name, source_seed, device
    )
    protocol_hash = file_sha256(cfg["data"]["protocol_file"])
    stream_hash = file_sha256(cfg["data"]["slice_stream_file"])
    checkpoint_hash = file_sha256(checkpoint)
    result_root = (
        Path(cfg["tta"]["results_dir"])
        / method_name
        / f"seed{source_seed}"
        / f"slice_random_{cfg['tta']['timing']}_{cfg['tta']['reset']}"
    )
    summaries: dict[str, Any] = {}
    target_orders: dict[str, Any] = {}
    class_names = {int(key): value for key, value in cfg["evaluation"]["class_names"].items()}
    classes = [int(value) for value in cfg["evaluation"]["classes"]]

    for vendor_index, vendor in enumerate(vendors):
        if cfg["tta"]["reset"] == "vendor" or (
            vendor_index == 0 and cfg["tta"]["reset"] != "never"
        ):
            method.reset()
        loader = build_target_slice_loader(
            vendor,
            cfg,
            order_seed=source_seed,
            batch_size=int(cfg["tta"]["batch_size"]),
        )
        dataset = loader.dataset
        if not isinstance(dataset, MMSTargetSliceDataset):
            raise TypeError("Target slice loader has an unexpected dataset type")
        if dataset.order_seed != source_seed:
            raise RuntimeError("Resolved slice order seed differs from the source checkpoint seed")
        target_orders[vendor] = {
            "order_seed": dataset.order_seed,
            "n_slices": len(dataset),
            "slice_order_sha256": dataset.slice_order_sha256,
            "slice_filter": dataset.slice_filter,
        }

        slice_records: list[dict[str, Any]] = []
        batch_records: list[dict[str, Any]] = []
        for batch_arrival_index, batch in enumerate(loader):
            predictions, adaptation, probe_payload = run_random_slice_batch(
                method, batch["image"], device
            )
            targets = dataset.load_masks(list(batch["mask_path"]))
            if probe_payload is not None:
                adaptation["_probe_payload"] = probe_payload
                label_probe = attach_method_label_probe([adaptation], targets)
            else:
                label_probe = None
            slice_ids = list(batch["slice_id"])
            batch_record = {
                "method": method_name,
                "source_seed": source_seed,
                "vendor": vendor,
                "batch_arrival_index": batch_arrival_index,
                "arrival_batch_size": int(predictions.shape[0]),
                "slice_ids": slice_ids,
                "adaptation": adaptation,
                "target_order_seed": dataset.order_seed,
                "slice_order_sha256": dataset.slice_order_sha256,
                "slice_filter": dataset.slice_filter,
            }
            if label_probe is not None:
                probe_name, probe_value = label_probe
                batch_record[probe_name] = probe_value
            batch_records.append(batch_record)

            for batch_position in range(int(predictions.shape[0])):
                metrics, gt_present = evaluate_slice(
                    predictions[batch_position].numpy(),
                    targets[batch_position].numpy(),
                    classes=classes,
                    class_names=class_names,
                )
                slice_records.append({
                    "method": method_name,
                    "profile_verified": bool(method_cfg["profile_verified"]),
                    "profile_kind": method_cfg["profile_kind"],
                    "source_seed": source_seed,
                    "target_order_seed": dataset.order_seed,
                    "slice_order_sha256": dataset.slice_order_sha256,
                    "slice_filter": dataset.slice_filter,
                    "initialization_profile": initialization_profile,
                    "method_seed": int(method_cfg["method_seed"]),
                    "vendor": vendor,
                    "patient_id": str(batch["patient_id"][batch_position]),
                    "phase": str(batch["phase"][batch_position]),
                    "z_index": int(batch["z_index"][batch_position]),
                    "slice_id": slice_ids[batch_position],
                    "slice_arrival_index": int(batch["slice_arrival_index"][batch_position]),
                    "batch_arrival_index": batch_arrival_index,
                    "batch_position": batch_position,
                    "arrival_batch_size": int(predictions.shape[0]),
                    "timing": cfg["tta"]["timing"],
                    "reset": cfg["tta"]["reset"],
                    "prediction_source": method.prediction_source,
                    "metrics": metrics,
                    "gt_present": gt_present,
                    "source_checkpoint_sha256": checkpoint_hash,
                    "protocol_sha256": protocol_hash,
                    "target_stream_sha256": stream_hash,
                })

        if len(slice_records) != len(dataset):
            raise RuntimeError(
                f"Random slice stream for Vendor {vendor} produced {len(slice_records)} "
                f"predictions for {len(dataset)} slices"
            )
        _write_jsonl(slice_records, result_root / f"vendor_{vendor}.jsonl")
        _write_jsonl(batch_records, result_root / f"vendor_{vendor}_batches.jsonl")
        summary = aggregate_slice_results(
            slice_records,
            bootstrap_resamples=int(cfg["evaluation"]["bootstrap_resamples"]),
            seed=int(cfg["evaluation"]["bootstrap_seed"]),
        )
        save_json(summary, result_root / f"vendor_{vendor}_summary.json")
        summaries[vendor] = summary

    manifest = {
        "method": method_name,
        "source_seed": source_seed,
        "initialization_profile": initialization_profile,
        "stream_mode": "slice_random",
        "vendors": vendors,
        "resolved_config": cfg,
        "resolved_method_config": method_cfg,
        "runtime": run_metadata(Path(__file__).resolve().parent),
        "source_checkpoint_sha256": checkpoint_hash,
        "fisher_sha256": fisher_hash,
        "protocol_sha256": protocol_hash,
        "target_stream_sha256": stream_hash,
        "target_order_policy": {**TARGET_SLICE_ORDER_POLICY, "vendor_order": vendors},
        "target_order_seed": source_seed,
        "target_orders": target_orders,
        "slice_filter": cfg["data"]["slice_filter"],
        "slice_metric_policy": dict(SLICE_METRIC_POLICY),
        "trainable_parameters": method.trainable_parameter_names(),
        "summaries": summaries,
    }
    save_json(manifest, result_root / "run_manifest.json")
    return manifest


def run_experiment(
    cfg: dict[str, Any],
    method_name: str,
    source_seed: int,
    vendors: list[str],
    device: torch.device,
) -> dict[str, Any]:
    stream_mode = str(cfg["tta"].get("stream_mode", "patient_volume"))
    if stream_mode == "slice_random":
        return run_slice_experiment(cfg, method_name, source_seed, vendors, device)
    if stream_mode != "patient_volume":
        raise ValueError(f"Unknown target stream mode: {stream_mode}")
    method, method_cfg, checkpoint, expected_initialization, fisher_hash = _prepare_experiment(
        cfg, method_name, source_seed, device
    )
    protocol_hash = file_sha256(cfg["data"]["protocol_file"])
    stream_hash = file_sha256(cfg["data"]["stream_file"])
    checkpoint_hash = file_sha256(checkpoint)
    initialization_profile = expected_initialization
    result_root = Path(cfg["tta"]["results_dir"]) / method_name / f"seed{source_seed}" / f"{cfg['tta']['timing']}_{cfg['tta']['reset']}"
    summaries: dict[str, Any] = {}
    target_orders: dict[str, Any] = {}
    for vendor_index, vendor in enumerate(vendors):
        if cfg["tta"]["reset"] == "vendor" or (vendor_index == 0 and cfg["tta"]["reset"] != "never"):
            method.reset()
        dataset = build_target_stream(vendor, cfg, order_seed=source_seed)
        if dataset.order_seed != source_seed:
            raise RuntimeError("Resolved target order seed differs from the source checkpoint seed")
        target_orders[vendor] = {
            "order_seed": dataset.order_seed,
            "patient_ids": dataset.patient_order,
            "target_order_sha256": dataset.target_order_sha256,
            "target_content_sha256": dataset.target_content_sha256,
            "n_slices": dataset.n_slices,
            "slice_filter": dataset.slice_filter,
        }
        records = []
        for volume_index in range(len(dataset)):
            if cfg["tta"]["reset"] == "patient":
                method.reset()
            volume = dataset[volume_index]
            prediction, adaptation_records = run_volume(
                method, volume["image"], int(cfg["tta"]["batch_size"]), device
            )
            target = dataset.load_mask(volume)
            label_probe = attach_method_label_probe(adaptation_records, target)
            classes = [int(value) for value in cfg["evaluation"]["classes"]]
            _validate_evaluation_target(
                target, classes, vendor, volume["patient_id"], volume["phase"]
            )
            class_names = {int(key): value for key, value in cfg["evaluation"]["class_names"].items()}
            scores = evaluate_volume(
                prediction.numpy(), target.numpy(),
                classes=classes,
                class_names=class_names,
            )
            record = {
                "method": method_name,
                "profile_verified": bool(method_cfg["profile_verified"]),
                "profile_kind": method_cfg["profile_kind"],
                "source_seed": source_seed,
                "target_order_seed": dataset.order_seed,
                "target_order_sha256": dataset.target_order_sha256,
                "target_content_sha256": dataset.target_content_sha256,
                "slice_filter": dataset.slice_filter,
                "initialization_profile": initialization_profile,
                "method_seed": int(method_cfg["method_seed"]),
                "vendor": vendor,
                "patient_id": volume["patient_id"],
                "phase": volume["phase"],
                "volume_id": volume["volume_id"],
                "n_slices": volume["n_slices"],
                "slice_ids": volume["slice_ids"],
                "z_indices": volume["z_indices"],
                "patient_arrival_index": volume["patient_arrival_index"],
                "volume_arrival_index": volume["volume_arrival_index"],
                "timing": cfg["tta"]["timing"],
                "reset": cfg["tta"]["reset"],
                "prediction_source": method.prediction_source,
                "metrics": scores,
                "adaptation": adaptation_records,
                "source_checkpoint_sha256": checkpoint_hash,
                "fisher_sha256": fisher_hash,
                "protocol_sha256": protocol_hash,
                "target_stream_sha256": stream_hash,
                "trainable_parameters": method.trainable_parameter_names(),
            }
            if label_probe is not None:
                probe_name, probe_value = label_probe
                record[probe_name] = probe_value
            records.append(record)
        _write_jsonl(records, result_root / f"vendor_{vendor}.jsonl")
        summary = aggregate_results(
            records,
            bootstrap_resamples=int(cfg["evaluation"]["bootstrap_resamples"]),
            seed=int(cfg["evaluation"]["bootstrap_seed"]),
        )
        save_json(summary, result_root / f"vendor_{vendor}_summary.json")
        summaries[vendor] = summary
    manifest = {
        "method": method_name,
        "source_seed": source_seed,
        "initialization_profile": initialization_profile,
        "stream_mode": "patient_volume",
        "vendors": vendors,
        "resolved_config": cfg,
        "resolved_method_config": method_cfg,
        "runtime": run_metadata(Path(__file__).resolve().parent),
        "source_checkpoint_sha256": checkpoint_hash,
        "fisher_sha256": fisher_hash,
        "protocol_sha256": protocol_hash,
        "target_stream_sha256": stream_hash,
        "target_order_policy": {**TARGET_ORDER_POLICY, "vendor_order": vendors},
        "target_order_seed": source_seed,
        "target_orders": target_orders,
        "slice_filter": cfg["data"]["slice_filter"],
        "trainable_parameters": method.trainable_parameter_names(),
        "summaries": summaries,
    }
    save_json(manifest, result_root / "run_manifest.json")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--source-seed", required=True, type=int)
    parser.add_argument("--vendors", nargs="+", choices=["B", "C", "D"])
    parser.add_argument(
        "--stream-mode",
        choices=["patient_volume", "slice_random"],
        help="Override the configured target arrival mode",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override the configured TTA arrival batch size",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.stream_mode is not None:
        cfg["tta"]["stream_mode"] = args.stream_mode
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("batch-size must be positive")
        cfg["tta"]["batch_size"] = args.batch_size
    vendors = args.vendors or list(cfg["experiment"]["target_vendors"])
    manifest = run_experiment(cfg, args.method, args.source_seed, vendors, get_device(args.device))
    print(json.dumps({"method": manifest["method"], "vendors": manifest["vendors"], "summaries": manifest["summaries"]}, indent=2))


if __name__ == "__main__":
    main()
