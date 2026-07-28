#!/usr/bin/env python3
"""Experiment 2: diagnose pixel-level complementarity of two tree filters.

For every ACDC-source/MMS-target task, the frozen source model produces one
probability map and its final decoder feature.  A shallow tree is guided by
the grayscale input image and a deep tree by the decoder feature.  The two
trees independently propagate the same source probabilities.  Target labels
are used only after propagation to measure where the trees agree, disagree,
and correct source-model errors.

The fixed scan contains four global spatial-temperature settings and three
window-only settings.  This script never adapts model parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from helper.EnergyLossTree import propagate_dual_tree_pseudo_labels
from helper.backbones.UNet import UNet
from helper.dataloader import TestLoader, _stack_images, _stack_masks


RESEARCH_ROOT = Path(__file__).resolve().parent
DEFAULT_BACKBONES = RESEARCH_ROOT / "backbone_params"
DEFAULT_DATASET = Path(
    "/rds/general/user/hs225/ephemeral/TTA_Project_data/dataset/Cardiac/MMS_normalized"
)
DEFAULT_OUTPUT = RESEARCH_ROOT / "myExperiment2_two_trees_complementaricity_results"

PATIENT_SETTINGS = (1, 2, 3)
SEEDS = (0, 1, 2, 3, 4)
VENDORS = ("A", "B", "C", "D")
NUM_CLASSES = 4
SHALLOW_SIGMA = 0.01
DEEP_SIGMA = 1.0

REGIONS = (
    "all",
    "boundary",
    "interior",
    "foreground_interior",
    "background_interior",
)
CLASS_SCOPES: Tuple[Tuple[str, int | None], ...] = (
    ("all", None),
    ("background", 0),
    ("rv", 1),
    ("myo", 2),
    ("lv", 3),
)

COUNT_FIELDS = (
    "total_pixels",
    "source_correct_pixels",
    "shallow_correct_pixels",
    "deep_correct_pixels",
    "disagreement_pixels",
    "shallow_correct_deep_wrong_pixels",
    "deep_correct_shallow_wrong_pixels",
    "source_error_pixels",
    "shallow_correction_pixels",
    "deep_correction_pixels",
    "union_correction_pixels",
    "intersection_correction_pixels",
    "shallow_exclusive_correction_pixels",
    "deep_exclusive_correction_pixels",
)
RATE_FIELDS = (
    "source_accuracy",
    "shallow_accuracy",
    "deep_accuracy",
    "disagreement_rate",
    "shallow_correct_deep_wrong_rate",
    "deep_correct_shallow_wrong_rate",
    "shallow_source_error_correction_rate",
    "deep_source_error_correction_rate",
    "union_source_error_correction_rate",
    "correction_jaccard",
    "shallow_exclusive_source_error_correction_rate",
    "deep_exclusive_source_error_correction_rate",
)

PATIENT_FIELDS = (
    "task_id",
    "patient_setting",
    "seed",
    "vendor",
    "vendor_name",
    "domain",
    "patient_id",
    "n_slices",
    "config_id",
    "mode",
    "spatial_temperature",
    "window_size",
    "stride",
    "overlap",
    "region",
    "class_scope",
) + COUNT_FIELDS + RATE_FIELDS

TASK_FIELDS = (
    "task_id",
    "patient_setting",
    "seed",
    "vendor",
    "vendor_name",
    "domain",
    "n_patients",
    "n_slices",
    "config_id",
    "mode",
    "spatial_temperature",
    "window_size",
    "stride",
    "overlap",
    "region",
    "class_scope",
) + COUNT_FIELDS + RATE_FIELDS


@dataclass(frozen=True)
class PropagationConfig:
    config_id: str
    mode: str
    spatial_temperature: float | None
    window_size: int | None
    stride: int | None
    overlap: int | None

    def propagation_kwargs(self) -> Dict[str, Any]:
        return {
            "window_size": self.window_size,
            "stride": self.stride,
            "spatial_temperature": self.spatial_temperature,
        }


PROPAGATION_CONFIGS: Tuple[PropagationConfig, ...] = (
    PropagationConfig("distance_t16", "distance", 16.0, None, None, None),
    PropagationConfig("distance_t64", "distance", 64.0, None, None, None),
    PropagationConfig("distance_t128", "distance", 128.0, None, None, None),
    PropagationConfig("distance_t256", "distance", 256.0, None, None, None),
    PropagationConfig("window_w64_o32", "window", None, 64, 32, 32),
    PropagationConfig("window_w128_o64", "window", None, 128, 64, 64),
    PropagationConfig("window_w256_o128", "window", None, 256, 128, 128),
)
CONFIG_BY_ID = {config.config_id: config for config in PROPAGATION_CONFIGS}
SCOPE_ENTRIES = tuple(
    (region, class_name)
    for region in REGIONS
    for class_name, _ in CLASS_SCOPES
)

Accumulator = MutableMapping[Tuple[str, str, str], Dict[str, int]]


def task_coordinates(task_id: int) -> Tuple[int, int, str]:
    """Map one of 60 tasks to shot, source seed, and MMS vendor."""
    if not 0 <= int(task_id) < 60:
        raise ValueError("task-id must be in [0, 59].")
    patient_setting = int(task_id) // 20 + 1
    remainder = int(task_id) % 20
    seed = remainder // 4
    vendor = VENDORS[remainder % 4]
    return patient_setting, seed, vendor


def checkpoint_path(root: Path, patient_setting: int, seed: int) -> Path:
    return (
        root
        / f"Patient{int(patient_setting)}"
        / f"Seed{int(seed)}"
        / "baseline_model_with_metadata.pt"
    )


def resolve_device(name: str) -> torch.device:
    text = str(name).strip().lower()
    if text == "auto":
        text = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return device


def set_seed(seed: int, cuda: bool) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if cuda:
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    temporary.replace(path)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if float(denominator) <= 0.0:
        return float("nan")
    return float(numerator) / float(denominator)


def finite_mean(values: Iterable[Any]) -> float:
    finite = [
        float(value)
        for value in values
        if value not in (None, "") and math.isfinite(float(value))
    ]
    return float(np.mean(np.asarray(finite, dtype=np.float64))) if finite else float("nan")


def finite_std(values: Iterable[Any]) -> float:
    finite = [
        float(value)
        for value in values
        if value not in (None, "") and math.isfinite(float(value))
    ]
    if not finite:
        return float("nan")
    if len(finite) == 1:
        return 0.0
    return float(np.std(np.asarray(finite, dtype=np.float64), ddof=1))


def validate_configurations() -> None:
    if len(PROPAGATION_CONFIGS) != 7:
        raise AssertionError("The experiment must contain exactly seven propagation configs.")
    if len(CONFIG_BY_ID) != len(PROPAGATION_CONFIGS):
        raise AssertionError("Propagation config ids must be unique.")
    for config in PROPAGATION_CONFIGS:
        if config.mode == "distance":
            if config.spatial_temperature is None or config.spatial_temperature <= 0:
                raise ValueError(f"Invalid distance config: {config}")
            if config.window_size is not None or config.stride is not None:
                raise ValueError(f"Distance config cannot enable windows: {config}")
        elif config.mode == "window":
            if config.spatial_temperature is not None:
                raise ValueError(f"Window config must disable distance decay: {config}")
            if config.window_size is None or config.stride is None or config.overlap is None:
                raise ValueError(f"Incomplete window config: {config}")
            if config.window_size - config.stride != config.overlap:
                raise ValueError(f"Window overlap/stride mismatch: {config}")
        else:
            raise ValueError(f"Unknown propagation mode: {config.mode}")


def validate_inputs(backbone_root: Path, dataset_root: Path) -> None:
    expected = {
        checkpoint_path(backbone_root, patient, seed).resolve()
        for patient in PATIENT_SETTINGS
        for seed in SEEDS
    }
    discovered = {
        path.resolve()
        for path in backbone_root.glob(
            "Patient*/Seed*/baseline_model_with_metadata.pt"
        )
    }
    missing = sorted(str(path) for path in expected - discovered)
    unexpected = sorted(str(path) for path in discovered - expected)
    if len(discovered) != 15 or missing or unexpected:
        raise RuntimeError(
            "Expected exactly 15 source backbones; "
            f"found={len(discovered)} missing={missing} unexpected={unexpected}"
        )
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"MMS root does not exist: {dataset_root}")
    for vendor in VENDORS:
        TestLoader(vendor=vendor, batch_size=1, dataset_root=dataset_root)


def load_backbone(
    path: Path,
    patient_setting: int,
    seed: int,
    device: torch.device,
) -> Tuple[UNet, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint is not a mapping: {path}")
    metadata = dict(checkpoint.get("metadata", {}))
    if int(metadata.get("labeled_cases_per_class", patient_setting)) != int(
        patient_setting
    ):
        raise ValueError(f"Patient setting mismatch in {path}")
    if int(metadata.get("seed", seed)) != int(seed):
        raise ValueError(f"Seed mismatch in {path}")
    weights = checkpoint.get(
        "teacher_state_dict",
        checkpoint.get("model_state_dict"),
    )
    if weights is None:
        raise KeyError(f"No teacher/model state dict in {path}")
    model = UNet(
        n_channels=1,
        n_classes=NUM_CLASSES,
        only_feature=False,
        bilinear=False,
    )
    model.load_state_dict(weights, strict=True)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, metadata


def group_by_patient(
    records: Sequence[Any],
    max_patients: int,
    max_slices: int,
) -> List[List[Any]]:
    groups: Dict[str, List[Any]] = {}
    order: List[str] = []
    for record in records:
        patient_id = str(record.patient_id)
        if patient_id not in groups:
            groups[patient_id] = []
            order.append(patient_id)
        groups[patient_id].append(record)
    selected = [groups[patient_id] for patient_id in order]
    if int(max_patients) > 0:
        selected = selected[: int(max_patients)]
    if int(max_slices) > 0:
        remaining = int(max_slices)
        limited: List[List[Any]] = []
        for group in selected:
            if remaining <= 0:
                break
            current = group[:remaining]
            if current:
                limited.append(current)
                remaining -= len(current)
        selected = limited
    return selected


def source_forward(
    model: UNet,
    images: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    output = model(images)
    if not isinstance(output, (tuple, list)) or len(output) != 2:
        raise TypeError("UNet must return (decoder_feature, logits).")
    feature, logits = output
    if not isinstance(feature, torch.Tensor) or not isinstance(logits, torch.Tensor):
        raise TypeError("UNet feature and logits must both be tensors.")
    if feature.ndim != 4 or logits.ndim != 4:
        raise ValueError("UNet feature/logits must have shape [B,C,H,W].")
    if logits.shape[1] != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} logits, got {logits.shape[1]}.")
    if feature.shape[0] != logits.shape[0] or feature.shape[-2:] != logits.shape[-2:]:
        raise ValueError("Decoder feature and logits have incompatible shapes.")
    probabilities = logits.softmax(dim=1)
    return feature.detach(), probabilities.detach()


def multiclass_boundary_band(
    labels: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """Return a dilated band around all four-neighbour semantic boundaries."""
    if labels.ndim != 3:
        raise ValueError(f"labels must have shape [B,H,W], got {tuple(labels.shape)}")
    if int(radius) < 0:
        raise ValueError("Boundary radius must be non-negative.")
    boundary = torch.zeros_like(labels, dtype=torch.bool)
    horizontal = labels[:, :, :-1] != labels[:, :, 1:]
    vertical = labels[:, :-1, :] != labels[:, 1:, :]
    boundary[:, :, :-1] |= horizontal
    boundary[:, :, 1:] |= horizontal
    boundary[:, :-1, :] |= vertical
    boundary[:, 1:, :] |= vertical
    if int(radius) > 0:
        kernel_size = 2 * int(radius) + 1
        boundary = (
            F.max_pool2d(
                boundary.unsqueeze(1).to(torch.float32),
                kernel_size=kernel_size,
                stride=1,
                padding=int(radius),
            )
            > 0
        ).squeeze(1)
    return boundary


def scope_masks(
    ground_truth: torch.Tensor,
    boundary_radius: int,
) -> Tuple[torch.Tensor, Tuple[Tuple[str, str], ...]]:
    boundary = multiclass_boundary_band(ground_truth, boundary_radius)
    interior = ~boundary
    foreground = ground_truth > 0
    region_masks = {
        "all": torch.ones_like(ground_truth, dtype=torch.bool),
        "boundary": boundary,
        "interior": interior,
        "foreground_interior": foreground & interior,
        "background_interior": (~foreground) & interior,
    }
    class_masks = {
        class_name: (
            torch.ones_like(ground_truth, dtype=torch.bool)
            if class_index is None
            else ground_truth == int(class_index)
        )
        for class_name, class_index in CLASS_SCOPES
    }
    masks = torch.stack(
        [
            region_masks[region] & class_masks[class_name]
            for region, class_name in SCOPE_ENTRIES
        ],
        dim=0,
    )
    return masks, SCOPE_ENTRIES


def pixel_count_matrix(
    source_labels: torch.Tensor,
    shallow_labels: torch.Tensor,
    deep_labels: torch.Tensor,
    ground_truth: torch.Tensor,
    scopes: torch.Tensor,
) -> torch.Tensor:
    """Calculate all requested raw counts in one GPU-to-CPU synchronization."""
    tensors = (source_labels, shallow_labels, deep_labels, ground_truth)
    if any(tensor.shape != ground_truth.shape for tensor in tensors):
        raise ValueError("All label tensors must share shape [B,H,W].")
    if scopes.ndim != 4 or scopes.shape[1:] != ground_truth.shape:
        raise ValueError("Scope masks must have shape [S,B,H,W].")

    source_correct = source_labels == ground_truth
    shallow_correct = shallow_labels == ground_truth
    deep_correct = deep_labels == ground_truth
    disagreement = shallow_labels != deep_labels
    source_error = ~source_correct
    shallow_correction = source_error & shallow_correct
    deep_correction = source_error & deep_correct
    union_correction = shallow_correction | deep_correction
    intersection_correction = shallow_correction & deep_correction

    indicators = torch.stack(
        (
            torch.ones_like(ground_truth, dtype=torch.bool),
            source_correct,
            shallow_correct,
            deep_correct,
            disagreement,
            shallow_correct & (~deep_correct),
            deep_correct & (~shallow_correct),
            source_error,
            shallow_correction,
            deep_correction,
            union_correction,
            intersection_correction,
            shallow_correction & (~deep_correction),
            deep_correction & (~shallow_correction),
        ),
        dim=0,
    )
    if indicators.shape[0] != len(COUNT_FIELDS):
        raise AssertionError("Indicator/count field mismatch.")
    counts = (
        scopes.unsqueeze(1) & indicators.unsqueeze(0)
    ).sum(dim=(2, 3, 4), dtype=torch.int64)
    return counts.detach().cpu()


def empty_accumulator() -> Accumulator:
    return defaultdict(lambda: {field: 0 for field in COUNT_FIELDS})


def update_accumulator(
    accumulator: Accumulator,
    config_id: str,
    counts: torch.Tensor,
) -> None:
    if counts.shape != (len(SCOPE_ENTRIES), len(COUNT_FIELDS)):
        raise ValueError(
            "Unexpected count matrix shape: "
            f"{tuple(counts.shape)} != {(len(SCOPE_ENTRIES), len(COUNT_FIELDS))}"
        )
    for scope_index, (region, class_scope) in enumerate(SCOPE_ENTRIES):
        target = accumulator[(config_id, region, class_scope)]
        for field_index, field in enumerate(COUNT_FIELDS):
            target[field] += int(counts[scope_index, field_index].item())


def derived_rates(counts: Mapping[str, int]) -> Dict[str, float]:
    return {
        "source_accuracy": safe_ratio(
            counts["source_correct_pixels"], counts["total_pixels"]
        ),
        "shallow_accuracy": safe_ratio(
            counts["shallow_correct_pixels"], counts["total_pixels"]
        ),
        "deep_accuracy": safe_ratio(
            counts["deep_correct_pixels"], counts["total_pixels"]
        ),
        "disagreement_rate": safe_ratio(
            counts["disagreement_pixels"], counts["total_pixels"]
        ),
        "shallow_correct_deep_wrong_rate": safe_ratio(
            counts["shallow_correct_deep_wrong_pixels"],
            counts["disagreement_pixels"],
        ),
        "deep_correct_shallow_wrong_rate": safe_ratio(
            counts["deep_correct_shallow_wrong_pixels"],
            counts["disagreement_pixels"],
        ),
        "shallow_source_error_correction_rate": safe_ratio(
            counts["shallow_correction_pixels"], counts["source_error_pixels"]
        ),
        "deep_source_error_correction_rate": safe_ratio(
            counts["deep_correction_pixels"], counts["source_error_pixels"]
        ),
        "union_source_error_correction_rate": safe_ratio(
            counts["union_correction_pixels"], counts["source_error_pixels"]
        ),
        "correction_jaccard": safe_ratio(
            counts["intersection_correction_pixels"],
            counts["union_correction_pixels"],
        ),
        "shallow_exclusive_source_error_correction_rate": safe_ratio(
            counts["shallow_exclusive_correction_pixels"],
            counts["source_error_pixels"],
        ),
        "deep_exclusive_source_error_correction_rate": safe_ratio(
            counts["deep_exclusive_correction_pixels"],
            counts["source_error_pixels"],
        ),
    }


def diagnostic_rows(
    accumulator: Mapping[Tuple[str, str, str], Mapping[str, int]],
    metadata: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for config in PROPAGATION_CONFIGS:
        for region, class_scope in SCOPE_ENTRIES:
            counts = dict(accumulator[(config.config_id, region, class_scope)])
            row = dict(metadata)
            row.update(asdict(config))
            row.update(region=region, class_scope=class_scope)
            row.update(counts)
            row.update(derived_rates(counts))
            rows.append(row)
    return rows


@torch.no_grad()
def run_task(args: argparse.Namespace) -> Dict[str, Any]:
    task_id = int(args.task_id)
    patient_setting, seed, vendor = task_coordinates(task_id)
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Tree complementarity tasks require a CUDA device.")
    set_seed(seed, cuda=True)

    output_root = Path(args.output_root)
    task_dir = output_root / "shards" / f"task_{task_id}"
    completion_path = task_dir / "completion.json"
    if completion_path.is_file() and args.resume and not args.overwrite:
        with completion_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        completed_ids = tuple(payload.get("config_ids", ()))
        expected_ids = tuple(config.config_id for config in PROPAGATION_CONFIGS)
        if payload.get("status") != "complete" or completed_ids != expected_ids:
            raise ValueError(f"Invalid completion marker: {completion_path}")
        print(f"[SKIP] complete task={task_id}", flush=True)
        return dict(payload)
    if task_dir.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"Task output exists; use --resume or --overwrite: {task_dir}"
        )
    if task_dir.exists() and args.overwrite:
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)

    backbone = checkpoint_path(
        Path(args.backbone_root),
        patient_setting,
        seed,
    )
    loader = TestLoader(
        vendor=vendor,
        batch_size=1,
        shuffle_all_slices=False,
        seed=seed,
        dataset_root=Path(args.dataset_root),
    )
    patient_groups = group_by_patient(
        loader.records,
        max_patients=int(args.max_patients),
        max_slices=int(args.max_slices),
    )
    if not patient_groups:
        raise RuntimeError("The selected target stream is empty.")

    model, checkpoint_metadata = load_backbone(
        backbone,
        patient_setting,
        seed,
        device,
    )
    patient_rows: List[Dict[str, Any]] = []
    task_accumulator = empty_accumulator()
    n_slices = 0

    print(
        f"[RUN] task={task_id} shot={patient_setting} seed={seed} "
        f"domain={loader.domain} patients={len(patient_groups)} "
        f"configs={len(PROPAGATION_CONFIGS)} boundary_radius={args.boundary_radius} "
        f"device={device}",
        flush=True,
    )

    for patient_step, records in enumerate(patient_groups, start=1):
        patient_accumulator = empty_accumulator()
        for record in records:
            n_slices += 1
            images = _stack_images([record]).to(device, non_blocking=True)
            ground_truth = _stack_masks([record]).to(
                device,
                non_blocking=True,
            ).long()
            feature, probabilities = source_forward(model, images)
            source_labels = probabilities.argmax(dim=1)
            scopes, _ = scope_masks(
                ground_truth,
                boundary_radius=int(args.boundary_radius),
            )

            for config in PROPAGATION_CONFIGS:
                targets = propagate_dual_tree_pseudo_labels(
                    probabilities,
                    images,
                    feature,
                    pseudo_label_weights=None,
                    shallow_sigma=SHALLOW_SIGMA,
                    deep_sigma=DEEP_SIGMA,
                    window_batch_size=int(args.window_batch_size),
                    **config.propagation_kwargs(),
                )
                shallow_labels = targets.shallow.argmax(dim=1)
                deep_labels = targets.deep.argmax(dim=1)
                counts = pixel_count_matrix(
                    source_labels,
                    shallow_labels,
                    deep_labels,
                    ground_truth,
                    scopes,
                )
                update_accumulator(
                    patient_accumulator,
                    config.config_id,
                    counts,
                )
                update_accumulator(
                    task_accumulator,
                    config.config_id,
                    counts,
                )

        patient_rows.extend(
            diagnostic_rows(
                patient_accumulator,
                {
                    "task_id": task_id,
                    "patient_setting": patient_setting,
                    "seed": seed,
                    "vendor": loader.vendor,
                    "vendor_name": loader.vendor_name,
                    "domain": loader.domain,
                    "patient_id": records[0].patient_id,
                    "n_slices": len(records),
                },
            )
        )
        print(
            f"[PATIENT] {patient_step}/{len(patient_groups)} "
            f"id={records[0].patient_id} slices={len(records)}",
            flush=True,
        )

    task_rows = diagnostic_rows(
        task_accumulator,
        {
            "task_id": task_id,
            "patient_setting": patient_setting,
            "seed": seed,
            "vendor": loader.vendor,
            "vendor_name": loader.vendor_name,
            "domain": loader.domain,
            "n_patients": len(patient_groups),
            "n_slices": n_slices,
        },
    )
    write_csv(
        task_dir / "patient_diagnostics.csv",
        patient_rows,
        PATIENT_FIELDS,
    )
    write_csv(
        task_dir / "config_summary.csv",
        task_rows,
        TASK_FIELDS,
    )
    write_json(
        task_dir / "run_config.json",
        {
            "task_id": task_id,
            "patient_setting": patient_setting,
            "seed": seed,
            "vendor": loader.vendor,
            "vendor_name": loader.vendor_name,
            "domain": loader.domain,
            "checkpoint": str(backbone.resolve()),
            "checkpoint_metadata": checkpoint_metadata,
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "output_root": str(output_root.resolve()),
            "n_patients": len(patient_groups),
            "n_slices": n_slices,
            "boundary_radius": int(args.boundary_radius),
            "shallow_guidance": "normalized_grayscale_input",
            "deep_guidance": "final_decoder_feature_before_classifier",
            "shallow_sigma": SHALLOW_SIGMA,
            "deep_sigma": DEEP_SIGMA,
            "pseudo_label_weights": "uniform_ones",
            "propagation_configs": [asdict(config) for config in PROPAGATION_CONFIGS],
            "window_batch_size": int(args.window_batch_size),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
        },
    )
    completion = {
        "status": "complete",
        "task_id": task_id,
        "config_ids": [config.config_id for config in PROPAGATION_CONFIGS],
        "n_patients": len(patient_groups),
        "n_slices": n_slices,
        "patient_rows": len(patient_rows),
        "summary_rows": len(task_rows),
    }
    write_json(completion_path, completion)
    print(
        f"[COMPLETE] task={task_id} patients={len(patient_groups)} "
        f"slices={n_slices} rows={len(patient_rows)}",
        flush=True,
    )
    return completion


def concatenate_patient_rows(
    paths: Sequence[Path],
    output_path: Path,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    row_count = 0
    with temporary.open("w", newline="", encoding="utf-8") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=list(PATIENT_FIELDS))
        writer.writeheader()
        for path in paths:
            with path.open(newline="", encoding="utf-8") as input_handle:
                reader = csv.DictReader(input_handle)
                if tuple(reader.fieldnames or ()) != PATIENT_FIELDS:
                    raise ValueError(f"Unexpected patient CSV schema: {path}")
                for row in reader:
                    writer.writerow(row)
                    row_count += 1
    temporary.replace(output_path)
    return row_count


def make_group_summary(task_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in task_rows:
        key = (
            str(row["config_id"]),
            str(row["patient_setting"]),
            str(row["domain"]),
            str(row["region"]),
            str(row["class_scope"]),
        )
        grouped[key].append(row)

    output: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        config_id, patient_setting, domain, region, class_scope = key
        rows = grouped[key]
        config = CONFIG_BY_ID[config_id]
        item: Dict[str, Any] = {
            "config_id": config_id,
            "mode": config.mode,
            "spatial_temperature": config.spatial_temperature,
            "window_size": config.window_size,
            "stride": config.stride,
            "overlap": config.overlap,
            "patient_setting": int(patient_setting),
            "domain": domain,
            "region": region,
            "class_scope": class_scope,
            "n_tasks": len(rows),
        }
        for field in COUNT_FIELDS:
            item[f"{field}_sum"] = sum(int(float(row[field])) for row in rows)
        for field in RATE_FIELDS:
            item[f"{field}_mean"] = finite_mean(row[field] for row in rows)
            item[f"{field}_std"] = finite_std(row[field] for row in rows)
        output.append(item)
    return output


def aggregate(output_root: Path) -> None:
    task_rows: List[Dict[str, Any]] = []
    patient_paths: List[Path] = []
    missing: List[int] = []
    failures: Dict[str, str] = {}
    expected_ids = tuple(config.config_id for config in PROPAGATION_CONFIGS)

    for task_id in range(60):
        task_dir = output_root / "shards" / f"task_{task_id}"
        completion_path = task_dir / "completion.json"
        if not completion_path.is_file():
            missing.append(task_id)
            continue
        try:
            with completion_path.open(encoding="utf-8") as handle:
                completion = json.load(handle)
            if completion.get("status") != "complete":
                raise ValueError(f"status={completion.get('status')}")
            if tuple(completion.get("config_ids", ())) != expected_ids:
                raise ValueError("config id mismatch")
            current_rows = read_csv(task_dir / "config_summary.csv")
            if len(current_rows) != len(PROPAGATION_CONFIGS) * len(SCOPE_ENTRIES):
                raise ValueError(f"config_summary rows={len(current_rows)}")
            task_rows.extend(current_rows)
            patient_path = task_dir / "patient_diagnostics.csv"
            if not patient_path.is_file():
                raise FileNotFoundError(patient_path)
            patient_paths.append(patient_path)
        except Exception as error:
            failures[str(task_id)] = f"{type(error).__name__}: {error}"

    task_rows.sort(
        key=lambda row: (
            int(row["task_id"]),
            str(row["config_id"]),
            str(row["region"]),
            str(row["class_scope"]),
        )
    )
    write_csv(output_root / "task_summary.csv", task_rows, TASK_FIELDS)
    patient_row_count = concatenate_patient_rows(
        patient_paths,
        output_root / "patient_diagnostics.csv",
    )
    group_rows = make_group_summary(task_rows)
    group_fields = (
        "config_id",
        "mode",
        "spatial_temperature",
        "window_size",
        "stride",
        "overlap",
        "patient_setting",
        "domain",
        "region",
        "class_scope",
        "n_tasks",
    ) + tuple(f"{field}_sum" for field in COUNT_FIELDS) + tuple(
        name
        for field in RATE_FIELDS
        for name in (f"{field}_mean", f"{field}_std")
    )
    write_csv(output_root / "group_summary.csv", group_rows, group_fields)
    write_json(
        output_root / "missing_tasks.json",
        {
            "expected": 60,
            "complete": 60 - len(missing) - len(failures),
            "missing": missing,
            "failures": failures,
            "patient_rows": patient_row_count,
            "task_rows": len(task_rows),
            "group_rows": len(group_rows),
        },
    )
    print(
        f"[AGGREGATE] complete={60-len(missing)-len(failures)}/60 "
        f"patient_rows={patient_row_count} missing={missing} failures={failures}",
        flush=True,
    )
    if missing or failures:
        raise RuntimeError("Two-tree complementarity matrix is incomplete.")


def cpu_smoke_test(args: argparse.Namespace) -> None:
    validate_configurations()
    expected_ids = (
        "distance_t16",
        "distance_t64",
        "distance_t128",
        "distance_t256",
        "window_w64_o32",
        "window_w128_o64",
        "window_w256_o128",
    )
    if tuple(config.config_id for config in PROPAGATION_CONFIGS) != expected_ids:
        raise AssertionError("Unexpected propagation scan ordering.")

    ground_truth = torch.tensor([[[0, 1, 1, 2]]], dtype=torch.long)
    source = torch.tensor([[[0, 0, 1, 0]]], dtype=torch.long)
    shallow = torch.tensor([[[0, 1, 1, 0]]], dtype=torch.long)
    deep = torch.tensor([[[0, 0, 1, 2]]], dtype=torch.long)
    scopes, entries = scope_masks(ground_truth, boundary_radius=0)
    counts = pixel_count_matrix(source, shallow, deep, ground_truth, scopes)
    all_index = entries.index(("all", "all"))
    all_counts = {
        field: int(counts[all_index, index].item())
        for index, field in enumerate(COUNT_FIELDS)
    }
    expected = {
        "disagreement_pixels": 2,
        "shallow_correct_deep_wrong_pixels": 1,
        "deep_correct_shallow_wrong_pixels": 1,
        "source_error_pixels": 2,
        "shallow_correction_pixels": 1,
        "deep_correction_pixels": 1,
        "union_correction_pixels": 2,
        "intersection_correction_pixels": 0,
    }
    for field, value in expected.items():
        if all_counts[field] != value:
            raise AssertionError(f"{field}: {all_counts[field]} != {value}")
    rates = derived_rates(all_counts)
    if rates["union_source_error_correction_rate"] != 1.0:
        raise AssertionError("Union correction rate smoke check failed.")
    if rates["correction_jaccard"] != 0.0:
        raise AssertionError("Correction Jaccard smoke check failed.")
    if not math.isnan(safe_ratio(0, 0)):
        raise AssertionError("Zero denominators must produce NaN.")

    overlap_gt = torch.tensor([[[0, 1]]], dtype=torch.long)
    overlap_source = torch.tensor([[[0, 0]]], dtype=torch.long)
    overlap_target = torch.tensor([[[0, 1]]], dtype=torch.long)
    overlap_scopes, overlap_entries = scope_masks(overlap_gt, boundary_radius=0)
    overlap_counts_tensor = pixel_count_matrix(
        overlap_source,
        overlap_target,
        overlap_target,
        overlap_gt,
        overlap_scopes,
    )
    overlap_all_index = overlap_entries.index(("all", "all"))
    overlap_counts = {
        field: int(overlap_counts_tensor[overlap_all_index, index].item())
        for index, field in enumerate(COUNT_FIELDS)
    }
    if derived_rates(overlap_counts)["correction_jaccard"] != 1.0:
        raise AssertionError("Overlapping correction Jaccard smoke check failed.")

    boundary_gt = torch.zeros((1, 11, 11), dtype=torch.long)
    boundary_gt[:, :, 5:] = 1
    band = multiclass_boundary_band(boundary_gt, radius=3)
    interior = ~band
    if bool((band & interior).any()) or not bool((band | interior).all()):
        raise AssertionError("Boundary/interior masks must partition the image.")
    if not bool(((boundary_gt > 0) & interior).any()):
        raise AssertionError("Foreground interior unexpectedly empty.")
    if not bool(((boundary_gt == 0) & interior).any()):
        raise AssertionError("Background interior unexpectedly empty.")

    backbone = checkpoint_path(Path(args.backbone_root), 1, 0)
    loader = TestLoader(
        vendor="A",
        batch_size=1,
        dataset_root=Path(args.dataset_root),
    )
    model, checkpoint_metadata = load_backbone(
        backbone,
        1,
        0,
        torch.device("cpu"),
    )
    json.dumps(checkpoint_metadata)
    image = _stack_images([loader.records[0]])
    mask = _stack_masks([loader.records[0]])
    with torch.no_grad():
        feature, probabilities = source_forward(model, image)
    if tuple(probabilities.shape) != (1, NUM_CLASSES, 256, 256):
        raise AssertionError(f"Unexpected probability shape: {probabilities.shape}")
    if feature.shape[0] != 1 or feature.shape[-2:] != (256, 256):
        raise AssertionError(f"Unexpected feature shape: {feature.shape}")
    if tuple(mask.shape) != (1, 256, 256):
        raise AssertionError(f"Unexpected mask shape: {mask.shape}")
    torch.testing.assert_close(
        probabilities.sum(dim=1),
        torch.ones_like(probabilities[:, 0]),
        rtol=1e-5,
        atol=1e-5,
    )

    with tempfile.TemporaryDirectory(prefix="two_trees_cpu_smoke_") as directory:
        root = Path(directory)
        accumulator = empty_accumulator()
        update_accumulator(accumulator, PROPAGATION_CONFIGS[0].config_id, counts)
        rows = diagnostic_rows(
            accumulator,
            {
                "task_id": 0,
                "patient_setting": 1,
                "seed": 0,
                "vendor": "A",
                "vendor_name": "Siemens",
                "domain": "vendor_A_Siemens",
                "patient_id": "smoke",
                "n_slices": 1,
            },
        )
        write_csv(root / "patient.csv", rows, PATIENT_FIELDS)
        if len(read_csv(root / "patient.csv")) != len(PROPAGATION_CONFIGS) * len(
            SCOPE_ENTRIES
        ):
            raise AssertionError("CSV smoke row count mismatch.")

    print(
        "[CPU SMOKE] PASS "
        f"configs={len(PROPAGATION_CONFIGS)} feature={tuple(feature.shape)} "
        f"probabilities={tuple(probabilities.shape)}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure pixel-level complementarity between grayscale-guided "
            "and decoder-feature-guided tree propagation."
        )
    )
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--cpu-smoke-test", action="store_true")
    parser.add_argument("--backbone-root", default=str(DEFAULT_BACKBONES))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--boundary-radius", type=int, default=3)
    parser.add_argument("--window-batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--max-slices", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    modes = sum(bool(value) for value in (args.aggregate_only, args.cpu_smoke_test))
    if modes > 1:
        raise ValueError("--aggregate-only and --cpu-smoke-test are mutually exclusive.")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if args.aggregate_only:
        if args.resume or args.overwrite:
            raise ValueError("--resume/--overwrite do not apply to --aggregate-only.")
        return
    if args.cpu_smoke_test:
        if args.resume or args.overwrite:
            raise ValueError("--resume/--overwrite do not apply to --cpu-smoke-test.")
        return
    if args.task_id is None:
        raise ValueError(
            "--task-id is required unless --aggregate-only or --cpu-smoke-test is used."
        )
    task_coordinates(args.task_id)
    if int(args.boundary_radius) < 0:
        raise ValueError("boundary-radius must be non-negative.")
    if int(args.window_batch_size) <= 0:
        raise ValueError("window-batch-size must be positive.")
    if min(int(args.max_patients), int(args.max_slices)) < 0:
        raise ValueError("max-patients/max-slices cannot be negative.")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    validate_args(args)
    validate_configurations()
    if args.cpu_smoke_test:
        cpu_smoke_test(args)
        return
    if args.aggregate_only:
        aggregate(Path(args.output_root))
        return
    validate_inputs(Path(args.backbone_root), Path(args.dataset_root))
    device = resolve_device(args.device)
    if device.type == "cuda":
        print(
            f"[CUDA] name={torch.cuda.get_device_name(device)} "
            f"torch={torch.__version__} cuda={torch.version.cuda} "
            f"cudnn={torch.backends.cudnn.version()}",
            flush=True,
        )
    run_task(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"[ERROR] {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        raise
