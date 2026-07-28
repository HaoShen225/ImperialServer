#!/usr/bin/env python3
"""Run InTEnt on the 15-backbone ACDC -> MMS experiment matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from helper.Evaluator import PatientStreamEvaluator
from helper.Intent import InTEnt
from helper.backbones.UNet import UNet
from helper.dataloader import TestLoader, _stack_images, _stack_masks


RESEARCH_ROOT = Path(__file__).resolve().parent
DEFAULT_BACKBONES = RESEARCH_ROOT / "backbone_params"
DEFAULT_DATASET = Path(
    "/rds/general/user/hs225/ephemeral/TTA_Project_data/dataset/Cardiac/MMS_normalized"
)
DEFAULT_OUTPUT = RESEARCH_ROOT / "intent_results"
DEFAULT_TEST_FRACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
PATIENT_SETTINGS = (1, 2, 3)
SEEDS = (0, 1, 2, 3, 4)
VENDORS = ("A", "B", "C", "D")
NUM_CLASSES = 4

SUMMARY_FIELDS = (
    "method",
    "task_id",
    "patient_setting",
    "seed",
    "vendor",
    "vendor_name",
    "domain",
    "checkpoint",
    "batch_size",
    "branch_count",
    "test_fractions",
    "paper_lambdas",
    "background_index",
    "entropy_mode",
    "eps",
    "episodic",
    "stream_order",
    "n_patients",
    "n_slices",
    "mean_entropy_span",
    "mean_max_weight",
    "dice_rv",
    "dice_myo",
    "dice_lv",
    "dice_mean",
    "hd95_rv",
    "hd95_myo",
    "hd95_lv",
    "hd95_mean",
    "output_dir",
)

TRACE_FIELDS = (
    "task_id",
    "patient_id",
    "phase",
    "z_index",
    "slice_id",
    "batch_index",
    "branch_index",
    "test_fraction",
    "paper_lambda",
    "balanced_entropy",
    "weight",
)

BRANCH_SUMMARY_FIELDS = (
    "task_id",
    "patient_setting",
    "seed",
    "vendor",
    "domain",
    "branch_index",
    "test_fraction",
    "paper_lambda",
    "n_slices",
    "mean_balanced_entropy",
    "mean_weight",
    "min_weight",
    "max_weight",
)


def task_coordinates(task_id: int) -> Tuple[int, int, str]:
    """Map one of 60 array tasks to shot, seed, and MMS vendor."""
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


def parse_test_fractions(value: str | Sequence[float]) -> Tuple[float, ...]:
    """Parse and validate source-to-test BN interpolation fractions."""
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",") if item.strip()]
        fractions = tuple(float(item) for item in parts)
    else:
        fractions = tuple(float(item) for item in value)
    if not fractions:
        raise ValueError("At least one test fraction is required.")
    if any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in fractions):
        raise ValueError("Test fractions must be finite values in [0, 1].")
    if len(set(fractions)) != len(fractions):
        raise ValueError("Test fractions must not contain duplicates.")
    return fractions


def validate_inputs(backbone_root: Path, dataset_root: Path) -> None:
    """Validate the fixed 15-backbone matrix and all four MMS domains."""
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
            "Expected exactly 15 backbones; "
            f"found={len(discovered)} missing={missing} unexpected={unexpected}"
        )
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"MMS root does not exist: {dataset_root}")
    for vendor in VENDORS:
        TestLoader(vendor=vendor, batch_size=1, dataset_root=dataset_root)


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
    rows: Sequence[Mapping[str, Any]],
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
        writer.writerows(dict(row) for row in rows)
    temporary.replace(path)


def finite_mean(values: Sequence[Any]) -> float:
    finite: List[float] = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            finite.append(number)
    return float(sum(finite) / len(finite)) if finite else float("nan")


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

    state = checkpoint.get(
        "teacher_state_dict",
        checkpoint.get("model_state_dict"),
    )
    if state is None:
        raise KeyError(f"No teacher/model state dict in {path}")
    model = UNet(
        n_channels=1,
        n_classes=NUM_CLASSES,
        only_feature=False,
        bilinear=False,
    )
    model.load_state_dict(state, strict=True)
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
            patient_records = group[:remaining]
            if patient_records:
                limited.append(patient_records)
                remaining -= len(patient_records)
        selected = limited
    return selected


def make_branch_summaries(
    trace_rows: Sequence[Mapping[str, Any]],
    *,
    task_id: int,
    patient_setting: int,
    seed: int,
    vendor: str,
    domain: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Mapping[str, Any]]] = {}
    for row in trace_rows:
        branch_index = int(row["branch_index"])
        grouped.setdefault(branch_index, []).append(row)

    summaries: List[Dict[str, Any]] = []
    for branch_index in sorted(grouped):
        rows = grouped[branch_index]
        weights = [float(row["weight"]) for row in rows]
        summaries.append(
            {
                "task_id": int(task_id),
                "patient_setting": int(patient_setting),
                "seed": int(seed),
                "vendor": str(vendor),
                "domain": str(domain),
                "branch_index": branch_index,
                "test_fraction": float(rows[0]["test_fraction"]),
                "paper_lambda": float(rows[0]["paper_lambda"]),
                "n_slices": len(rows),
                "mean_balanced_entropy": finite_mean(
                    [row["balanced_entropy"] for row in rows]
                ),
                "mean_weight": finite_mean(weights),
                "min_weight": min(weights),
                "max_weight": max(weights),
            }
        )
    return summaries


def save_incremental_outputs(
    task_dir: Path,
    trace_rows: Sequence[Mapping[str, Any]],
    evaluator: PatientStreamEvaluator,
    *,
    task_id: int,
    patient_setting: int,
    seed: int,
    vendor: str,
    domain: str,
) -> None:
    write_csv(
        task_dir / "integration_trace.csv",
        trace_rows,
        TRACE_FIELDS,
    )
    branch_summaries = make_branch_summaries(
        trace_rows,
        task_id=task_id,
        patient_setting=patient_setting,
        seed=seed,
        vendor=vendor,
        domain=domain,
    )
    write_csv(
        task_dir / "branch_summary.csv",
        branch_summaries,
        BRANCH_SUMMARY_FIELDS,
    )
    evaluator.save_csv(task_dir)


def run_task(args: argparse.Namespace) -> Dict[str, Any]:
    task_id = int(args.task_id)
    patient_setting, seed, vendor = task_coordinates(task_id)
    test_fractions = parse_test_fractions(args.test_fractions)
    device = resolve_device(args.device)
    set_seed(seed, cuda=device.type == "cuda")

    task_dir = Path(args.output_root) / "shards" / f"task_{task_id}"
    completion_path = task_dir / "completion.json"
    if completion_path.is_file() and args.resume and not args.overwrite:
        with completion_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("status") != "complete":
            raise ValueError(
                f"Invalid completion status in {completion_path}: "
                f"{payload.get('status')}"
            )
        print(f"[SKIP] complete task={task_id}", flush=True)
        return dict(payload["summary"])
    if task_dir.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"Task output exists; use --resume or --overwrite: {task_dir}"
        )

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
        raise RuntimeError("The selected test stream is empty.")

    model, metadata = load_backbone(
        backbone,
        patient_setting,
        seed,
        device,
    )
    intent = InTEnt(
        model,
        test_fractions=test_fractions,
        background_index=int(args.background_index),
        eps=float(args.eps),
    )
    backbone_id = (
        f"MeanTeacher_Patient{patient_setting}_Seed{seed}_teacher_InTEnt"
    )
    evaluator = PatientStreamEvaluator(
        domain=loader.domain,
        seed=seed,
        backbone_id=backbone_id,
    )
    trace_rows: List[Dict[str, Any]] = []
    entropy_spans: List[float] = []
    max_weights: List[float] = []
    batch_index = 0

    print(
        f"[RUN] task={task_id} Patient{patient_setting} Seed{seed} "
        f"{loader.domain} patients={len(patient_groups)} "
        f"slices={sum(len(group) for group in patient_groups)} "
        f"branches={len(test_fractions)} device={device}",
        flush=True,
    )
    for patient_step, records in enumerate(patient_groups, start=1):
        predictions: List[torch.Tensor] = []
        for record in records:
            batch_index += 1
            images = _stack_images([record]).to(
                device,
                non_blocking=device.type == "cuda",
            )
            result = intent.forward_with_details(images)
            probabilities = result.probabilities
            if not bool(torch.isfinite(probabilities).all()):
                raise RuntimeError(
                    f"Non-finite InTEnt probabilities for slice {record.slice_id}"
                )
            predictions.append(
                torch.argmax(probabilities, dim=1).detach().cpu()
            )

            entropies = result.entropies[:, 0].detach().cpu()
            weights = result.weights[:, 0].detach().cpu()
            if not torch.isclose(
                weights.sum(),
                torch.tensor(1.0, dtype=weights.dtype),
                rtol=1e-5,
                atol=1e-6,
            ):
                raise RuntimeError(
                    f"InTEnt weights do not sum to one for slice {record.slice_id}"
                )
            entropy_spans.append(
                float((entropies.max() - entropies.min()).item())
            )
            max_weights.append(float(weights.max().item()))
            for branch_index, fraction in enumerate(test_fractions):
                trace_rows.append(
                    {
                        "task_id": task_id,
                        "patient_id": record.patient_id,
                        "phase": record.phase,
                        "z_index": record.z_index,
                        "slice_id": record.slice_id,
                        "batch_index": batch_index,
                        "branch_index": branch_index,
                        "test_fraction": float(fraction),
                        "paper_lambda": float(1.0 - fraction),
                        "balanced_entropy": float(
                            entropies[branch_index].item()
                        ),
                        "weight": float(weights[branch_index].item()),
                    }
                )

        evaluator.update(
            torch.cat(predictions, dim=0),
            _stack_masks(records),
            [record.meta(include_mask_path=True) for record in records],
            step=patient_step,
            backbone_id=backbone_id,
        )
        save_incremental_outputs(
            task_dir,
            trace_rows,
            evaluator,
            task_id=task_id,
            patient_setting=patient_setting,
            seed=seed,
            vendor=loader.vendor,
            domain=loader.domain,
        )
        print(
            f"[PATIENT] {patient_step}/{len(patient_groups)} "
            f"id={records[0].patient_id}",
            flush=True,
        )

    metrics = evaluator.seed_summary()
    n_slices = batch_index
    fractions_text = "|".join(f"{value:g}" for value in test_fractions)
    lambdas_text = "|".join(
        f"{1.0 - value:g}"
        for value in test_fractions
    )
    summary: Dict[str, Any] = {
        "method": "InTEnt",
        "task_id": task_id,
        "patient_setting": patient_setting,
        "seed": seed,
        "vendor": loader.vendor,
        "vendor_name": loader.vendor_name,
        "domain": loader.domain,
        "checkpoint": str(backbone.resolve()),
        "batch_size": 1,
        "branch_count": len(test_fractions),
        "test_fractions": fractions_text,
        "paper_lambdas": lambdas_text,
        "background_index": int(args.background_index),
        "entropy_mode": "categorical_foreground_background_balanced",
        "eps": float(args.eps),
        "episodic": True,
        "stream_order": "patient,phase,z_index",
        "n_patients": int(metrics["n_patients"]),
        "n_slices": n_slices,
        "mean_entropy_span": finite_mean(entropy_spans),
        "mean_max_weight": finite_mean(max_weights),
        "dice_rv": metrics["dice_rv"],
        "dice_myo": metrics["dice_myo"],
        "dice_lv": metrics["dice_lv"],
        "dice_mean": metrics["dice_mean"],
        "hd95_rv": metrics["hd95_rv"],
        "hd95_myo": metrics["hd95_myo"],
        "hd95_lv": metrics["hd95_lv"],
        "hd95_mean": metrics["hd95_mean"],
        "output_dir": str(task_dir.resolve()),
    }
    config = dict(summary)
    config.update(
        {
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "backbone_root": str(Path(args.backbone_root).resolve()),
            "checkpoint_metadata": metadata,
            "test_fractions": list(test_fractions),
            "paper_lambdas": [1.0 - value for value in test_fractions],
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "max_patients": int(args.max_patients),
            "max_slices": int(args.max_slices),
        }
    )
    write_json(task_dir / "run_config.json", config)
    write_csv(
        task_dir / "run_summary.csv",
        [summary],
        SUMMARY_FIELDS,
    )
    write_json(
        completion_path,
        {"status": "complete", "summary": summary},
    )
    print(
        f"[COMPLETE] task={task_id} slices={n_slices} "
        f"dice={float(summary['dice_mean']):.6f} "
        f"hd95={float(summary['hd95_mean']):.6f}",
        flush=True,
    )
    return summary


def aggregate(output_root: Path) -> None:
    rows: List[Dict[str, Any]] = []
    missing: List[int] = []
    failures: Dict[str, str] = {}
    for task_id in range(60):
        completion = (
            output_root
            / "shards"
            / f"task_{task_id}"
            / "completion.json"
        )
        if not completion.is_file():
            missing.append(task_id)
            continue
        try:
            with completion.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("status") != "complete":
                raise ValueError(f"status={payload.get('status')}")
            summary = dict(payload["summary"])
            if int(summary["task_id"]) != task_id:
                raise ValueError(
                    f"task_id={summary.get('task_id')} expected={task_id}"
                )
            rows.append(summary)
        except Exception as error:
            failures[str(task_id)] = f"{type(error).__name__}: {error}"

    rows.sort(key=lambda row: int(row["task_id"]))
    write_csv(output_root / "run_summary.csv", rows, SUMMARY_FIELDS)
    write_json(
        output_root / "missing_tasks.json",
        {
            "expected": 60,
            "complete": len(rows),
            "missing": missing,
            "failures": failures,
        },
    )
    print(
        f"[AGGREGATE] complete={len(rows)}/60 "
        f"missing={missing} failures={failures}",
        flush=True,
    )
    if len(rows) != 60:
        raise RuntimeError(f"InTEnt matrix incomplete: {len(rows)}/60 tasks")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or aggregate ACDC -> MMS InTEnt tasks."
    )
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--backbone-root", default=str(DEFAULT_BACKBONES))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--test-fractions",
        default=",".join(str(value) for value in DEFAULT_TEST_FRACTIONS),
        help=(
            "Comma-separated weights of single-image test BN statistics; "
            "paper lambda equals 1-test_fraction."
        ),
    )
    parser.add_argument("--background-index", type=int, default=0)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-patients",
        type=int,
        default=0,
        help="0 evaluates every patient.",
    )
    parser.add_argument(
        "--max-slices",
        type=int,
        default=0,
        help="0 evaluates every selected slice; intended for smoke tests.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if args.task_id is None:
        raise ValueError("--task-id is required unless --aggregate-only is used.")
    task_coordinates(int(args.task_id))
    parse_test_fractions(args.test_fractions)
    if not 0 <= int(args.background_index) < NUM_CLASSES:
        raise ValueError(
            f"background-index must be in [0, {NUM_CLASSES - 1}]."
        )
    if not math.isfinite(float(args.eps)) or float(args.eps) <= 0:
        raise ValueError("eps must be a finite positive value.")
    if min(int(args.max_patients), int(args.max_slices)) < 0:
        raise ValueError("max-patients and max-slices cannot be negative.")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.aggregate_only:
        if args.resume or args.overwrite:
            raise ValueError(
                "--resume/--overwrite do not apply to --aggregate-only."
            )
        aggregate(Path(args.output_root))
        return

    validate_args(args)
    validate_inputs(
        Path(args.backbone_root),
        Path(args.dataset_root),
    )
    device = resolve_device(args.device)
    if device.type == "cuda":
        print(
            f"[CUDA] name={torch.cuda.get_device_name(device)} "
            f"torch={torch.__version__} cuda={torch.version.cuda} "
            f"cudnn={torch.backends.cudnn.version()}",
            flush=True,
        )
    else:
        print(f"[CPU] torch={torch.__version__}", flush=True)
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
