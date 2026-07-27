#!/usr/bin/env python3
"""Run SAR with the 15 fixed BN-UNet backbones on ACDC -> MMS."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch

from helper.Evaluator import PatientStreamEvaluator
from helper.Sar import SAR, SAM, collect_params, configure_model
from helper.backbones.UNet import UNet
from helper.dataloader import TestLoader, _stack_images, _stack_masks


RESEARCH_ROOT = Path(__file__).resolve().parent
DEFAULT_BACKBONES = RESEARCH_ROOT / "backbone_params"
DEFAULT_DATASET = Path(
    "/rds/general/user/hs225/ephemeral/TTA_Project_data/dataset/Cardiac/MMS_normalized"
)
DEFAULT_OUTPUT = RESEARCH_ROOT / "sar_results"
PATIENT_SETTINGS = (1, 2, 3)
SEEDS = (0, 1, 2, 3, 4)
VENDORS = ("A", "B", "C", "D")
SUMMARY_FIELDS = (
    "method", "task_id", "patient_setting", "seed", "vendor", "vendor_name", "domain",
    "checkpoint", "batch_size", "optimizer", "learning_rate", "momentum", "rho", "adaptive",
    "steps", "episodic", "stream_order", "num_classes", "margin_e0", "ema_decay",
    "reset_constant_em", "adapted_parameter_names", "n_patients", "n_slices", "n_updates",
    "update_rate", "reliable_rate_first", "reliable_rate_second", "reset_count",
    "dice_rv", "dice_myo", "dice_lv", "dice_mean", "hd95_rv", "hd95_myo", "hd95_lv",
    "hd95_mean", "mean_entropy_first", "mean_entropy_second", "mean_sam_grad_norm", "mean_ema",
    "output_dir",
)
TRACE_FIELDS = (
    "task_id", "patient_id", "phase", "z_index", "slice_id", "batch_index",
    "entropy_first", "entropy_second", "reliable_count_first", "reliable_count_second",
    "sam_grad_norm", "ema", "updated", "reset_triggered",
)


def task_coordinates(task_id: int) -> tuple[int, int, str]:
    if not 0 <= int(task_id) < 60:
        raise ValueError("task-id must be in [0, 59].")
    patient_setting = int(task_id) // 20 + 1
    remainder = int(task_id) % 20
    return patient_setting, remainder // 4, VENDORS[remainder % 4]


def checkpoint_path(root: Path, patient_setting: int, seed: int) -> Path:
    return root / f"Patient{patient_setting}" / f"Seed{seed}" / "baseline_model_with_metadata.pt"


def validate_inputs(backbone_root: Path, dataset_root: Path) -> None:
    expected = {
        checkpoint_path(backbone_root, patient, seed).resolve()
        for patient in PATIENT_SETTINGS for seed in SEEDS
    }
    discovered = {
        path.resolve() for path in backbone_root.glob("Patient*/Seed*/baseline_model_with_metadata.pt")
    }
    missing = sorted(str(path) for path in expected - discovered)
    unexpected = sorted(str(path) for path in discovered - expected)
    if len(discovered) != 15 or missing or unexpected:
        raise RuntimeError(
            f"Expected exactly 15 backbones; found={len(discovered)} missing={missing} "
            f"unexpected={unexpected}"
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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
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
    path: Path, patient_setting: int, seed: int, device: torch.device
) -> tuple[UNet, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint is not a mapping: {path}")
    metadata = dict(checkpoint.get("metadata", {}))
    if int(metadata.get("labeled_cases_per_class", patient_setting)) != patient_setting:
        raise ValueError(f"Patient setting mismatch in {path}")
    if int(metadata.get("seed", seed)) != seed:
        raise ValueError(f"Seed mismatch in {path}")
    state = checkpoint.get("teacher_state_dict", checkpoint.get("model_state_dict"))
    if state is None:
        raise KeyError(f"No teacher/model state dict in {path}")
    model = UNet(n_channels=1, n_classes=4, only_feature=False, bilinear=False)
    model.load_state_dict(state, strict=True)
    return model.to(device), metadata


def group_by_patient(records: Sequence[Any]) -> List[List[Any]]:
    groups: Dict[str, List[Any]] = {}
    order: List[str] = []
    for record in records:
        key = str(record.patient_id)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(record)
    return [groups[key] for key in order]


def run_task(args: argparse.Namespace) -> Dict[str, Any]:
    task_id = int(args.task_id)
    patient_setting, seed, vendor = task_coordinates(task_id)
    device = resolve_device(args.device)
    set_seed(seed, cuda=device.type == "cuda")
    task_dir = Path(args.output_root) / "shards" / f"task_{task_id}"
    completion_path = task_dir / "completion.json"
    if completion_path.is_file() and args.resume and not args.overwrite:
        with completion_path.open("r", encoding="utf-8") as handle:
            return dict(json.load(handle)["summary"])
    if task_dir.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"Task output exists; use --resume or --overwrite: {task_dir}")

    backbone = checkpoint_path(Path(args.backbone_root), patient_setting, seed)
    loader = TestLoader(
        vendor=vendor, batch_size=1, shuffle_all_slices=False, seed=seed,
        dataset_root=Path(args.dataset_root),
    )
    patient_groups = group_by_patient(loader.records)
    if int(args.max_patients) > 0:
        patient_groups = patient_groups[: int(args.max_patients)]
    model, metadata = load_backbone(backbone, patient_setting, seed, device)
    configure_model(model)
    params, parameter_names = collect_params(model)
    optimizer = SAM(
        params, torch.optim.SGD, lr=float(args.learning_rate), momentum=float(args.momentum),
        rho=float(args.rho), adaptive=bool(args.adaptive),
    )
    sar = SAR(
        model, optimizer, steps=int(args.steps), episodic=bool(args.episodic),
        margin_e0=float(args.margin_e0), num_classes=int(args.num_classes),
        reset_constant_em=float(args.reset_constant_em), ema_decay=float(args.ema_decay),
    )
    backbone_id = f"MeanTeacher_Patient{patient_setting}_Seed{seed}_teacher_SAR_BN"
    evaluator = PatientStreamEvaluator(domain=loader.domain, seed=seed, backbone_id=backbone_id)
    trace_rows: List[Dict[str, Any]] = []
    batch_index = 0
    print(
        f"[RUN] task={task_id} Patient{patient_setting} Seed{seed} {loader.domain} "
        f"patients={len(patient_groups)} device={device}", flush=True,
    )
    for patient_step, records in enumerate(patient_groups, start=1):
        predictions: List[torch.Tensor] = []
        for record in records:
            batch_index += 1
            images = _stack_images([record]).to(device, non_blocking=device.type == "cuda")
            logits = sar(images)
            predictions.append(torch.argmax(logits, dim=1).detach().cpu())
            stats = dict(sar.last_stats)
            trace_rows.append(
                {
                    "task_id": task_id, "patient_id": record.patient_id, "phase": record.phase,
                    "z_index": record.z_index, "slice_id": record.slice_id, "batch_index": batch_index,
                    "entropy_first": stats.get("entropy_first"),
                    "entropy_second": stats.get("entropy_second"),
                    "reliable_count_first": stats.get("reliable_count_first", 0),
                    "reliable_count_second": stats.get("reliable_count_second", 0),
                    "sam_grad_norm": stats.get("sam_grad_norm"), "ema": stats.get("ema"),
                    "updated": bool(stats.get("updated", False)),
                    "reset_triggered": bool(stats.get("reset_triggered", False)),
                }
            )
        evaluator.update(
            torch.cat(predictions, dim=0), _stack_masks(records),
            [record.meta(include_mask_path=True) for record in records],
            step=patient_step, backbone_id=backbone_id,
        )
        write_csv(task_dir / "adaptation_trace.csv", trace_rows, TRACE_FIELDS)
        evaluator.save_csv(task_dir)
        print(f"[PATIENT] {patient_step}/{len(patient_groups)} id={records[0].patient_id}", flush=True)

    metrics = evaluator.seed_summary()
    n_slices = len(trace_rows)
    n_updates = sum(int(bool(row["updated"])) for row in trace_rows)
    summary: Dict[str, Any] = {
        "method": "SAR-BN", "task_id": task_id, "patient_setting": patient_setting,
        "seed": seed, "vendor": loader.vendor, "vendor_name": loader.vendor_name,
        "domain": loader.domain, "checkpoint": str(backbone.resolve()), "batch_size": 1,
        "optimizer": "SAM(SGD)", "learning_rate": float(args.learning_rate),
        "momentum": float(args.momentum), "rho": float(args.rho), "adaptive": bool(args.adaptive),
        "steps": int(args.steps), "episodic": bool(args.episodic),
        "stream_order": "patient,phase,z_index", "num_classes": int(args.num_classes),
        "margin_e0": float(args.margin_e0), "ema_decay": float(args.ema_decay),
        "reset_constant_em": float(args.reset_constant_em),
        "adapted_parameter_names": "|".join(parameter_names),
        "n_patients": int(metrics["n_patients"]), "n_slices": n_slices,
        "n_updates": n_updates, "update_rate": n_updates / n_slices if n_slices else float("nan"),
        "reliable_rate_first": finite_mean([row["reliable_count_first"] for row in trace_rows]),
        "reliable_rate_second": finite_mean([row["reliable_count_second"] for row in trace_rows]),
        "reset_count": sum(int(bool(row["reset_triggered"])) for row in trace_rows),
        "dice_rv": metrics["dice_rv"], "dice_myo": metrics["dice_myo"],
        "dice_lv": metrics["dice_lv"], "dice_mean": metrics["dice_mean"],
        "hd95_rv": metrics["hd95_rv"], "hd95_myo": metrics["hd95_myo"],
        "hd95_lv": metrics["hd95_lv"], "hd95_mean": metrics["hd95_mean"],
        "mean_entropy_first": finite_mean([row["entropy_first"] for row in trace_rows]),
        "mean_entropy_second": finite_mean([row["entropy_second"] for row in trace_rows]),
        "mean_sam_grad_norm": finite_mean([row["sam_grad_norm"] for row in trace_rows]),
        "mean_ema": finite_mean([row["ema"] for row in trace_rows]),
        "output_dir": str(task_dir.resolve()),
    }
    config = dict(summary)
    config.update(
        dataset_root=str(Path(args.dataset_root).resolve()), checkpoint_metadata=metadata,
        adapted_parameter_names=parameter_names, torch_version=torch.__version__,
        cuda_version=torch.version.cuda, device=str(device),
    )
    write_json(task_dir / "run_config.json", config)
    write_csv(task_dir / "run_summary.csv", [summary], SUMMARY_FIELDS)
    write_json(completion_path, {"status": "complete", "summary": summary})
    print(
        f"[COMPLETE] task={task_id} updates={n_updates}/{n_slices} resets={summary['reset_count']} "
        f"dice={summary['dice_mean']:.6f} hd95={summary['hd95_mean']:.6f}", flush=True,
    )
    return summary


def aggregate(output_root: Path) -> None:
    rows: List[Dict[str, Any]] = []
    missing: List[int] = []
    failures: Dict[str, str] = {}
    for task_id in range(60):
        completion = output_root / "shards" / f"task_{task_id}" / "completion.json"
        if not completion.is_file():
            missing.append(task_id)
            continue
        try:
            with completion.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("status") != "complete":
                raise ValueError(f"status={payload.get('status')}")
            rows.append(dict(payload["summary"]))
        except Exception as error:
            failures[str(task_id)] = f"{type(error).__name__}: {error}"
    rows.sort(key=lambda row: int(row["task_id"]))
    write_csv(output_root / "run_summary.csv", rows, SUMMARY_FIELDS)
    write_json(
        output_root / "missing_tasks.json",
        {"expected": 60, "complete": len(rows), "missing": missing, "failures": failures},
    )
    print(f"[AGGREGATE] complete={len(rows)}/60 missing={missing} failures={failures}", flush=True)
    if len(rows) != 60:
        raise RuntimeError(f"SAR matrix incomplete: {len(rows)}/60 tasks")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or aggregate ACDC -> MMS SAR tasks.")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--backbone-root", default=str(DEFAULT_BACKBONES))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--learning-rate", type=float, default=1.5625e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--episodic", action="store_true")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--margin-e0", type=float, default=0.4 * math.log(4))
    parser.add_argument("--ema-decay", type=float, default=0.9)
    parser.add_argument("--reset-constant-em", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if args.aggregate_only:
        aggregate(Path(args.output_root))
        return
    if args.task_id is None:
        raise ValueError("--task-id is required unless --aggregate-only is used.")
    if (
        args.learning_rate <= 0 or args.momentum < 0 or args.rho < 0 or args.steps <= 0
        or args.num_classes < 2 or args.margin_e0 <= 0 or not 0 <= args.ema_decay < 1
        or args.reset_constant_em < 0 or args.max_patients < 0
    ):
        raise ValueError("Invalid SAR hyperparameters.")
    validate_inputs(Path(args.backbone_root), Path(args.dataset_root))
    device = resolve_device(args.device)
    if device.type == "cuda":
        print(
            f"[CUDA] name={torch.cuda.get_device_name(device)} torch={torch.__version__} "
            f"cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()}", flush=True,
        )
    run_task(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
