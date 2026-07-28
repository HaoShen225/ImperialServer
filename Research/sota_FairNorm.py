#!/usr/bin/env python3
"""FairNorm：用 Batch size=1 公平复现 Norm 测试时归一化基线。

实验口径
========
本脚本使用 ``Research/backbone_params`` 中的 15 个 Mean Teacher teacher
backbone，在 ACDC -> MMS 的 3-shot × 5-seed × 4-domain 共 60 个任务上
评估 Norm。与已有 ``sota_tent.py`` 中 Batch size=8 的 Norm 相比，本实验
把测试 batch 严格固定为 1，以便与 InTEnt、GraTA、SAR、SPEGC 等逐切片
方案在 batch size 上公平比较。

所有模型参数（包括 BN 的 gamma/beta）始终冻结。每个 BatchNorm2d 层
关闭 running statistics，并仅使用当前一张目标切片在 N/H/W 维计算的
均值与方差。不同切片之间不累计统计量、梯度或优化器状态，因此该方案
是逐切片、无参数更新、与测试流顺序无关的 episodic Norm。
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from helper.Evaluator import PatientStreamEvaluator
from helper.Tent import model_logits
from helper.backbones.UNet import UNet
from helper.dataloader import TestLoader, _stack_images, _stack_masks


RESEARCH_ROOT = Path(__file__).resolve().parent
DEFAULT_BACKBONES = RESEARCH_ROOT / "backbone_params"
DEFAULT_DATASET = Path(
    "/rds/general/user/hs225/ephemeral/TTA_Project_data/dataset/Cardiac/MMS_normalized"
)
DEFAULT_OUTPUT = RESEARCH_ROOT / "FairNorm_results"

PATIENT_SETTINGS = (1, 2, 3)
SEEDS = (0, 1, 2, 3, 4)
VENDORS = ("A", "B", "C", "D")
NUM_CLASSES = 4
BATCH_SIZE = 1
METHOD = "FairNorm"
NORMALIZATION = "current_slice_batch_statistics"

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
    "normalization",
    "bn_affine",
    "bn_layer_count",
    "optimizer",
    "steps",
    "episodic",
    "stream_dependent",
    "stream_order",
    "adapted_parameter_names",
    "n_patients",
    "n_slices",
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


def task_coordinates(task_id: int) -> Tuple[int, int, str]:
    """Map a canonical array id to shot, seed, and MMS vendor."""
    value = int(task_id)
    if not 0 <= value < 60:
        raise ValueError("--task-id must be in [0, 59].")
    patient_setting = value // 20 + 1
    remainder = value % 20
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


def resolve_device(value: str) -> torch.device:
    text = str(value).strip().lower()
    if text == "auto":
        text = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return device


def set_seed(seed: int, *, cuda: bool) -> None:
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
        TestLoader(
            vendor=vendor,
            batch_size=BATCH_SIZE,
            dataset_root=dataset_root,
        )


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
    stored_patient = metadata.get("labeled_cases_per_class")
    stored_seed = metadata.get("seed")
    if stored_patient is not None and int(stored_patient) != int(patient_setting):
        raise ValueError(f"Patient setting mismatch in {path}: {stored_patient}")
    if stored_seed is not None and int(stored_seed) != int(seed):
        raise ValueError(f"Seed mismatch in {path}: {stored_seed}")

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
    return model, metadata


def configure_fairnorm(model: nn.Module) -> Tuple[str, ...]:
    """Freeze the model and make every BN use only the current slice."""
    model.eval()
    model.requires_grad_(False)
    names: List[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.BatchNorm2d):
            continue
        names.append(name)
        module.train()
        module.track_running_stats = False
        module.running_mean = None
        module.running_var = None

    if not names:
        raise RuntimeError("FairNorm requires at least one BatchNorm2d layer.")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("FairNorm must freeze every model parameter.")
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if not module.training:
                raise AssertionError("FairNorm BatchNorm2d layers must use batch mode.")
            if module.track_running_stats:
                raise AssertionError("FairNorm must disable BN running statistics.")
            if module.running_mean is not None or module.running_var is not None:
                raise AssertionError("FairNorm BN running statistics must be absent.")
    return tuple(names)


def fairnorm_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Run a strictly singleton FairNorm forward pass."""
    if images.ndim != 4 or int(images.shape[0]) != BATCH_SIZE:
        raise ValueError(
            "FairNorm requires input shape [1,C,H,W]; "
            f"received {tuple(images.shape)}"
        )
    with torch.no_grad():
        logits = model_logits(model(images))
    if logits.ndim != 4 or int(logits.shape[0]) != BATCH_SIZE:
        raise RuntimeError(
            "FairNorm logits must have shape [1,C,H,W]; "
            f"received {tuple(logits.shape)}"
        )
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("FairNorm produced non-finite logits.")
    return logits


def group_by_patient(
    records: Sequence[Any],
    *,
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


def _completed_summary(
    completion_path: Path,
    *,
    expected_task_id: int,
) -> Dict[str, Any]:
    with completion_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") != "complete":
        raise ValueError(
            f"Invalid completion status in {completion_path}: "
            f"{payload.get('status')}"
        )
    summary = dict(payload["summary"])
    if int(summary.get("task_id", -1)) != int(expected_task_id):
        raise ValueError(
            f"Task id mismatch in {completion_path}: {summary.get('task_id')}"
        )
    if summary.get("method") != METHOD:
        raise ValueError(
            f"Method mismatch in {completion_path}: {summary.get('method')}"
        )
    if int(summary.get("batch_size", -1)) != BATCH_SIZE:
        raise ValueError(
            f"Batch-size mismatch in {completion_path}: "
            f"{summary.get('batch_size')}"
        )
    if summary.get("normalization") != NORMALIZATION:
        raise ValueError(
            f"Normalization mismatch in {completion_path}: "
            f"{summary.get('normalization')}"
        )
    return summary


def run_task(args: argparse.Namespace) -> Dict[str, Any]:
    task_id = int(args.task_id)
    patient_setting, seed, vendor = task_coordinates(task_id)
    device = resolve_device(args.device)
    set_seed(seed, cuda=device.type == "cuda")

    output_root = Path(args.output_root)
    task_dir = output_root / "shards" / f"task_{task_id}"
    completion_path = task_dir / "completion.json"
    if completion_path.is_file() and args.resume and not args.overwrite:
        summary = _completed_summary(
            completion_path,
            expected_task_id=task_id,
        )
        print(f"[SKIP] complete task={task_id}", flush=True)
        return summary
    if task_dir.exists():
        if args.overwrite or args.resume:
            shutil.rmtree(task_dir)
        else:
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
        batch_size=BATCH_SIZE,
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
        raise RuntimeError("The selected FairNorm test stream is empty.")

    model, checkpoint_metadata = load_backbone(
        backbone,
        patient_setting,
        seed,
        device,
    )
    bn_names = configure_fairnorm(model)
    backbone_id = (
        f"MeanTeacher_Patient{patient_setting}_Seed{seed}_teacher_FairNorm"
    )
    evaluator = PatientStreamEvaluator(
        domain=loader.domain,
        seed=seed,
        backbone_id=backbone_id,
    )
    n_slices = 0

    print(
        f"[RUN] task={task_id} shot={patient_setting} seed={seed} "
        f"domain={loader.domain} patients={len(patient_groups)} "
        f"slices={sum(len(group) for group in patient_groups)} "
        f"batch_size={BATCH_SIZE} bn_layers={len(bn_names)} device={device}",
        flush=True,
    )
    for patient_step, records in enumerate(patient_groups, start=1):
        predictions: List[torch.Tensor] = []
        for record in records:
            images = _stack_images([record]).to(
                device,
                non_blocking=device.type == "cuda",
            )
            logits = fairnorm_logits(model, images)
            predictions.append(
                torch.argmax(logits, dim=1).detach().cpu()
            )
            n_slices += 1

        evaluator.update(
            torch.cat(predictions, dim=0),
            _stack_masks(records),
            [record.meta(include_mask_path=True) for record in records],
            step=patient_step,
            backbone_id=backbone_id,
        )
        evaluator.save_csv(task_dir)
        print(
            f"[PATIENT] {patient_step}/{len(patient_groups)} "
            f"id={records[0].patient_id} slices={len(records)}",
            flush=True,
        )

    metrics = evaluator.seed_summary()
    summary: Dict[str, Any] = {
        "method": METHOD,
        "task_id": task_id,
        "patient_setting": patient_setting,
        "seed": seed,
        "vendor": loader.vendor,
        "vendor_name": loader.vendor_name,
        "domain": loader.domain,
        "checkpoint": str(backbone.resolve()),
        "batch_size": BATCH_SIZE,
        "normalization": NORMALIZATION,
        "bn_affine": "source_frozen",
        "bn_layer_count": len(bn_names),
        "optimizer": "none",
        "steps": 0,
        "episodic": True,
        "stream_dependent": False,
        "stream_order": "patient,phase,z_index",
        "adapted_parameter_names": "",
        "n_patients": int(metrics["n_patients"]),
        "n_slices": n_slices,
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
            "checkpoint_metadata": checkpoint_metadata,
            "bn_layer_names": list(bn_names),
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


def rebuild_summary(
    output_root: Path,
    *,
    require_complete: bool,
) -> Dict[str, Any]:
    """Atomically rebuild global output while array tasks run concurrently."""
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".summary.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
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
                rows.append(
                    _completed_summary(
                        completion,
                        expected_task_id=task_id,
                    )
                )
            except Exception as error:
                failures[str(task_id)] = (
                    f"{type(error).__name__}: {error}"
                )

        rows.sort(key=lambda row: int(row["task_id"]))
        complete = len(rows) == 60 and not failures
        status: Dict[str, Any] = {
            "expected_tasks": 60,
            "completed_tasks": len(rows),
            "complete": complete,
            "missing_task_ids": missing,
            "failures": failures,
        }
        write_csv(
            output_root / "run_summary.csv",
            rows,
            SUMMARY_FIELDS,
        )
        write_json(output_root / "matrix_status.json", status)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    print(
        f"[AGGREGATE] complete={len(rows)}/60 "
        f"missing={len(missing)} failures={len(failures)}",
        flush=True,
    )
    if require_complete and not bool(status["complete"]):
        raise RuntimeError(
            "FairNorm matrix is incomplete: "
            f"complete={len(rows)}/60 missing={missing} failures={failures}"
        )
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or aggregate Batch-size-1 FairNorm on ACDC -> MMS."
        )
    )
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--backbone-root", default=str(DEFAULT_BACKBONES))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-patients",
        type=int,
        default=0,
        help="Debug limit; 0 evaluates every patient.",
    )
    parser.add_argument(
        "--max-slices",
        type=int,
        default=0,
        help="Debug limit across selected patients; 0 evaluates every slice.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if int(args.max_patients) < 0 or int(args.max_slices) < 0:
        raise ValueError("--max-patients/--max-slices must be non-negative.")
    if args.aggregate_only:
        rebuild_summary(
            Path(args.output_root),
            require_complete=True,
        )
        return
    if args.task_id is None:
        raise ValueError("--task-id is required unless --aggregate-only is used.")
    task_coordinates(int(args.task_id))
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
    rebuild_summary(
        Path(args.output_root),
        require_complete=False,
    )


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
