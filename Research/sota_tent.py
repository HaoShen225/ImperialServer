#!/usr/bin/env python3
"""Reproduce Source, test-time Norm, and Tent on ACDC -> MMS.

The experiment matrix contains the 15 Mean Teacher source backbones in
``Research/backbone_params`` and the four MMS vendor domains. Each of the 60
backbone-domain cells is evaluated independently with Source, Norm, and Tent.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from helper.Evaluator import PatientStreamEvaluator
from helper.Tent import Tent, collect_params, configure_model, model_logits
from helper.backbones.UNet import UNet
from helper.dataloader import TestLoader, _stack_images, _stack_masks


RESEARCH_ROOT = Path(__file__).resolve().parent
DEFAULT_BACKBONE_ROOT = RESEARCH_ROOT / "backbone_params"
DEFAULT_DATASET_ROOT = Path(
    "/rds/general/user/hs225/ephemeral/TTA_Project_data/dataset/Cardiac/MMS_normalized"
)
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "tent_results"
PATIENT_SETTINGS = (1, 2, 3)
SEEDS = (0, 1, 2, 3, 4)
VENDORS = ("A", "B", "C", "D")
METHODS = ("source", "norm", "tent")
NUM_CLASSES = 4

SUMMARY_FIELDS = (
    "method",
    "patient_setting",
    "seed",
    "vendor",
    "vendor_name",
    "domain",
    "checkpoint",
    "batch_size",
    "learning_rate",
    "optimizer",
    "steps",
    "episodic",
    "stream_order",
    "adapted_parameter_names",
    "n_patients",
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


def parse_csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_csv_strings(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def set_seed(seed: int, *, cuda: bool) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if cuda:
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(value: str) -> torch.device:
    name = str(value).strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    temporary.replace(path)


def checkpoint_path(backbone_root: Path, patient_setting: int, seed: int) -> Path:
    return backbone_root / f"Patient{int(patient_setting)}" / f"Seed{int(seed)}" / "baseline_model_with_metadata.pt"


def validate_full_backbone_matrix(backbone_root: Path) -> None:
    expected = {
        checkpoint_path(backbone_root, patient_setting, seed).resolve()
        for patient_setting in PATIENT_SETTINGS
        for seed in SEEDS
    }
    missing = sorted(str(path) for path in expected if not path.is_file())
    discovered = {
        path.resolve()
        for path in backbone_root.glob("Patient*/Seed*/baseline_model_with_metadata.pt")
    }
    unexpected = sorted(str(path) for path in discovered - expected)
    if missing or unexpected or len(discovered) != 15:
        raise RuntimeError(
            "The backbone matrix must contain exactly Patient1-3 x Seed0-4. "
            f"missing={missing}, unexpected={unexpected}, discovered={len(discovered)}"
        )


def load_backbone(
    path: Path,
    *,
    patient_setting: int,
    seed: int,
    device: torch.device,
) -> tuple[UNet, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint {path} is not a mapping.")

    metadata = dict(checkpoint.get("metadata", {}))
    stored_patient = metadata.get("labeled_cases_per_class")
    stored_seed = metadata.get("seed")
    if stored_patient is not None and int(stored_patient) != int(patient_setting):
        raise ValueError(f"Checkpoint patient setting mismatch in {path}: {stored_patient}")
    if stored_seed is not None and int(stored_seed) != int(seed):
        raise ValueError(f"Checkpoint seed mismatch in {path}: {stored_seed}")

    state = checkpoint.get("teacher_state_dict", checkpoint.get("model_state_dict"))
    if state is None:
        raise KeyError(f"Checkpoint {path} has no teacher/model state dict.")
    model = UNet(n_channels=1, n_classes=NUM_CLASSES, only_feature=False, bilinear=False)
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model, metadata


def configure_norm(model: nn.Module) -> nn.Module:
    """Use independent current-batch BN statistics without optimization."""
    model.train()
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
    return model


def group_records_by_patient(records: Sequence[Any]) -> List[List[Any]]:
    groups: Dict[str, List[Any]] = {}
    order: List[str] = []
    for record in records:
        patient_id = str(record.patient_id)
        if patient_id not in groups:
            groups[patient_id] = []
            order.append(patient_id)
        groups[patient_id].append(record)
    return [groups[patient_id] for patient_id in order]


def batched(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), int(batch_size)):
        yield values[start : start + int(batch_size)]


def infer_patient(
    model: nn.Module,
    method: str,
    records: Sequence[Any],
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    predictions: List[torch.Tensor] = []
    for record_batch in batched(records, batch_size):
        images = _stack_images(record_batch).to(device, non_blocking=device.type == "cuda")
        if method == "tent":
            logits = model(images)
        else:
            with torch.no_grad():
                logits = model_logits(model(images))
        predictions.append(torch.argmax(logits, dim=1).detach().cpu())
    return torch.cat(predictions, dim=0)


def setup_method(
    base_model: nn.Module,
    method: str,
    *,
    learning_rate: float,
    steps: int,
    episodic: bool,
) -> tuple[nn.Module, List[str], str, float]:
    if method == "source":
        base_model.eval()
        base_model.requires_grad_(False)
        return base_model, [], "none", 0.0
    if method == "norm":
        return configure_norm(base_model), [], "none", 0.0
    if method == "tent":
        configure_model(base_model)
        params, names = collect_params(base_model)
        optimizer = torch.optim.Adam(
            params,
            lr=float(learning_rate),
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )
        return Tent(base_model, optimizer, steps=int(steps), episodic=bool(episodic)), names, "Adam", float(learning_rate)
    raise ValueError(f"Unsupported method: {method}")


def evaluate_run(
    *,
    args: argparse.Namespace,
    device: torch.device,
    patient_setting: int,
    seed: int,
    vendor: str,
    method: str,
) -> Dict[str, Any]:
    checkpoint = checkpoint_path(Path(args.backbone_root), patient_setting, seed)
    loader = TestLoader(
        vendor=vendor,
        batch_size=int(args.batch_size),
        shuffle_all_slices=False,
        seed=int(seed),
        dataset_root=Path(args.dataset_root),
    )
    patient_groups = group_records_by_patient(loader.records)
    if int(args.max_patients) > 0:
        patient_groups = patient_groups[: int(args.max_patients)]

    run_dir = (
        Path(args.output_root)
        / f"Patient{int(patient_setting)}"
        / f"Seed{int(seed)}"
        / loader.domain
        / method
    )
    completed_path = run_dir / "run_summary.json"
    if completed_path.is_file() and args.resume and not args.overwrite:
        with completed_path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        print(f"[SKIP] complete {method} Patient{patient_setting} Seed{seed} {loader.domain}", flush=True)
        return dict(row)
    if run_dir.exists() and not args.overwrite and not args.resume:
        raise FileExistsError(f"Output exists; use --resume or --overwrite: {run_dir}")

    set_seed(seed, cuda=device.type == "cuda")
    base_model, checkpoint_metadata = load_backbone(
        checkpoint,
        patient_setting=patient_setting,
        seed=seed,
        device=device,
    )
    adapted_model, adapted_names, optimizer_name, effective_lr = setup_method(
        base_model,
        method,
        learning_rate=float(args.learning_rate),
        steps=int(args.steps),
        episodic=bool(args.episodic),
    )
    backbone_id = f"MeanTeacher_Patient{patient_setting}_Seed{seed}_teacher_{method}"
    evaluator = PatientStreamEvaluator(domain=loader.domain, seed=seed, backbone_id=backbone_id)

    print(
        f"[RUN] {method} Patient{patient_setting} Seed{seed} {loader.domain} "
        f"patients={len(patient_groups)} slices={sum(len(group) for group in patient_groups)}",
        flush=True,
    )
    for patient_step, records in enumerate(patient_groups, start=1):
        predictions = infer_patient(
            adapted_model,
            method,
            records,
            device=device,
            batch_size=int(args.batch_size),
        )
        masks = _stack_masks(records)
        meta = [record.meta(include_mask_path=True) for record in records]
        evaluator.update(predictions, masks, meta, step=patient_step, backbone_id=backbone_id)

    evaluator.save_csv(run_dir)
    metrics = evaluator.seed_summary()
    row: Dict[str, Any] = {
        "method": method.capitalize(),
        "patient_setting": int(patient_setting),
        "seed": int(seed),
        "vendor": loader.vendor,
        "vendor_name": loader.vendor_name,
        "domain": loader.domain,
        "checkpoint": str(checkpoint.resolve()),
        "batch_size": int(args.batch_size),
        "learning_rate": effective_lr,
        "optimizer": optimizer_name,
        "steps": int(args.steps) if method == "tent" else 0,
        "episodic": bool(args.episodic) if method == "tent" else False,
        "stream_order": "patient,phase,z_index",
        "adapted_parameter_names": "|".join(adapted_names),
        "n_patients": int(metrics["n_patients"]),
        "dice_rv": metrics["dice_rv"],
        "dice_myo": metrics["dice_myo"],
        "dice_lv": metrics["dice_lv"],
        "dice_mean": metrics["dice_mean"],
        "hd95_rv": metrics["hd95_rv"],
        "hd95_myo": metrics["hd95_myo"],
        "hd95_lv": metrics["hd95_lv"],
        "hd95_mean": metrics["hd95_mean"],
        "output_dir": str(run_dir.resolve()),
    }
    config = dict(row)
    config.update(
        {
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "checkpoint_metadata": checkpoint_metadata,
            "adapted_parameter_names": adapted_names,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
        }
    )
    write_json(run_dir / "run_config.json", config)
    write_json(completed_path, row)
    print(
        f"[DONE] {method} Patient{patient_setting} Seed{seed} {loader.domain} "
        f"dice={float(row['dice_mean']):.6f} hd95={float(row['hd95_mean']):.6f}",
        flush=True,
    )
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce Source/Norm/Tent for ACDC -> MMS.")
    parser.add_argument("--backbone-root", default=str(DEFAULT_BACKBONE_ROOT))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--patients", default="1,2,3")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--vendors", default="A,B,C,D")
    parser.add_argument("--methods", default="source,norm,tent")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--episodic", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-patients", type=int, default=0, help="0 evaluates every patient.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> tuple[List[int], List[int], List[str], List[str]]:
    patients = parse_csv_ints(args.patients)
    seeds = parse_csv_ints(args.seeds)
    vendors = [item.upper() for item in parse_csv_strings(args.vendors)]
    methods = [item.lower() for item in parse_csv_strings(args.methods)]
    if not patients or any(item not in PATIENT_SETTINGS for item in patients):
        raise ValueError(f"--patients must be drawn from {PATIENT_SETTINGS}")
    if not seeds or any(item not in SEEDS for item in seeds):
        raise ValueError(f"--seeds must be drawn from {SEEDS}")
    if not vendors or any(item not in VENDORS for item in vendors):
        raise ValueError(f"--vendors must be drawn from {VENDORS}")
    if not methods or any(item not in METHODS for item in methods):
        raise ValueError(f"--methods must be drawn from {METHODS}")
    if int(args.batch_size) <= 0 or int(args.steps) <= 0 or int(args.max_patients) < 0:
        raise ValueError("batch-size/steps must be positive and max-patients must be non-negative.")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if not Path(args.dataset_root).is_dir():
        raise FileNotFoundError(f"MMS dataset root does not exist: {args.dataset_root}")
    validate_full_backbone_matrix(Path(args.backbone_root))
    return patients, seeds, vendors, methods


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    patients, seeds, vendors, methods = validate_args(args)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(
            f"[DEVICE] {device} name={torch.cuda.get_device_name(device)} "
            f"torch={torch.__version__} cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()}",
            flush=True,
        )
    else:
        print(f"[DEVICE] cpu torch={torch.__version__}", flush=True)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, Any]] = []
    total = len(patients) * len(seeds) * len(vendors) * len(methods)
    completed = 0
    for patient_setting in patients:
        for seed in seeds:
            for vendor in vendors:
                for method in methods:
                    row = evaluate_run(
                        args=args,
                        device=device,
                        patient_setting=patient_setting,
                        seed=seed,
                        vendor=vendor,
                        method=method,
                    )
                    summaries.append(row)
                    completed += 1
                    summaries.sort(key=lambda item: (int(item["patient_setting"]), int(item["seed"]), str(item["vendor"]), str(item["method"])))
                    write_csv(output_root / "run_summary.csv", summaries)
                    print(f"[PROGRESS] {completed}/{total}", flush=True)

    print(f"[COMPLETE] wrote {len(summaries)} rows to {output_root / 'run_summary.csv'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
