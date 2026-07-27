#!/usr/bin/env python3
"""Run SPEGC with the 15 fixed U-Net backbones on ACDC -> MMS."""

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
from helper.Spegc import SPEGC, SemanticPromptGraphClustering
from helper.backbones.UNet import UNet
from helper.dataloader import TestLoader, _stack_images, _stack_masks


RESEARCH_ROOT = Path(__file__).resolve().parent
DEFAULT_BACKBONES = RESEARCH_ROOT / "backbone_params"
DEFAULT_DATASET = Path(
    "/rds/general/user/hs225/ephemeral/TTA_Project_data/dataset/Cardiac/MMS_normalized"
)
DEFAULT_OUTPUT = RESEARCH_ROOT / "spegc_results"
PATIENT_SETTINGS = (1, 2, 3)
SEEDS = (0, 1, 2, 3, 4)
VENDORS = ("A", "B", "C", "D")
SUMMARY_FIELDS = (
    "method", "task_id", "patient_setting", "seed", "vendor", "vendor_name", "domain",
    "checkpoint", "batch_size", "optimizer", "learning_rate", "momentum", "weight_decay",
    "steps", "episodic", "stream_order", "detach_backbone", "feature_dim", "num_centroids",
    "target_clusters", "num_prompts", "density_temperature", "sinkhorn_temperature",
    "sinkhorn_iterations", "commonality_weight", "mc_passes", "dropout_probability",
    "keep_ratio", "sample_dist", "pool_size", "min_pool_size", "background_index",
    "max_nodes", "adapted_parameter_names", "n_patients", "n_slices", "n_updates",
    "update_rate", "empty_graph_count", "skip_empty_current_graph", "skip_pool_not_ready",
    "skip_insufficient_graphs", "skip_insufficient_nodes", "skip_nonfinite_loss",
    "skip_nonfinite_gradient", "skip_no_gradients", "mean_sampled_nodes",
    "mean_foreground_pixels", "mean_foreground_ratio", "mean_reliable_pixels",
    "mean_total_nodes", "mean_candidate_edges", "mean_edge_budget", "mean_graph_loss",
    "mean_commonality_loss", "mean_total_loss", "mean_gradient_norm",
    "max_selected_self_edge_mass", "backbone_parameter_drift_l2", "graph_parameter_drift_l2",
    "dice_rv", "dice_myo", "dice_lv", "dice_mean", "hd95_rv", "hd95_myo", "hd95_lv",
    "hd95_mean", "output_dir",
)
TRACE_FIELDS = (
    "task_id", "patient_id", "phase", "z_index", "slice_id", "batch_index", "updated",
    "skip_reason", "pool_size_before", "pool_size_after", "graph_count",
    "current_graph_count", "sampled_nodes", "foreground_pixels", "foreground_ratio",
    "reliable_pixels", "total_nodes", "candidate_edges", "edge_budget", "graph_loss",
    "commonality_loss", "total_loss", "gradient_norm", "selected_self_edge_mass",
    "detach_backbone",
)


def task_coordinates(task_id: int) -> Tuple[int, int, str]:
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
) -> Tuple[UNet, Dict[str, Any]]:
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


def group_by_patient(records: Sequence[Any], max_patients: int, max_slices: int) -> List[List[Any]]:
    groups: Dict[str, List[Any]] = {}
    order: List[str] = []
    for record in records:
        key = str(record.patient_id)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(record)
    result = [groups[key] for key in order]
    if int(max_patients) > 0:
        result = result[: int(max_patients)]
    if int(max_slices) > 0:
        remaining = int(max_slices)
        limited: List[List[Any]] = []
        for group in result:
            if remaining <= 0:
                break
            selected = group[:remaining]
            if selected:
                limited.append(selected)
                remaining -= len(selected)
        result = limited
    return result


def configure_adaptation(
    model: UNet,
    graph_module: SemanticPromptGraphClustering,
    detach_backbone: bool,
) -> Tuple[List[torch.nn.Parameter], List[str]]:
    """Select the effective SPEGC parameters without changing U-Net structure."""
    model.eval()
    model.requires_grad_(False)
    graph_module.train()
    graph_module.requires_grad_(True)
    parameters: List[torch.nn.Parameter] = []
    names: List[str] = []
    if not detach_backbone:
        for name, parameter in model.named_parameters():
            if name.startswith("outc."):
                continue
            parameter.requires_grad_(True)
            parameters.append(parameter)
            names.append(f"backbone.{name}")
    for name, parameter in graph_module.named_parameters():
        parameters.append(parameter)
        names.append(f"graph.{name}")
    return parameters, names


@torch.no_grad()
def parameter_drift_l2(module: torch.nn.Module, initial_state: Mapping[str, torch.Tensor]) -> float:
    """Return the global L2 distance from an initial module state."""
    squared = torch.zeros((), device=next(module.parameters()).device, dtype=torch.float64)
    for name, parameter in module.named_parameters():
        initial = initial_state[name].to(device=parameter.device, dtype=parameter.dtype)
        squared += (parameter.detach() - initial).double().square().sum()
    return float(torch.sqrt(squared).cpu())


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
    patient_groups = group_by_patient(loader.records, args.max_patients, args.max_slices)
    if not patient_groups:
        raise RuntimeError("The selected test stream is empty.")
    model, metadata = load_backbone(backbone, patient_setting, seed, device)
    graph_module = SemanticPromptGraphClustering(
        feature_dim=int(args.feature_dim), num_centroids=int(args.num_centroids),
        target_clusters=int(args.target_clusters), num_prompts=int(args.num_prompts),
        density_temperature=float(args.density_temperature),
        sinkhorn_temperature=float(args.sinkhorn_temperature),
        commonality_weight=float(args.commonality_weight),
        sinkhorn_iterations=int(args.sinkhorn_iterations),
    ).to(device)
    params, parameter_names = configure_adaptation(model, graph_module, bool(args.detach_backbone))
    optimizer = torch.optim.SGD(
        params, lr=float(args.learning_rate), momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
    )
    spegc = SPEGC(
        model, graph_module, optimizer, steps=int(args.steps), episodic=bool(args.episodic),
        detach_backbone=bool(args.detach_backbone), pool_size=int(args.pool_size),
        min_pool_size=int(args.min_pool_size), mc_passes=int(args.mc_passes),
        dropout_probability=float(args.dropout_probability), keep_ratio=float(args.keep_ratio),
        sample_dist=int(args.sample_dist), background_index=int(args.background_index),
        max_nodes=None if int(args.max_nodes) == 0 else int(args.max_nodes),
    )

    method = "SPEGC-official-detach-phase1" if args.detach_backbone else "SPEGC-online-phase1"
    backbone_id = f"MeanTeacher_Patient{patient_setting}_Seed{seed}_teacher_{method}"
    evaluator = PatientStreamEvaluator(domain=loader.domain, seed=seed, backbone_id=backbone_id)
    trace_rows: List[Dict[str, Any]] = []
    batch_index = 0
    print(
        f"[RUN] task={task_id} Patient{patient_setting} Seed{seed} {loader.domain} "
        f"patients={len(patient_groups)} device={device} method={method}", flush=True,
    )
    for patient_step, records in enumerate(patient_groups, start=1):
        predictions: List[torch.Tensor] = []
        for record in records:
            batch_index += 1
            images = _stack_images([record]).to(device, non_blocking=device.type == "cuda")
            logits = spegc(images)
            predictions.append(torch.argmax(logits, dim=1).detach().cpu())
            stats = dict(spegc.last_stats)
            trace_rows.append(
                {
                    "task_id": task_id, "patient_id": record.patient_id, "phase": record.phase,
                    "z_index": record.z_index, "slice_id": record.slice_id,
                    "batch_index": batch_index, "updated": bool(stats.get("updated", False)),
                    "skip_reason": stats.get("skip_reason"),
                    "pool_size_before": stats.get("pool_size_before", 0),
                    "pool_size_after": stats.get("pool_size_after", 0),
                    "graph_count": stats.get("graph_count", 0),
                    "current_graph_count": stats.get("current_graph_count", 0),
                    "sampled_nodes": stats.get("sampled_nodes", 0),
                    "foreground_pixels": stats.get("foreground_pixels", 0),
                    "foreground_ratio": stats.get("foreground_ratio", 0.0),
                    "reliable_pixels": stats.get("reliable_pixels", 0),
                    "total_nodes": stats.get("total_nodes", 0),
                    "candidate_edges": stats.get("candidate_edges", 0),
                    "edge_budget": stats.get("edge_budget", 0),
                    "graph_loss": stats.get("graph_loss"),
                    "commonality_loss": stats.get("commonality_loss"),
                    "total_loss": stats.get("total_loss"),
                    "gradient_norm": stats.get("gradient_norm"),
                    "selected_self_edge_mass": stats.get("selected_self_edge_mass", 0.0),
                    "detach_backbone": bool(args.detach_backbone),
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
    skip_counts: Dict[str, int] = {}
    for row in trace_rows:
        reason = row.get("skip_reason")
        if reason:
            skip_counts[str(reason)] = skip_counts.get(str(reason), 0) + 1
    backbone_drift = parameter_drift_l2(model, spegc.model_state)
    graph_drift = parameter_drift_l2(graph_module, spegc.graph_state)
    summary: Dict[str, Any] = {
        "method": method, "task_id": task_id, "patient_setting": patient_setting, "seed": seed,
        "vendor": loader.vendor, "vendor_name": loader.vendor_name, "domain": loader.domain,
        "checkpoint": str(backbone.resolve()), "batch_size": 1, "optimizer": "SGD",
        "learning_rate": float(args.learning_rate), "momentum": float(args.momentum),
        "weight_decay": float(args.weight_decay), "steps": int(args.steps),
        "episodic": bool(args.episodic), "stream_order": "patient,phase,z_index",
        "detach_backbone": bool(args.detach_backbone), "feature_dim": int(args.feature_dim),
        "num_centroids": int(args.num_centroids), "target_clusters": int(args.target_clusters),
        "num_prompts": int(args.num_prompts), "density_temperature": float(args.density_temperature),
        "sinkhorn_temperature": float(args.sinkhorn_temperature),
        "sinkhorn_iterations": int(args.sinkhorn_iterations),
        "commonality_weight": float(args.commonality_weight), "mc_passes": int(args.mc_passes),
        "dropout_probability": float(args.dropout_probability), "keep_ratio": float(args.keep_ratio),
        "sample_dist": int(args.sample_dist), "pool_size": int(args.pool_size),
        "min_pool_size": int(args.min_pool_size), "background_index": int(args.background_index),
        "max_nodes": int(args.max_nodes), "adapted_parameter_names": "|".join(parameter_names),
        "n_patients": int(metrics["n_patients"]), "n_slices": n_slices, "n_updates": n_updates,
        "update_rate": n_updates / n_slices if n_slices else float("nan"),
        "empty_graph_count": sum(int(row["current_graph_count"] == 0) for row in trace_rows),
        "skip_empty_current_graph": skip_counts.get("empty_current_graph", 0),
        "skip_pool_not_ready": skip_counts.get("pool_not_ready", 0),
        "skip_insufficient_graphs": skip_counts.get("insufficient_graphs", 0),
        "skip_insufficient_nodes": skip_counts.get("insufficient_nodes", 0),
        "skip_nonfinite_loss": skip_counts.get("nonfinite_loss", 0),
        "skip_nonfinite_gradient": skip_counts.get("nonfinite_gradient", 0),
        "skip_no_gradients": skip_counts.get("no_gradients", 0),
        "mean_sampled_nodes": finite_mean([row["sampled_nodes"] for row in trace_rows]),
        "mean_foreground_pixels": finite_mean([row["foreground_pixels"] for row in trace_rows]),
        "mean_foreground_ratio": finite_mean([row["foreground_ratio"] for row in trace_rows]),
        "mean_reliable_pixels": finite_mean([row["reliable_pixels"] for row in trace_rows]),
        "mean_total_nodes": finite_mean([row["total_nodes"] for row in trace_rows]),
        "mean_candidate_edges": finite_mean([row["candidate_edges"] for row in trace_rows]),
        "mean_edge_budget": finite_mean([row["edge_budget"] for row in trace_rows]),
        "mean_graph_loss": finite_mean([row["graph_loss"] for row in trace_rows]),
        "mean_commonality_loss": finite_mean([row["commonality_loss"] for row in trace_rows]),
        "mean_total_loss": finite_mean([row["total_loss"] for row in trace_rows]),
        "mean_gradient_norm": finite_mean([row["gradient_norm"] for row in trace_rows]),
        "max_selected_self_edge_mass": max(
            (float(row["selected_self_edge_mass"]) for row in trace_rows), default=0.0
        ),
        "backbone_parameter_drift_l2": backbone_drift,
        "graph_parameter_drift_l2": graph_drift,
        "dice_rv": metrics["dice_rv"], "dice_myo": metrics["dice_myo"],
        "dice_lv": metrics["dice_lv"], "dice_mean": metrics["dice_mean"],
        "hd95_rv": metrics["hd95_rv"], "hd95_myo": metrics["hd95_myo"],
        "hd95_lv": metrics["hd95_lv"], "hd95_mean": metrics["hd95_mean"],
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
        f"[COMPLETE] task={task_id} updates={n_updates}/{n_slices} "
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
        raise RuntimeError(f"SPEGC matrix incomplete: {len(rows)}/60 tasks")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or aggregate ACDC -> MMS SPEGC tasks.")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--backbone-root", default=str(DEFAULT_BACKBONES))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--episodic", action="store_true")
    parser.add_argument("--detach-backbone", action="store_true")
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--num-centroids", type=int, default=32)
    parser.add_argument("--target-clusters", type=int, default=48)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--density-temperature", type=float, default=0.1)
    parser.add_argument("--sinkhorn-temperature", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iterations", type=int, default=20)
    parser.add_argument("--commonality-weight", type=float, default=0.2)
    parser.add_argument("--mc-passes", type=int, default=4)
    parser.add_argument("--dropout-probability", type=float, default=0.1)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--sample-dist", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=3)
    parser.add_argument("--min-pool-size", type=int, default=1)
    parser.add_argument("--background-index", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=0, help="0 keeps official sparse sampling only.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--max-slices", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.learning_rate, args.steps, args.feature_dim, args.num_centroids, args.target_clusters,
        args.num_prompts, args.density_temperature, args.sinkhorn_temperature,
        args.sinkhorn_iterations, args.mc_passes, args.sample_dist, args.pool_size,
    )
    if any(float(value) <= 0 for value in positive):
        raise ValueError("Positive SPEGC hyperparameters must be greater than zero.")
    if args.momentum < 0 or args.weight_decay < 0 or args.commonality_weight < 0:
        raise ValueError("Optimizer and loss weights cannot be negative.")
    if not 0 <= args.dropout_probability < 1 or not 0 < args.keep_ratio <= 1:
        raise ValueError("Invalid dropout probability or reliable keep ratio.")
    if not 0 <= args.min_pool_size <= args.pool_size:
        raise ValueError("min-pool-size must satisfy 0 <= min <= pool-size.")
    if min(args.max_nodes, args.max_patients, args.max_slices) < 0:
        raise ValueError("Stream and node limits cannot be negative.")
    if args.feature_dim != 64:
        raise ValueError("The current fixed U-Net exposes a 64-channel final decoder feature.")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if args.aggregate_only:
        aggregate(Path(args.output_root))
        return
    if args.task_id is None:
        raise ValueError("--task-id is required unless --aggregate-only is used.")
    validate_args(args)
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
