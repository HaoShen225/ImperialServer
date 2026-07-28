#!/usr/bin/env python3
"""Train fully supervised MMS source backbones for clean TTA experiments.

The default experiment matrix contains twenty independent runs:

    source domain A, seeds 0/1/2/3/4
    source domain B, seeds 0/1/2/3/4
    source domain C, seeds 0/1/2/3/4
    source domain D, seeds 0/1/2/3/4

For a selected source domain, every patient with a mask in the normalized MMS
dataset is used for supervised training.  This is intentional: the normalized
MMS copy stores masks for all four domains, whereas the official
``Training/Labeled`` partition exists only for vendors A and B.  Checkpoint
metadata records the original MMS partitions and the source domain so that
downstream TTA experiments can exclude the source domain from target testing.

The training objective and U-Net architecture match the supervised component
of ``Research/backbone_training_MeanTeacher.py``: categorical cross entropy
plus foreground soft-Dice loss.  No pseudo-label, consistency, EMA teacher, or
unlabeled loss is used.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from helper.backbones.UNet import UNet  # noqa: E402
from helper.dataloader import PROCESSED_SUBDIR, TestLoader, VENDOR_INFO  # noqa: E402


DEFAULT_DATASET_ROOT = Path(
    "/rds/general/user/hs225/ephemeral/TTA_Project_data/dataset/Cardiac/MMS_normalized"
)
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "backbone_params_cleanSource"
CANONICAL_VENDORS = ("A", "B", "C", "D")
CANONICAL_SEEDS = (0, 1, 2, 3, 4)
NUM_EXPECTED_RUNS = len(CANONICAL_VENDORS) * len(CANONICAL_SEEDS)
NUM_CLASSES = 4
LABEL_MAP = {
    "0": "background",
    "1": "rv",
    "2": "myo",
    "3": "lv",
}


def parse_csv_values(text: str) -> List[str]:
    """Parse a comma-separated CLI value while preserving order."""
    return [item.strip() for item in str(text).split(",") if item.strip()]


def parse_seeds(text: str) -> List[int]:
    """Parse and validate a comma-separated subset of canonical seeds."""
    seeds = [int(value) for value in parse_csv_values(text)]
    if not seeds:
        raise ValueError("--seeds produced an empty list.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must not contain duplicates.")
    invalid = [seed for seed in seeds if seed not in CANONICAL_SEEDS]
    if invalid:
        raise ValueError(f"Seeds must be chosen from {CANONICAL_SEEDS}; got {invalid}.")
    return seeds


def resolve_vendor(value: str) -> str:
    """Resolve a vendor id, name, or domain name to A/B/C/D."""
    text = str(value).strip().lower()
    aliases: Dict[str, str] = {}
    for vendor, info in VENDOR_INFO.items():
        aliases[vendor.lower()] = vendor
        aliases[str(info["vendor_name"]).lower()] = vendor
        aliases[str(info["domain"]).lower()] = vendor
        aliases[f"vendor_{vendor.lower()}"] = vendor
        aliases[f"vendor {vendor.lower()}"] = vendor
    if text not in aliases:
        valid = ", ".join(CANONICAL_VENDORS)
        raise ValueError(f"Unknown MMS domain {value!r}; expected one of {valid}.")
    return aliases[text]


def parse_domains(text: str) -> List[str]:
    """Parse and validate a comma-separated subset of MMS domains."""
    domains = [resolve_vendor(value) for value in parse_csv_values(text)]
    if not domains:
        raise ValueError("--domains produced an empty list.")
    if len(set(domains)) != len(domains):
        raise ValueError("--domains must not contain duplicates.")
    return domains


def task_coordinates(task_id: int) -> tuple[str, int]:
    """Map the canonical task id 0..19 to source-domain and seed."""
    task_id = int(task_id)
    if not 0 <= task_id < NUM_EXPECTED_RUNS:
        raise ValueError(f"--task-id must be in [0, {NUM_EXPECTED_RUNS - 1}].")
    domain_index, seed_index = divmod(task_id, len(CANONICAL_SEEDS))
    return CANONICAL_VENDORS[domain_index], CANONICAL_SEEDS[seed_index]


def task_id_for_run(vendor: str, seed: int) -> int:
    """Return the canonical task id for a source-domain/seed pair."""
    resolved = resolve_vendor(vendor)
    if int(seed) not in CANONICAL_SEEDS:
        raise ValueError(f"Seed must be one of {CANONICAL_SEEDS}; got {seed}.")
    return CANONICAL_VENDORS.index(resolved) * len(CANONICAL_SEEDS) + int(seed)


def resolve_device(value: str) -> torch.device:
    """Resolve ``auto``, CPU, or CUDA and reject unavailable CUDA requests."""
    text = str(value).strip().lower()
    if text == "auto":
        text = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for one independent training run."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)


def finite_mean(values: Iterable[Any]) -> float:
    """Return the mean of finite numeric values, or NaN if none exist."""
    finite: List[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite.append(number)
    return float(sum(finite) / len(finite)) if finite else float("nan")


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a sibling temporary file."""
    output = Path(path)
    ensure_dir(output.parent)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output)


def atomic_write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    """Write a sequence of mappings to CSV through a temporary file."""
    output = Path(path)
    ensure_dir(output.parent)
    if fieldnames is None:
        discovered: List[str] = []
        for row in rows:
            for key in row:
                if key not in discovered:
                    discovered.append(key)
        fieldnames = discovered
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(dict(row) for row in rows)
    os.replace(temporary, output)


def atomic_torch_save(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically save a PyTorch checkpoint in the destination directory."""
    output = Path(path)
    ensure_dir(output.parent)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, output)


def _read_patient_partitions(dataset_root: Path, vendor: str) -> Dict[str, str]:
    manifest = dataset_root / "manifests" / "patients.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"MMS patient manifest not found: {manifest}")
    info = VENDOR_INFO[vendor]
    partitions: Dict[str, str] = {}
    with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if (
                str(row.get("vendor", "")).strip() == vendor
                and str(row.get("domain", "")).strip() == info["domain"]
            ):
                patient_id = str(row["patient_id"]).strip()
                partitions[patient_id] = str(row.get("original_part", "")).strip()
    if not partitions:
        raise RuntimeError(f"No manifest patients found for source domain {info['domain']}.")
    return partitions


class MMSFullyLabeledDomainDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """All mask-bearing 2-D slices from one normalized MMS vendor domain."""

    def __init__(
        self,
        vendor: str,
        dataset_root: str | Path = DEFAULT_DATASET_ROOT,
        *,
        processed_subdir: str | Path = PROCESSED_SUBDIR,
        validate_files: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.processed_subdir = Path(processed_subdir)
        self.vendor = resolve_vendor(vendor)
        info = VENDOR_INFO[self.vendor]
        self.vendor_name = info["vendor_name"]
        self.domain = info["domain"]

        loader = TestLoader(
            vendor=self.vendor,
            batch_size=1,
            shuffle_all_slices=False,
            seed=0,
            dataset_root=self.dataset_root,
            processed_subdir=self.processed_subdir,
        )
        self.records = tuple(loader.records)
        self.patient_ids = tuple(sorted(loader.patient_ids))
        self.slice_count = len(self.records)
        self.patient_partitions = _read_patient_partitions(self.dataset_root, self.vendor)
        self.partition_patient_counts = dict(
            sorted(Counter(self.patient_partitions.values()).items())
        )
        self.partition_slice_counts = dict(
            sorted(
                Counter(
                    self.patient_partitions.get(record.patient_id, "unknown")
                    for record in self.records
                ).items()
            )
        )

        record_patients = {record.patient_id for record in self.records}
        manifest_patients = set(self.patient_partitions)
        if record_patients != manifest_patients:
            raise RuntimeError(
                f"Manifest/metadata patients differ for {self.domain}: "
                f"missing={sorted(manifest_patients - record_patients)}, "
                f"extra={sorted(record_patients - manifest_patients)}"
            )
        if not self.records:
            raise RuntimeError(f"No labeled slices found for {self.domain}.")

        if validate_files:
            missing_images = [
                str(record.image_path) for record in self.records if not record.image_path.is_file()
            ]
            missing_masks = [
                str(record.mask_path)
                for record in self.records
                if record.mask_path is None or not record.mask_path.is_file()
            ]
            if missing_images or missing_masks:
                raise FileNotFoundError(
                    f"Missing MMS files for {self.domain}: "
                    f"images={missing_images[:5]}, masks={missing_masks[:5]}"
                )

    def __len__(self) -> int:
        return self.slice_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[int(index)]
        image_array = np.load(record.image_path).astype(np.float32, copy=False)
        if image_array.ndim != 2:
            raise ValueError(
                f"Expected a 2-D image at {record.image_path}, got {image_array.shape}."
            )
        if record.mask_path is None:
            raise ValueError(f"Slice {record.slice_id} has no mask path.")
        mask_array = np.load(record.mask_path).astype(np.int64, copy=False)
        if mask_array.ndim != 2:
            raise ValueError(
                f"Expected a 2-D mask at {record.mask_path}, got {mask_array.shape}."
            )
        if image_array.shape != mask_array.shape:
            raise ValueError(
                f"Image/mask shapes differ for {record.slice_id}: "
                f"{image_array.shape} vs {mask_array.shape}."
            )
        minimum = int(mask_array.min())
        maximum = int(mask_array.max())
        if minimum < 0 or maximum >= NUM_CLASSES:
            raise ValueError(
                f"Mask labels for {record.slice_id} must be in [0, {NUM_CLASSES - 1}], "
                f"got [{minimum}, {maximum}]."
            )
        image = torch.from_numpy(image_array.copy()).unsqueeze(0).float()
        mask = torch.from_numpy(mask_array.copy()).long()
        return image, mask


def seed_data_worker(_worker_id: int) -> None:
    """Seed NumPy/Python inside a PyTorch data-loader worker."""
    worker_seed = int(torch.initial_seed() % (2**32))
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def epoch_data_loader(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Create a deterministic, independently shuffled loader for one epoch."""
    generator = torch.Generator()
    generator.manual_seed(int(seed) * 1_000_003 + int(epoch))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=False,
        worker_init_fn=seed_data_worker,
        generator=generator,
    )


def build_model() -> UNet:
    """Build the U-Net used by the repository's existing TTA methods."""
    return UNet(n_channels=1, n_classes=NUM_CLASSES, only_feature=False, bilinear=False)


def model_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Extract logits from tensor, tuple/list, or mapping model output."""
    output = model(images)
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("The model returned an empty sequence.")
        logits = output[-1]
    elif isinstance(output, Mapping):
        if "logits" not in output:
            raise KeyError("A mapping model output must contain 'logits'.")
        logits = output["logits"]
    else:
        raise TypeError(f"Unsupported model output type: {type(output).__name__}.")
    if not isinstance(logits, torch.Tensor):
        raise TypeError("Extracted model logits must be a torch.Tensor.")
    return logits


def one_hot_mask(mask: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Convert ``[B,H,W]`` integer masks to ``[B,C,H,W]`` one-hot masks."""
    return F.one_hot(mask.long(), num_classes=int(num_classes)).permute(0, 3, 1, 2).float()


def soft_dice_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_classes: int = NUM_CLASSES,
    include_background: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute per-image foreground soft-Dice loss."""
    probabilities = torch.softmax(logits, dim=1)
    targets = one_hot_mask(mask, num_classes=num_classes)
    class_ids = (
        range(int(num_classes))
        if include_background
        else range(1, int(num_classes))
    )
    losses: List[torch.Tensor] = []
    for class_id in class_ids:
        prediction = probabilities[:, class_id]
        target = targets[:, class_id]
        intersection = (prediction * target).sum(dim=(1, 2))
        denominator = prediction.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        losses.append(1.0 - (2.0 * intersection + float(eps)) / (denominator + float(eps)))
    if not losses:
        return logits.sum() * 0.0
    return torch.cat([loss.reshape(-1) for loss in losses]).mean()


def supervised_segmentation_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    dice_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, cross-entropy, and foreground soft-Dice losses."""
    cross_entropy = F.cross_entropy(logits, mask.long())
    dice = soft_dice_loss(logits, mask, num_classes=NUM_CLASSES)
    total = cross_entropy + float(dice_weight) * dice
    return total, cross_entropy, dice


def run_directory(output_root: str | Path, vendor: str, seed: int) -> Path:
    """Return the required ``A/seed0`` output path for one backbone."""
    return Path(output_root) / resolve_vendor(vendor) / f"seed{int(seed)}"


def backbone_id(vendor: str, seed: int) -> str:
    """Return a stable human-readable id for a clean source backbone."""
    info = VENDOR_INFO[resolve_vendor(vendor)]
    return f"CleanMMS_{info['domain']}_Seed{int(seed)}_UNet"


def protocol_signature(args: argparse.Namespace, vendor: str, seed: int) -> Dict[str, Any]:
    """Fields that must match before an interrupted run can be resumed."""
    return {
        "method": "FullySupervisedMMS",
        "architecture": "UNet",
        "source_vendor": resolve_vendor(vendor),
        "source_domain": VENDOR_INFO[resolve_vendor(vendor)]["domain"],
        "seed": int(seed),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "processed_subdir": str(Path(args.processed_subdir)),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "dice_weight": float(args.dice_weight),
        "grad_clip": float(args.grad_clip),
        "max_steps_per_epoch": int(args.max_steps_per_epoch),
        "deterministic": not bool(args.allow_nondeterministic),
    }


def _training_state_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    log_rows: Sequence[Mapping[str, Any]],
    signature: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_log": [dict(row) for row in log_rows],
        "protocol_signature": dict(signature),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_training_state(
    *,
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_signature: Mapping[str, Any],
) -> tuple[int, List[Dict[str, Any]]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Resume checkpoint is not a mapping: {checkpoint_path}")
    stored_signature = checkpoint.get("protocol_signature")
    if stored_signature != dict(expected_signature):
        raise RuntimeError(
            "Refusing to resume with a different training protocol. "
            f"stored={stored_signature}, requested={dict(expected_signature)}"
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    random.setstate(checkpoint["python_random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"])
    cuda_state = checkpoint.get("cuda_random_state", [])
    if torch.cuda.is_available() and cuda_state:
        torch.cuda.set_rng_state_all(cuda_state)
    epoch = int(checkpoint.get("epoch", 0))
    rows = [dict(row) for row in checkpoint.get("training_log", [])]
    return epoch, rows


def _final_checkpoint_payload(
    *,
    model: nn.Module,
    args: argparse.Namespace,
    dataset: MMSFullyLabeledDomainDataset,
    vendor: str,
    seed: int,
    log_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    final_row = dict(log_rows[-1]) if log_rows else {}
    target_vendors = [value for value in CANONICAL_VENDORS if value != vendor]
    metadata: Dict[str, Any] = {
        "method": "FullySupervisedMMS",
        "primary_model": "model",
        "architecture": "UNet",
        "n_channels": 1,
        "num_classes": NUM_CLASSES,
        "label_map": LABEL_MAP,
        "backbone_id": backbone_id(vendor, seed),
        "seed": int(seed),
        "source_vendor": vendor,
        "source_vendor_name": dataset.vendor_name,
        "source_domain": dataset.domain,
        "target_vendors_for_tta": target_vendors,
        "target_domains_for_tta": [VENDOR_INFO[value]["domain"] for value in target_vendors],
        "uses_all_mask_bearing_source_patients": True,
        "source_patient_ids": list(dataset.patient_ids),
        "source_patient_count": len(dataset.patient_ids),
        "source_slice_count": len(dataset),
        "original_partition_patient_counts": dataset.partition_patient_counts,
        "original_partition_slice_counts": dataset.partition_slice_counts,
        "dataset_root": str(dataset.dataset_root.resolve()),
        "processed_subdir": str(dataset.processed_subdir),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "optimizer": "AdamW",
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "dice_weight": float(args.dice_weight),
        "grad_clip": float(args.grad_clip),
        "max_steps_per_epoch": int(args.max_steps_per_epoch),
        "debug_limited_training": int(args.max_steps_per_epoch) > 0,
        "deterministic": not bool(args.allow_nondeterministic),
        "final_epoch": int(final_row.get("epoch", 0) or 0),
        "final_train_loss": final_row.get("train_loss", float("nan")),
        "final_cross_entropy": final_row.get("cross_entropy", float("nan")),
        "final_dice_loss": final_row.get("dice_loss", float("nan")),
        "args": vars(args),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    return {
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
    }


def _install_baseline_alias(final_path: Path, baseline_path: Path) -> str:
    """Create the conventional baseline filename without duplicating if possible."""
    temporary = baseline_path.with_suffix(baseline_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    storage = "hardlink"
    try:
        os.link(final_path, temporary)
    except OSError:
        storage = "copy"
        shutil.copyfile(final_path, temporary)
    os.replace(temporary, baseline_path)
    return storage


@dataclass(frozen=True)
class RunResult:
    """Serializable result of one source-domain/seed training run."""

    status: str
    task_id: int
    source_vendor: str
    source_domain: str
    seed: int
    run_dir: str
    checkpoint: str
    source_patient_count: int
    source_slice_count: int
    epochs: int
    final_train_loss: float
    final_cross_entropy: float
    final_dice_loss: float
    best_train_loss: float
    debug_limited_training: bool

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _load_completed_result(completion_path: Path) -> RunResult:
    with completion_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    summary = payload.get("summary", payload)
    if not isinstance(summary, Mapping):
        raise TypeError(f"Completion summary is not a mapping: {completion_path}")
    return RunResult(
        status=str(summary["status"]),
        task_id=int(summary["task_id"]),
        source_vendor=str(summary["source_vendor"]),
        source_domain=str(summary["source_domain"]),
        seed=int(summary["seed"]),
        run_dir=str(summary["run_dir"]),
        checkpoint=str(summary["checkpoint"]),
        source_patient_count=int(summary["source_patient_count"]),
        source_slice_count=int(summary["source_slice_count"]),
        epochs=int(summary["epochs"]),
        final_train_loss=float(summary["final_train_loss"]),
        final_cross_entropy=float(summary["final_cross_entropy"]),
        final_dice_loss=float(summary["final_dice_loss"]),
        best_train_loss=float(summary["best_train_loss"]),
        debug_limited_training=bool(summary["debug_limited_training"]),
    )


def train_one_run(
    *,
    args: argparse.Namespace,
    device: torch.device,
    vendor: str,
    seed: int,
) -> RunResult:
    """Train one fully supervised source backbone."""
    vendor = resolve_vendor(vendor)
    seed = int(seed)
    set_seed(seed, deterministic=not bool(args.allow_nondeterministic))
    if int(args.cpu_threads) > 0:
        torch.set_num_threads(int(args.cpu_threads))

    run_path = run_directory(args.output_root, vendor, seed)
    completion_path = run_path / "completion.json"
    final_path = run_path / "checkpoint_final.pt"
    baseline_path = run_path / "baseline_model_with_metadata.pt"
    latest_path = run_path / "checkpoint_last.pt"

    if completion_path.is_file() and bool(args.resume) and not bool(args.overwrite):
        result = _load_completed_result(completion_path)
        if not Path(result.checkpoint).is_file():
            raise FileNotFoundError(
                f"Completion exists but its checkpoint is missing: {result.checkpoint}"
            )
        print(f"[SKIP] Domain{vendor} Seed{seed}: complete")
        return result

    protocol_files = (
        completion_path,
        final_path,
        baseline_path,
        latest_path,
        run_path / "training_log.csv",
        run_path / "run_config.json",
    )
    if (
        any(path.exists() for path in protocol_files)
        and not bool(args.resume)
        and not bool(args.overwrite)
    ):
        raise FileExistsError(
            f"Run output already exists at {run_path}; use --resume or --overwrite."
        )

    ensure_dir(run_path)
    if bool(args.overwrite):
        # A stale completion marker or resumable state must never survive the
        # start of a replacement run: a later --resume could otherwise accept
        # artifacts from the run being replaced.
        for path in (completion_path, latest_path, baseline_path, final_path):
            if path.exists():
                path.unlink()
    dataset = MMSFullyLabeledDomainDataset(
        vendor,
        dataset_root=Path(args.dataset_root),
        processed_subdir=Path(args.processed_subdir),
        validate_files=not bool(args.skip_file_validation),
    )
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    signature = protocol_signature(args, vendor, seed)
    log_rows: List[Dict[str, Any]] = []
    completed_epoch = 0

    if bool(args.resume) and latest_path.is_file() and not bool(args.overwrite):
        completed_epoch, log_rows = _restore_training_state(
            checkpoint_path=latest_path,
            model=model,
            optimizer=optimizer,
            expected_signature=signature,
        )
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        print(f"[RESUME] Domain{vendor} Seed{seed} from epoch {completed_epoch}")
    elif bool(args.resume) and not bool(args.overwrite) and any(
        path.exists() for path in protocol_files
    ):
        # A job interrupted before its first periodic checkpoint has no model
        # state worth restoring. Start that run again from epoch one; the
        # deterministic seed/epoch loader reproduces the same training order.
        print(
            f"[RESTART] Domain{vendor} Seed{seed}: incomplete run has no "
            "checkpoint_last.pt; restarting from epoch 1",
            flush=True,
        )

    config = {
        **signature,
        "task_id": task_id_for_run(vendor, seed),
        "run_dir": str(run_path.resolve()),
        "source_vendor_name": dataset.vendor_name,
        "source_patient_ids": list(dataset.patient_ids),
        "source_patient_count": len(dataset.patient_ids),
        "source_slice_count": len(dataset),
        "original_partition_patient_counts": dataset.partition_patient_counts,
        "original_partition_slice_counts": dataset.partition_slice_counts,
        "uses_all_mask_bearing_source_patients": True,
        "target_vendors_for_tta": [
            value for value in CANONICAL_VENDORS if value != vendor
        ],
        "epochs": int(args.epochs),
        "num_workers": int(args.num_workers),
        "max_steps_per_epoch": int(args.max_steps_per_epoch),
        "debug_limited_training": int(args.max_steps_per_epoch) > 0,
        "device": str(device),
        "args": vars(args),
    }
    atomic_write_json(run_path / "run_config.json", config)

    print(
        f"[TRAIN] task={task_id_for_run(vendor, seed)} Domain{vendor} "
        f"({dataset.domain}) Seed{seed} patients={len(dataset.patient_ids)} "
        f"slices={len(dataset)} device={device}",
        flush=True,
    )
    for epoch in range(completed_epoch + 1, int(args.epochs) + 1):
        model.train()
        loader = epoch_data_loader(
            dataset,
            batch_size=int(args.batch_size),
            seed=seed,
            epoch=epoch,
            num_workers=int(args.num_workers),
            pin_memory=device.type == "cuda",
        )
        total_losses: List[float] = []
        cross_entropies: List[float] = []
        dice_losses: List[float] = []
        steps = 0

        for step, (images, masks) in enumerate(loader, start=1):
            if int(args.max_steps_per_epoch) > 0 and step > int(args.max_steps_per_epoch):
                break
            images = images.to(device, non_blocking=device.type == "cuda")
            masks = masks.to(device, non_blocking=device.type == "cuda")
            logits = model_logits(model, images)
            loss, cross_entropy, dice = supervised_segmentation_loss(
                logits,
                masks,
                dice_weight=float(args.dice_weight),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()

            steps += 1
            total_losses.append(float(loss.detach().cpu()))
            cross_entropies.append(float(cross_entropy.detach().cpu()))
            dice_losses.append(float(dice.detach().cpu()))

        if steps == 0:
            raise RuntimeError(f"Epoch {epoch} produced no optimizer steps.")
        row: Dict[str, Any] = {
            "method": "FullySupervisedMMS",
            "task_id": task_id_for_run(vendor, seed),
            "source_vendor": vendor,
            "source_domain": dataset.domain,
            "seed": seed,
            "epoch": epoch,
            "steps": steps,
            "train_loss": finite_mean(total_losses),
            "cross_entropy": finite_mean(cross_entropies),
            "dice_loss": finite_mean(dice_losses),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "source_patient_count": len(dataset.patient_ids),
            "source_slice_count": len(dataset),
        }
        log_rows.append(row)
        atomic_write_csv(run_path / "training_log.csv", log_rows)

        should_checkpoint = (
            epoch == int(args.epochs)
            or epoch % int(args.checkpoint_every) == 0
        )
        if should_checkpoint:
            atomic_torch_save(
                latest_path,
                _training_state_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    log_rows=log_rows,
                    signature=signature,
                ),
            )
        print(
            f"  epoch {epoch:03d}/{int(args.epochs):03d} steps={steps:04d} "
            f"loss={row['train_loss']:.6f} ce={row['cross_entropy']:.6f} "
            f"dice={row['dice_loss']:.6f}",
            flush=True,
        )

    if not log_rows:
        raise RuntimeError("Training finished without any logged epochs.")
    payload = _final_checkpoint_payload(
        model=model,
        args=args,
        dataset=dataset,
        vendor=vendor,
        seed=seed,
        log_rows=log_rows,
    )
    atomic_torch_save(final_path, payload)
    alias_storage = _install_baseline_alias(final_path, baseline_path)
    if latest_path.exists():
        latest_path.unlink()

    final_row = log_rows[-1]
    result = RunResult(
        status="complete",
        task_id=task_id_for_run(vendor, seed),
        source_vendor=vendor,
        source_domain=dataset.domain,
        seed=seed,
        run_dir=str(run_path.resolve()),
        checkpoint=str(baseline_path.resolve()),
        source_patient_count=len(dataset.patient_ids),
        source_slice_count=len(dataset),
        epochs=int(final_row["epoch"]),
        final_train_loss=float(final_row["train_loss"]),
        final_cross_entropy=float(final_row["cross_entropy"]),
        final_dice_loss=float(final_row["dice_loss"]),
        best_train_loss=min(float(row["train_loss"]) for row in log_rows),
        debug_limited_training=int(args.max_steps_per_epoch) > 0,
    )
    atomic_write_json(
        completion_path,
        {
            "status": "complete",
            "checkpoint_alias_storage": alias_storage,
            "summary": result.as_dict(),
        },
    )
    print(
        f"[COMPLETE] Domain{vendor} Seed{seed} "
        f"loss={result.final_train_loss:.6f} checkpoint={baseline_path}",
        flush=True,
    )
    return result


RUN_SUMMARY_FIELDS = (
    "status",
    "task_id",
    "source_vendor",
    "source_domain",
    "seed",
    "source_patient_count",
    "source_slice_count",
    "epochs",
    "final_train_loss",
    "final_cross_entropy",
    "final_dice_loss",
    "best_train_loss",
    "debug_limited_training",
    "checkpoint",
    "run_dir",
)


def rebuild_run_summary(output_root: str | Path) -> List[Dict[str, Any]]:
    """Rebuild the global summary safely while array tasks finish concurrently."""
    root = Path(output_root)
    ensure_dir(root)
    lock_path = root / ".run_summary.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        rows: List[Dict[str, Any]] = []
        for vendor in CANONICAL_VENDORS:
            for seed in CANONICAL_SEEDS:
                completion = run_directory(root, vendor, seed) / "completion.json"
                if not completion.is_file():
                    continue
                result = _load_completed_result(completion)
                rows.append(result.as_dict())
        rows.sort(key=lambda row: int(row["task_id"]))
        atomic_write_csv(root / "run_summary.csv", rows, RUN_SUMMARY_FIELDS)
        missing = sorted(
            set(range(NUM_EXPECTED_RUNS)) - {int(row["task_id"]) for row in rows}
        )
        atomic_write_json(
            root / "matrix_status.json",
            {
                "expected_runs": NUM_EXPECTED_RUNS,
                "completed_runs": len(rows),
                "complete": len(rows) == NUM_EXPECTED_RUNS,
                "missing_task_ids": missing,
            },
        )
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return rows


def validate_args(args: argparse.Namespace) -> tuple[List[str], List[int]]:
    """Validate CLI arguments and return selected domains and seeds."""
    domains = parse_domains(args.domains)
    seeds = parse_seeds(args.seeds)
    if bool(args.resume) and bool(args.overwrite):
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if int(args.epochs) <= 0:
        raise ValueError("--epochs must be positive.")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive.")
    if float(args.learning_rate) <= 0.0:
        raise ValueError("--learning-rate must be positive.")
    if float(args.weight_decay) < 0.0:
        raise ValueError("--weight-decay must be non-negative.")
    if float(args.dice_weight) < 0.0:
        raise ValueError("--dice-weight must be non-negative.")
    if float(args.grad_clip) < 0.0:
        raise ValueError("--grad-clip must be non-negative.")
    if int(args.num_workers) < 0:
        raise ValueError("--num-workers must be non-negative.")
    if int(args.max_steps_per_epoch) < 0:
        raise ValueError("--max-steps-per-epoch must be non-negative.")
    if int(args.checkpoint_every) <= 0:
        raise ValueError("--checkpoint-every must be positive.")
    if int(args.cpu_threads) < 0:
        raise ValueError("--cpu-threads must be non-negative.")
    if args.task_id is not None:
        task_coordinates(int(args.task_id))
    dataset_root = Path(args.dataset_root)
    if not bool(args.aggregate_only) and not dataset_root.is_dir():
        raise FileNotFoundError(f"MMS dataset root does not exist: {dataset_root}")
    return domains, seeds


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train 4 MMS source domains x 5 seeds as fully supervised clean TTA backbones."
        )
    )
    parser.add_argument(
        "--domains",
        default=",".join(CANONICAL_VENDORS),
        help="Comma-separated source domains; default: A,B,C,D.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in CANONICAL_SEEDS),
        help="Comma-separated seeds; default: 0,1,2,3,4.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        help="Run one canonical task in [0,19], ordered A0..A4,B0..B4,C0..C4,D0..D4.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only rebuild run_summary.csv and matrix_status.json.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save resumable state every N epochs and at the final epoch.",
    )
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=0,
        help="Debug limit; 0 uses every labeled source slice.",
    )
    parser.add_argument("--cpu-threads", type=int, default=0, help="0 keeps the PyTorch default.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--processed-subdir", default=str(PROCESSED_SUBDIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--allow-nondeterministic",
        action="store_true",
        help="Allow faster non-deterministic CUDA kernels.",
    )
    parser.add_argument(
        "--skip-file-validation",
        action="store_true",
        help="Skip the up-front existence check for every source image and mask.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    domains, seeds = validate_args(args)
    ensure_dir(args.output_root)

    if bool(args.aggregate_only):
        rows = rebuild_run_summary(args.output_root)
        print(
            f"[AGGREGATE] {len(rows)}/{NUM_EXPECTED_RUNS} completed runs: "
            f"{Path(args.output_root) / 'run_summary.csv'}"
        )
        return

    device = resolve_device(args.device)
    if args.task_id is not None:
        runs = [task_coordinates(int(args.task_id))]
    else:
        runs = [(vendor, seed) for vendor in domains for seed in seeds]

    print(
        f"[DEVICE] {device} runs={len(runs)} deterministic={not args.allow_nondeterministic}",
        flush=True,
    )
    for vendor, seed in runs:
        train_one_run(
            args=args,
            device=device,
            vendor=vendor,
            seed=seed,
        )
        rebuild_run_summary(args.output_root)

    rows = rebuild_run_summary(args.output_root)
    print(
        f"[DONE] canonical matrix {len(rows)}/{NUM_EXPECTED_RUNS}: "
        f"{Path(args.output_root) / 'run_summary.csv'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
