from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence

import numpy as np
import torch


ACDC_ROOT = Path(r"D:\Running_Place\PostGraduateProjects\dataset\Cardiac\ACDC_normalized")
MMS_ROOT = Path(r"D:\Running_Place\PostGraduateProjects\dataset\Cardiac\MMS_normalized")
PROCESSED_SUBDIR = Path("processed") / "2d_1p5mm_256"

FOREGROUND_CLASSES = (1, 2, 3)
FOREGROUND_CLASS_NAMES: Mapping[int, str] = {
    1: "rv",
    2: "myo",
    3: "lv",
}

PHASE_ORDER: Mapping[str, int] = {
    "ED": 0,
    "ES": 1,
}

VENDOR_INFO: Mapping[str, Dict[str, str]] = {
    "A": {"vendor": "A", "vendor_name": "Siemens", "domain": "vendor_A_Siemens"},
    "B": {"vendor": "B", "vendor_name": "Philips", "domain": "vendor_B_Philips"},
    "C": {"vendor": "C", "vendor_name": "GE", "domain": "vendor_C_GE"},
    "D": {"vendor": "D", "vendor_name": "Canon", "domain": "vendor_D_Canon"},
}


@dataclass(frozen=True)
class SliceRecord:
    slice_id: str
    patient_id: str
    image_path: Path
    mask_path: Path | None
    phase: str
    z_index: int
    group: str = ""
    vendor: str = ""
    vendor_name: str = ""
    domain: str = ""
    has_fg: int | None = None
    fg_pixels: int | None = None
    rv_pixels: int | None = None
    myo_pixels: int | None = None
    lv_pixels: int | None = None

    def meta(self, *, include_mask_path: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "slice_id": self.slice_id,
            "patient_id": self.patient_id,
            "phase": self.phase,
            "z_index": int(self.z_index),
            "image_path": str(self.image_path),
        }
        if include_mask_path and self.mask_path is not None:
            out["mask_path"] = str(self.mask_path)
        if self.group:
            out["group"] = self.group
        if self.vendor:
            out["vendor"] = self.vendor
        if self.vendor_name:
            out["vendor_name"] = self.vendor_name
        if self.domain:
            out["domain"] = self.domain
        for name in ("has_fg", "fg_pixels", "rv_pixels", "myo_pixels", "lv_pixels"):
            value = getattr(self, name)
            if value is not None:
                out[name] = int(value)
        return out


def _read_csv(path: str | Path) -> List[Dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _phase_key(phase: Any) -> tuple[int, str]:
    text = str(phase).strip().upper()
    return (PHASE_ORDER.get(text, 99), text)


def _record_key(record: SliceRecord) -> tuple[str, tuple[int, str], int, str]:
    return (record.patient_id, _phase_key(record.phase), int(record.z_index), record.slice_id)


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except Exception:
        return default


def _resolve_dataset_path(dataset_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return dataset_root / path


def _load_image_tensor(path: Path) -> torch.Tensor:
    arr = np.load(path).astype(np.float32, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image array at {path}, got shape {arr.shape}")
    return torch.from_numpy(arr)[None, ...].float()


def _load_mask_tensor(path: Path) -> torch.Tensor:
    arr = np.load(path).astype(np.int64, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask array at {path}, got shape {arr.shape}")
    return torch.from_numpy(arr).long()


def _stack_images(records: Sequence[SliceRecord]) -> torch.Tensor:
    return torch.stack([_load_image_tensor(record.image_path) for record in records], dim=0)


def _stack_masks(records: Sequence[SliceRecord]) -> torch.Tensor:
    masks: List[torch.Tensor] = []
    for record in records:
        if record.mask_path is None:
            raise ValueError(f"Record {record.slice_id} has no mask_path")
        masks.append(_load_mask_tensor(record.mask_path))
    return torch.stack(masks, dim=0)


def _take_cyclic(records: Sequence[SliceRecord], start: int, count: int) -> List[SliceRecord]:
    if count <= 0:
        return []
    if not records:
        raise RuntimeError("Cannot sample from an empty record list.")
    n = len(records)
    return [records[(start + i) % n] for i in range(count)]


class TrainLoader:
    """ACDC semi-supervised source-domain loader with fixed labeled/unlabeled batches."""

    def __init__(
        self,
        labeled_cases_per_class: int,
        seed: int,
        batch_size: int,
        labeled_batch_size: int,
        dataset_root: str | Path = ACDC_ROOT,
        *,
        processed_subdir: str | Path = PROCESSED_SUBDIR,
        shuffle_labeled: bool = True,
        shuffle_unlabeled: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.processed_subdir = Path(processed_subdir)
        self.labeled_cases_per_class = int(labeled_cases_per_class)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.labeled_batch_size = int(labeled_batch_size)
        self.unlabeled_batch_size = self.batch_size - self.labeled_batch_size
        self.shuffle_labeled = bool(shuffle_labeled)
        self.shuffle_unlabeled = bool(shuffle_unlabeled)
        self._epoch = 0

        if self.labeled_cases_per_class <= 0:
            raise ValueError("labeled_cases_per_class must be positive.")
        if self.batch_size <= 1:
            raise ValueError("batch_size must be greater than 1.")
        if self.labeled_batch_size <= 0 or self.labeled_batch_size >= self.batch_size:
            raise ValueError("labeled_batch_size must satisfy 0 < labeled_batch_size < batch_size.")

        patients_by_group = self._load_train_patients_by_group()
        self.labeled_patients_by_group = self._select_labeled_patients(patients_by_group)
        self.labeled_patients = [
            patient_id
            for group in sorted(self.labeled_patients_by_group)
            for patient_id in self.labeled_patients_by_group[group]
        ]
        labeled_set = set(self.labeled_patients)
        all_train_patients = sorted([patient_id for ids in patients_by_group.values() for patient_id in ids])
        self.unlabeled_patients = [patient_id for patient_id in all_train_patients if patient_id not in labeled_set]

        records = self._load_acdc_train_records(set(all_train_patients))
        self.labeled_records = [record for record in records if record.patient_id in labeled_set]
        self.unlabeled_records = [record for record in records if record.patient_id not in labeled_set]
        self.labeled_slice_count = len(self.labeled_records)
        self.unlabeled_slice_count = len(self.unlabeled_records)

        if not self.labeled_records:
            raise RuntimeError("No labeled slices were found.")
        if not self.unlabeled_records:
            raise RuntimeError("No unlabeled slices were found.")

    def _load_train_patients_by_group(self) -> Dict[str, List[str]]:
        rows = _read_csv(self.dataset_root / "manifests" / "patients.csv")
        groups: Dict[str, List[str]] = {}
        for row in rows:
            if str(row.get("output_split", "")).strip().lower() != "train":
                continue
            patient_id = str(row["patient_id"]).strip()
            group = str(row["group"]).strip()
            groups.setdefault(group, []).append(patient_id)
        if not groups:
            raise RuntimeError(f"No ACDC train patients found in {self.dataset_root}")
        return {group: sorted(patient_ids) for group, patient_ids in sorted(groups.items())}

    def _select_labeled_patients(self, patients_by_group: Mapping[str, Sequence[str]]) -> Dict[str, List[str]]:
        selected: Dict[str, List[str]] = {}
        for group in sorted(patients_by_group):
            candidates = list(patients_by_group[group])
            if len(candidates) < self.labeled_cases_per_class:
                raise ValueError(
                    f"Group {group} has only {len(candidates)} train patients, "
                    f"cannot select {self.labeled_cases_per_class}."
                )
            rng = random.Random(self.seed)
            rng.shuffle(candidates)
            selected[group] = candidates[: self.labeled_cases_per_class]
        return selected

    def _load_acdc_train_records(self, allowed_patients: set[str]) -> List[SliceRecord]:
        rows = _read_jsonl(self.dataset_root / self.processed_subdir / "metadata.jsonl")
        records: List[SliceRecord] = []
        for row in rows:
            if str(row.get("output_split", "")).strip().lower() != "train":
                continue
            patient_id = str(row["patient_id"]).strip()
            if patient_id not in allowed_patients:
                continue
            records.append(
                SliceRecord(
                    slice_id=str(row["slice_id"]),
                    patient_id=patient_id,
                    image_path=_resolve_dataset_path(self.dataset_root, row["image"]),
                    mask_path=_resolve_dataset_path(self.dataset_root, row["mask"]),
                    phase=str(row.get("phase", "")),
                    z_index=int(row.get("z_index", 0)),
                    group=str(row.get("group", "")),
                    has_fg=_safe_int(row.get("has_fg")),
                    fg_pixels=_safe_int(row.get("fg_pixels")),
                    rv_pixels=_safe_int(row.get("rv_pixels")),
                    myo_pixels=_safe_int(row.get("myo_pixels")),
                    lv_pixels=_safe_int(row.get("lv_pixels")),
                )
            )
        records = sorted(records, key=_record_key)
        if not records:
            raise RuntimeError(f"No ACDC train slice records found under {self.dataset_root}")
        return records

    def __len__(self) -> int:
        return int(math.ceil(len(self.unlabeled_records) / float(self.unlabeled_batch_size)))

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        epoch_seed = self.seed + self._epoch
        self._epoch += 1

        labeled_records = list(self.labeled_records)
        unlabeled_records = list(self.unlabeled_records)
        if self.shuffle_labeled:
            random.Random(epoch_seed + 1000003).shuffle(labeled_records)
        if self.shuffle_unlabeled:
            random.Random(epoch_seed).shuffle(unlabeled_records)

        labeled_pos = 0
        unlabeled_pos = 0
        for _step in range(len(self)):
            labeled = _take_cyclic(labeled_records, labeled_pos, self.labeled_batch_size)
            unlabeled = _take_cyclic(unlabeled_records, unlabeled_pos, self.unlabeled_batch_size)
            labeled_pos += self.labeled_batch_size
            unlabeled_pos += self.unlabeled_batch_size
            yield {
                "labeled_images": _stack_images(labeled),
                "labeled_masks": _stack_masks(labeled),
                "unlabeled_images": _stack_images(unlabeled),
                "labeled_meta": [record.meta(include_mask_path=True) for record in labeled],
                "unlabeled_meta": [record.meta(include_mask_path=False) for record in unlabeled],
            }


class TestLoader:
    """MMS target-domain loader grouped by MRI vendor."""

    def __init__(
        self,
        vendor: str,
        batch_size: int,
        shuffle_all_slices: bool = False,
        seed: int = 0,
        dataset_root: str | Path = MMS_ROOT,
        *,
        processed_subdir: str | Path = PROCESSED_SUBDIR,
        drop_last: bool = False,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.processed_subdir = Path(processed_subdir)
        self.batch_size = int(batch_size)
        self.shuffle_all_slices = bool(shuffle_all_slices)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        info = self._resolve_vendor(vendor)
        self.vendor = info["vendor"]
        self.vendor_name = info["vendor_name"]
        self.domain = info["domain"]
        self.domain_dir = self.dataset_root / self.processed_subdir / self.domain

        manifest_patients = self._manifest_patients()
        self.records = self._load_mms_records(manifest_patients)
        if self.shuffle_all_slices:
            self.records = list(self.records)
            random.Random(self.seed).shuffle(self.records)
        else:
            self.records = sorted(self.records, key=_record_key)
        self.patient_ids = sorted({record.patient_id for record in self.records})
        self.slice_count = len(self.records)

    def _resolve_vendor(self, vendor: str) -> Dict[str, str]:
        text = str(vendor).strip()
        key = text.lower()
        aliases: Dict[str, str] = {}
        for vendor_id, info in VENDOR_INFO.items():
            aliases[vendor_id.lower()] = vendor_id
            aliases[info["vendor_name"].lower()] = vendor_id
            aliases[info["domain"].lower()] = vendor_id
            aliases[f"vendor_{vendor_id.lower()}"] = vendor_id
            aliases[f"vendor {vendor_id.lower()}"] = vendor_id
        if key not in aliases:
            valid = ", ".join(sorted({v for v in aliases}))
            raise ValueError(f"Unknown MMS vendor {vendor!r}. Valid aliases include: {valid}")
        return dict(VENDOR_INFO[aliases[key]])

    def _manifest_patients(self) -> set[str]:
        rows = _read_csv(self.dataset_root / "manifests" / "patients.csv")
        patient_ids = {
            str(row["patient_id"]).strip()
            for row in rows
            if str(row.get("vendor", "")).strip() == self.vendor
            and str(row.get("domain", "")).strip() == self.domain
        }
        if not patient_ids:
            raise RuntimeError(f"No MMS manifest patients found for vendor {self.vendor} ({self.domain})")
        return patient_ids

    def _load_mms_records(self, manifest_patients: set[str]) -> List[SliceRecord]:
        rows = _read_jsonl(self.domain_dir / "metadata.jsonl")
        records: List[SliceRecord] = []
        metadata_patients: set[str] = set()
        for row in rows:
            patient_id = str(row["patient_id"]).strip()
            metadata_patients.add(patient_id)
            if patient_id not in manifest_patients:
                continue
            records.append(
                SliceRecord(
                    slice_id=str(row["slice_id"]),
                    patient_id=patient_id,
                    image_path=_resolve_dataset_path(self.dataset_root, row["image"]),
                    mask_path=_resolve_dataset_path(self.dataset_root, row["mask"]),
                    phase=str(row.get("phase", "")),
                    z_index=int(row.get("z_index", 0)),
                    group=str(row.get("pathology", "")),
                    vendor=str(row.get("vendor", self.vendor)),
                    vendor_name=str(row.get("vendor_name", self.vendor_name)),
                    domain=str(row.get("domain", self.domain)),
                    has_fg=_safe_int(row.get("has_fg")),
                    fg_pixels=_safe_int(row.get("fg_pixels")),
                    rv_pixels=_safe_int(row.get("rv_pixels")),
                    myo_pixels=_safe_int(row.get("myo_pixels")),
                    lv_pixels=_safe_int(row.get("lv_pixels")),
                )
            )
        missing_in_metadata = manifest_patients - metadata_patients
        extra_in_metadata = metadata_patients - manifest_patients
        if missing_in_metadata or extra_in_metadata:
            raise RuntimeError(
                f"MMS metadata/manifest mismatch for {self.domain}: "
                f"missing={sorted(missing_in_metadata)}, extra={sorted(extra_in_metadata)}"
            )
        if not records:
            raise RuntimeError(f"No MMS records found for {self.domain}")
        return records

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.records) // self.batch_size
        return int(math.ceil(len(self.records) / float(self.batch_size)))

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        limit = len(self.records)
        if self.drop_last:
            limit = (limit // self.batch_size) * self.batch_size
        for start in range(0, limit, self.batch_size):
            records = self.records[start : start + self.batch_size]
            if not records:
                continue
            yield {
                "images": _stack_images(records),
                "masks": _stack_masks(records),
                "meta": [record.meta(include_mask_path=True) for record in records],
            }


__all__ = [
    "ACDC_ROOT",
    "MMS_ROOT",
    "FOREGROUND_CLASSES",
    "FOREGROUND_CLASS_NAMES",
    "TrainLoader",
    "TestLoader",
]
