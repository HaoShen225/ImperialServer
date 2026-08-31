"""M&Ms data semantics and the authoritative streaming protocol."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    return _read_json(cfg["data"]["protocol_file"])


def _resolve_dataset_path(cfg: dict[str, Any], relative: str) -> Path:
    return Path(cfg["data"]["root"]) / relative


class MMSSourceDataset(Dataset):
    """Vendor-A slice dataset. It intentionally performs no augmentation."""

    def __init__(self, records: Sequence[dict[str, str]], data_root: str | Path):
        self.records = list(records)
        self.data_root = Path(data_root)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        image = np.load(self.data_root / row["image"], allow_pickle=False)
        mask = np.load(self.data_root / row["mask"], allow_pickle=False)
        return {
            "image": torch.from_numpy(image).float().unsqueeze(0),
            "mask": torch.from_numpy(mask.astype(np.int64, copy=False)),
            "patient_id": row["patient_id"],
            "phase": row["phase"],
            "z_index": int(row["z_index"]),
        }


def _source_records(cfg: dict[str, Any], split: str) -> list[dict[str, str]]:
    if split not in {"train", "val"}:
        raise ValueError(f"Unknown source split: {split}")
    protocol = load_protocol(cfg)
    patient_ids = set(protocol["source"][split]["patient_ids"])
    rows = _read_csv(cfg["data"]["slices_manifest"])
    records = [row for row in rows if row["vendor"] == "A" and row["patient_id"] in patient_ids]
    records.sort(key=lambda row: (row["patient_id"], 0 if row["phase"] == "ED" else 1, int(row["z_index"])))
    return records


def build_source_loaders(
    cfg: dict[str, Any],
    seed: int,
    batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    data_root = cfg["data"]["root"]
    train = MMSSourceDataset(_source_records(cfg, "train"), data_root)
    val = MMSSourceDataset(_source_records(cfg, "val"), data_root)
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "batch_size": batch_size or int(cfg["source"]["batch_size"]),
        "num_workers": int(cfg["data"]["num_workers"]),
        "pin_memory": False,
    }
    return (
        DataLoader(train, shuffle=True, generator=generator, **loader_kwargs),
        DataLoader(val, shuffle=False, **loader_kwargs),
    )


class MMSTargetVolumeDataset(Dataset):
    """Image-only target volumes; masks are loaded explicitly after inference."""

    def __init__(self, volumes: Sequence[dict[str, Any]], data_root: str | Path):
        self.volumes = list(volumes)
        self.data_root = Path(data_root)

    def __len__(self) -> int:
        return len(self.volumes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        volume = self.volumes[index]
        images = [np.load(self.data_root / row["image"], allow_pickle=False) for row in volume["slices"]]
        return {
            "volume_id": volume["volume_id"],
            "patient_id": volume["patient_id"],
            "phase": volume["phase"],
            "vendor": volume["vendor"],
            "image": torch.from_numpy(np.stack(images)).float().unsqueeze(1),
            "mask_paths": [row["mask"] for row in volume["slices"]],
            "z_indices": [int(row["z_index"]) for row in volume["slices"]],
        }

    def load_mask(self, volume: dict[str, Any]) -> torch.Tensor:
        masks = [np.load(self.data_root / path, allow_pickle=False) for path in volume["mask_paths"]]
        return torch.from_numpy(np.stack(masks).astype(np.int64, copy=False))


def _volume_index(cfg: dict[str, Any]) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(cfg["data"]["slices_manifest"]):
        grouped[(row["vendor"], row["patient_id"], row["phase"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["z_index"]))
    return grouped


def build_target_stream(vendor: str, cfg: dict[str, Any]) -> MMSTargetVolumeDataset:
    stream_cfg = _read_json(cfg["data"]["stream_file"])
    if vendor not in stream_cfg["target_vendors"]:
        raise ValueError(f"Vendor {vendor!r} is not in the authoritative target stream")
    protocol = load_protocol(cfg)
    target_cfg = protocol["targets"][vendor]
    patients = target_cfg["patient_ids"]
    excluded_parts = set(stream_cfg.get("excluded_original_parts", []))
    index = _volume_index(cfg)
    volumes: list[dict[str, Any]] = []
    for patient_id in patients:
        patient_volumes: list[dict[str, Any]] = []
        original_parts: set[str] = set()
        for phase in stream_cfg["phase_order"]:
            slices = index.get((vendor, patient_id, phase))
            if not slices:
                raise ValueError(f"Missing target volume {vendor}/{patient_id}/{phase}")
            phase_parts = {row["original_part"] for row in slices}
            if len(phase_parts) != 1:
                raise ValueError(f"Mixed original parts in target volume {vendor}/{patient_id}/{phase}")
            original_parts.update(phase_parts)
            patient_volumes.append({
                "volume_id": f"{patient_id}_{phase}",
                "patient_id": patient_id,
                "phase": phase,
                "vendor": vendor,
                "slices": slices,
            })
        if original_parts & excluded_parts:
            continue
        if len(original_parts) != 1:
            raise ValueError(f"Mixed original parts across target patient {vendor}/{patient_id}")
        volumes.extend(patient_volumes)
    patient_count = len({volume["patient_id"] for volume in volumes})
    expected_count = int(target_cfg["counts"]["patients"])
    if patient_count != expected_count:
        raise ValueError(
            f"Target stream count mismatch for vendor {vendor}: expected {expected_count}, got {patient_count}"
        )
    return MMSTargetVolumeDataset(volumes, cfg["data"]["root"])


def build_source_validation_volumes(cfg: dict[str, Any]) -> MMSTargetVolumeDataset:
    records = _source_records(cfg, "val")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in records:
        grouped[(row["patient_id"], row["phase"])].append(row)
    volumes = []
    for (patient_id, phase), slices in sorted(grouped.items(), key=lambda item: (item[0][0], 0 if item[0][1] == "ED" else 1)):
        slices.sort(key=lambda row: int(row["z_index"]))
        volumes.append({"volume_id": f"{patient_id}_{phase}", "patient_id": patient_id, "phase": phase, "vendor": "A", "slices": slices})
    return MMSTargetVolumeDataset(volumes, cfg["data"]["root"])


def split_volume_into_batches(images: torch.Tensor, batch_size: int) -> Iterable[torch.Tensor]:
    if images.ndim != 4:
        raise ValueError(f"Expected [Z,C,H,W], got {tuple(images.shape)}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, images.shape[0], batch_size):
        yield images[start : start + batch_size]
