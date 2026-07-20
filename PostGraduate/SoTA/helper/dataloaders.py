from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from torch.utils.data import DataLoader, Dataset


DATA_ROOT = Path(r"D:\Running_Place\PostGraduateProjects\dataset\Spine\SPIDER_domain_strict_3Foreground")
SPLIT_ROOT = DATA_ROOT / "split" / "PostGraduateProject"
DEFAULT_SPLIT_CSV = SPLIT_ROOT / "spider_symphonytim_t1_fewshot_source_to_5domains_seed0-4.csv"

SOURCE_DOMAIN_NAME = "SIEMENS_SymphonyTim_37743_T1-TSE"
TARGET_DOMAIN_NAMES = (
    "Philips_Medical_Systems_Ingenia_70714_T1-TSE",
    "Philips_Medical_Systems_Ingenia_70714_T2-TSE",
    "SIEMENS_Aera_141639_T1-TSE",
    "SIEMENS_Aera_141639_T2-TSE",
    "SIEMENS_SymphonyTim_37743_T2-TSE",
)

DOMAIN_NAMES = (
    "Philips_Medical_Systems_Ingenia_70714_T1-TSE",
    "Philips_Medical_Systems_Ingenia_70714_T2-TSE",
    "SIEMENS_Aera_141639_T1-TSE",
    "SIEMENS_Aera_141639_T2-TSE",
    SOURCE_DOMAIN_NAME,
    "SIEMENS_SymphonyTim_37743_T2-TSE",
)

DOMAIN_ALIASES: Mapping[str, str] = {
    "philips_t1": "Philips_Medical_Systems_Ingenia_70714_T1-TSE",
    "philips_t2": "Philips_Medical_Systems_Ingenia_70714_T2-TSE",
    "aera_t1": "SIEMENS_Aera_141639_T1-TSE",
    "siemens_aera_t1": "SIEMENS_Aera_141639_T1-TSE",
    "aera_t2": "SIEMENS_Aera_141639_T2-TSE",
    "siemens_aera_t2": "SIEMENS_Aera_141639_T2-TSE",
    "symphonytim_t1": SOURCE_DOMAIN_NAME,
    "siemens_symphonytim_t1": SOURCE_DOMAIN_NAME,
    "symphonytim_t2": "SIEMENS_SymphonyTim_37743_T2-TSE",
    "siemens_symphonytim_t2": "SIEMENS_SymphonyTim_37743_T2-TSE",
    **{name.lower(): name for name in DOMAIN_NAMES},
}

VERTEBRAE_LABELS = tuple(list(range(1, 26)) + list(range(101, 126)))
DISC_LABELS = tuple(range(201, 226))
FOREGROUND_CLASSES = (1, 2)
FOREGROUND_CLASS_NAMES: Mapping[int, str] = {
    1: "vertebrae",
    2: "intervertebral_discs",
}


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    file: str
    image_path: Path
    mask_path: Path
    domain: str
    split_role: str = ""


def _numeric_key(text: str) -> tuple[int, str]:
    value = str(text)
    return (0, f"{int(value):012d}") if value.isdigit() else (1, value)


def normalize_case_id(value: Any) -> str:
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return str(value).strip()


def parse_case_id(stem: str) -> str:
    return normalize_case_id(str(stem).split("_", 1)[0])


def resolve_domain_name(domain: str | Path) -> str:
    text = Path(domain).name if Path(domain).exists() else str(domain).strip()
    return DOMAIN_ALIASES.get(text.lower(), text)


def resolve_domain(domain: str | Path, data_root: str | Path = DATA_ROOT) -> Path:
    candidate = Path(domain)
    if candidate.exists():
        return candidate.resolve()

    root = Path(data_root)
    text = str(domain).strip()
    names = [text]
    alias = DOMAIN_ALIASES.get(text.lower())
    if alias is not None:
        names.insert(0, alias)
    for name in names:
        path = root / name
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Unknown SPIDER domain {domain!r} under {root}")


def collect_case_records(domain: str | Path, data_root: str | Path = DATA_ROOT) -> List[CaseRecord]:
    domain_path = resolve_domain(domain, data_root=data_root)
    img_root = domain_path / "images"
    mask_root = domain_path / "masks"
    records: List[CaseRecord] = []
    for img_path in sorted(img_root.glob("*.mha"), key=lambda p: _numeric_key(parse_case_id(p.stem))):
        mask_path = mask_root / img_path.name
        if not mask_path.exists():
            continue
        records.append(
            CaseRecord(
                case_id=parse_case_id(img_path.stem),
                file=img_path.name,
                image_path=img_path,
                mask_path=mask_path,
                domain=domain_path.name,
            )
        )
    if not records:
        raise RuntimeError(f"No paired .mha image/mask records found in {domain_path}")
    return records


def read_split_rows(split_csv: str | Path = DEFAULT_SPLIT_CSV) -> List[Dict[str, str]]:
    path = Path(split_csv)
    if not path.exists():
        raise FileNotFoundError(f"Split CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def split_case_records(
    *,
    shot: int,
    seed: int,
    role: str,
    domain: str | Path | None = None,
    data_root: str | Path = DATA_ROOT,
    split_csv: str | Path = DEFAULT_SPLIT_CSV,
) -> List[CaseRecord]:
    domain_name = resolve_domain_name(domain) if domain is not None else None
    root = Path(data_root)
    records: List[CaseRecord] = []
    for row in read_split_rows(split_csv):
        if int(row["shot"]) != int(shot) or int(row["seed"]) != int(seed):
            continue
        if str(row["role"]) != str(role):
            continue
        if domain_name is not None and row["domain"] != domain_name:
            continue
        records.append(
            CaseRecord(
                case_id=normalize_case_id(row["patient_id"]),
                file=row["file"],
                image_path=root / row["image"],
                mask_path=root / row["mask"],
                domain=row["domain"],
                split_role=row["role"],
            )
        )
    return sorted(records, key=lambda r: (r.domain, _numeric_key(r.case_id), r.file))


def remap_mask_to_3class(mask: np.ndarray) -> np.ndarray:
    arr = mask.astype(np.int32)
    out = np.zeros_like(arr, dtype=np.int64)
    out[np.isin(arr, np.asarray(VERTEBRAE_LABELS, dtype=np.int32))] = 1
    out[np.isin(arr, np.asarray(DISC_LABELS, dtype=np.int32))] = 2
    return out


def select_middle_sagittal_indices(mask_zyx: np.ndarray, num_slices: int = 9) -> List[int]:
    x_any = np.where((mask_zyx > 0).sum(axis=(0, 1)) > 0)[0]
    x_dim = int(mask_zyx.shape[2])
    if len(x_any) > 0:
        center = int(round((int(x_any[0]) + int(x_any[-1])) / 2.0))
    else:
        center = x_dim // 2
    half = int(num_slices) // 2
    idxs = list(range(center - half, center + half + 1))
    if len(idxs) > int(num_slices):
        idxs = idxs[: int(num_slices)]
    return [min(max(int(x), 0), x_dim - 1) for x in idxs]


def normalize_slice_to_01(img2d: np.ndarray, clip_min: float = -3.0, clip_max: float = 3.0) -> np.ndarray:
    img2d = np.nan_to_num(img2d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    img2d = np.clip(img2d, clip_min, clip_max)
    img2d = (img2d - clip_min) / max(clip_max - clip_min, 1e-8)
    return np.clip(img2d, 0.0, 1.0).astype(np.float32)


def resize_np_2d(arr: np.ndarray, size_hw: tuple[int, int], is_mask: bool) -> np.ndarray:
    t = torch.from_numpy(arr[None, None, ...]).float()
    if is_mask:
        out = F.interpolate(t, size=size_hw, mode="nearest")
    else:
        out = F.interpolate(t, size=size_hw, mode="bilinear", align_corners=False)
    return out[0, 0].cpu().numpy()


def _resize_hw(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return (int(value), int(value))
    vals = tuple(int(v) for v in value)
    if len(vals) != 2:
        raise ValueError(f"resize_hw must contain 2 values, got {value!r}")
    return vals


def _spacing_for_resized_sagittal_stack(
    itk_image: sitk.Image,
    original_shape_zyx: tuple[int, int, int],
    resize_hw: tuple[int, int],
) -> tuple[tuple[float, float, float], tuple[float, float]]:
    spacing_xyz = tuple(float(x) for x in itk_image.GetSpacing())
    if len(spacing_xyz) < 3:
        spacing_xyz = (1.0, 1.0, 1.0)
    z_dim, y_dim, _x_dim = original_shape_zyx
    resize_h, resize_w = resize_hw
    x_spacing = spacing_xyz[0]
    z_spacing = spacing_xyz[2] * float(z_dim) / float(resize_h)
    y_spacing = spacing_xyz[1] * float(y_dim) / float(resize_w)
    return (x_spacing, z_spacing, y_spacing), (z_spacing, y_spacing)


def _slice_indices_for_policy(
    sem_mask_zyx: np.ndarray,
    *,
    slice_policy: str,
    num_middle_slices: int,
) -> List[int]:
    policy = str(slice_policy).strip().lower()
    if policy in {"center9", "center", "middle"}:
        return select_middle_sagittal_indices(sem_mask_zyx, int(num_middle_slices))
    if policy in {"all", "all_filtered"}:
        return list(range(int(sem_mask_zyx.shape[2])))
    raise ValueError(f"Unknown slice_policy {slice_policy!r}; expected center9 or all.")


def load_case_slices(
    record: CaseRecord,
    *,
    resize_hw: int | Sequence[int] = 224,
    min_fg_ratio: float = 0.05,
    clip_min: float = -3.0,
    clip_max: float = 3.0,
    slice_policy: str = "center9",
    num_middle_slices: int = 9,
    filter_min_fg: bool = False,
) -> List[Dict[str, Any]]:
    size_hw = _resize_hw(resize_hw)
    img_itk = sitk.ReadImage(str(record.image_path))
    mask_itk = sitk.ReadImage(str(record.mask_path))
    img_zyx = sitk.GetArrayFromImage(img_itk).astype(np.float32)
    raw_mask_zyx = sitk.GetArrayFromImage(mask_itk).astype(np.int32)
    sem_mask_zyx = remap_mask_to_3class(raw_mask_zyx)
    case_spacing, slice_spacing = _spacing_for_resized_sagittal_stack(img_itk, img_zyx.shape, size_hw)
    sagittal_indices = _slice_indices_for_policy(
        sem_mask_zyx,
        slice_policy=slice_policy,
        num_middle_slices=num_middle_slices,
    )
    if str(slice_policy).strip().lower() == "all_filtered":
        filter_min_fg = True

    out: List[Dict[str, Any]] = []
    for position, x_idx in enumerate(sagittal_indices):
        img2d = normalize_slice_to_01(img_zyx[:, :, x_idx], clip_min=clip_min, clip_max=clip_max)
        mask2d = sem_mask_zyx[:, :, x_idx]
        img2d = resize_np_2d(img2d, size_hw, is_mask=False)
        mask2d = resize_np_2d(mask2d.astype(np.float32), size_hw, is_mask=True).astype(np.int64)
        mask2d = np.clip(mask2d, 0, 2).astype(np.int64)
        fg_ratio = float(np.mean(mask2d > 0))
        if bool(filter_min_fg) and fg_ratio < float(min_fg_ratio):
            continue
        out.append(
            {
                "case_id": record.case_id,
                "file": record.file,
                "domain": record.domain,
                "split_role": record.split_role,
                "sagittal_x_index": int(x_idx),
                "slice_position": int(position),
                "foreground_ratio": fg_ratio,
                "image": img2d.astype(np.float32),
                "mask": mask2d.astype(np.int64),
                "case_spacing": case_spacing,
                "slice_spacing": slice_spacing,
            }
        )
    return out


class SpiderSliceDataset(Dataset):
    def __init__(
        self,
        items: Sequence[Dict[str, Any]],
        *,
        domain_path: Path,
        case_ids: Sequence[str],
        selected_case_ids: Sequence[str] | None = None,
        min_fg_ratio: float = 0.05,
        resize_hw: int | Sequence[int] = 224,
        slice_policy: str = "center9",
        num_middle_slices: int = 9,
        filter_min_fg: bool = False,
        split_csv: str | Path | None = None,
        split_role: str = "",
    ):
        self.items = list(items)
        self.domain_path = Path(domain_path)
        self.case_ids = [normalize_case_id(x) for x in case_ids]
        self.selected_case_ids = [normalize_case_id(x) for x in (selected_case_ids or case_ids)]
        self.min_fg_ratio = float(min_fg_ratio)
        self.resize_hw = _resize_hw(resize_hw)
        self.slice_policy = str(slice_policy)
        self.num_middle_slices = int(num_middle_slices)
        self.filter_min_fg = bool(filter_min_fg)
        self.split_csv = str(split_csv) if split_csv is not None else ""
        self.split_role = str(split_role or "")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        img = torch.from_numpy(item["image"])[None, ...].float()
        mask = torch.from_numpy(item["mask"]).long()
        meta = {
            "case_id": item["case_id"],
            "file": item["file"],
            "domain": item["domain"],
            "split_role": item.get("split_role", ""),
            "sagittal_x_index": item["sagittal_x_index"],
            "slice_position": item.get("slice_position", 0),
            "foreground_ratio": item["foreground_ratio"],
        }
        return img, mask, meta

    def items_for_case(self, case_id: str) -> List[Dict[str, Any]]:
        key = normalize_case_id(case_id)
        return [item for item in self.items if item["case_id"] == key]

    def grouped_case_ids(self) -> List[str]:
        return sorted({item["case_id"] for item in self.items}, key=_numeric_key)

    def slice_indices_by_case(self) -> Dict[str, List[int]]:
        return {case_id: [int(item["sagittal_x_index"]) for item in self.items_for_case(case_id)] for case_id in self.grouped_case_ids()}


def _build_dataset_from_records(
    records: Sequence[CaseRecord],
    *,
    domain_path: Path,
    selected_case_ids: Sequence[str],
    min_fg_ratio: float,
    resize_hw: int | Sequence[int],
    slice_policy: str = "center9",
    num_middle_slices: int = 9,
    filter_min_fg: bool = False,
    split_csv: str | Path | None = None,
    split_role: str = "",
) -> SpiderSliceDataset:
    items: List[Dict[str, Any]] = []
    for record in records:
        items.extend(
            load_case_slices(
                record,
                resize_hw=resize_hw,
                min_fg_ratio=min_fg_ratio,
                slice_policy=slice_policy,
                num_middle_slices=num_middle_slices,
                filter_min_fg=filter_min_fg,
            )
        )
    if not items:
        raise RuntimeError("No slices were loaded for the requested dataset.")
    return SpiderSliceDataset(
        items,
        domain_path=domain_path,
        case_ids=[record.case_id for record in records],
        selected_case_ids=selected_case_ids,
        min_fg_ratio=min_fg_ratio,
        resize_hw=resize_hw,
        slice_policy=slice_policy,
        num_middle_slices=num_middle_slices,
        filter_min_fg=filter_min_fg,
        split_csv=split_csv,
        split_role=split_role,
    )


def enumerate_fewshot_splits(
    domain: str | Path = SOURCE_DOMAIN_NAME,
    shots: Iterable[int] = (3, 4, 5),
    seeds: Iterable[int] = range(5),
    *,
    data_root: str | Path = DATA_ROOT,
    split_csv: str | Path = DEFAULT_SPLIT_CSV,
    use_split: bool = True,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if use_split and Path(split_csv).exists():
        for shot in shots:
            for seed in seeds:
                records = split_case_records(
                    shot=int(shot),
                    seed=int(seed),
                    role="source_train",
                    domain=domain,
                    data_root=data_root,
                    split_csv=split_csv,
                )
                rows.append(
                    {
                        "domain": resolve_domain_name(domain),
                        "shot": int(shot),
                        "seed": int(seed),
                        "case_ids": [record.case_id for record in records],
                        "split_csv": str(split_csv),
                    }
                )
        return rows

    records = collect_case_records(domain, data_root=data_root)
    case_ids = sorted({record.case_id for record in records}, key=_numeric_key)
    for shot in shots:
        for seed in seeds:
            selected = random.Random(int(seed)).sample(case_ids, int(shot))
            rows.append({"domain": resolve_domain(domain, data_root).name, "shot": int(shot), "seed": int(seed), "case_ids": selected})
    return rows


def build_train_dataset(
    domain: str | Path,
    shot: int,
    seed: int,
    *,
    data_root: str | Path = DATA_ROOT,
    min_fg_ratio: float = 0.05,
    resize_hw: int | Sequence[int] = 224,
    split_csv: str | Path | None = DEFAULT_SPLIT_CSV,
    use_split: bool = True,
    slice_policy: str = "center9",
    num_middle_slices: int = 9,
    filter_min_fg: bool = False,
) -> SpiderSliceDataset:
    if int(shot) not in {3, 4, 5}:
        raise ValueError(f"shot must be one of 3, 4, 5; got {shot}")
    if int(seed) not in {0, 1, 2, 3, 4}:
        raise ValueError(f"seed must be one of 0, 1, 2, 3, 4; got {seed}")
    domain_path = resolve_domain(domain, data_root=data_root)

    if bool(use_split) and split_csv is not None:
        selected_records = split_case_records(
            shot=int(shot),
            seed=int(seed),
            role="source_train",
            domain=domain_path.name,
            data_root=data_root,
            split_csv=split_csv,
        )
        if not selected_records:
            raise RuntimeError(f"No source_train rows found for domain={domain_path.name} shot={shot} seed={seed} in {split_csv}")
        return _build_dataset_from_records(
            selected_records,
            domain_path=domain_path,
            selected_case_ids=[record.case_id for record in selected_records],
            min_fg_ratio=min_fg_ratio,
            resize_hw=resize_hw,
            slice_policy=slice_policy,
            num_middle_slices=num_middle_slices,
            filter_min_fg=filter_min_fg,
            split_csv=split_csv,
            split_role="source_train",
        )

    records = collect_case_records(domain_path, data_root=data_root)
    case_ids = sorted({record.case_id for record in records}, key=_numeric_key)
    selected = random.Random(int(seed)).sample(case_ids, int(shot))
    selected_set = set(selected)
    selected_records = [record for record in records if record.case_id in selected_set]
    return _build_dataset_from_records(
        selected_records,
        domain_path=domain_path,
        selected_case_ids=selected,
        min_fg_ratio=min_fg_ratio,
        resize_hw=resize_hw,
        slice_policy=slice_policy,
        num_middle_slices=num_middle_slices,
        filter_min_fg=filter_min_fg,
    )


def build_test_dataset(
    domain: str | Path,
    exclude_case_ids: Sequence[str] | None = None,
    *,
    data_root: str | Path = DATA_ROOT,
    min_fg_ratio: float = 0.05,
    resize_hw: int | Sequence[int] = 224,
    max_cases: int | None = None,
    shot: int | None = None,
    seed: int | None = None,
    split_csv: str | Path | None = DEFAULT_SPLIT_CSV,
    split_role: str | None = None,
    use_split: bool | None = None,
    slice_policy: str = "center9",
    num_middle_slices: int = 9,
    filter_min_fg: bool = False,
) -> SpiderSliceDataset:
    domain_path = resolve_domain(domain, data_root=data_root)
    excluded = {normalize_case_id(x) for x in (exclude_case_ids or [])}
    should_use_split = bool(split_role and shot is not None and seed is not None) if use_split is None else bool(use_split)

    if should_use_split:
        if split_csv is None:
            raise ValueError("split_csv is required when use_split=True")
        if shot is None or seed is None:
            raise ValueError("shot and seed are required when use_split=True")
        records = split_case_records(
            shot=int(shot),
            seed=int(seed),
            role=str(split_role),
            domain=domain_path.name,
            data_root=data_root,
            split_csv=split_csv,
        )
        records = [record for record in records if record.case_id not in excluded]
        if max_cases is not None and int(max_cases) > 0:
            records = records[: int(max_cases)]
        if not records:
            raise RuntimeError(f"No split test cases for domain={domain_path.name} role={split_role} shot={shot} seed={seed}")
        return _build_dataset_from_records(
            records,
            domain_path=domain_path,
            selected_case_ids=[record.case_id for record in records],
            min_fg_ratio=min_fg_ratio,
            resize_hw=resize_hw,
            slice_policy=slice_policy,
            num_middle_slices=num_middle_slices,
            filter_min_fg=filter_min_fg,
            split_csv=split_csv,
            split_role=str(split_role or ""),
        )

    records = [record for record in collect_case_records(domain_path, data_root=data_root) if record.case_id not in excluded]
    records = sorted(records, key=lambda r: _numeric_key(r.case_id))
    if max_cases is not None and int(max_cases) > 0:
        records = records[: int(max_cases)]
    if not records:
        raise RuntimeError(f"No test cases left for {domain_path.name} after excluding {sorted(excluded)}")
    return _build_dataset_from_records(
        records,
        domain_path=domain_path,
        selected_case_ids=[record.case_id for record in records],
        min_fg_ratio=min_fg_ratio,
        resize_hw=resize_hw,
        slice_policy=slice_policy,
        num_middle_slices=num_middle_slices,
        filter_min_fg=filter_min_fg,
    )


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, device: str | torch.device = "cpu") -> DataLoader:
    dev = torch.device(device)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        drop_last=False,
        pin_memory=(dev.type == "cuda"),
    )


def _finite_mean(values: Sequence[float]) -> float:
    xs = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(xs)) if xs else float("nan")


def _normalize_spacing(spacing: Sequence[float] | None, ndim: int) -> tuple[float, ...]:
    if spacing is None:
        return tuple([1.0] * int(ndim))
    vals = tuple(float(x) for x in spacing)
    if len(vals) == int(ndim):
        return vals
    if len(vals) > int(ndim):
        return vals[-int(ndim) :]
    return tuple(list(vals) + [1.0] * (int(ndim) - len(vals)))


def _class_dice(pred: np.ndarray, gt: np.ndarray, cls: int, eps: float = 1e-6) -> float:
    pred_c = pred == cls
    gt_c = gt == cls
    if not np.any(gt_c):
        return float("nan")
    denom = float(pred_c.sum() + gt_c.sum())
    if denom <= 0:
        return float("nan")
    return float((2.0 * np.logical_and(pred_c, gt_c).sum() + eps) / (denom + eps))


def case_dice_by_class(pred: np.ndarray, gt: np.ndarray, classes: Sequence[int] = FOREGROUND_CLASSES) -> Dict[int, float]:
    pred_arr = np.asarray(pred)
    gt_arr = np.asarray(gt)
    return {int(cls): _class_dice(pred_arr, gt_arr, int(cls)) for cls in classes}


def case_dice(pred: np.ndarray, gt: np.ndarray, classes: Sequence[int] = FOREGROUND_CLASSES) -> float:
    return _finite_mean(list(case_dice_by_class(pred, gt, classes=classes).values()))


def slice_dice_by_class(pred: np.ndarray, gt: np.ndarray, classes: Sequence[int] = FOREGROUND_CLASSES) -> Dict[int, float]:
    pred_arr = np.asarray(pred)
    gt_arr = np.asarray(gt)
    if pred_arr.ndim == 2:
        return case_dice_by_class(pred_arr, gt_arr, classes=classes)
    out: Dict[int, float] = {}
    for cls in classes:
        out[int(cls)] = _finite_mean([_class_dice(pred_arr[i], gt_arr[i], int(cls)) for i in range(pred_arr.shape[0])])
    return out


def slice_dice(pred: np.ndarray, gt: np.ndarray, classes: Sequence[int] = FOREGROUND_CLASSES) -> float:
    return _finite_mean(list(slice_dice_by_class(pred, gt, classes=classes).values()))


def _surface(mask: np.ndarray) -> np.ndarray:
    mask_bool = np.asarray(mask).astype(bool)
    if not np.any(mask_bool):
        return np.zeros_like(mask_bool, dtype=bool)
    structure = ndi.generate_binary_structure(mask_bool.ndim, 1)
    eroded = ndi.binary_erosion(mask_bool, structure=structure, border_value=0)
    return np.logical_xor(mask_bool, eroded)


def _diagonal_length(shape: Sequence[int], spacing: Sequence[float]) -> float:
    return float(math.sqrt(sum(((max(int(dim) - 1, 1)) * float(sp)) ** 2 for dim, sp in zip(shape, spacing))))


def _binary_hd95(pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float]) -> float:
    pred_bool = np.asarray(pred).astype(bool)
    gt_bool = np.asarray(gt).astype(bool)
    spacing = _normalize_spacing(spacing, pred_bool.ndim)
    if not np.any(gt_bool):
        return float("nan")
    if not np.any(pred_bool):
        return _diagonal_length(gt_bool.shape, spacing)

    pred_surface = _surface(pred_bool)
    gt_surface = _surface(gt_bool)
    if not np.any(pred_surface) or not np.any(gt_surface):
        return _diagonal_length(gt_bool.shape, spacing)

    dt_to_gt = ndi.distance_transform_edt(~gt_surface, sampling=spacing)
    dt_to_pred = ndi.distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate([dt_to_gt[pred_surface], dt_to_pred[gt_surface]])
    return float(np.percentile(distances, 95.0)) if distances.size else 0.0


def case_hd95_by_class(
    pred: np.ndarray,
    gt: np.ndarray,
    classes: Sequence[int] = FOREGROUND_CLASSES,
    spacing: Sequence[float] | None = None,
) -> Dict[int, float]:
    pred_arr = np.asarray(pred)
    gt_arr = np.asarray(gt)
    sp = _normalize_spacing(spacing, pred_arr.ndim)
    out: Dict[int, float] = {}
    for cls in classes:
        gt_c = gt_arr == int(cls)
        out[int(cls)] = _binary_hd95(pred_arr == int(cls), gt_c, sp) if np.any(gt_c) else float("nan")
    return out


def case_hd95(
    pred: np.ndarray,
    gt: np.ndarray,
    classes: Sequence[int] = FOREGROUND_CLASSES,
    spacing: Sequence[float] | None = None,
) -> float:
    return _finite_mean(list(case_hd95_by_class(pred, gt, classes=classes, spacing=spacing).values()))


def slice_hd95_by_class(
    pred: np.ndarray,
    gt: np.ndarray,
    classes: Sequence[int] = FOREGROUND_CLASSES,
    spacing: Sequence[float] | None = None,
) -> Dict[int, float]:
    pred_arr = np.asarray(pred)
    gt_arr = np.asarray(gt)
    if pred_arr.ndim == 2:
        return case_hd95_by_class(pred_arr, gt_arr, classes=classes, spacing=spacing)
    sp = _normalize_spacing(spacing, pred_arr.ndim)
    slice_sp = sp[-2:]
    out: Dict[int, float] = {}
    for cls in classes:
        out[int(cls)] = _finite_mean(
            [
                case_hd95(pred_arr[i], gt_arr[i], classes=(int(cls),), spacing=slice_sp)
                for i in range(pred_arr.shape[0])
            ]
        )
    return out


def slice_hd95(
    pred: np.ndarray,
    gt: np.ndarray,
    classes: Sequence[int] = FOREGROUND_CLASSES,
    spacing: Sequence[float] | None = None,
) -> float:
    return _finite_mean(list(slice_hd95_by_class(pred, gt, classes=classes, spacing=spacing).values()))


def _add_class_metric_columns(row: Dict[str, Any], prefix: str, values: Mapping[int, float]) -> None:
    for cls, value in values.items():
        name = FOREGROUND_CLASS_NAMES.get(int(cls), f"class_{int(cls)}")
        row[f"{prefix}_{name}"] = float(value)


@torch.no_grad()
def run_test_flow(
    model: torch.nn.Module,
    domain: str | Path,
    exclude_case_ids: Sequence[str] | None = None,
    *,
    data_root: str | Path = DATA_ROOT,
    min_fg_ratio: float = 0.05,
    resize_hw: int | Sequence[int] = 224,
    batch_size: int = 4,
    device: str | torch.device = "cpu",
    max_cases: int | None = None,
    shot: int | None = None,
    seed: int | None = None,
    split_csv: str | Path | None = DEFAULT_SPLIT_CSV,
    split_role: str | None = None,
    use_split: bool | None = None,
    eval_set: str = "test",
    slice_policy: str = "center9",
    num_middle_slices: int = 9,
    filter_min_fg: bool = False,
) -> Dict[str, Any]:
    dev = torch.device(device)
    dataset = build_test_dataset(
        domain,
        exclude_case_ids=exclude_case_ids,
        data_root=data_root,
        min_fg_ratio=min_fg_ratio,
        resize_hw=resize_hw,
        max_cases=max_cases,
        shot=shot,
        seed=seed,
        split_csv=split_csv,
        split_role=split_role,
        use_split=use_split,
        slice_policy=slice_policy,
        num_middle_slices=num_middle_slices,
        filter_min_fg=filter_min_fg,
    )
    model.eval()
    rows: List[Dict[str, Any]] = []
    for case_id in dataset.grouped_case_ids():
        items = dataset.items_for_case(case_id)
        images = torch.stack([torch.from_numpy(item["image"])[None, ...].float() for item in items], dim=0)
        gt_stack = np.stack([item["mask"] for item in items], axis=0).astype(np.int64)
        preds = []
        for start in range(0, images.shape[0], int(batch_size)):
            batch = images[start : start + int(batch_size)].to(dev)
            logits = model(batch)
            if isinstance(logits, Mapping):
                logits = logits["logits"]
            preds.append(torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64))
        pred_stack = np.concatenate(preds, axis=0)
        case_spacing = items[0]["case_spacing"]
        slice_spacing = items[0]["slice_spacing"]
        case_dice_cls = case_dice_by_class(pred_stack, gt_stack)
        case_hd95_cls = case_hd95_by_class(pred_stack, gt_stack, spacing=case_spacing)
        slice_dice_cls = slice_dice_by_class(pred_stack, gt_stack)
        slice_hd95_cls = slice_hd95_by_class(pred_stack, gt_stack, spacing=slice_spacing)
        row = {
            "case_id": case_id,
            "n_slices": int(len(items)),
            "sagittal_x_indices": "|".join(str(int(item["sagittal_x_index"])) for item in items),
            "case_dice": _finite_mean(list(case_dice_cls.values())),
            "case_hd95": _finite_mean(list(case_hd95_cls.values())),
            "slice_dice": _finite_mean(list(slice_dice_cls.values())),
            "slice_hd95": _finite_mean(list(slice_hd95_cls.values())),
        }
        _add_class_metric_columns(row, "case_dice", case_dice_cls)
        _add_class_metric_columns(row, "case_hd95", case_hd95_cls)
        _add_class_metric_columns(row, "slice_dice", slice_dice_cls)
        _add_class_metric_columns(row, "slice_hd95", slice_hd95_cls)
        rows.append(row)

    summary = {
        "case_dice": _finite_mean([row["case_dice"] for row in rows]),
        "case_hd95": _finite_mean([row["case_hd95"] for row in rows]),
        "slice_dice": _finite_mean([row["slice_dice"] for row in rows]),
        "slice_hd95": _finite_mean([row["slice_hd95"] for row in rows]),
    }
    for prefix in ("case_dice", "case_hd95", "slice_dice", "slice_hd95"):
        for cls, name in FOREGROUND_CLASS_NAMES.items():
            summary[f"{prefix}_{name}"] = _finite_mean([row[f"{prefix}_{name}"] for row in rows])

    return {
        "domain": dataset.domain_path.name,
        "eval_set": str(eval_set),
        "split_role": str(split_role or ""),
        "excluded_case_ids": sorted({normalize_case_id(x) for x in (exclude_case_ids or [])}, key=_numeric_key),
        "n_cases": int(len(rows)),
        "n_slices": int(len(dataset)),
        "case_rows": rows,
        "summary": summary,
        "slice_policy": dataset.slice_policy,
        "num_middle_slices": int(dataset.num_middle_slices),
        "split_csv": dataset.split_csv,
    }
