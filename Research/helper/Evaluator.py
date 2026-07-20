from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from scipy import ndimage as ndi


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


def _phase_key(phase: Any) -> tuple[int, str]:
    text = str(phase).strip().upper()
    return (PHASE_ORDER.get(text, 99), text)


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except Exception:
        return default


def _finite_values(values: Sequence[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            number = float(value)
        except Exception:
            continue
        if np.isfinite(number):
            out.append(number)
    return out


def _finite_mean(values: Sequence[Any]) -> float:
    xs = _finite_values(values)
    return float(np.mean(xs)) if xs else float("nan")


def _finite_var(values: Sequence[Any], ddof: int = 1) -> float:
    xs = _finite_values(values)
    if not xs:
        return float("nan")
    if len(xs) <= int(ddof):
        return 0.0
    return float(np.var(xs, ddof=int(ddof)))


def _normalize_spacing(spacing: Sequence[float] | None, ndim: int) -> tuple[float, ...]:
    if spacing is None:
        return tuple([1.0] * int(ndim))
    vals = tuple(float(v) for v in spacing)
    if len(vals) == int(ndim):
        return vals
    if len(vals) > int(ndim):
        return vals[-int(ndim) :]
    return tuple(list(vals) + [1.0] * (int(ndim) - len(vals)))


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
    sp = _normalize_spacing(spacing, pred_bool.ndim)
    if not np.any(gt_bool):
        return float("nan")
    if not np.any(pred_bool):
        return _diagonal_length(gt_bool.shape, sp)

    pred_surface = _surface(pred_bool)
    gt_surface = _surface(gt_bool)
    if not np.any(pred_surface) or not np.any(gt_surface):
        return _diagonal_length(gt_bool.shape, sp)

    dt_to_gt = ndi.distance_transform_edt(~gt_surface, sampling=sp)
    dt_to_pred = ndi.distance_transform_edt(~pred_surface, sampling=sp)
    distances = np.concatenate([dt_to_gt[pred_surface], dt_to_pred[gt_surface]])
    return float(np.percentile(distances, 95.0)) if distances.size else 0.0


def _class_dice(pred: np.ndarray, gt: np.ndarray, cls: int, eps: float = 1e-6) -> float:
    pred_c = np.asarray(pred) == int(cls)
    gt_c = np.asarray(gt) == int(cls)
    if not np.any(gt_c):
        return float("nan")
    denom = float(pred_c.sum() + gt_c.sum())
    if denom <= 0:
        return float("nan")
    return float((2.0 * np.logical_and(pred_c, gt_c).sum() + eps) / (denom + eps))


def _dice_by_class(pred: np.ndarray, gt: np.ndarray, classes: Sequence[int]) -> Dict[int, float]:
    return {int(cls): _class_dice(pred, gt, int(cls)) for cls in classes}


def _hd95_by_class(
    pred: np.ndarray,
    gt: np.ndarray,
    classes: Sequence[int],
    spacing: Sequence[float],
) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for cls in classes:
        gt_c = np.asarray(gt) == int(cls)
        out[int(cls)] = _binary_hd95(np.asarray(pred) == int(cls), gt_c, spacing) if np.any(gt_c) else float("nan")
    return out


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_label_array(value: Any, classes: Sequence[int] = FOREGROUND_CLASSES) -> np.ndarray:
    arr = _to_numpy(value)
    if arr.ndim == 4 and arr.shape[1] >= max(classes) + 1:
        arr = np.argmax(arr, axis=1)
    elif arr.ndim == 3 and arr.shape[0] >= max(classes) + 1 and np.issubdtype(arr.dtype, np.floating):
        arr = np.argmax(arr, axis=0)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.rint(arr)
    return arr.astype(np.int64, copy=False)


def _as_slice_stack(value: Any, classes: Sequence[int] = FOREGROUND_CLASSES) -> np.ndarray:
    arr = _to_label_array(value, classes=classes)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected labels with shape [H,W] or [N,H,W], got {arr.shape}")
    return arr


def _normalize_meta(meta: Sequence[Mapping[str, Any]] | None, n_items: int) -> List[Dict[str, Any]]:
    if meta is None:
        return [{"patient_id": "patient0", "phase": "volume", "z_index": i, "slice_id": f"slice{i:03d}"} for i in range(n_items)]
    if len(meta) != int(n_items):
        raise ValueError(f"meta length {len(meta)} does not match number of slices {n_items}")
    return [dict(item) for item in meta]


def _row_with_class_metrics(
    *,
    base: Mapping[str, Any],
    values: Mapping[int, float],
    classes: Sequence[int],
) -> Dict[str, Any]:
    row = dict(base)
    for cls in classes:
        row[FOREGROUND_CLASS_NAMES.get(int(cls), f"class_{int(cls)}")] = float(values[int(cls)])
    row["mean"] = _finite_mean([values[int(cls)] for cls in classes])
    return row


def _summary_from_rows(rows: Sequence[Mapping[str, Any]], classes: Sequence[int]) -> Dict[str, float]:
    summary: Dict[str, float] = {
        "mean": _finite_mean([float(row.get("mean", float("nan"))) for row in rows]),
    }
    for cls in classes:
        name = FOREGROUND_CLASS_NAMES.get(int(cls), f"class_{int(cls)}")
        summary[name] = _finite_mean([float(row.get(name, float("nan"))) for row in rows])
    return summary


def _volume_groups(meta: Sequence[Mapping[str, Any]]) -> Dict[tuple[str, str], List[int]]:
    groups: Dict[tuple[str, str], List[int]] = {}
    for idx, item in enumerate(meta):
        patient_id = str(item.get("patient_id", "patient0"))
        phase = str(item.get("phase", "volume")).upper()
        groups.setdefault((patient_id, phase), []).append(idx)
    return groups


def _sort_group_indices(indices: Sequence[int], meta: Sequence[Mapping[str, Any]]) -> List[int]:
    return sorted(indices, key=lambda i: (_safe_int(meta[i].get("z_index"), i), str(meta[i].get("slice_id", ""))))


def _patient_rows_from_volume_rows(
    volume_rows: Sequence[Mapping[str, Any]],
    classes: Sequence[int],
) -> List[Dict[str, Any]]:
    by_patient: Dict[str, List[Mapping[str, Any]]] = {}
    for row in volume_rows:
        by_patient.setdefault(str(row["patient_id"]), []).append(row)

    patient_rows: List[Dict[str, Any]] = []
    for patient_id in sorted(by_patient):
        rows = sorted(by_patient[patient_id], key=lambda row: _phase_key(row.get("phase", "")))
        item: Dict[str, Any] = {
            "patient_id": patient_id,
            "phases": "|".join(str(row.get("phase", "")) for row in rows),
            "n_volumes": int(len(rows)),
            "n_slices": int(sum(int(row.get("n_slices", 0)) for row in rows)),
        }
        for cls in classes:
            name = FOREGROUND_CLASS_NAMES.get(int(cls), f"class_{int(cls)}")
            item[name] = _finite_mean([float(row.get(name, float("nan"))) for row in rows])
        item["mean"] = _finite_mean([float(item[FOREGROUND_CLASS_NAMES.get(int(cls), f"class_{int(cls)}")]) for cls in classes])
        patient_rows.append(item)
    return patient_rows


def SliceWiceDice(
    pred: Any,
    gt: Any,
    meta: Sequence[Mapping[str, Any]] | None = None,
    *,
    classes: Sequence[int] = FOREGROUND_CLASSES,
) -> Dict[str, Any]:
    pred_arr = _as_slice_stack(pred, classes=classes)
    gt_arr = _as_slice_stack(gt, classes=classes)
    if pred_arr.shape != gt_arr.shape:
        raise ValueError(f"pred shape {pred_arr.shape} does not match gt shape {gt_arr.shape}")
    metas = _normalize_meta(meta, pred_arr.shape[0])

    rows: List[Dict[str, Any]] = []
    for idx in range(pred_arr.shape[0]):
        values = _dice_by_class(pred_arr[idx], gt_arr[idx], classes)
        rows.append(
            _row_with_class_metrics(
                base={
                    "index": int(idx),
                    "slice_id": metas[idx].get("slice_id", ""),
                    "patient_id": metas[idx].get("patient_id", ""),
                    "phase": metas[idx].get("phase", ""),
                    "z_index": _safe_int(metas[idx].get("z_index"), idx),
                },
                values=values,
                classes=classes,
            )
        )
    return {"metric": "slice_dice", "per_slice": rows, "summary": _summary_from_rows(rows, classes)}


def SliceWiceHD95(
    pred: Any,
    gt: Any,
    meta: Sequence[Mapping[str, Any]] | None = None,
    *,
    classes: Sequence[int] = FOREGROUND_CLASSES,
    spacing: Sequence[float] = (1.5, 1.5),
) -> Dict[str, Any]:
    pred_arr = _as_slice_stack(pred, classes=classes)
    gt_arr = _as_slice_stack(gt, classes=classes)
    if pred_arr.shape != gt_arr.shape:
        raise ValueError(f"pred shape {pred_arr.shape} does not match gt shape {gt_arr.shape}")
    metas = _normalize_meta(meta, pred_arr.shape[0])
    sp = _normalize_spacing(spacing, 2)

    rows: List[Dict[str, Any]] = []
    for idx in range(pred_arr.shape[0]):
        values = _hd95_by_class(pred_arr[idx], gt_arr[idx], classes, sp)
        rows.append(
            _row_with_class_metrics(
                base={
                    "index": int(idx),
                    "slice_id": metas[idx].get("slice_id", ""),
                    "patient_id": metas[idx].get("patient_id", ""),
                    "phase": metas[idx].get("phase", ""),
                    "z_index": _safe_int(metas[idx].get("z_index"), idx),
                },
                values=values,
                classes=classes,
            )
        )
    return {"metric": "slice_hd95", "per_slice": rows, "summary": _summary_from_rows(rows, classes)}


def PatientWiceDice(
    pred: Any,
    gt: Any,
    meta: Sequence[Mapping[str, Any]] | None = None,
    *,
    classes: Sequence[int] = FOREGROUND_CLASSES,
) -> Dict[str, Any]:
    pred_arr = _as_slice_stack(pred, classes=classes)
    gt_arr = _as_slice_stack(gt, classes=classes)
    if pred_arr.shape != gt_arr.shape:
        raise ValueError(f"pred shape {pred_arr.shape} does not match gt shape {gt_arr.shape}")
    metas = _normalize_meta(meta, pred_arr.shape[0])

    volume_rows: List[Dict[str, Any]] = []
    for (patient_id, phase), indices in sorted(_volume_groups(metas).items()):
        ordered = _sort_group_indices(indices, metas)
        pred_stack = pred_arr[ordered]
        gt_stack = gt_arr[ordered]
        values = _dice_by_class(pred_stack, gt_stack, classes)
        volume_rows.append(
            _row_with_class_metrics(
                base={
                    "patient_id": patient_id,
                    "phase": phase,
                    "n_slices": int(len(ordered)),
                    "z_indices": "|".join(str(_safe_int(metas[i].get("z_index"), i)) for i in ordered),
                },
                values=values,
                classes=classes,
            )
        )

    patient_rows = _patient_rows_from_volume_rows(volume_rows, classes)
    return {
        "metric": "patient_dice",
        "per_volume": volume_rows,
        "per_patient": patient_rows,
        "summary": _summary_from_rows(patient_rows, classes),
    }


def PatientWiceHD95(
    pred: Any,
    gt: Any,
    meta: Sequence[Mapping[str, Any]] | None = None,
    *,
    classes: Sequence[int] = FOREGROUND_CLASSES,
    spacing: Sequence[float] = (1.0, 1.5, 1.5),
) -> Dict[str, Any]:
    pred_arr = _as_slice_stack(pred, classes=classes)
    gt_arr = _as_slice_stack(gt, classes=classes)
    if pred_arr.shape != gt_arr.shape:
        raise ValueError(f"pred shape {pred_arr.shape} does not match gt shape {gt_arr.shape}")
    metas = _normalize_meta(meta, pred_arr.shape[0])
    sp = _normalize_spacing(spacing, 3)

    volume_rows: List[Dict[str, Any]] = []
    for (patient_id, phase), indices in sorted(_volume_groups(metas).items()):
        ordered = _sort_group_indices(indices, metas)
        pred_stack = pred_arr[ordered]
        gt_stack = gt_arr[ordered]
        values = _hd95_by_class(pred_stack, gt_stack, classes, sp)
        volume_rows.append(
            _row_with_class_metrics(
                base={
                    "patient_id": patient_id,
                    "phase": phase,
                    "n_slices": int(len(ordered)),
                    "z_indices": "|".join(str(_safe_int(metas[i].get("z_index"), i)) for i in ordered),
                },
                values=values,
                classes=classes,
            )
        )

    patient_rows = _patient_rows_from_volume_rows(volume_rows, classes)
    return {
        "metric": "patient_hd95",
        "per_volume": volume_rows,
        "per_patient": patient_rows,
        "summary": _summary_from_rows(patient_rows, classes),
    }


def _class_name(cls: int) -> str:
    return FOREGROUND_CLASS_NAMES.get(int(cls), f"class_{int(cls)}")


def _metric_columns(prefix: str, classes: Sequence[int]) -> List[str]:
    return [f"{prefix}_{_class_name(int(cls))}" for cls in classes] + [f"{prefix}_mean"]


def _merge_metric_rows(
    *,
    dice_rows: Sequence[Mapping[str, Any]],
    hd95_rows: Sequence[Mapping[str, Any]],
    id_fields: Sequence[str],
    common: Mapping[str, Any],
    classes: Sequence[int],
) -> List[Dict[str, Any]]:
    if len(dice_rows) != len(hd95_rows):
        raise ValueError(f"dice row count {len(dice_rows)} does not match hd95 row count {len(hd95_rows)}")

    rows: List[Dict[str, Any]] = []
    for dice_row, hd95_row in zip(dice_rows, hd95_rows):
        row: Dict[str, Any] = dict(common)
        for field in id_fields:
            row[field] = dice_row.get(field, hd95_row.get(field, ""))
        for cls in classes:
            name = _class_name(int(cls))
            row[f"dice_{name}"] = float(dice_row.get(name, float("nan")))
            row[f"hd95_{name}"] = float(hd95_row.get(name, float("nan")))
        row["dice_mean"] = float(dice_row.get("mean", float("nan")))
        row["hd95_mean"] = float(hd95_row.get("mean", float("nan")))
        rows.append(row)
    return rows


def _summary_from_metric_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    domain: str,
    seed: int,
    backbone_id: str,
    classes: Sequence[int],
    n_items_key: str,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "domain": domain,
        "seed": int(seed),
        "backbone_id": backbone_id,
        "n_items": int(len(rows)),
        n_items_key: int(len(rows)),
    }
    for metric in ("dice", "hd95"):
        class_values: List[float] = []
        for cls in classes:
            column = f"{metric}_{_class_name(int(cls))}"
            value = _finite_mean([row.get(column, float("nan")) for row in rows])
            summary[column] = value
            class_values.append(value)
        summary[f"{metric}_mean"] = _finite_mean(class_values)
    return summary


def _summary_backbone_id(rows: Sequence[Mapping[str, Any]], default: str) -> str:
    ids = sorted({str(row.get("backbone_id", "")) for row in rows if str(row.get("backbone_id", ""))})
    if len(ids) == 1:
        return ids[0]
    if len(ids) > 1:
        return "|".join(ids)
    return str(default)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _aggregate_seed_summaries(
    seed_summaries: Sequence[Mapping[str, Any]],
    *,
    classes: Sequence[int],
) -> Dict[str, Any]:
    rows = [dict(row) for row in seed_summaries]
    domains = sorted({str(row.get("domain", "")) for row in rows if str(row.get("domain", ""))})
    backbone_ids = sorted({str(row.get("backbone_id", "")) for row in rows if str(row.get("backbone_id", ""))})
    out: Dict[str, Any] = {
        "domain": domains[0] if len(domains) == 1 else "|".join(domains),
        "backbone_id": backbone_ids[0] if len(backbone_ids) == 1 else "|".join(backbone_ids),
        "n_seeds": int(len(rows)),
        "seeds": "|".join(str(row.get("seed", "")) for row in rows),
        "n_items": int(sum(int(row.get("n_items", 0) or 0) for row in rows)),
    }
    for metric in ("dice", "hd95"):
        for cls in classes:
            column = f"{metric}_{_class_name(int(cls))}"
            values = [row.get(column, float("nan")) for row in rows]
            out[f"avg_{column}"] = _finite_mean(values)
            out[f"var_{column}"] = _finite_var(values, ddof=1)
        mean_column = f"{metric}_mean"
        values = [row.get(mean_column, float("nan")) for row in rows]
        out[f"avg_{mean_column}"] = _finite_mean(values)
        out[f"var_{mean_column}"] = _finite_var(values, ddof=1)
    return out


class SliceWiceEvaluation:
    """Evaluator for single-slice continual TTA streams."""

    def __init__(
        self,
        domain: str,
        seed: int,
        classes: Sequence[int] = FOREGROUND_CLASSES,
        slice_spacing: Sequence[float] = (1.5, 1.5),
        backbone_id: str = "",
    ) -> None:
        self.domain = str(domain)
        self.seed = int(seed)
        self.classes = tuple(int(cls) for cls in classes)
        self.slice_spacing = tuple(float(v) for v in slice_spacing)
        self.backbone_id = str(backbone_id)
        self.rows: List[Dict[str, Any]] = []
        self._update_count = 0

    def update(
        self,
        pred: Any,
        gt: Any,
        meta: Sequence[Mapping[str, Any]] | None,
        step: int | None = None,
        backbone_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        current_step = self._update_count if step is None else int(step)
        self._update_count += 1
        current_backbone_id = self.backbone_id if backbone_id is None else str(backbone_id)

        dice = SliceWiceDice(pred, gt, meta, classes=self.classes)
        hd95 = SliceWiceHD95(pred, gt, meta, classes=self.classes, spacing=self.slice_spacing)
        rows = _merge_metric_rows(
            dice_rows=dice["per_slice"],
            hd95_rows=hd95["per_slice"],
            id_fields=("slice_id", "patient_id", "phase", "z_index"),
            common={
                "domain": self.domain,
                "seed": self.seed,
                "step": current_step,
                "backbone_id": current_backbone_id,
            },
            classes=self.classes,
        )
        self.rows.extend(rows)
        return rows

    def seed_summary(self) -> Dict[str, Any]:
        return _summary_from_metric_rows(
            rows=self.rows,
            domain=self.domain,
            seed=self.seed,
            backbone_id=_summary_backbone_id(self.rows, self.backbone_id),
            classes=self.classes,
            n_items_key="n_slices",
        )

    def save_csv(self, output_dir: str | Path) -> Dict[str, Path]:
        out = Path(output_dir)
        row_fields = [
            "domain",
            "seed",
            "step",
            "backbone_id",
            "slice_id",
            "patient_id",
            "phase",
            "z_index",
            *_metric_columns("dice", self.classes),
            *_metric_columns("hd95", self.classes),
        ]
        summary_fields = [
            "domain",
            "seed",
            "backbone_id",
            "n_items",
            "n_slices",
            *_metric_columns("dice", self.classes),
            *_metric_columns("hd95", self.classes),
        ]
        paths = {
            "slice_rows": out / "slice_rows.csv",
            "seed_summary": out / "seed_summary.csv",
        }
        _write_csv(paths["slice_rows"], self.rows, row_fields)
        _write_csv(paths["seed_summary"], [self.seed_summary()], summary_fields)
        return paths

    @staticmethod
    def aggregate_seed_summaries(
        seed_summaries: Sequence[Mapping[str, Any]],
        classes: Sequence[int] = FOREGROUND_CLASSES,
    ) -> Dict[str, Any]:
        return _aggregate_seed_summaries(seed_summaries, classes=tuple(int(cls) for cls in classes))


class PatientStreamEvaluator:
    """Evaluator for patient-wise continual TTA streams."""

    def __init__(
        self,
        domain: str,
        seed: int,
        classes: Sequence[int] = FOREGROUND_CLASSES,
        patient_spacing: Sequence[float] = (1.0, 1.5, 1.5),
        backbone_id: str = "",
    ) -> None:
        self.domain = str(domain)
        self.seed = int(seed)
        self.classes = tuple(int(cls) for cls in classes)
        self.patient_spacing = tuple(float(v) for v in patient_spacing)
        self.backbone_id = str(backbone_id)
        self.patient_rows: List[Dict[str, Any]] = []
        self.volume_rows: List[Dict[str, Any]] = []
        self._update_count = 0

    def update(
        self,
        pred: Any,
        gt: Any,
        meta: Sequence[Mapping[str, Any]] | None,
        step: int | None = None,
        backbone_id: str | None = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        metas = _normalize_meta(meta, _as_slice_stack(gt, classes=self.classes).shape[0])
        patient_ids = sorted({str(item.get("patient_id", "")) for item in metas})
        if len(patient_ids) != 1:
            raise ValueError(
                "PatientStreamEvaluator.update expects exactly one complete patient; "
                f"got patient_ids={patient_ids}"
            )

        current_step = self._update_count if step is None else int(step)
        self._update_count += 1
        current_backbone_id = self.backbone_id if backbone_id is None else str(backbone_id)

        dice = PatientWiceDice(pred, gt, metas, classes=self.classes)
        hd95 = PatientWiceHD95(pred, gt, metas, classes=self.classes, spacing=self.patient_spacing)
        common = {
            "domain": self.domain,
            "seed": self.seed,
            "step": current_step,
            "backbone_id": current_backbone_id,
        }
        patient_rows = _merge_metric_rows(
            dice_rows=dice["per_patient"],
            hd95_rows=hd95["per_patient"],
            id_fields=("patient_id", "phases", "n_volumes", "n_slices"),
            common=common,
            classes=self.classes,
        )
        volume_rows = _merge_metric_rows(
            dice_rows=dice["per_volume"],
            hd95_rows=hd95["per_volume"],
            id_fields=("patient_id", "phase", "n_slices", "z_indices"),
            common=common,
            classes=self.classes,
        )
        self.patient_rows.extend(patient_rows)
        self.volume_rows.extend(volume_rows)
        return {"patient_rows": patient_rows, "volume_rows": volume_rows}

    def seed_summary(self) -> Dict[str, Any]:
        summary = _summary_from_metric_rows(
            rows=self.patient_rows,
            domain=self.domain,
            seed=self.seed,
            backbone_id=_summary_backbone_id(self.patient_rows, self.backbone_id),
            classes=self.classes,
            n_items_key="n_patients",
        )
        return summary

    def save_csv(self, output_dir: str | Path) -> Dict[str, Path]:
        out = Path(output_dir)
        patient_fields = [
            "domain",
            "seed",
            "step",
            "backbone_id",
            "patient_id",
            "phases",
            "n_volumes",
            "n_slices",
            *_metric_columns("dice", self.classes),
            *_metric_columns("hd95", self.classes),
        ]
        volume_fields = [
            "domain",
            "seed",
            "step",
            "backbone_id",
            "patient_id",
            "phase",
            "n_slices",
            "z_indices",
            *_metric_columns("dice", self.classes),
            *_metric_columns("hd95", self.classes),
        ]
        summary_fields = [
            "domain",
            "seed",
            "backbone_id",
            "n_items",
            "n_patients",
            *_metric_columns("dice", self.classes),
            *_metric_columns("hd95", self.classes),
        ]
        paths = {
            "patient_rows": out / "patient_rows.csv",
            "volume_rows": out / "volume_rows.csv",
            "seed_summary": out / "seed_summary.csv",
        }
        _write_csv(paths["patient_rows"], self.patient_rows, patient_fields)
        _write_csv(paths["volume_rows"], self.volume_rows, volume_fields)
        _write_csv(paths["seed_summary"], [self.seed_summary()], summary_fields)
        return paths

    @staticmethod
    def aggregate_seed_summaries(
        seed_summaries: Sequence[Mapping[str, Any]],
        classes: Sequence[int] = FOREGROUND_CLASSES,
    ) -> Dict[str, Any]:
        return _aggregate_seed_summaries(seed_summaries, classes=tuple(int(cls) for cls in classes))


SliceWiseDice = SliceWiceDice
SliceWiseHD95 = SliceWiceHD95
PatientWiseDice = PatientWiceDice
PatientWiseHD95 = PatientWiceHD95
SliceWiseEvaluation = SliceWiceEvaluation


__all__ = [
    "FOREGROUND_CLASSES",
    "FOREGROUND_CLASS_NAMES",
    "SliceWiceDice",
    "SliceWiceHD95",
    "PatientWiceDice",
    "PatientWiceHD95",
    "SliceWiseDice",
    "SliceWiseHD95",
    "PatientWiseDice",
    "PatientWiseHD95",
    "SliceWiceEvaluation",
    "SliceWiseEvaluation",
    "PatientStreamEvaluator",
]
