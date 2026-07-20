from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

import Disperssion_Ablabtins7_WhyHD95Deterioriates as ab7
from helper.dataloaders import (
    DATA_ROOT,
    FOREGROUND_CLASSES,
    FOREGROUND_CLASS_NAMES,
    collect_case_records,
    load_case_slices,
    normalize_case_id,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SELECTION_CSV = PROJECT_ROOT / "TrainingData" / "WhyHD95Deterioriates_RadialGate" / "diagnostic_case_metrics.csv"
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "TrainingData" / "HD95OutlierAudit_RadialGate"
DEFAULT_CAUSE_RESULT_ROOT = PROJECT_ROOT / "TrainingData" / "WhyThereIsOutliers_RadialGate"
DEFAULT_RADIALGATE_CHECKPOINT_ROOT = ab7.DEFAULT_RADIALGATE_CHECKPOINT_ROOT
DEFAULT_SADG_CHECKPOINT_ROOT = ab7.DEFAULT_SADG_CHECKPOINT_ROOT

CAUSE_RADIAL_NONE = "RadialGate-none"
CAUSE_REFLECT_FFT = "RadialGate-ReflectFFT"
CAUSE_SPATIAL_BLEND = "RadialGate-SpatialBlend"
CAUSE_REFLECT_PAD_INFERENCE = "RadialGate-ReflectPadInference"
DEFAULT_CAUSE_VARIANTS = (
    CAUSE_RADIAL_NONE,
    CAUSE_REFLECT_FFT,
    CAUSE_SPATIAL_BLEND,
    CAUSE_REFLECT_PAD_INFERENCE,
)

ANNOTATION_LABELS = (
    "fov_border_artifact",
    "small_fp_error",
    "plausible_anatomy_gt_missing",
    "boundary_overexpansion",
    "gt_artifact_discontinuity",
    "ambiguous",
)


def resolve_path(path: str | Path) -> Path:
    return ab7.resolve_path(path, base=PROJECT_ROOT)


def ensure_dir(path: Path) -> Path:
    return ab7.ensure_dir(path)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ab7.write_csv(path, rows)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def finite_mean(values: Iterable[Any]) -> float:
    xs = [finite_float(v) for v in values]
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def sanitize_name(text: Any) -> str:
    out = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text))
    return out.strip("_") or "item"


def parse_int_csv(text: Any) -> List[int]:
    values: List[int] = []
    for part in str(text).replace("|", ",").split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def parse_str_csv(text: Any) -> List[str]:
    return [part.strip() for part in str(text).replace("|", ",").split(",") if part.strip()]


def margin_tag(margin: int) -> str:
    return f"m{int(margin)}"


def choose_device(device_arg: str) -> torch.device:
    text = str(device_arg).strip().lower()
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(text)


def parse_case_ids(text: Any) -> List[str]:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    return [normalize_case_id(x) for x in str(text).replace(",", "|").split("|") if str(x).strip()]


def class_name(cls: int) -> str:
    return str(FOREGROUND_CLASS_NAMES.get(int(cls), f"class_{int(cls)}"))


def select_run_cases(args: argparse.Namespace) -> List[Dict[str, Any]]:
    selection_csv = resolve_path(args.selection_csv)
    if not selection_csv.exists():
        raise FileNotFoundError(f"Selection CSV does not exist: {selection_csv}")
    df = pd.read_csv(selection_csv)
    key = ["shot", "seed", "domain", "case_id"]
    required = set(key + ["diagnostic_variant", "case_dice", "case_hd95", "pred_to_gt_p95", "gt_to_pred_p95"])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{selection_csv} is missing columns: {missing}")

    sadg = df[df["diagnostic_variant"] == ab7.SADG_NONE].copy()
    radial = df[df["diagnostic_variant"] == ab7.RADIAL_NONE].copy()
    if sadg.empty or radial.empty:
        raise ValueError("Selection CSV must contain both SADG-none and RadialGate-none rows.")

    merged = radial.merge(sadg, on=key, suffixes=("_radial", "_sadg"))
    merged["dice_delta"] = merged["case_dice_radial"] - merged["case_dice_sadg"]
    merged["hd95_delta"] = merged["case_hd95_radial"] - merged["case_hd95_sadg"]
    merged["pred_to_gt_p95_delta"] = merged["pred_to_gt_p95_radial"] - merged["pred_to_gt_p95_sadg"]
    merged["gt_to_pred_p95_delta"] = merged["gt_to_pred_p95_radial"] - merged["gt_to_pred_p95_sadg"]

    if bool(args.require_dice_up):
        merged = merged[merged["dice_delta"] > 0]
    if bool(args.require_hd95_up):
        merged = merged[merged["hd95_delta"] > 0]
    if bool(args.require_pred_to_gt_dominant):
        merged = merged[merged["pred_to_gt_p95_radial"] > merged["gt_to_pred_p95_radial"]]

    if str(args.selection_unit).strip().lower() != "run_case":
        raise ValueError("Only --selection_unit run_case is implemented for this audit.")

    selected_rows: List[Dict[str, Any]] = []
    for domain, group in merged.groupby("domain", sort=True):
        top = group.sort_values(["hd95_delta", "dice_delta"], ascending=[False, False]).head(int(args.top_cases_per_domain))
        if len(top) < int(args.top_cases_per_domain):
            print(f"[WARN] domain={domain} only has {len(top)} eligible rows.", flush=True)
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            out: Dict[str, Any] = {
                "selected_rank": int(rank),
                "shot": int(row["shot"]),
                "seed": int(row["seed"]),
                "domain": str(row["domain"]),
                "case_id": normalize_case_id(row["case_id"]),
                "selection_unit": "run_case",
                "case_dice_sadg": finite_float(row["case_dice_sadg"]),
                "case_dice_radial": finite_float(row["case_dice_radial"]),
                "case_hd95_sadg": finite_float(row["case_hd95_sadg"]),
                "case_hd95_radial": finite_float(row["case_hd95_radial"]),
                "dice_delta": finite_float(row["dice_delta"]),
                "hd95_delta": finite_float(row["hd95_delta"]),
                "pred_to_gt_p95_sadg": finite_float(row["pred_to_gt_p95_sadg"]),
                "pred_to_gt_p95_radial": finite_float(row["pred_to_gt_p95_radial"]),
                "gt_to_pred_p95_sadg": finite_float(row["gt_to_pred_p95_sadg"]),
                "gt_to_pred_p95_radial": finite_float(row["gt_to_pred_p95_radial"]),
                "pred_to_gt_p95_delta": finite_float(row["pred_to_gt_p95_delta"]),
                "gt_to_pred_p95_delta": finite_float(row["gt_to_pred_p95_delta"]),
                "radialgate_checkpoint_path": str(row.get("checkpoint_path_radial", "")),
                "sadg_checkpoint_path": str(row.get("checkpoint_path_sadg", "")),
                "train_case_ids": str(row.get("train_case_ids_radial", "")),
                "sagittal_x_indices": str(row.get("sagittal_x_indices_radial", "")),
            }
            for cls in FOREGROUND_CLASSES:
                name = class_name(int(cls))
                for metric in ("case_dice", "case_hd95", "pred_to_gt_p95", "gt_to_pred_p95", "volume_bias"):
                    radial_col = f"{metric}_{name}_radial"
                    sadg_col = f"{metric}_{name}_sadg"
                    if radial_col in row:
                        out[f"{metric}_{name}_radial"] = finite_float(row[radial_col])
                    if sadg_col in row:
                        out[f"{metric}_{name}_sadg"] = finite_float(row[sadg_col])
            selected_rows.append(out)
    return selected_rows


def load_single_case_items(args: argparse.Namespace, domain: str, case_id: str) -> List[Dict[str, Any]]:
    records = collect_case_records(domain, data_root=resolve_path(args.data_root))
    case_key = normalize_case_id(case_id)
    matches = [record for record in records if normalize_case_id(record.case_id) == case_key]
    if not matches:
        raise RuntimeError(f"Could not find case_id={case_id} in domain={domain}")
    items = load_case_slices(
        matches[0],
        resize_hw=int(args.resize_hw),
        min_fg_ratio=float(args.min_fg_ratio),
        slice_policy=str(args.slice_policy),
        num_middle_slices=int(args.num_middle_slices),
        filter_min_fg=bool(args.filter_min_fg),
    )
    if not items:
        raise RuntimeError(f"No slices left for case_id={case_id} in domain={domain}; check filtering settings.")
    return items


@torch.no_grad()
def predict_case(
    model: torch.nn.Module,
    items: Sequence[Dict[str, Any]],
    *,
    device: torch.device,
    eval_batch_size: int,
) -> Dict[str, Any]:
    images = torch.stack([torch.from_numpy(item["image"])[None, ...].float() for item in items], dim=0)
    preds: List[np.ndarray] = []
    model.eval()
    for start in range(0, int(images.shape[0]), int(eval_batch_size)):
        batch = images[start:start + int(eval_batch_size)].to(device)
        logits = model(batch)
        if isinstance(logits, Mapping):
            logits = logits["logits"]
        preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64))
    return {
        "pred_stack": np.concatenate(preds, axis=0),
        "gt_stack": np.stack([item["mask"] for item in items], axis=0).astype(np.int64),
        "image_stack": np.stack([item["image"] for item in items], axis=0).astype(np.float32),
        "case_spacing": tuple(float(x) for x in items[0]["case_spacing"]),
        "slice_spacing": tuple(float(x) for x in items[0]["slice_spacing"]),
        "sagittal_x_indices": [int(item["sagittal_x_index"]) for item in items],
    }


class ReflectFFTStyleLayer(torch.nn.Module):
    def __init__(self, base_layer: torch.nn.Module, pad_px: int):
        super().__init__()
        self.base_layer = base_layer
        self.pad_px = int(pad_px)
        if self.pad_px < 0:
            raise ValueError("reflect FFT pad must be non-negative.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_px <= 0:
            return self.base_layer(x)
        pad = int(self.pad_px)
        xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        X = torch.fft.fft2(xp, norm=self.base_layer.fft_norm)
        Xs = torch.fft.fftshift(X, dim=(-2, -1))
        hp, wp = int(xp.shape[-2]), int(xp.shape[-1])
        h_lf, w_lf = int(self.base_layer.H_lf), int(self.base_layer.W_lf)
        u0 = hp // 2 - h_lf // 2
        v0 = wp // 2 - w_lf // 2
        u1 = u0 + h_lf
        v1 = v0 + w_lf
        X_lf = Xs[:, :, u0:u1, v0:v1]

        A_lf = torch.abs(X_lf)
        phase_lf = X_lf / (A_lf + float(self.base_layer.eps))
        T = self.base_layer.symmetric_template().to(dtype=A_lf.dtype, device=A_lf.device)
        rho = self.base_layer.rho_map().to(dtype=A_lf.dtype, device=A_lf.device)
        logA_cal = (1.0 - rho) * torch.log(A_lf + float(self.base_layer.eps)) + rho * T
        X_lf_cal = torch.exp(logA_cal) * phase_lf

        mask = self.base_layer.cosine_mask.to(dtype=X_lf.real.dtype, device=X_lf.device)
        X_lf_new = mask * X_lf_cal + (1.0 - mask) * X_lf
        Xs_new = Xs.clone()
        Xs_new[:, :, u0:u1, v0:v1] = X_lf_new
        X_new = torch.fft.ifftshift(Xs_new, dim=(-2, -1))
        out = torch.fft.ifft2(X_new, norm=self.base_layer.fft_norm).real
        return out[:, :, pad:-pad, pad:-pad]


class ReflectFFTStyleModel(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, style_layer: torch.nn.Module, pad_px: int):
        super().__init__()
        self.backbone = backbone
        self.reflect_style_layer = ReflectFFTStyleLayer(style_layer, pad_px)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self.reflect_style_layer(x)
        return self.backbone(x, return_features=return_features)


class SpatialBlendStyleModel(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, style_layer: torch.nn.Module, margin_px: int):
        super().__init__()
        self.backbone = backbone
        self.style_layer = style_layer
        self.margin_px = int(margin_px)
        if self.margin_px <= 0:
            raise ValueError("spatial blend margin must be positive.")

    def spatial_ramp(self, x: torch.Tensor) -> torch.Tensor:
        h, w = int(x.shape[-2]), int(x.shape[-1])
        z = torch.arange(h, dtype=x.dtype, device=x.device)[:, None]
        y = torch.arange(w, dtype=x.dtype, device=x.device)[None, :]
        d = torch.minimum(torch.minimum(z, float(h - 1) - z), torch.minimum(y, float(w - 1) - y))
        t = (d / float(self.margin_px)).clamp(0.0, 1.0)
        ramp = 0.5 - 0.5 * torch.cos(float(np.pi) * t)
        return ramp[None, None, :, :]

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x_rg = self.style_layer(x)
        ramp = self.spatial_ramp(x)
        x_blend = x + ramp * (x_rg - x)
        return self.backbone(x_blend, return_features=return_features)


class ReflectPadInferenceModel(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, style_layer: torch.nn.Module, pad_px: int):
        super().__init__()
        self.backbone = backbone
        self.style_layer = style_layer
        self.pad_px = int(pad_px)
        if self.pad_px < 0:
            raise ValueError("reflect inference pad must be non-negative.")

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self.style_layer(x)
        if self.pad_px <= 0:
            return self.backbone(x, return_features=return_features)
        pad = int(self.pad_px)
        xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        out = self.backbone(xp, return_features=return_features)
        if isinstance(out, Mapping):
            out = dict(out)
            out["logits"] = out["logits"][:, :, pad:-pad, pad:-pad]
            return out
        return out[:, :, pad:-pad, pad:-pad]


def build_cause_models(radial_model: torch.nn.Module, args: argparse.Namespace) -> Dict[str, torch.nn.Module]:
    backbone = ab7.unwrap_backbone(radial_model)
    style_layer = ab7.unwrap_style_layer(radial_model)
    if style_layer is None:
        raise ValueError("RadialGate model does not contain a style layer.")
    requested = parse_str_csv(args.cause_variants)
    valid = set(DEFAULT_CAUSE_VARIANTS)
    unknown = [variant for variant in requested if variant not in valid]
    if unknown:
        raise ValueError(f"Unknown cause variants: {unknown}. Valid variants: {sorted(valid)}")
    out: Dict[str, torch.nn.Module] = {}
    for variant in requested:
        if variant == CAUSE_RADIAL_NONE:
            out[variant] = radial_model
        elif variant == CAUSE_REFLECT_FFT:
            out[variant] = ReflectFFTStyleModel(backbone, style_layer, int(args.reflect_fft_pad_px))
        elif variant == CAUSE_SPATIAL_BLEND:
            out[variant] = SpatialBlendStyleModel(backbone, style_layer, int(args.spatial_blend_margin_px))
        elif variant == CAUSE_REFLECT_PAD_INFERENCE:
            out[variant] = ReflectPadInferenceModel(backbone, style_layer, int(args.reflect_inference_pad_px))
        out[variant].eval()
    return out


def diag_surface(mask: np.ndarray) -> np.ndarray:
    return ab7._diag_surface(mask)


def diagonal_length(shape: Sequence[int], spacing: Sequence[float]) -> float:
    return ab7._diag_diagonal_length(shape, spacing)


def distance_to_gt_surface(gt_mask: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
    gt_surface = diag_surface(gt_mask)
    if not np.any(gt_surface):
        return np.full(gt_mask.shape, diagonal_length(gt_mask.shape, spacing), dtype=np.float32)
    return ndi.distance_transform_edt(~gt_surface, sampling=spacing).astype(np.float32)


def component_labels(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    return ab7.component_sizes(mask)


def mode_positive(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=np.int64)
    values = values[values > 0]
    if values.size == 0:
        return 0
    counts = np.bincount(values)
    return int(np.argmax(counts))


def border_distance_maps(shape: Sequence[int], spacing: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    if len(shape) != 3:
        raise ValueError(f"Expected [slices,H,W] stack shape, got {shape}")
    h, w = int(shape[1]), int(shape[2])
    z = np.arange(h, dtype=np.float32)[:, None]
    y = np.arange(w, dtype=np.float32)[None, :]
    border_px = np.minimum(
        np.minimum(z, float(h - 1) - z),
        np.minimum(y, float(w - 1) - y),
    ).astype(np.float32)
    sp = ab7._diag_normalize_spacing(spacing, 3)
    z_spacing = float(sp[1])
    y_spacing = float(sp[2])
    border_mm = np.minimum(
        np.minimum(z * z_spacing, (float(h - 1) - z) * z_spacing),
        np.minimum(y * y_spacing, (float(w - 1) - y) * y_spacing),
    ).astype(np.float32)
    return border_px, border_mm


def add_border_stats(
    item: Dict[str, Any],
    *,
    coords: np.ndarray,
    border_px_map: np.ndarray,
    border_mm_map: np.ndarray,
    margins: Sequence[int],
) -> None:
    if coords.size == 0:
        item["border_distance_px_min"] = float("nan")
        item["border_distance_px_mean"] = float("nan")
        item["border_distance_mm_min"] = float("nan")
        item["border_distance_mm_mean"] = float("nan")
        for margin in margins:
            tag = margin_tag(int(margin))
            item[f"is_border_artifact_{tag}"] = False
            item[f"border_far_point_ratio_{tag}"] = float("nan")
        return
    z_idx = coords[:, 1].astype(np.int64)
    y_idx = coords[:, 2].astype(np.int64)
    px_values = border_px_map[z_idx, y_idx]
    mm_values = border_mm_map[z_idx, y_idx]
    item["border_distance_px_min"] = float(np.min(px_values))
    item["border_distance_px_mean"] = float(np.mean(px_values))
    item["border_distance_mm_min"] = float(np.min(mm_values))
    item["border_distance_mm_mean"] = float(np.mean(mm_values))
    for margin in margins:
        tag = margin_tag(int(margin))
        hits = px_values < float(margin)
        item[f"is_border_artifact_{tag}"] = bool(np.any(hits))
        item[f"border_far_point_ratio_{tag}"] = float(np.mean(hits))


def point_border_values(
    coord: Sequence[int],
    border_px_map: np.ndarray,
    border_mm_map: np.ndarray,
) -> tuple[float, float]:
    z = int(coord[1])
    y = int(coord[2])
    return float(border_px_map[z, y]), float(border_mm_map[z, y])


def apply_border_prune(pred: np.ndarray, margin_px: int) -> np.ndarray:
    out = np.asarray(pred).astype(np.int64, copy=True)
    margin = int(margin_px)
    if margin <= 0:
        return out
    if out.ndim != 3:
        raise ValueError(f"Expected [slices,H,W] prediction stack, got shape={out.shape}")
    h, w = int(out.shape[1]), int(out.shape[2])
    z = np.arange(h)[:, None]
    y = np.arange(w)[None, :]
    border_mask = np.minimum(np.minimum(z, h - 1 - z), np.minimum(y, w - 1 - y)) < margin
    out[:, border_mask] = 0
    return out


def threshold_for_class(row: Mapping[str, Any], cls: int, args: argparse.Namespace, dist_values: np.ndarray) -> float:
    mode = str(args.threshold_mode).strip().lower()
    name = class_name(cls)
    class_p95 = finite_float(row.get(f"pred_to_gt_p95_{name}_radial"), default=float("nan"))
    if not np.isfinite(class_p95):
        class_p95 = float(np.percentile(dist_values, 95.0)) if dist_values.size else finite_float(row.get("pred_to_gt_p95_radial"), 0.0)
    if mode == "adaptive_p95":
        return float(max(float(args.threshold_min_mm), float(args.threshold_p95_fraction) * class_p95))
    if mode == "fixed":
        return float(args.threshold_min_mm)
    raise ValueError(f"Unknown threshold_mode={args.threshold_mode!r}")


def nearest_gt_class_at(
    coord: Sequence[int],
    gt_distance_by_class: Mapping[int, np.ndarray],
) -> tuple[int, str, float]:
    z = tuple(int(x) for x in coord)
    best_cls = 0
    best_dist = float("inf")
    for cls, dist_map in gt_distance_by_class.items():
        value = finite_float(dist_map[z], default=float("inf"))
        if value < best_dist:
            best_cls = int(cls)
            best_dist = float(value)
    return best_cls, class_name(best_cls) if best_cls else "", best_dist


def heuristic_label(
    *,
    border_distance_px_min: float,
    border_artifact_primary_margin_px: int,
    small_component_flag: bool,
    is_largest_pred_component: bool,
    volume_bias: float,
    nearest_gt_class_id: int,
    cls: int,
    nearest_gt_distance_mm: float,
    threshold_mm: float,
) -> str:
    if np.isfinite(border_distance_px_min) and float(border_distance_px_min) < float(border_artifact_primary_margin_px):
        return "fov_border_artifact"
    if small_component_flag and not is_largest_pred_component:
        return "small_fp_error"
    if nearest_gt_class_id > 0 and int(nearest_gt_class_id) != int(cls) and nearest_gt_distance_mm <= threshold_mm:
        return "plausible_anatomy_gt_missing"
    if is_largest_pred_component and np.isfinite(volume_bias) and volume_bias > 0.1:
        return "boundary_overexpansion"
    return "ambiguous"


def extract_far_boundary(
    *,
    row: Mapping[str, Any],
    case_data: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    pred = np.asarray(case_data["pred_stack"], dtype=np.int64)
    gt = np.asarray(case_data["gt_stack"], dtype=np.int64)
    spacing = tuple(float(x) for x in case_data["case_spacing"])
    sagittal_x_indices = list(case_data["sagittal_x_indices"])
    border_margins = parse_int_csv(args.border_margins_px)
    border_px_map, border_mm_map = border_distance_maps(pred.shape, spacing)

    point_rows: List[Dict[str, Any]] = []
    component_rows: List[Dict[str, Any]] = []
    border_component_rows: List[Dict[str, Any]] = []
    far_masks_by_class: Dict[int, np.ndarray] = {}
    far_component_labels_by_class: Dict[int, np.ndarray] = {}
    gt_distance_by_class = {int(cls): distance_to_gt_surface(gt == int(cls), spacing) for cls in FOREGROUND_CLASSES}

    for cls in FOREGROUND_CLASSES:
        cls = int(cls)
        name = class_name(cls)
        pred_c = pred == cls
        gt_c = gt == cls
        pred_surface = diag_surface(pred_c)
        if not np.any(pred_surface):
            far_masks_by_class[cls] = np.zeros_like(pred_c, dtype=bool)
            far_component_labels_by_class[cls] = np.zeros_like(pred_c, dtype=np.int32)
            continue
        dist_to_gt = distance_to_gt_surface(gt_c, spacing)
        surface_distances = dist_to_gt[pred_surface]
        threshold_mm = threshold_for_class(row, cls, args, surface_distances)
        far_mask = pred_surface & (dist_to_gt > threshold_mm)
        far_labels, far_sizes, far_n = component_labels(far_mask)
        far_masks_by_class[cls] = far_mask
        far_component_labels_by_class[cls] = far_labels

        pred_labels, pred_sizes, pred_n = component_labels(pred_c)
        largest_pred_label = int(np.argmax(pred_sizes) + 1) if pred_sizes.size else 0
        largest_pred_size = int(np.max(pred_sizes)) if pred_sizes.size else 0
        small_threshold = max(
            int(args.prune_min_voxels),
            int(math.ceil(float(args.prune_min_largest_ratio) * float(largest_pred_size))),
        ) if largest_pred_size > 0 else int(args.prune_min_voxels)
        volume_bias = finite_float(row.get(f"volume_bias_{name}_radial"), default=float("nan"))

        candidate_components: List[Dict[str, Any]] = []
        for comp_id in range(1, int(far_n) + 1):
            comp_mask = far_labels == comp_id
            coords = np.argwhere(comp_mask)
            if coords.size == 0:
                continue
            distances = dist_to_gt[comp_mask]
            max_local = int(np.argmax(distances))
            max_coord = coords[max_local]
            max_distance = float(distances[max_local])
            pred_comp_id = mode_positive(pred_labels[comp_mask])
            pred_comp_size = int(pred_sizes[pred_comp_id - 1]) if pred_comp_id > 0 and pred_comp_id <= len(pred_sizes) else 0
            nearest_cls, nearest_name, nearest_dist = nearest_gt_class_at(max_coord, gt_distance_by_class)
            is_largest = bool(pred_comp_id > 0 and pred_comp_id == largest_pred_label)
            small_flag = bool(pred_comp_size > 0 and pred_comp_size < small_threshold)
            comp_item = {
                "class_id": cls,
                "class_name": name,
                "component_id": int(comp_id),
                "component_voxels": int(coords.shape[0]),
                "max_distance_mm": max_distance,
                "mean_distance_mm": float(np.mean(distances)),
                "median_distance_mm": float(np.median(distances)),
                "threshold_mm": float(threshold_mm),
                "max_point_slice_pos": int(max_coord[0]),
                "max_point_sagittal_x_index": int(sagittal_x_indices[int(max_coord[0])]),
                "max_point_z": int(max_coord[1]),
                "max_point_y": int(max_coord[2]),
                "pred_component_id": int(pred_comp_id),
                "pred_component_size": int(pred_comp_size),
                "largest_pred_component_size": int(largest_pred_size),
                "small_component_threshold": int(small_threshold),
                "is_largest_pred_component": bool(is_largest),
                "small_component_flag": bool(small_flag),
                "nearest_gt_class_id": int(nearest_cls),
                "nearest_gt_class_name": nearest_name,
                "nearest_gt_distance_mm": float(nearest_dist),
                "volume_bias": volume_bias,
            }
            add_border_stats(
                comp_item,
                coords=coords,
                border_px_map=border_px_map,
                border_mm_map=border_mm_map,
                margins=border_margins,
            )
            comp_item["heuristic_suggestion"] = heuristic_label(
                border_distance_px_min=finite_float(comp_item.get("border_distance_px_min")),
                border_artifact_primary_margin_px=int(args.border_artifact_primary_margin_px),
                small_component_flag=small_flag,
                is_largest_pred_component=is_largest,
                volume_bias=volume_bias,
                nearest_gt_class_id=nearest_cls,
                cls=cls,
                nearest_gt_distance_mm=nearest_dist,
                threshold_mm=threshold_mm,
            )
            candidate_components.append(comp_item)

        selected_components = sorted(candidate_components, key=lambda x: (x["max_distance_mm"], x["component_voxels"]), reverse=True)
        selected_component_ids = {int(item["component_id"]) for item in selected_components[: int(args.top_components_per_case)]}
        for rank, comp in enumerate(selected_components, start=1):
            border_row = common_case_fields(row)
            border_row.update(comp)
            border_row["component_rank_in_class"] = int(rank)
            border_row["component_selected_for_review"] = bool(int(comp["component_id"]) in selected_component_ids)
            border_component_rows.append(border_row)
        for rank, comp in enumerate(selected_components[: int(args.top_components_per_case)], start=1):
            comp_row = common_case_fields(row)
            comp_row.update(comp)
            comp_row["component_rank_in_class"] = int(rank)
            comp_row["component_selected_for_review"] = True
            component_rows.append(comp_row)

        coords_all = np.argwhere(far_mask)
        for coord in coords_all:
            comp_id = int(far_labels[tuple(coord)])
            border_px, border_mm = point_border_values(coord, border_px_map, border_mm_map)
            point_row = common_case_fields(row)
            point_row.update(
                {
                    "class_id": cls,
                    "class_name": name,
                    "component_id": comp_id,
                    "component_selected_for_review": bool(comp_id in selected_component_ids),
                    "slice_pos": int(coord[0]),
                    "sagittal_x_index": int(sagittal_x_indices[int(coord[0])]),
                    "z": int(coord[1]),
                    "y": int(coord[2]),
                    "distance_mm": float(dist_to_gt[tuple(coord)]),
                    "threshold_mm": float(threshold_mm),
                    "pred_component_id": int(pred_labels[tuple(coord)]),
                    "border_distance_px": float(border_px),
                    "border_distance_mm": float(border_mm),
                }
            )
            for margin in border_margins:
                tag = margin_tag(int(margin))
                point_row[f"is_border_artifact_{tag}"] = bool(border_px < float(margin))
            point_rows.append(point_row)

    return point_rows, component_rows, border_component_rows, far_masks_by_class, far_component_labels_by_class


def common_case_fields(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "shot": int(row["shot"]),
        "seed": int(row["seed"]),
        "domain": str(row["domain"]),
        "case_id": normalize_case_id(row["case_id"]),
        "selected_rank": int(row.get("selected_rank", 0)),
        "dice_delta": finite_float(row.get("dice_delta")),
        "hd95_delta": finite_float(row.get("hd95_delta")),
        "case_dice_sadg": finite_float(row.get("case_dice_sadg")),
        "case_dice_radial": finite_float(row.get("case_dice_radial")),
        "case_hd95_sadg": finite_float(row.get("case_hd95_sadg")),
        "case_hd95_radial": finite_float(row.get("case_hd95_radial")),
        "pred_to_gt_p95_radial": finite_float(row.get("pred_to_gt_p95_radial")),
        "gt_to_pred_p95_radial": finite_float(row.get("gt_to_pred_p95_radial")),
    }


def border_pruned_variant_name(args: argparse.Namespace, margin_px: int) -> str:
    return f"{args.border_pruned_variant_prefix}-{margin_tag(int(margin_px))}"


def evaluate_prediction_stack(
    *,
    row: Mapping[str, Any],
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: Sequence[float],
    args: argparse.Namespace,
    variant: str,
    border_margin_px: int,
    is_border_pruned: bool,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    dice_cls = ab7.case_dice_by_class(pred, gt)
    hd95_cls = ab7.case_hd95_by_class(pred, gt, spacing=spacing)
    class_diagnostics: Dict[int, Dict[str, float]] = {}
    for cls in FOREGROUND_CLASSES:
        class_diagnostics[int(cls)] = ab7.class_component_diagnostics(
            pred,
            gt,
            cls=int(cls),
            spacing=spacing,
            args=args,
        )

    case_row: Dict[str, Any] = {
        **common_case_fields(row),
        "diagnostic_variant": variant,
        "border_margin_px": int(border_margin_px),
        "is_border_pruned": bool(is_border_pruned),
        "case_dice": finite_mean(dice_cls.values()),
        "case_hd95": finite_mean(hd95_cls.values()),
        "pred_to_gt_p95": finite_mean(diag.get("pred_to_gt_p95") for diag in class_diagnostics.values()),
        "gt_to_pred_p95": finite_mean(diag.get("gt_to_pred_p95") for diag in class_diagnostics.values()),
        "directed_p95_gap": finite_mean(diag.get("directed_p95_gap") for diag in class_diagnostics.values()),
        "volume_bias": finite_mean(diag.get("volume_bias") for diag in class_diagnostics.values()),
    }
    class_rows: List[Dict[str, Any]] = []
    for cls in FOREGROUND_CLASSES:
        cls = int(cls)
        name = class_name(cls)
        diag = class_diagnostics[cls]
        case_row[f"case_dice_{name}"] = finite_float(dice_cls.get(cls))
        case_row[f"case_hd95_{name}"] = finite_float(hd95_cls.get(cls))
        for key in ("pred_to_gt_p95", "gt_to_pred_p95", "directed_p95_gap", "volume_bias"):
            case_row[f"{key}_{name}"] = finite_float(diag.get(key))
        class_rows.append(
            {
                **common_case_fields(row),
                "diagnostic_variant": variant,
                "border_margin_px": int(border_margin_px),
                "is_border_pruned": bool(is_border_pruned),
                "class_id": cls,
                "class_name": name,
                "case_dice": finite_float(dice_cls.get(cls)),
                "case_hd95": finite_float(hd95_cls.get(cls)),
                "pred_to_gt_p95": finite_float(diag.get("pred_to_gt_p95")),
                "gt_to_pred_p95": finite_float(diag.get("gt_to_pred_p95")),
                "directed_p95_gap": finite_float(diag.get("directed_p95_gap")),
                "volume_bias": finite_float(diag.get("volume_bias")),
            }
        )
    return case_row, class_rows


def evaluate_border_pruned_variants(
    *,
    row: Mapping[str, Any],
    case_data: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pred = np.asarray(case_data["pred_stack"], dtype=np.int64)
    gt = np.asarray(case_data["gt_stack"], dtype=np.int64)
    spacing = tuple(float(x) for x in case_data["case_spacing"])
    margins = parse_int_csv(args.border_pruned_margins_px)
    case_rows: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []

    base_case, base_class = evaluate_prediction_stack(
        row=row,
        pred=pred,
        gt=gt,
        spacing=spacing,
        args=args,
        variant=ab7.RADIAL_NONE,
        border_margin_px=0,
        is_border_pruned=False,
    )
    case_rows.append(base_case)
    class_rows.extend(base_class)

    for margin in margins:
        pruned_pred = apply_border_prune(pred, int(margin))
        case_row, cls_rows = evaluate_prediction_stack(
            row=row,
            pred=pruned_pred,
            gt=gt,
            spacing=spacing,
            args=args,
            variant=border_pruned_variant_name(args, int(margin)),
            border_margin_px=int(margin),
            is_border_pruned=True,
        )
        case_rows.append(case_row)
        class_rows.extend(cls_rows)
    return case_rows, class_rows


def summarize_border_pruned_eval(case_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not case_rows:
        return []
    df = pd.DataFrame(case_rows)
    metric_cols = [
        "case_dice",
        "case_hd95",
        "pred_to_gt_p95",
        "gt_to_pred_p95",
        "directed_p95_gap",
        "volume_bias",
    ]
    rows: List[Dict[str, Any]] = []
    for (domain, variant, margin), group in df.groupby(["domain", "diagnostic_variant", "border_margin_px"], sort=True):
        item: Dict[str, Any] = {
            "domain": domain,
            "diagnostic_variant": variant,
            "border_margin_px": int(margin),
            "n_cases": int(len(group)),
        }
        for metric in metric_cols:
            item[metric] = finite_mean(group[metric])
        rows.append(item)
    for (variant, margin), group in df.groupby(["diagnostic_variant", "border_margin_px"], sort=True):
        item = {
            "domain": "ALL_DOMAINS",
            "diagnostic_variant": variant,
            "border_margin_px": int(margin),
            "n_cases": int(len(group)),
        }
        for metric in metric_cols:
            item[metric] = finite_mean(group[metric])
        rows.append(item)
    return rows


def summarize_border_pruned_domain_class(class_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not class_rows:
        return []
    df = pd.DataFrame(class_rows)
    metric_cols = ["case_dice", "case_hd95", "pred_to_gt_p95", "gt_to_pred_p95", "directed_p95_gap", "volume_bias"]
    rows: List[Dict[str, Any]] = []
    for (domain, variant, margin, cls, name), group in df.groupby(
        ["domain", "diagnostic_variant", "border_margin_px", "class_id", "class_name"],
        sort=True,
    ):
        item: Dict[str, Any] = {
            "domain": domain,
            "diagnostic_variant": variant,
            "border_margin_px": int(margin),
            "class_id": int(cls),
            "class_name": name,
            "n_cases": int(len(group)),
        }
        for metric in metric_cols:
            item[metric] = finite_mean(group[metric])
        rows.append(item)
    return rows


def summarize_border_pruned_delta(case_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not case_rows:
        return []
    df = pd.DataFrame(case_rows)
    key = ["shot", "seed", "domain", "case_id"]
    metrics = ["case_dice", "case_hd95", "pred_to_gt_p95", "gt_to_pred_p95", "directed_p95_gap", "volume_bias"]
    base = df[df["diagnostic_variant"] == ab7.RADIAL_NONE][key + metrics].copy()
    pruned = df[df["diagnostic_variant"] != ab7.RADIAL_NONE][
        key + ["diagnostic_variant", "border_margin_px"] + metrics
    ].copy()
    if base.empty or pruned.empty:
        return []
    base = base.rename(columns={metric: f"{metric}_base" for metric in metrics})
    pruned = pruned.rename(columns={metric: f"{metric}_pruned" for metric in metrics})
    merged = pruned.merge(base, on=key)
    for metric in metrics:
        merged[f"{metric}_delta"] = merged[f"{metric}_pruned"] - merged[f"{metric}_base"]
    merged["hd95_reduction_mm"] = -merged["case_hd95_delta"]
    merged["hd95_reduction_pct"] = np.where(
        merged["case_hd95_base"].astype(float) > 0,
        (merged["case_hd95_base"] - merged["case_hd95_pruned"]) / merged["case_hd95_base"],
        np.nan,
    )
    merged["dice_drop_small"] = merged["case_dice_delta"] > -0.01
    merged["hd95_improved_5mm"] = merged["case_hd95_delta"] <= -5.0
    merged["border_artifact_evidence"] = merged["dice_drop_small"] & merged["hd95_improved_5mm"]

    rows: List[Dict[str, Any]] = []
    group_cols = ["domain", "diagnostic_variant", "border_margin_px"]
    for key_vals, group in merged.groupby(group_cols, sort=True):
        domain, variant, margin = key_vals
        item = {
            "domain": domain,
            "diagnostic_variant": variant,
            "border_margin_px": int(margin),
            "n_cases": int(len(group)),
        }
        for metric in metrics:
            item[f"{metric}_delta_mean"] = finite_mean(group[f"{metric}_delta"])
        item["hd95_reduction_mm_mean"] = finite_mean(group["hd95_reduction_mm"])
        item["hd95_reduction_pct_mean"] = finite_mean(group["hd95_reduction_pct"])
        item["dice_drop_small_rate"] = float(np.mean(group["dice_drop_small"]))
        item["hd95_improved_5mm_rate"] = float(np.mean(group["hd95_improved_5mm"]))
        item["border_artifact_evidence_rate"] = float(np.mean(group["border_artifact_evidence"]))
        rows.append(item)
    for (variant, margin), group in merged.groupby(["diagnostic_variant", "border_margin_px"], sort=True):
        item = {
            "domain": "ALL_DOMAINS",
            "diagnostic_variant": variant,
            "border_margin_px": int(margin),
            "n_cases": int(len(group)),
        }
        for metric in metrics:
            item[f"{metric}_delta_mean"] = finite_mean(group[f"{metric}_delta"])
        item["hd95_reduction_mm_mean"] = finite_mean(group["hd95_reduction_mm"])
        item["hd95_reduction_pct_mean"] = finite_mean(group["hd95_reduction_pct"])
        item["dice_drop_small_rate"] = float(np.mean(group["dice_drop_small"]))
        item["hd95_improved_5mm_rate"] = float(np.mean(group["hd95_improved_5mm"]))
        item["border_artifact_evidence_rate"] = float(np.mean(group["border_artifact_evidence"]))
        rows.append(item)
    return rows


def threshold_row_for_variant(selection_row: Mapping[str, Any], case_metric_row: Mapping[str, Any]) -> Dict[str, Any]:
    row = dict(selection_row)
    row["pred_to_gt_p95_radial"] = finite_float(case_metric_row.get("pred_to_gt_p95"))
    row["gt_to_pred_p95_radial"] = finite_float(case_metric_row.get("gt_to_pred_p95"))
    for cls in FOREGROUND_CLASSES:
        name = class_name(int(cls))
        for metric in ("pred_to_gt_p95", "gt_to_pred_p95", "volume_bias"):
            value = case_metric_row.get(f"{metric}_{name}")
            if value is not None:
                row[f"{metric}_{name}_radial"] = finite_float(value)
    return row


def summarize_outlier_cause_border_artifacts(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    margin_cols = sorted([col for col in df.columns if col.startswith("is_border_artifact_m")])
    out: List[Dict[str, Any]] = []
    for (domain, variant), group in df.groupby(["domain", "diagnostic_variant"], sort=True):
        item: Dict[str, Any] = {
            "domain": domain,
            "diagnostic_variant": variant,
            "n_components": int(len(group)),
            "max_distance_mm_mean": finite_mean(group.get("max_distance_mm", [])),
            "border_distance_px_min_mean": finite_mean(group.get("border_distance_px_min", [])),
            "border_distance_mm_min_mean": finite_mean(group.get("border_distance_mm_min", [])),
        }
        for col in margin_cols:
            item[f"{col}_ratio"] = finite_mean(group[col])
            item[f"{col}_count"] = int(np.nansum(group[col].astype(bool)))
        out.append(item)
    for variant, group in df.groupby("diagnostic_variant", sort=True):
        item = {
            "domain": "ALL_DOMAINS",
            "diagnostic_variant": variant,
            "n_components": int(len(group)),
            "max_distance_mm_mean": finite_mean(group.get("max_distance_mm", [])),
            "border_distance_px_min_mean": finite_mean(group.get("border_distance_px_min", [])),
            "border_distance_mm_min_mean": finite_mean(group.get("border_distance_mm_min", [])),
        }
        for col in margin_cols:
            item[f"{col}_ratio"] = finite_mean(group[col])
            item[f"{col}_count"] = int(np.nansum(group[col].astype(bool)))
        out.append(item)
    return out


def prepare_output_path(path: str | Path, overwrite: bool) -> Path:
    result_root = resolve_path(path)
    if result_root.exists() and bool(overwrite):
        shutil.rmtree(result_root)
    elif result_root.exists() and any(result_root.iterdir()):
        raise FileExistsError(f"Result root already exists and is not empty: {result_root}. Use --overwrite.")
    ensure_dir(result_root)
    return result_root


def write_cause_report(
    result_root: Path,
    selected_rows: Sequence[Dict[str, Any]],
    eval_rows: Sequence[Dict[str, Any]],
    delta_rows: Sequence[Dict[str, Any]],
    border_rows: Sequence[Dict[str, Any]],
    variant_names: Sequence[str],
) -> None:
    lines: List[str] = []
    lines.append("# RadialGate Outlier Cause Diagnostic Report")
    lines.append("")
    lines.append("本报告只做 diagnostic-only 复评，不重新训练、不修改 checkpoint。")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Selected run-cases: {len(selected_rows)}")
    lines.append(f"- Variants: {', '.join(str(x) for x in variant_names)}")
    lines.append("")
    lines.append("## Overall Delta vs RadialGate-none")
    lines.append("")
    lines.append("| Variant | Dice Delta | HD95 Delta | HD95 Reduction | Evidence Rate | Interpretation |")
    lines.append("|---|---:|---:|---:|---:|---|")
    supported: List[str] = []
    for row in delta_rows:
        if row.get("domain") != "ALL_DOMAINS":
            continue
        variant = str(row.get("diagnostic_variant", ""))
        dice_delta = finite_float(row.get("case_dice_delta_mean"))
        hd95_delta = finite_float(row.get("case_hd95_delta_mean"))
        evidence = finite_float(row.get("border_artifact_evidence_rate"))
        support = bool(np.isfinite(hd95_delta) and hd95_delta <= -5.0 and np.isfinite(dice_delta) and dice_delta > -0.01)
        if support:
            supported.append(variant)
        if variant == CAUSE_REFLECT_FFT:
            interpretation = "supports FFT periodic boundary hypothesis" if support else "weak FFT-boundary evidence"
        elif variant == CAUSE_SPATIAL_BLEND:
            interpretation = "supports edge-overcalibration hypothesis" if support else "weak edge-overcalibration evidence"
        elif variant == CAUSE_REFLECT_PAD_INFERENCE:
            interpretation = "supports CNN padding sensitivity hypothesis" if support else "weak CNN-padding evidence"
        else:
            interpretation = ""
        lines.append(
            f"| {variant} | {dice_delta:.4f} | {hd95_delta:.2f} | "
            f"{finite_float(row.get('hd95_reduction_mm_mean')):.2f} | {evidence:.1%} | {interpretation} |"
        )
    lines.append("")
    if set((CAUSE_REFLECT_FFT, CAUSE_SPATIAL_BLEND, CAUSE_REFLECT_PAD_INFERENCE)).issubset(set(supported)):
        lines.append("结论：三种干预均满足显著改善标准，更像是频域边界效应与 CNN padding 敏感性耦合。")
    elif supported:
        lines.append("结论：获得支持的假设为：" + ", ".join(supported))
    else:
        lines.append("结论：三种干预均未达到默认显著改善标准，需要继续检查小孤立假阳性、结构外扩或 GT 问题。")
    lines.append("")
    lines.append("## Domain Delta vs RadialGate-none")
    lines.append("")
    lines.append("| Domain | Variant | Dice Delta | HD95 Delta | Evidence Rate |")
    lines.append("|---|---|---:|---:|---:|")
    for row in delta_rows:
        if row.get("domain") == "ALL_DOMAINS":
            continue
        lines.append(
            f"| {row.get('domain', '')} | {row.get('diagnostic_variant', '')} | "
            f"{finite_float(row.get('case_dice_delta_mean')):.4f} | "
            f"{finite_float(row.get('case_hd95_delta_mean')):.2f} | "
            f"{finite_float(row.get('border_artifact_evidence_rate')):.1%} |"
        )
    lines.append("")
    lines.append("## Border Component Ratio")
    lines.append("")
    lines.append("| Domain | Variant | Components | Border<5px | Border<10px | Border<15px |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in border_rows:
        if row.get("domain") != "ALL_DOMAINS":
            continue
        lines.append(
            f"| {row.get('domain', '')} | {row.get('diagnostic_variant', '')} | {int(row.get('n_components', 0) or 0)} | "
            f"{finite_float(row.get('is_border_artifact_m5_ratio'), 0.0):.1%} | "
            f"{finite_float(row.get('is_border_artifact_m10_ratio'), 0.0):.1%} | "
            f"{finite_float(row.get('is_border_artifact_m15_ratio'), 0.0):.1%} |"
        )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    for name in (
        "selected_run_cases.csv",
        "outlier_cause_case_metrics.csv",
        "outlier_cause_eval_metrics.csv",
        "outlier_cause_domain_class_metrics.csv",
        "outlier_cause_delta_summary.csv",
        "outlier_cause_border_artifact_summary.csv",
        "run_config.json",
    ):
        lines.append(f"- `{name}`")
    (result_root / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def rgba_overlay(mask: np.ndarray, color: Sequence[float], alpha: float) -> np.ndarray:
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[np.asarray(mask).astype(bool)] = [float(color[0]), float(color[1]), float(color[2]), float(alpha)]
    return rgba


def boundary2d(mask: np.ndarray) -> np.ndarray:
    return diag_surface(np.asarray(mask).astype(bool))


def add_slice_overlay(
    ax: plt.Axes,
    image: np.ndarray,
    gt_slice: np.ndarray,
    radial_slice: np.ndarray,
    sadg_slice: np.ndarray | None,
    far_slice: np.ndarray,
    title: str,
) -> None:
    ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    ax.imshow(rgba_overlay(boundary2d(gt_slice > 0), (0.0, 1.0, 0.0), 0.88))
    if sadg_slice is not None:
        ax.imshow(rgba_overlay(boundary2d(sadg_slice > 0), (0.0, 0.85, 1.0), 0.78))
    ax.imshow(rgba_overlay(boundary2d(radial_slice > 0), (1.0, 0.0, 0.0), 0.78))
    ax.imshow(rgba_overlay(far_slice > 0, (1.0, 0.75, 0.0), 0.92))
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def save_overview_png(
    *,
    row: Mapping[str, Any],
    case_data: Mapping[str, Any],
    sadg_pred_stack: np.ndarray | None,
    far_masks_by_class: Mapping[int, np.ndarray],
    out_dir: Path,
    max_panels: int = 6,
) -> Path:
    ensure_dir(out_dir)
    image = np.asarray(case_data["image_stack"])
    gt = np.asarray(case_data["gt_stack"])
    radial = np.asarray(case_data["pred_stack"])
    far_union = np.zeros_like(radial, dtype=bool)
    for mask in far_masks_by_class.values():
        far_union |= np.asarray(mask).astype(bool)
    counts = far_union.reshape(far_union.shape[0], -1).sum(axis=1)
    if np.any(counts > 0):
        slice_ids = list(np.argsort(-counts)[:max_panels])
    else:
        mid = far_union.shape[0] // 2
        slice_ids = [mid]
    slice_ids = sorted(int(x) for x in slice_ids)
    cols = min(3, len(slice_ids))
    rows = int(math.ceil(len(slice_ids) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False)
    sagittal_x = list(case_data["sagittal_x_indices"])
    for ax, slice_pos in zip(axes.reshape(-1), slice_ids):
        sadg_slice = None if sadg_pred_stack is None else sadg_pred_stack[slice_pos]
        title = f"slice={slice_pos} x={sagittal_x[slice_pos]} far={int(counts[slice_pos])}"
        add_slice_overlay(
            ax,
            image[slice_pos],
            gt[slice_pos],
            radial[slice_pos],
            sadg_slice,
            far_union[slice_pos],
            title,
        )
    for ax in axes.reshape(-1)[len(slice_ids):]:
        ax.axis("off")
    fig.suptitle(
        f"{row['domain']} | shot{row['shot']} seed{row['seed']} case{row['case_id']} | "
        f"Dice +{finite_float(row.get('dice_delta')):.3f}, HD95 +{finite_float(row.get('hd95_delta')):.2f}",
        fontsize=10,
    )
    fig.tight_layout()
    out_path = out_dir / "overview.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_component_crop_png(
    *,
    component: Mapping[str, Any],
    case_data: Mapping[str, Any],
    sadg_pred_stack: np.ndarray | None,
    far_masks_by_class: Mapping[int, np.ndarray],
    out_dir: Path,
    crop_radius: int,
) -> Path:
    cls = int(component["class_id"])
    slice_pos = int(component["max_point_slice_pos"])
    z = int(component["max_point_z"])
    y = int(component["max_point_y"])
    image = np.asarray(case_data["image_stack"])
    gt = np.asarray(case_data["gt_stack"])
    radial = np.asarray(case_data["pred_stack"])
    h, w = image.shape[1], image.shape[2]
    z0, z1 = max(0, z - crop_radius), min(h, z + crop_radius + 1)
    y0, y1 = max(0, y - crop_radius), min(w, y + crop_radius + 1)
    far_slice = np.asarray(far_masks_by_class.get(cls, np.zeros_like(radial, dtype=bool)))[slice_pos]
    sadg_slice = None if sadg_pred_stack is None else sadg_pred_stack[slice_pos, z0:z1, y0:y1]

    fig, ax = plt.subplots(1, 1, figsize=(4.2, 4.2))
    add_slice_overlay(
        ax,
        image[slice_pos, z0:z1, y0:y1],
        gt[slice_pos, z0:z1, y0:y1] == cls,
        radial[slice_pos, z0:z1, y0:y1] == cls,
        None if sadg_slice is None else sadg_slice == cls,
        far_slice[z0:z1, y0:y1],
        (
            f"{component['class_name']} comp={component['component_id']} "
            f"max={finite_float(component['max_distance_mm']):.1f}mm"
        ),
    )
    ax.scatter([y - y0], [z - z0], s=24, c="yellow", marker="x")
    fig.tight_layout()
    out_path = out_dir / (
        f"component_rank{int(component['component_rank_in_class']):02d}_"
        f"class{cls}_{sanitize_name(component['class_name'])}_"
        f"comp{int(component['component_id'])}.png"
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def build_annotation_queue(component_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in component_rows:
        item = dict(row)
        item["reviewer_category"] = ""
        item["reviewer_notes"] = ""
        item["reviewer_name"] = ""
        item["allowed_categories"] = "|".join(ANNOTATION_LABELS)
        rows.append(item)
    return rows


def summarize_annotations(result_root: Path) -> List[Dict[str, Any]]:
    queue_path = result_root / "annotation_queue.csv"
    if not queue_path.exists():
        raise FileNotFoundError(f"No annotation queue found: {queue_path}")
    df = pd.read_csv(queue_path)
    if "reviewer_category" not in df.columns:
        raise ValueError("annotation_queue.csv has no reviewer_category column.")
    df["reviewer_category"] = df["reviewer_category"].fillna("").astype(str).str.strip()
    labeled = df[df["reviewer_category"] != ""].copy()
    rows: List[Dict[str, Any]] = []
    if labeled.empty:
        rows.append({"domain": "ALL", "n_labeled": 0})
        return rows
    for domain, group in labeled.groupby("domain", sort=True):
        counts = Counter(group["reviewer_category"])
        row = {"domain": domain, "n_labeled": int(len(group))}
        for label in ANNOTATION_LABELS:
            row[f"count_{label}"] = int(counts.get(label, 0))
            row[f"ratio_{label}"] = float(counts.get(label, 0) / len(group))
        noise = counts.get("plausible_anatomy_gt_missing", 0) + counts.get("gt_artifact_discontinuity", 0)
        row["label_noise_ratio"] = float(noise / len(group))
        rows.append(row)
    counts = Counter(labeled["reviewer_category"])
    all_row = {"domain": "ALL", "n_labeled": int(len(labeled))}
    for label in ANNOTATION_LABELS:
        all_row[f"count_{label}"] = int(counts.get(label, 0))
        all_row[f"ratio_{label}"] = float(counts.get(label, 0) / len(labeled))
    noise = counts.get("plausible_anatomy_gt_missing", 0) + counts.get("gt_artifact_discontinuity", 0)
    all_row["label_noise_ratio"] = float(noise / len(labeled))
    rows.append(all_row)
    return rows


def write_report(
    result_root: Path,
    selected_rows: Sequence[Dict[str, Any]],
    component_rows: Sequence[Dict[str, Any]],
    annotation_summary: Sequence[Dict[str, Any]],
    border_component_rows: Sequence[Dict[str, Any]] | None = None,
    border_pruned_delta_rows: Sequence[Dict[str, Any]] | None = None,
) -> None:
    border_component_rows = list(border_component_rows or [])
    border_pruned_delta_rows = list(border_pruned_delta_rows or [])
    margins = sorted(
        {
            int(str(key).split("_m")[-1])
            for row in border_component_rows
            for key in row
            if key.startswith("is_border_artifact_m")
        }
    )
    lines: List[str] = []
    lines.append("# RadialGate HD95 远距离预测点审计报告")
    lines.append("")
    lines.append("本报告只做诊断复评，不重新训练模型。筛选条件为 RadialGate 相比 SADG Dice 上升、HD95 上升，并且 RadialGate 的 pred-to-GT p95 大于 GT-to-pred p95。")
    lines.append("")
    lines.append("## Case Selection")
    lines.append("")
    lines.append(f"- Selected run-cases: {len(selected_rows)}")
    lines.append(f"- Far components queued for review: {len(component_rows)}")
    lines.append(f"- Far components in border artifact analysis: {len(border_component_rows)}")
    lines.append("")
    by_domain: Dict[str, List[Dict[str, Any]]] = {}
    for row in selected_rows:
        by_domain.setdefault(str(row["domain"]), []).append(row)
    comp_by_domain: Dict[str, List[Dict[str, Any]]] = {}
    for row in component_rows:
        comp_by_domain.setdefault(str(row["domain"]), []).append(row)
    border_by_domain: Dict[str, List[Dict[str, Any]]] = {}
    for row in border_component_rows:
        border_by_domain.setdefault(str(row["domain"]), []).append(row)
    if margins:
        margin_headers = " | ".join(f"Border<{m}px" for m in margins)
        margin_sep = "|".join(["---:"] * len(margins))
        lines.append(f"| Domain | Selected | Mean Dice Delta | Mean HD95 Delta | Review Components | All Far Components | {margin_headers} |")
        lines.append(f"|---|---:|---:|---:|---:|---:|{margin_sep}|")
    else:
        lines.append("| Domain | Selected | Mean Dice Delta | Mean HD95 Delta | Review Components | All Far Components |")
        lines.append("|---|---:|---:|---:|---:|---:|")
    for domain in sorted(by_domain):
        cases = by_domain[domain]
        comps = comp_by_domain.get(domain, [])
        border_comps = border_by_domain.get(domain, [])
        margin_cells: List[str] = []
        for margin in margins:
            key = f"is_border_artifact_{margin_tag(margin)}"
            ratio = finite_mean(bool(item.get(key, False)) for item in border_comps)
            margin_cells.append(f"{ratio:.1%}" if np.isfinite(ratio) else "nan")
        tail = " | " + " | ".join(margin_cells) if margin_cells else ""
        lines.append(
            f"| {domain} | {len(cases)} | {finite_mean(row.get('dice_delta') for row in cases):.4f} | "
            f"{finite_mean(row.get('hd95_delta') for row in cases):.2f} | {len(comps)} | {len(border_comps)}{tail} |"
        )
    lines.append("")
    lines.append("## Heuristic Suggestions")
    lines.append("")
    counts = Counter(str(row.get("heuristic_suggestion", "")) for row in component_rows)
    if component_rows:
        for label, count in counts.most_common():
            lines.append(f"- `{label}`: {count} ({count / len(component_rows):.1%})")
    else:
        lines.append("- No far components were extracted.")
    if border_pruned_delta_rows:
        lines.append("")
        lines.append("## Border-Pruned Re-Evaluation")
        lines.append("")
        lines.append("| Domain | Variant | Dice Delta | HD95 Delta | HD95 Reduction | HD95 Reduction % | Evidence Rate |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for row in border_pruned_delta_rows:
            if row.get("domain") != "ALL_DOMAINS":
                continue
            lines.append(
                f"| {row.get('domain', '')} | {row.get('diagnostic_variant', '')} | "
                f"{finite_float(row.get('case_dice_delta_mean')):.4f} | "
                f"{finite_float(row.get('case_hd95_delta_mean')):.2f} | "
                f"{finite_float(row.get('hd95_reduction_mm_mean')):.2f} | "
                f"{finite_float(row.get('hd95_reduction_pct_mean')):.1%} | "
                f"{finite_float(row.get('border_artifact_evidence_rate')):.1%} |"
            )
        lines.append("")
        lines.append("Evidence rate: cases with HD95 improvement >= 5mm and Dice drop < 0.01.")
    lines.append("")
    lines.append("## Manual Annotation Rule")
    lines.append("")
    lines.append("请在 `annotation_queue.csv` 的 `reviewer_category` 中填入以下之一：")
    for label in ANNOTATION_LABELS:
        lines.append(f"- `{label}`")
    lines.append("")
    lines.append("解释标准：若 `plausible_anatomy_gt_missing + gt_artifact_discontinuity` ≥ 30%，标签噪声是 HD95 恶化的重要因素；若 ≥ 50%，标签噪声可视为主要因素。")
    if annotation_summary:
        lines.append("")
        lines.append("## Manual Annotation Summary")
        lines.append("")
        lines.append("| Domain | Labeled | Label Noise Ratio | Small FP Ratio | Boundary Expansion Ratio |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in annotation_summary:
            lines.append(
                f"| {row.get('domain', '')} | {int(row.get('n_labeled', 0) or 0)} | "
                f"{finite_float(row.get('label_noise_ratio'), 0.0):.1%} | "
                f"{finite_float(row.get('ratio_small_fp_error'), 0.0):.1%} | "
                f"{finite_float(row.get('ratio_boundary_overexpansion'), 0.0):.1%} |"
            )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    for name in (
        "selected_run_cases.csv",
        "far_boundary_points.csv",
        "far_boundary_components.csv",
        "border_artifact_analysis.csv",
        "border_pruned_case_metrics.csv",
        "border_pruned_eval_metrics.csv",
        "border_pruned_domain_class_metrics.csv",
        "border_pruned_delta_summary.csv",
        "annotation_queue.csv",
        "annotation_summary.csv",
    ):
        lines.append(f"- `{name}`")
    lines.append("- `figures/<domain>/shotX_seedY_caseZ/overview.png`")
    (result_root / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def checkpoint_for(row: Mapping[str, Any], family: str, args: argparse.Namespace) -> Path:
    if family == "radial":
        raw = str(row.get("radialgate_checkpoint_path", "")).strip()
        if raw:
            return Path(raw)
        return resolve_path(args.radialgate_checkpoint_root) / f"shot{int(row['shot'])}" / f"Seed{int(row['seed'])}" / "baseline_model_with_metadata.pt"
    raw = str(row.get("sadg_checkpoint_path", "")).strip()
    if raw:
        return Path(raw)
    return resolve_path(args.sadg_checkpoint_root) / f"shot{int(row['shot'])}" / f"Seed{int(row['seed'])}" / "baseline_model_with_metadata.pt"


def prepare_output_root(args: argparse.Namespace) -> Path:
    result_root = resolve_path(args.result_root)
    if result_root.exists() and bool(args.overwrite):
        shutil.rmtree(result_root)
    elif result_root.exists() and any(result_root.iterdir()):
        raise FileExistsError(f"Result root already exists and is not empty: {result_root}. Use --overwrite.")
    ensure_dir(result_root)
    return result_root


def run_cause_experiment(args: argparse.Namespace) -> None:
    result_root = prepare_output_path(args.cause_result_root, bool(args.overwrite))
    device = choose_device(args.device)
    variants = parse_str_csv(args.cause_variants)
    print(f"[DEVICE] {device}", flush=True)
    print(f"[RESULT_ROOT] {result_root}", flush=True)
    print(f"[VARIANTS] {','.join(variants)}", flush=True)

    selected_rows = select_run_cases(args)
    write_csv(result_root / "selected_run_cases.csv", selected_rows)
    write_json(
        result_root / "run_config.json",
        {
            "args": vars(args),
            "device": str(device),
            "selection_csv": str(resolve_path(args.selection_csv)),
            "radialgate_checkpoint_root": str(resolve_path(args.radialgate_checkpoint_root)),
            "cause_variants": variants,
            "interpretation_criteria": {
                "hd95_delta_threshold_mm": -5.0,
                "dice_delta_min": -0.01,
            },
        },
    )
    print(f"[SELECT] selected_run_cases={len(selected_rows)}", flush=True)

    all_case_rows: List[Dict[str, Any]] = []
    all_class_rows: List[Dict[str, Any]] = []
    all_border_component_rows: List[Dict[str, Any]] = []
    grouped: Dict[tuple[int, int], List[Dict[str, Any]]] = {}
    for row in selected_rows:
        grouped.setdefault((int(row["shot"]), int(row["seed"])), []).append(row)

    for (shot, seed), rows in sorted(grouped.items()):
        radial_ckpt = checkpoint_for(rows[0], "radial", args)
        radial_model, _radial_meta = ab7.load_radialgate_checkpoint_model(args, device, radial_ckpt)
        cause_models = build_cause_models(radial_model, args)
        print(f"[LOAD] shot={shot} seed={seed} rows={len(rows)}", flush=True)
        for row in sorted(rows, key=lambda item: (str(item["domain"]), int(item["selected_rank"]))):
            items = load_single_case_items(args, str(row["domain"]), str(row["case_id"]))
            for variant in variants:
                model = cause_models[variant]
                case_data = predict_case(model, items, device=device, eval_batch_size=int(args.eval_batch_size))
                case_row, class_rows = evaluate_prediction_stack(
                    row=row,
                    pred=np.asarray(case_data["pred_stack"], dtype=np.int64),
                    gt=np.asarray(case_data["gt_stack"], dtype=np.int64),
                    spacing=tuple(float(x) for x in case_data["case_spacing"]),
                    args=args,
                    variant=variant,
                    border_margin_px=0,
                    is_border_pruned=False,
                )
                all_case_rows.append(case_row)
                all_class_rows.extend(class_rows)

                threshold_row = threshold_row_for_variant(row, case_row)
                _points, _review_components, border_component_rows, _far_masks, _far_labels = extract_far_boundary(
                    row=threshold_row,
                    case_data=case_data,
                    args=args,
                )
                for comp in border_component_rows:
                    comp["diagnostic_variant"] = variant
                    comp["threshold_source"] = "variant_adaptive_p95"
                    comp["reflect_fft_pad_px"] = int(args.reflect_fft_pad_px) if variant == CAUSE_REFLECT_FFT else 0
                    comp["spatial_blend_margin_px"] = int(args.spatial_blend_margin_px) if variant == CAUSE_SPATIAL_BLEND else 0
                    comp["reflect_inference_pad_px"] = int(args.reflect_inference_pad_px) if variant == CAUSE_REFLECT_PAD_INFERENCE else 0
                all_border_component_rows.extend(border_component_rows)
                print(
                    f"[CASE] variant={variant} domain={row['domain']} shot={row['shot']} seed={row['seed']} "
                    f"case={row['case_id']} dice={case_row['case_dice']:.4f} hd95={case_row['case_hd95']:.2f} "
                    f"border_components={len(border_component_rows)}",
                    flush=True,
                )
        del cause_models
        del radial_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    eval_rows = summarize_border_pruned_eval(all_case_rows)
    domain_class_rows = summarize_border_pruned_domain_class(all_class_rows)
    delta_rows = summarize_border_pruned_delta(all_case_rows)
    border_summary_rows = summarize_outlier_cause_border_artifacts(all_border_component_rows)
    write_csv(result_root / "outlier_cause_case_metrics.csv", all_case_rows)
    write_csv(result_root / "outlier_cause_eval_metrics.csv", eval_rows)
    write_csv(result_root / "outlier_cause_domain_class_metrics.csv", domain_class_rows)
    write_csv(result_root / "outlier_cause_delta_summary.csv", delta_rows)
    write_csv(result_root / "outlier_cause_border_artifact_summary.csv", border_summary_rows)
    write_csv(result_root / "outlier_cause_border_artifact_components.csv", all_border_component_rows)
    write_cause_report(result_root, selected_rows, eval_rows, delta_rows, border_summary_rows, variants)
    print(
        f"[DONE] case_rows={len(all_case_rows)} class_rows={len(all_class_rows)} "
        f"border_component_rows={len(all_border_component_rows)}",
        flush=True,
    )


def run_audit(args: argparse.Namespace) -> None:
    result_root = prepare_output_root(args)
    device = choose_device(args.device)
    print(f"[DEVICE] {device}", flush=True)
    print(f"[RESULT_ROOT] {result_root}", flush=True)

    selected_rows = select_run_cases(args)
    write_csv(result_root / "selected_run_cases.csv", selected_rows)
    write_json(
        result_root / "run_config.json",
        {
            "args": vars(args),
            "device": str(device),
            "selection_csv": str(resolve_path(args.selection_csv)),
            "radialgate_checkpoint_root": str(resolve_path(args.radialgate_checkpoint_root)),
            "sadg_checkpoint_root": str(resolve_path(args.sadg_checkpoint_root)),
            "annotation_labels": list(ANNOTATION_LABELS),
        },
    )
    print(f"[SELECT] selected_run_cases={len(selected_rows)}", flush=True)

    all_point_rows: List[Dict[str, Any]] = []
    all_component_rows: List[Dict[str, Any]] = []
    all_border_component_rows: List[Dict[str, Any]] = []
    all_border_pruned_case_rows: List[Dict[str, Any]] = []
    all_border_pruned_class_rows: List[Dict[str, Any]] = []
    grouped: Dict[tuple[int, int], List[Dict[str, Any]]] = {}
    for row in selected_rows:
        grouped.setdefault((int(row["shot"]), int(row["seed"])), []).append(row)

    for (shot, seed), rows in sorted(grouped.items()):
        radial_ckpt = checkpoint_for(rows[0], "radial", args)
        radial_model, _radial_meta = ab7.load_radialgate_checkpoint_model(args, device, radial_ckpt)
        sadg_model = None
        if bool(args.include_sadg_overlay):
            sadg_ckpt = checkpoint_for(rows[0], "sadg", args)
            sadg_model, _sadg_meta = ab7.load_sadg_checkpoint_model(args, device, sadg_ckpt)
        print(f"[LOAD] shot={shot} seed={seed} rows={len(rows)}", flush=True)
        for row in sorted(rows, key=lambda item: (str(item["domain"]), int(item["selected_rank"]))):
            case_dir = (
                result_root
                / "figures"
                / sanitize_name(row["domain"])
                / f"shot{int(row['shot'])}_seed{int(row['seed'])}_case{sanitize_name(row['case_id'])}"
            )
            items = load_single_case_items(args, str(row["domain"]), str(row["case_id"]))
            radial_data = predict_case(radial_model, items, device=device, eval_batch_size=int(args.eval_batch_size))
            sadg_pred = None
            if sadg_model is not None:
                sadg_data = predict_case(sadg_model, items, device=device, eval_batch_size=int(args.eval_batch_size))
                sadg_pred = np.asarray(sadg_data["pred_stack"], dtype=np.int64)

            if bool(args.enable_border_pruned_eval):
                pruned_case_rows, pruned_class_rows = evaluate_border_pruned_variants(
                    row=row,
                    case_data=radial_data,
                    args=args,
                )
                all_border_pruned_case_rows.extend(pruned_case_rows)
                all_border_pruned_class_rows.extend(pruned_class_rows)

            point_rows, component_rows, border_component_rows, far_masks_by_class, _far_labels_by_class = extract_far_boundary(
                row=row,
                case_data=radial_data,
                args=args,
            )
            overview_path = save_overview_png(
                row=row,
                case_data=radial_data,
                sadg_pred_stack=sadg_pred,
                far_masks_by_class=far_masks_by_class,
                out_dir=case_dir,
                max_panels=int(args.overview_max_panels),
            )
            crop_by_component: Dict[tuple[int, int], str] = {}
            for comp in component_rows:
                crop_path = save_component_crop_png(
                    component=comp,
                    case_data=radial_data,
                    sadg_pred_stack=sadg_pred,
                    far_masks_by_class=far_masks_by_class,
                    out_dir=case_dir,
                    crop_radius=int(args.crop_radius),
                )
                comp["overview_png"] = str(overview_path)
                comp["component_crop_png"] = str(crop_path)
                crop_by_component[(int(comp["class_id"]), int(comp["component_id"]))] = str(crop_path)
            for comp in border_component_rows:
                comp["overview_png"] = str(overview_path)
                comp["component_crop_png"] = crop_by_component.get((int(comp["class_id"]), int(comp["component_id"])), "")
            for point in point_rows:
                point["overview_png"] = str(overview_path)
            all_point_rows.extend(point_rows)
            all_component_rows.extend(component_rows)
            all_border_component_rows.extend(border_component_rows)
            print(
                f"[CASE] domain={row['domain']} shot={row['shot']} seed={row['seed']} case={row['case_id']} "
                f"far_points={len(point_rows)} far_components={len(component_rows)} border_components={len(border_component_rows)}",
                flush=True,
            )
        del radial_model
        if sadg_model is not None:
            del sadg_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    annotation_rows = build_annotation_queue(all_component_rows)
    write_csv(result_root / "far_boundary_points.csv", all_point_rows)
    write_csv(result_root / "far_boundary_components.csv", all_component_rows)
    write_csv(result_root / "border_artifact_analysis.csv", all_border_component_rows)
    border_pruned_eval_rows = summarize_border_pruned_eval(all_border_pruned_case_rows)
    border_pruned_class_rows = summarize_border_pruned_domain_class(all_border_pruned_class_rows)
    border_pruned_delta_rows = summarize_border_pruned_delta(all_border_pruned_case_rows)
    write_csv(result_root / "border_pruned_case_metrics.csv", all_border_pruned_case_rows)
    write_csv(result_root / "border_pruned_eval_metrics.csv", border_pruned_eval_rows)
    write_csv(result_root / "border_pruned_domain_class_metrics.csv", border_pruned_class_rows)
    write_csv(result_root / "border_pruned_delta_summary.csv", border_pruned_delta_rows)
    write_csv(result_root / "annotation_queue.csv", annotation_rows)
    annotation_summary = summarize_annotations(result_root)
    write_csv(result_root / "annotation_summary.csv", annotation_summary)
    write_report(
        result_root,
        selected_rows,
        all_component_rows,
        annotation_summary,
        all_border_component_rows,
        border_pruned_delta_rows,
    )
    print(
        f"[DONE] points={len(all_point_rows)} components={len(all_component_rows)} "
        f"border_components={len(all_border_component_rows)} border_pruned_case_rows={len(all_border_pruned_case_rows)}",
        flush=True,
    )


def run_summarize_annotations(args: argparse.Namespace) -> None:
    result_root = resolve_path(args.result_root)
    rows = summarize_annotations(result_root)
    write_csv(result_root / "annotation_summary.csv", rows)
    selected_path = result_root / "selected_run_cases.csv"
    component_path = result_root / "far_boundary_components.csv"
    border_component_path = result_root / "border_artifact_analysis.csv"
    border_delta_path = result_root / "border_pruned_delta_summary.csv"
    selected_rows = pd.read_csv(selected_path).to_dict("records") if selected_path.exists() else []
    component_rows = pd.read_csv(component_path).to_dict("records") if component_path.exists() else []
    border_component_rows = pd.read_csv(border_component_path).to_dict("records") if border_component_path.exists() else []
    border_delta_rows = pd.read_csv(border_delta_path).to_dict("records") if border_delta_path.exists() else []
    write_report(result_root, selected_rows, component_rows, rows, border_component_rows, border_delta_rows)
    print(f"[ANNOTATION_SUMMARY] wrote {result_root / 'annotation_summary.csv'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose why RadialGate creates HD95 outliers at image/FOV borders.")
    parser.add_argument("--data_root", default=str(DATA_ROOT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selection_csv", default=str(DEFAULT_SELECTION_CSV))
    parser.add_argument("--result_root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--cause_result_root", default=str(DEFAULT_CAUSE_RESULT_ROOT))
    parser.add_argument("--cause_variants", default=",".join(DEFAULT_CAUSE_VARIANTS))
    parser.add_argument("--reflect_fft_pad_px", type=int, default=32)
    parser.add_argument("--spatial_blend_margin_px", type=int, default=15)
    parser.add_argument("--reflect_inference_pad_px", type=int, default=16)
    parser.add_argument("--top_cases_per_domain", type=int, default=20)
    parser.add_argument("--selection_unit", default="run_case", choices=("run_case",))
    parser.add_argument("--require_dice_up", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require_hd95_up", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require_pred_to_gt_dominant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threshold_mode", default="adaptive_p95", choices=("adaptive_p95", "fixed"))
    parser.add_argument("--threshold_p95_fraction", type=float, default=0.8)
    parser.add_argument("--threshold_min_mm", type=float, default=16.0)
    parser.add_argument("--top_components_per_case", type=int, default=10)
    parser.add_argument("--include_sadg_overlay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--border_margins_px", default="5,10,15")
    parser.add_argument("--border_artifact_primary_margin_px", type=int, default=10)
    parser.add_argument("--enable_border_pruned_eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--border_pruned_margins_px", default="5,10,15")
    parser.add_argument("--border_pruned_variant_prefix", default="RadialGate-border-pruned")
    parser.add_argument("--border_pruned_only_selected_cases", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--slice_policy", default="all_filtered", choices=("center9", "all", "all_filtered"))
    parser.add_argument("--filter_min_fg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resize_hw", type=int, default=224)
    parser.add_argument("--min_fg_ratio", type=float, default=0.05)
    parser.add_argument("--num_middle_slices", type=int, default=9)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--base_ch", type=int, default=16)
    parser.add_argument("--latent_ch", type=int, default=64)
    parser.add_argument("--style_alpha", type=float, default=0.08)
    parser.add_argument("--style_rho", type=float, default=0.5)
    parser.add_argument("--style_eps", type=float, default=1e-6)
    parser.add_argument("--fft_norm", default="ortho", choices=("backward", "ortho", "forward"))
    parser.add_argument("--gate_num_bins", type=int, default=8)
    parser.add_argument("--gate_rho_max", type=float, default=1.0)
    parser.add_argument("--radialgate_checkpoint_root", default=str(DEFAULT_RADIALGATE_CHECKPOINT_ROOT))
    parser.add_argument("--sadg_checkpoint_root", default=str(DEFAULT_SADG_CHECKPOINT_ROOT))
    parser.add_argument("--prune_min_voxels", type=int, default=20)
    parser.add_argument("--prune_min_largest_ratio", type=float, default=0.01)
    parser.add_argument("--overview_max_panels", type=int, default=6)
    parser.add_argument("--crop_radius", type=int, default=48)
    parser.add_argument("--summarize_annotations", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.summarize_annotations):
        run_summarize_annotations(args)
    else:
        run_cause_experiment(args)


if __name__ == "__main__":
    main()
