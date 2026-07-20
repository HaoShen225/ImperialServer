from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi

from helper.backbone import ModelConfig, build_model
from helper.backbone_losses import (
    apply_sadg_perturbation_from_loss,
    backbone_training_loss,
    restore_sadg_perturbation,
)
from helper.dataloaders import (
    DATA_ROOT,
    DOMAIN_NAMES,
    FOREGROUND_CLASS_NAMES,
    SOURCE_DOMAIN_NAME,
    build_train_dataset,
    make_loader,
    run_test_flow,
)
from helper.load_model_params import extract_metadata, extract_state_dict


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_RESULT_ROOT = PROJECT_ROOT / "TrainingData" / "DisperssionWithSADG_GeometryPreservingRadialGate"
DEFAULT_BACKBONE_ROOT = PROJECT_ROOT / "backbones" / "GeometryPreservingRadialGate"
DEFAULT_PROMPT_ROOT = PROJECT_ROOT / "backbones" / "Prompts" / "GeometryPreservingRadialGate"

SEG_SADG_ABLATION = "Seg-SADG"
TEACHER_METHOD = "SADG"
STUDENT_METHOD = "LF-Barycenter-RadialGate-GeometryPreserving"
FIXED_SADG_RHO = 0.05


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(name: str) -> torch.device:
    if str(name).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_path(path: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base / p


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def value_tag(name: str, value: float) -> str:
    text = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return f"{name}{text}"


def sadg_rho_tag(value: float) -> str:
    return value_tag("sadg", value)


def finite_values(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            out.append(number)
    return out


def finite_mean(values: Iterable[Any]) -> float:
    vals = finite_values(values)
    return float(np.mean(vals)) if vals else float("nan")


def finite_std(values: Iterable[Any]) -> float:
    vals = finite_values(values)
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def model_config(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        in_ch=1,
        num_classes=3,
        base_ch=int(args.base_ch),
        latent_ch=int(args.latent_ch),
        model_type="d_l2_disp_bn",
        use_l2_norm=True,
        use_batch_norm=True,
    )


def lowfreq_window(H: int, W: int, alpha: float) -> Dict[str, int]:
    r_h = max(0, int(round(float(alpha) * int(H) / 2.0)))
    r_w = max(0, int(round(float(alpha) * int(W) / 2.0)))
    c_h = int(H) // 2
    c_w = int(W) // 2
    u0 = c_h - r_h
    u1 = c_h + r_h + 1
    v0 = c_w - r_w
    v1 = c_w + r_w + 1
    if u0 < 0 or v0 < 0 or u1 > int(H) or v1 > int(W):
        raise ValueError(f"Invalid low-frequency window for H={H}, W={W}, alpha={alpha}.")
    return {
        "radius_h": r_h,
        "radius_w": r_w,
        "size_h": u1 - u0,
        "size_w": v1 - v0,
        "u0": u0,
        "u1": u1,
        "v0": v0,
        "v1": v1,
    }


def radial_cosine_mask(h_lf: int, w_lf: int, device: torch.device | None = None) -> torch.Tensor:
    yy = torch.arange(int(h_lf), dtype=torch.float32, device=device)
    xx = torch.arange(int(w_lf), dtype=torch.float32, device=device)
    y, x = torch.meshgrid(yy, xx, indexing="ij")
    cy = (float(h_lf) - 1.0) / 2.0
    cx = (float(w_lf) - 1.0) / 2.0
    ry = max(cy, 1.0)
    rx = max(cx, 1.0)
    radius = torch.sqrt(((y - cy) / ry).pow(2) + ((x - cx) / rx).pow(2)).clamp(0.0, 1.0)
    return (0.5 + 0.5 * torch.cos(float(np.pi) * radius))[None, None, ...]


def radial_bin_map(h_lf: int, w_lf: int, num_bins: int, device: torch.device | None = None) -> torch.Tensor:
    yy = torch.arange(int(h_lf), dtype=torch.float32, device=device)
    xx = torch.arange(int(w_lf), dtype=torch.float32, device=device)
    y, x = torch.meshgrid(yy, xx, indexing="ij")
    cy = (float(h_lf) - 1.0) / 2.0
    cx = (float(w_lf) - 1.0) / 2.0
    ry = max(cy, 1.0)
    rx = max(cx, 1.0)
    radius = torch.sqrt(((y - cy) / ry).pow(2) + ((x - cx) / rx).pow(2)).clamp(0.0, 1.0)
    bins = torch.floor(radius * int(num_bins)).long().clamp(0, int(num_bins) - 1)
    return bins[None, None, ...]


def bounded_logit(value: float, *, eps: float = 1e-6) -> float:
    v = min(max(float(value), float(eps)), 1.0 - float(eps))
    return float(np.log(v / (1.0 - v)))


class LowFreqAmplitudeTemplateLayer(nn.Module):
    def __init__(
        self,
        *,
        H: int,
        W: int,
        alpha: float,
        style_rho: float,
        init_template: torch.Tensor,
        gate_num_bins: int = 8,
        gate_rho_max: float = 1.0,
        eps: float = 1e-6,
        fft_norm: str = "ortho",
    ):
        super().__init__()
        self.H = int(H)
        self.W = int(W)
        self.alpha = float(alpha)
        self.style_rho = float(style_rho)
        self.gate_num_bins = int(gate_num_bins)
        self.gate_rho_max = float(gate_rho_max)
        self.eps = float(eps)
        self.fft_norm = str(fft_norm)
        if self.gate_num_bins <= 0:
            raise ValueError("gate_num_bins must be positive.")
        if self.gate_rho_max <= 0.0:
            raise ValueError("gate_rho_max must be positive.")
        if self.style_rho < 0.0 or self.style_rho > self.gate_rho_max:
            raise ValueError("style_rho must be in [0, gate_rho_max].")

        window = lowfreq_window(self.H, self.W, self.alpha)
        self.radius_h = int(window["radius_h"])
        self.radius_w = int(window["radius_w"])
        self.H_lf = int(window["size_h"])
        self.W_lf = int(window["size_w"])
        self.u0 = int(window["u0"])
        self.u1 = int(window["u1"])
        self.v0 = int(window["v0"])
        self.v1 = int(window["v1"])

        template = init_template.detach().clone().float()
        expected_shape = (1, 1, self.H_lf, self.W_lf)
        if tuple(template.shape) != expected_shape:
            raise ValueError(f"init_template shape must be {expected_shape}, got {tuple(template.shape)}")
        self.register_buffer("T0", template.clone())
        self.register_buffer("cosine_mask", radial_cosine_mask(self.H_lf, self.W_lf))
        self.register_buffer("radial_bin_map", radial_bin_map(self.H_lf, self.W_lf, self.gate_num_bins))

        init_ratio = float(self.style_rho) / float(self.gate_rho_max)
        self.gate_logits = nn.Parameter(torch.full((self.gate_num_bins,), bounded_logit(init_ratio), dtype=torch.float32))

    def symmetric_template(self) -> torch.Tensor:
        return 0.5 * (self.T0 + torch.flip(self.T0, dims=(-2, -1)))

    def rho_bins(self) -> torch.Tensor:
        return float(self.gate_rho_max) * torch.sigmoid(self.gate_logits)

    def rho_map(self) -> torch.Tensor:
        idx = self.radial_bin_map.to(device=self.gate_logits.device)
        return self.rho_bins()[idx]

    def regularization_losses(
        self,
        *,
        lambda_anchor: float,
        lambda_smooth: float,
        lambda_mono: float,
    ) -> Dict[str, torch.Tensor]:
        rho = self.rho_bins()
        target = rho.new_full(rho.shape, float(self.style_rho))
        anchor = F.mse_loss(rho, target)
        if rho.numel() > 1:
            diff = rho[1:] - rho[:-1]
            smooth = diff.pow(2).mean()
            mono = F.relu(diff).pow(2).mean()
        else:
            smooth = rho.sum() * 0.0
            mono = rho.sum() * 0.0
        total = float(lambda_anchor) * anchor + float(lambda_smooth) * smooth + float(lambda_mono) * mono
        return {
            "gate_reg_loss": total,
            "gate_anchor_loss": anchor,
            "gate_smooth_loss": smooth,
            "gate_mono_loss": mono,
        }

    def radialgate_image(self, x: torch.Tensor) -> torch.Tensor:
        X = torch.fft.fft2(x, norm=self.fft_norm)
        Xs = torch.fft.fftshift(X, dim=(-2, -1))
        X_lf = Xs[:, :, self.u0:self.u1, self.v0:self.v1]

        A_lf = torch.abs(X_lf)
        phase_lf = X_lf / (A_lf + self.eps)
        T = self.symmetric_template().to(dtype=A_lf.dtype, device=A_lf.device)
        rho = self.rho_map().to(dtype=A_lf.dtype, device=A_lf.device)
        logA_cal = (1.0 - rho) * torch.log(A_lf + self.eps) + rho * T
        X_lf_cal = torch.exp(logA_cal) * phase_lf

        mask = self.cosine_mask.to(dtype=X_lf.real.dtype, device=X_lf.device)
        X_lf_new = mask * X_lf_cal + (1.0 - mask) * X_lf

        Xs_new = Xs.clone()
        Xs_new[:, :, self.u0:self.u1, self.v0:self.v1] = X_lf_new
        X_new = torch.fft.ifftshift(Xs_new, dim=(-2, -1))
        return torch.fft.ifft2(X_new, norm=self.fft_norm).real

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.radialgate_image(x)

    def scalar_metadata(self) -> Dict[str, Any]:
        rho = self.rho_bins().detach().cpu().float().numpy()
        return {
            "lf_radius_h": self.radius_h,
            "lf_radius_w": self.radius_w,
            "lf_size_h": self.H_lf,
            "lf_size_w": self.W_lf,
            "rho_bin_min": float(np.min(rho)),
            "rho_bin_max": float(np.max(rho)),
            "rho_bin_mean": float(np.mean(rho)),
            "rho_bins": "|".join(f"{float(v):.8g}" for v in rho),
        }

    def payload(self) -> Dict[str, Any]:
        rho_bins = self.rho_bins().detach().cpu()
        return {
            "schema_version": 1,
            "style_method": STUDENT_METHOD,
            "T0": self.T0.detach().cpu(),
            "T0_symmetric": self.symmetric_template().detach().cpu(),
            "cosine_mask": self.cosine_mask.detach().cpu(),
            "radial_bin_map": self.radial_bin_map.detach().cpu(),
            "rho_bins": rho_bins,
            "gate_logits": self.gate_logits.detach().cpu(),
            "config": {
                "H": self.H,
                "W": self.W,
                "style_alpha": self.alpha,
                "style_rho": self.style_rho,
                "gate_num_bins": self.gate_num_bins,
                "gate_rho_max": self.gate_rho_max,
                "style_eps": self.eps,
                "fft_norm": self.fft_norm,
                "lf_radius_h": self.radius_h,
                "lf_radius_w": self.radius_w,
                "lf_size_h": self.H_lf,
                "lf_size_w": self.W_lf,
            },
        }


class StyleCanonicalizedModel(nn.Module):
    def __init__(self, backbone: nn.Module, style_layer: LowFreqAmplitudeTemplateLayer | None = None):
        super().__init__()
        self.backbone = backbone
        self.style_layer = style_layer

    def forward(self, x: torch.Tensor, return_features: bool = False):
        if self.style_layer is not None:
            x = self.style_layer(x)
        return self.backbone(x, return_features=return_features)


def unwrap_backbone(model: nn.Module) -> nn.Module:
    return model.backbone if isinstance(model, StyleCanonicalizedModel) else model


def unwrap_style_layer(model: nn.Module) -> LowFreqAmplitudeTemplateLayer | None:
    return model.style_layer if isinstance(model, StyleCanonicalizedModel) else None


def teacher_run_dir(args: argparse.Namespace, shot: int, seed: int) -> Path:
    return (
        resolve_path(args.new_backbone_root)
        / SEG_SADG_ABLATION
        / TEACHER_METHOD
        / sadg_rho_tag(float(args.sadg_rho))
        / f"shot{int(shot)}"
        / f"Seed{int(seed)}"
    )


def student_run_dir(args: argparse.Namespace, shot: int, seed: int) -> Path:
    return (
        resolve_path(args.new_backbone_root)
        / SEG_SADG_ABLATION
        / "BaryRadGateGeometryPreserving"
        / value_tag("a", float(args.style_alpha))
        / value_tag("rho", float(args.style_rho))
        / f"gb{int(args.gate_num_bins)}"
        / sadg_rho_tag(float(args.sadg_rho))
        / f"shot{int(shot)}"
        / f"Seed{int(seed)}"
    )


def student_prompt_dir(args: argparse.Namespace, shot: int, seed: int) -> Path:
    return (
        resolve_path(args.prompt_root)
        / SEG_SADG_ABLATION
        / "BaryRadGateGeometryPreserving"
        / value_tag("a", float(args.style_alpha))
        / value_tag("rho", float(args.style_rho))
        / f"gb{int(args.gate_num_bins)}"
        / sadg_rho_tag(float(args.sadg_rho))
        / f"shot{int(shot)}"
        / f"Seed{int(seed)}"
    )


def build_train_data(args: argparse.Namespace, shot: int, seed: int):
    return build_train_dataset(
        SOURCE_DOMAIN_NAME,
        shot=int(shot),
        seed=int(seed),
        data_root=Path(args.data_root),
        min_fg_ratio=float(args.min_fg_ratio),
        resize_hw=int(args.resize_hw),
        split_csv=None,
        use_split=False,
        slice_policy=str(args.slice_policy),
        num_middle_slices=int(args.num_middle_slices),
        filter_min_fg=bool(args.filter_min_fg),
    )


def compute_source_log_template(train_dataset, *, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    H, W = train_dataset.resize_hw
    window = lowfreq_window(int(H), int(W), float(args.style_alpha))
    total = torch.zeros(1, 1, int(window["size_h"]), int(window["size_w"]), device=device)
    count = 0
    loader = make_loader(train_dataset, batch_size=int(args.batch_size), shuffle=False, device=device)
    with torch.no_grad():
        for img, _mask, _meta in loader:
            img = img.to(device)
            X = torch.fft.fftshift(torch.fft.fft2(img, norm=str(args.fft_norm)), dim=(-2, -1))
            X_lf = X[:, :, int(window["u0"]):int(window["u1"]), int(window["v0"]):int(window["v1"])]
            total += torch.log(torch.abs(X_lf) + float(args.style_eps)).sum(dim=0, keepdim=True)
            count += int(img.shape[0])
    if count <= 0:
        raise RuntimeError("Cannot compute source log-amplitude template from an empty training dataset.")
    return (total / float(count)).detach()


def build_teacher_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    return StyleCanonicalizedModel(build_model(model_config(args)).to(device), None).to(device)


def build_student_model(args: argparse.Namespace, device: torch.device, init_template: torch.Tensor) -> nn.Module:
    style_layer = LowFreqAmplitudeTemplateLayer(
        H=int(args.resize_hw),
        W=int(args.resize_hw),
        alpha=float(args.style_alpha),
        style_rho=float(args.style_rho),
        init_template=init_template.to(device),
        gate_num_bins=int(args.gate_num_bins),
        gate_rho_max=float(args.gate_rho_max),
        eps=float(args.style_eps),
        fft_norm=str(args.fft_norm),
    )
    return StyleCanonicalizedModel(build_model(model_config(args)).to(device), style_layer.to(device)).to(device)


def safe_torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_model_files(run_dir: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(run_dir)
    final_path = run_dir / "checkpoint_final.pt"
    baseline_path = run_dir / "baseline_model_with_metadata.pt"
    torch.save(payload, final_path)
    shutil.copyfile(final_path, baseline_path)


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    payload = safe_torch_load(checkpoint_path, map_location=device)
    unwrap_backbone(model).load_state_dict(extract_state_dict(payload), strict=True)
    layer = unwrap_style_layer(model)
    if layer is not None:
        if not isinstance(payload, Mapping) or "style_layer_state_dict" not in payload:
            raise ValueError(f"Checkpoint {checkpoint_path} does not contain style_layer_state_dict.")
        layer.load_state_dict(payload["style_layer_state_dict"], strict=True)
    metadata = extract_metadata(payload)
    metadata["checkpoint_path"] = str(checkpoint_path)
    return metadata


def checkpoint_payload(
    *,
    model: nn.Module,
    args: argparse.Namespace,
    role: str,
    shot: int,
    seed: int,
    train_dataset,
    train_rows: Sequence[Mapping[str, Any]],
    teacher_info: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    layer = unwrap_style_layer(model)
    teacher_info = dict(teacher_info or {})
    metadata = {
        "experiment": "DisperssionWithSADG_GeometryPreservingRadialGate",
        "role": role,
        "ablation": SEG_SADG_ABLATION,
        "style_method": TEACHER_METHOD if role == "teacher" else STUDENT_METHOD,
        "source_domain": SOURCE_DOMAIN_NAME,
        "shot": int(shot),
        "seed": int(seed),
        "train_case_ids": list(map(str, train_dataset.selected_case_ids)),
        "model_config": vars(model_config(args)),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "dice_weight": float(args.dice_weight),
        "lambda_disp": float(args.lambda_disp),
        "disp_margin": float(args.disp_margin),
        "sadg_rho": float(args.sadg_rho),
        "sadg_eps": float(args.sadg_eps),
        "style_alpha": float(args.style_alpha),
        "style_rho": float(args.style_rho),
        "style_eps": float(args.style_eps),
        "fft_norm": str(args.fft_norm),
        "gate_num_bins": int(args.gate_num_bins),
        "gate_rho_max": float(args.gate_rho_max),
        "lambda_gate_anchor": float(args.lambda_gate_anchor),
        "lambda_gate_smooth": float(args.lambda_gate_smooth),
        "lambda_gate_mono": float(args.lambda_gate_mono),
        "geometry_preserving": role == "student",
        "lambda_geo": float(args.lambda_geo) if role == "student" else 0.0,
        "geo_sdt_clip_px": float(args.geo_sdt_clip_px),
        "geo_eps": float(args.geo_eps),
        "geo_loss_type": "fg_sdt_weighted_l1" if role == "student" else "none",
        "slice_policy": str(args.slice_policy),
        "num_middle_slices": int(args.num_middle_slices),
        "filter_min_fg": bool(args.filter_min_fg),
        "min_fg_ratio": float(args.min_fg_ratio),
        "resize_hw": int(args.resize_hw),
        "max_train_steps": int(args.max_train_steps),
        "final_epoch": int(train_rows[-1]["epoch"]) if train_rows else 0,
        "teacher_checkpoint": teacher_info.get("teacher_checkpoint", ""),
        "teacher_train_case_ids": teacher_info.get("teacher_train_case_ids", []),
        "teacher_case_ids_match": bool(teacher_info.get("teacher_case_ids_match", False)),
    }
    payload: Dict[str, Any] = {
        "model_state_dict": unwrap_backbone(model).state_dict(),
        "metadata": metadata,
    }
    if layer is not None:
        payload["style_layer_state_dict"] = layer.state_dict()
        payload["style_layer_config"] = layer.payload()["config"]
    return payload


def save_style_artifacts(prompt_dir: Path, model: nn.Module, train_dataset, args: argparse.Namespace, teacher_info: Mapping[str, Any]) -> None:
    layer = unwrap_style_layer(model)
    if layer is None:
        return
    ensure_dir(prompt_dir)
    payload = layer.payload()
    payload.update(
        {
            "train_case_ids": list(map(str, train_dataset.selected_case_ids)),
            "teacher_checkpoint": teacher_info.get("teacher_checkpoint", ""),
            "teacher_train_case_ids": teacher_info.get("teacher_train_case_ids", []),
            "teacher_case_ids_match": bool(teacher_info.get("teacher_case_ids_match", False)),
        }
    )
    torch.save(payload, prompt_dir / "style_layer_final.pt")
    torch.save(
        {
            "rho_bins": layer.rho_bins().detach().cpu(),
            "gate_logits": layer.gate_logits.detach().cpu(),
            "gate_num_bins": int(layer.gate_num_bins),
            "gate_rho_max": float(layer.gate_rho_max),
            "style_rho": float(layer.style_rho),
        },
        prompt_dir / "gate_final.pt",
    )
    torch.save(
        {
            "T0": layer.T0.detach().cpu(),
            "T0_symmetric": layer.symmetric_template().detach().cpu(),
            "train_case_ids": list(map(str, train_dataset.selected_case_ids)),
            "style_alpha": float(args.style_alpha),
            "style_eps": float(args.style_eps),
            "fft_norm": str(args.fft_norm),
        },
        prompt_dir / "template_source_barycenter.pt",
    )


def compute_training_losses(model: nn.Module, img: torch.Tensor, mask: torch.Tensor, args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    out = model(img, return_features=True)
    losses = backbone_training_loss(
        out["logits"],
        mask,
        out["features"]["dec1"],
        num_classes=3,
        dice_weight=float(args.dice_weight),
        lambda_disp=float(args.lambda_disp),
        disp_margin=float(args.disp_margin),
    )
    losses["logits"] = out["logits"]
    return losses


def zero_tensor(model: nn.Module, device: torch.device) -> torch.Tensor:
    ref = next(model.parameters(), None)
    return ref.sum() * 0.0 if ref is not None else torch.tensor(0.0, device=device)


def foreground_sdt_abs_weights(mask: torch.Tensor, *, d_clip: float, device: torch.device) -> torch.Tensor:
    labels = mask.detach().cpu().numpy()
    if labels.ndim == 4:
        labels = labels[:, 0]
    if labels.ndim != 3:
        raise ValueError(f"Expected mask shape [B,H,W] or [B,1,H,W], got {tuple(mask.shape)}")
    clip = max(float(d_clip), 1e-6)
    weights: List[np.ndarray] = []
    structure = ndi.generate_binary_structure(2, 1)
    for label in labels:
        fg = np.asarray(label) > 0
        if np.any(fg):
            eroded = ndi.binary_erosion(fg, structure=structure, border_value=0)
            surface = np.logical_xor(fg, eroded)
            if not np.any(surface):
                surface = fg
            dist = ndi.distance_transform_edt(~surface).astype(np.float32)
        else:
            dist = np.full(fg.shape, clip, dtype=np.float32)
        weights.append(np.clip(dist, 0.0, clip).astype(np.float32) / clip)
    return torch.from_numpy(np.stack(weights, axis=0)).to(device=device)


def geometry_preserving_losses(
    *,
    student_logits: torch.Tensor,
    teacher_model: nn.Module,
    img: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        teacher_out = teacher_model(img, return_features=True)
        teacher_logits = teacher_out["logits"] if isinstance(teacher_out, Mapping) else teacher_out

    p_student = torch.softmax(student_logits, dim=1)[:, 1:, ...].sum(dim=1)
    p_teacher = torch.softmax(teacher_logits, dim=1)[:, 1:, ...].sum(dim=1)
    if p_teacher.shape[-2:] != p_student.shape[-2:]:
        p_teacher = F.interpolate(p_teacher[:, None, ...], size=p_student.shape[-2:], mode="bilinear", align_corners=False)[:, 0]

    weights = foreground_sdt_abs_weights(mask, d_clip=float(args.geo_sdt_clip_px), device=device)
    if weights.shape[-2:] != p_student.shape[-2:]:
        weights = F.interpolate(weights[:, None, ...], size=p_student.shape[-2:], mode="nearest")[:, 0]
    weights = weights.to(dtype=p_student.dtype, device=p_student.device)
    geo = (weights * (p_student - p_teacher).abs()).sum() / (weights.sum() + float(args.geo_eps))
    return {
        "geo_style_loss": float(args.lambda_geo) * geo,
        "geo_loss": geo,
        "geo_weight_mean": weights.mean().detach(),
    }


def train_step(
    *,
    model: nn.Module,
    teacher_model: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    img: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
    sadg_params: Sequence[torch.nn.Parameter],
    device: torch.device,
) -> Dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    epsilon_losses = compute_training_losses(model, img, mask, args)
    epsilon_loss = epsilon_losses["seg_loss"]
    sadg_state = apply_sadg_perturbation_from_loss(
        sadg_params,
        epsilon_loss,
        rho=float(args.sadg_rho),
        eps=float(args.sadg_eps),
        device=device,
    )
    perturbations = sadg_state["perturbations"]
    grad_norm = sadg_state["grad_norm"]
    perturb_norm = sadg_state["perturb_norm"]

    update_losses: Dict[str, torch.Tensor] | None = None
    gate_losses: Dict[str, torch.Tensor] | None = None
    geo_losses: Dict[str, torch.Tensor] | None = None
    optimizer.zero_grad(set_to_none=True)
    try:
        update_losses = compute_training_losses(model, img, mask, args)
        layer = unwrap_style_layer(model)
        if layer is None:
            zero = zero_tensor(model, device)
            gate_losses = {
                "gate_reg_loss": zero,
                "gate_anchor_loss": zero,
                "gate_smooth_loss": zero,
                "gate_mono_loss": zero,
            }
            geo_losses = {
                "geo_style_loss": zero,
                "geo_loss": zero,
                "geo_weight_mean": zero.detach(),
            }
        else:
            gate_losses = layer.regularization_losses(
                lambda_anchor=float(args.lambda_gate_anchor),
                lambda_smooth=float(args.lambda_gate_smooth),
                lambda_mono=float(args.lambda_gate_mono),
            )
            if teacher_model is None:
                raise RuntimeError("Geometry-preserving student requires a frozen SADG teacher.")
            geo_losses = geometry_preserving_losses(
                student_logits=update_losses["logits"],
                teacher_model=teacher_model,
                img=img,
                mask=mask,
                args=args,
                device=device,
            )
        total_loss = update_losses["loss"] + gate_losses["gate_reg_loss"] + geo_losses["geo_style_loss"]
        total_loss.backward()
    finally:
        restore_sadg_perturbation(perturbations)

    optimizer.step()
    if update_losses is None or gate_losses is None or geo_losses is None:
        raise RuntimeError("SADG update losses were not computed.")
    layer_meta = unwrap_style_layer(model).scalar_metadata() if unwrap_style_layer(model) is not None else {}
    total_style = gate_losses["gate_reg_loss"] + geo_losses["geo_style_loss"]
    return {
        "train_loss": float((update_losses["loss"] + total_style).detach().cpu()),
        "train_seg_loss": float(update_losses["seg_loss"].detach().cpu()),
        "train_disp_loss": float(update_losses["disp_loss"].detach().cpu()),
        "style_reg_loss": float(total_style.detach().cpu()),
        "gate_reg_loss": float(gate_losses["gate_reg_loss"].detach().cpu()),
        "gate_anchor_loss": float(gate_losses["gate_anchor_loss"].detach().cpu()),
        "gate_smooth_loss": float(gate_losses["gate_smooth_loss"].detach().cpu()),
        "gate_mono_loss": float(gate_losses["gate_mono_loss"].detach().cpu()),
        "geo_loss": float(geo_losses["geo_loss"].detach().cpu()),
        "geo_style_loss": float(geo_losses["geo_style_loss"].detach().cpu()),
        "geo_weight_mean": float(geo_losses["geo_weight_mean"].detach().cpu()),
        "epsilon_loss": float(epsilon_loss.detach().cpu()),
        "sadg_grad_norm": float(grad_norm.detach().cpu()),
        "perturb_norm": float(perturb_norm),
        **layer_meta,
    }


def train_model(
    *,
    args: argparse.Namespace,
    device: torch.device,
    role: str,
    shot: int,
    seed: int,
    train_dataset,
    teacher_model: nn.Module | None = None,
    teacher_info: Mapping[str, Any] | None = None,
) -> tuple[nn.Module, List[Dict[str, Any]], Path]:
    set_seed(int(seed))
    if role == "teacher":
        model = build_teacher_model(args, device)
        run_dir = teacher_run_dir(args, int(shot), int(seed))
    elif role == "student":
        init_template = compute_source_log_template(train_dataset, args=args, device=device)
        model = build_student_model(args, device, init_template)
        run_dir = student_run_dir(args, int(shot), int(seed))
    else:
        raise ValueError(f"Unknown role {role!r}.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    sadg_params = [p for p in unwrap_backbone(model).parameters() if p.requires_grad]
    loader = make_loader(train_dataset, batch_size=int(args.batch_size), shuffle=True, device=device)
    log_rows: List[Dict[str, Any]] = []

    print(
        f"[TRAIN] role={role} shot={shot} seed={seed} sadg_rho={float(args.sadg_rho):.6g} "
        f"cases={train_dataset.selected_case_ids} slices={len(train_dataset)}",
        flush=True,
    )
    for epoch in range(1, int(args.epochs) + 1):
        epoch_rows: List[Dict[str, float]] = []
        for step_idx, (img, mask, _meta) in enumerate(loader, start=1):
            img = img.to(device)
            mask = mask.to(device)
            step_row = train_step(
                model=model,
                teacher_model=teacher_model,
                optimizer=optimizer,
                img=img,
                mask=mask,
                args=args,
                sadg_params=sadg_params,
                device=device,
            )
            epoch_rows.append(step_row)
            if int(args.max_train_steps) > 0 and step_idx >= int(args.max_train_steps):
                break

        layer = unwrap_style_layer(model)
        row: Dict[str, Any] = {
            "role": role,
            "ablation": SEG_SADG_ABLATION,
            "style_method": TEACHER_METHOD if role == "teacher" else STUDENT_METHOD,
            "shot": int(shot),
            "seed": int(seed),
            "epoch": int(epoch),
            "train_loss": finite_mean(r.get("train_loss") for r in epoch_rows),
            "train_seg_loss": finite_mean(r.get("train_seg_loss") for r in epoch_rows),
            "train_disp_loss": finite_mean(r.get("train_disp_loss") for r in epoch_rows),
            "style_reg_loss": finite_mean(r.get("style_reg_loss") for r in epoch_rows),
            "gate_reg_loss": finite_mean(r.get("gate_reg_loss") for r in epoch_rows),
            "gate_anchor_loss": finite_mean(r.get("gate_anchor_loss") for r in epoch_rows),
            "gate_smooth_loss": finite_mean(r.get("gate_smooth_loss") for r in epoch_rows),
            "gate_mono_loss": finite_mean(r.get("gate_mono_loss") for r in epoch_rows),
            "geo_loss": finite_mean(r.get("geo_loss") for r in epoch_rows),
            "geo_style_loss": finite_mean(r.get("geo_style_loss") for r in epoch_rows),
            "geo_weight_mean": finite_mean(r.get("geo_weight_mean") for r in epoch_rows),
            "epsilon_loss": finite_mean(r.get("epsilon_loss") for r in epoch_rows),
            "sadg_grad_norm": finite_mean(r.get("sadg_grad_norm") for r in epoch_rows),
            "perturb_norm": finite_mean(r.get("perturb_norm") for r in epoch_rows),
            "sadg_rho": float(args.sadg_rho),
            "lambda_disp": float(args.lambda_disp),
            "dice_weight": float(args.dice_weight),
            "lambda_geo": float(args.lambda_geo) if role == "student" else 0.0,
            "teacher_checkpoint": (teacher_info or {}).get("teacher_checkpoint", ""),
            "teacher_case_ids_match": bool((teacher_info or {}).get("teacher_case_ids_match", False)),
            "train_case_ids": "|".join(str(x) for x in train_dataset.selected_case_ids),
        }
        if layer is not None:
            row.update(layer.scalar_metadata())
        log_rows.append(row)
        print(
            f"[EPOCH] role={role} shot={shot} seed={seed} epoch={epoch:03d} "
            f"loss={float(row['train_loss']):.6f} seg={float(row['train_seg_loss']):.6f} "
            f"disp={float(row['train_disp_loss']):.6f} eps={float(row['epsilon_loss']):.6f} "
            f"geo={float(row['geo_loss']):.6f} perturb={float(row['perturb_norm']):.6f}",
            flush=True,
        )

    payload = checkpoint_payload(
        model=model,
        args=args,
        role=role,
        shot=int(shot),
        seed=int(seed),
        train_dataset=train_dataset,
        train_rows=log_rows,
        teacher_info=teacher_info,
    )
    save_model_files(run_dir, payload)
    write_json(run_dir / "dataset_metadata.json", payload["metadata"])
    write_json(run_dir / "run_config.json", vars(args))
    write_csv(run_dir / "training_log.csv", log_rows)
    if role == "student":
        save_style_artifacts(student_prompt_dir(args, int(shot), int(seed)), model, train_dataset, args, teacher_info or {})
    return model, log_rows, run_dir


def load_or_train_teacher(
    *,
    args: argparse.Namespace,
    device: torch.device,
    shot: int,
    seed: int,
    train_dataset,
) -> tuple[nn.Module, Dict[str, Any]]:
    run_dir = teacher_run_dir(args, int(shot), int(seed))
    checkpoint = run_dir / "baseline_model_with_metadata.pt"
    if checkpoint.exists() and bool(args.resume) and not bool(args.overwrite):
        model = build_teacher_model(args, device)
        metadata = load_checkpoint(model, checkpoint, device)
        print(f"[LOAD] teacher shot={shot} seed={seed}: {checkpoint}", flush=True)
    elif checkpoint.exists() and not bool(args.overwrite) and not bool(args.resume):
        raise FileExistsError(f"Teacher checkpoint exists and --resume is disabled: {checkpoint}")
    else:
        model, _rows, _run_dir = train_model(
            args=args,
            device=device,
            role="teacher",
            shot=int(shot),
            seed=int(seed),
            train_dataset=train_dataset,
        )
        metadata = extract_metadata(safe_torch_load(checkpoint, map_location=device))

    current_ids = [str(x) for x in train_dataset.selected_case_ids]
    teacher_ids = [str(x) for x in metadata.get("train_case_ids", [])]
    case_ids_match = teacher_ids == current_ids
    if not case_ids_match:
        raise RuntimeError(
            "Teacher train_case_ids do not match current source split: "
            f"teacher={teacher_ids}, current={current_ids}, checkpoint={checkpoint}"
        )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {
        "teacher_checkpoint": str(checkpoint),
        "teacher_train_case_ids": teacher_ids,
        "teacher_case_ids_match": bool(case_ids_match),
    }


def load_or_train_student(
    *,
    args: argparse.Namespace,
    device: torch.device,
    shot: int,
    seed: int,
    train_dataset,
    teacher_model: nn.Module,
    teacher_info: Mapping[str, Any],
) -> tuple[nn.Module, List[Dict[str, Any]], Path, Dict[str, Any]]:
    run_dir = student_run_dir(args, int(shot), int(seed))
    checkpoint = run_dir / "baseline_model_with_metadata.pt"
    if checkpoint.exists() and bool(args.resume) and not bool(args.overwrite):
        init_template = compute_source_log_template(train_dataset, args=args, device=device)
        model = build_student_model(args, device, init_template)
        metadata = load_checkpoint(model, checkpoint, device)
        print(f"[LOAD] student shot={shot} seed={seed}: {checkpoint}", flush=True)
        return model, read_csv(run_dir / "training_log.csv"), run_dir, metadata
    if checkpoint.exists() and not bool(args.overwrite) and not bool(args.resume):
        raise FileExistsError(f"Student checkpoint exists and --resume is disabled: {checkpoint}")

    model, train_rows, _run_dir = train_model(
        args=args,
        device=device,
        role="student",
        shot=int(shot),
        seed=int(seed),
        train_dataset=train_dataset,
        teacher_model=teacher_model,
        teacher_info=teacher_info,
    )
    metadata = extract_metadata(safe_torch_load(checkpoint, map_location=device))
    return model, train_rows, run_dir, metadata


def annotate_eval_row(
    row: Dict[str, Any],
    *,
    args: argparse.Namespace,
    shot: int,
    seed: int,
    domain: str,
    train_case_ids: Sequence[str],
    output_dir: Path,
    prompt_dir: Path,
    checkpoint_source: str,
    model: nn.Module,
    teacher_info: Mapping[str, Any],
) -> Dict[str, Any]:
    layer = unwrap_style_layer(model)
    window = lowfreq_window(int(args.resize_hw), int(args.resize_hw), float(args.style_alpha))
    meta = layer.scalar_metadata() if layer is not None else {}
    out = {
        "ablation": SEG_SADG_ABLATION,
        "style_method": STUDENT_METHOD,
        "domain": domain,
        "shot": int(shot),
        "seed": int(seed),
        "style_alpha": float(args.style_alpha),
        "style_rho": float(args.style_rho),
        "sadg_rho": float(args.sadg_rho),
        "lambda_disp": float(args.lambda_disp),
        "dice_weight": float(args.dice_weight),
        "lambda_gate_anchor": float(args.lambda_gate_anchor),
        "lambda_gate_smooth": float(args.lambda_gate_smooth),
        "lambda_gate_mono": float(args.lambda_gate_mono),
        "lambda_geo": float(args.lambda_geo),
        "geo_sdt_clip_px": float(args.geo_sdt_clip_px),
        "geo_eps": float(args.geo_eps),
        "geometry_preserving": True,
        "geo_loss_type": "fg_sdt_weighted_l1",
        "teacher_checkpoint": teacher_info.get("teacher_checkpoint", ""),
        "teacher_train_case_ids": "|".join(str(x) for x in teacher_info.get("teacher_train_case_ids", [])),
        "teacher_case_ids_match": bool(teacher_info.get("teacher_case_ids_match", False)),
        "train_case_ids": "|".join(str(x) for x in train_case_ids),
        "excluded_case_ids": "|".join(str(x) for x in train_case_ids),
        "lf_radius_h": int(window["radius_h"]),
        "lf_radius_w": int(window["radius_w"]),
        "lf_size_h": int(window["size_h"]),
        "lf_size_w": int(window["size_w"]),
        "checkpoint_source": checkpoint_source,
        "output_dir": str(output_dir),
        "prompt_dir": str(prompt_dir),
    }
    out.update(meta)
    out.update(row)
    out["domain"] = domain
    return out


def evaluate_student(
    *,
    args: argparse.Namespace,
    model: nn.Module,
    device: torch.device,
    shot: int,
    seed: int,
    train_case_ids: Sequence[str],
    output_dir: Path,
    prompt_dir: Path,
    checkpoint_source: str,
    teacher_info: Mapping[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    eval_rows: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []
    for domain in DOMAIN_NAMES:
        metrics = run_test_flow(
            model,
            domain,
            exclude_case_ids=train_case_ids,
            data_root=Path(args.data_root),
            min_fg_ratio=float(args.min_fg_ratio),
            resize_hw=int(args.resize_hw),
            batch_size=int(args.eval_batch_size),
            device=device,
            max_cases=int(args.max_test_cases) if int(args.max_test_cases) > 0 else None,
            use_split=False,
            eval_set="all_domains",
            slice_policy=str(args.slice_policy),
            num_middle_slices=int(args.num_middle_slices),
            filter_min_fg=bool(args.filter_min_fg),
        )
        row = annotate_eval_row(
            {
                "n_cases": int(metrics["n_cases"]),
                "n_slices": int(metrics["n_slices"]),
                "slice_policy": metrics["slice_policy"],
                "num_middle_slices": int(metrics["num_middle_slices"]),
                **metrics["summary"],
            },
            args=args,
            shot=int(shot),
            seed=int(seed),
            domain=metrics["domain"],
            train_case_ids=train_case_ids,
            output_dir=output_dir,
            prompt_dir=prompt_dir,
            checkpoint_source=checkpoint_source,
            model=model,
            teacher_info=teacher_info,
        )
        eval_rows.append(row)
        for case_row in metrics["case_rows"]:
            case_rows.append(
                annotate_eval_row(
                    dict(case_row),
                    args=args,
                    shot=int(shot),
                    seed=int(seed),
                    domain=metrics["domain"],
                    train_case_ids=train_case_ids,
                    output_dir=output_dir,
                    prompt_dir=prompt_dir,
                    checkpoint_source=checkpoint_source,
                    model=model,
                    teacher_info=teacher_info,
                )
            )
        print(
            f"[EVAL] shot={shot} seed={seed} domain={metrics['domain']} "
            f"case_dice={float(row['case_dice']):.6f} case_hd95={float(row['case_hd95']):.6f}",
            flush=True,
        )
    return eval_rows, case_rows


def summarize_eval_groups(eval_rows: Sequence[Mapping[str, Any]], group_fields: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for row in eval_rows:
        key = tuple(row.get(field, "") for field in group_fields)
        groups.setdefault(key, []).append(row)
    out: List[Dict[str, Any]] = []
    metric_fields = [
        "case_dice",
        "case_hd95",
        "slice_dice",
        "slice_hd95",
        "rho_bin_mean",
        "lambda_geo",
    ]
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        summary = {field: value for field, value in zip(group_fields, key)}
        summary["n_rows"] = int(len(rows))
        summary["n_cases_total"] = int(sum(int(float(row.get("n_cases", 0) or 0)) for row in rows))
        for metric in metric_fields:
            summary[f"{metric}_mean"] = finite_mean(row.get(metric) for row in rows)
            summary[f"{metric}_std"] = finite_std(row.get(metric) for row in rows)
        out.append(summary)
    return out


def experiment_summary_rows(eval_rows: Sequence[Mapping[str, Any]], training_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[int, int], List[Mapping[str, Any]]] = {}
    for row in eval_rows:
        groups.setdefault((int(row.get("shot", 0)), int(row.get("seed", 0))), []).append(row)
    train_groups: Dict[tuple[int, int], List[Mapping[str, Any]]] = {}
    for row in training_rows:
        train_groups.setdefault((int(row.get("shot", 0)), int(row.get("seed", 0))), []).append(row)
    out: List[Dict[str, Any]] = []
    for (shot, seed), rows in sorted(groups.items()):
        train = train_groups.get((shot, seed), [])
        last_train = train[-1] if train else {}
        out.append(
            {
                "ablation": SEG_SADG_ABLATION,
                "style_method": STUDENT_METHOD,
                "shot": int(shot),
                "seed": int(seed),
                "case_dice_mean": finite_mean(row.get("case_dice") for row in rows),
                "case_hd95_mean": finite_mean(row.get("case_hd95") for row in rows),
                "slice_dice_mean": finite_mean(row.get("slice_dice") for row in rows),
                "slice_hd95_mean": finite_mean(row.get("slice_hd95") for row in rows),
                "final_train_loss": last_train.get("train_loss", float("nan")),
                "final_geo_loss": last_train.get("geo_loss", float("nan")),
                "final_rho_bin_mean": last_train.get("rho_bin_mean", float("nan")),
            }
        )
    return out


def write_analysis_report(result_root: Path, overall_rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Geometry-Preserving RadialGate 精简版结果报告",
        "",
        "本脚本仅公开评估 Geometry-Preserving RadialGate；SADG 只作为同 shot/seed 的 frozen teacher 内部使用。",
        "",
        "## Overall 15-run Domain Summary",
        "",
        "| Domain | Dice | HD95 | n_rows | n_cases | rho_mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in overall_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("domain", "")),
                    f"{float(row.get('case_dice_mean', float('nan'))):.4f}",
                    f"{float(row.get('case_hd95_mean', float('nan'))):.2f}",
                    str(int(row.get("n_rows", 0))),
                    str(int(row.get("n_cases_total", 0))),
                    f"{float(row.get('rho_bin_mean_mean', float('nan'))):.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 文件索引",
            "",
            "- `training_curves.csv`: student 训练曲线。",
            "- `eval_metrics.csv`: domain-level Dice/HD95。",
            "- `eval_case_metrics.csv`: case-level Dice/HD95。",
            "- `domain_5seed_summary.csv`: 每个 shot/domain 的 5-seed 平均。",
            "- `overall_15run_domain_summary.csv`: 每个 domain 跨 3 shots x 5 seeds 平均。",
        ]
    )
    (result_root / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_all_outputs(
    result_root: Path,
    *,
    training_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
    case_rows: Sequence[Mapping[str, Any]],
) -> None:
    write_csv(result_root / "training_curves.csv", training_rows)
    write_csv(result_root / "eval_metrics.csv", eval_rows)
    write_csv(result_root / "eval_case_metrics.csv", case_rows)
    experiments = experiment_summary_rows(eval_rows, training_rows)
    domain_5seed = summarize_eval_groups(eval_rows, ("shot", "domain"))
    overall = summarize_eval_groups(eval_rows, ("domain",))
    write_csv(result_root / "experiment_summary.csv", experiments)
    write_csv(result_root / "domain_5seed_summary.csv", domain_5seed)
    write_csv(result_root / "overall_15run_domain_summary.csv", overall)
    write_analysis_report(result_root, overall)


def run(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    result_root = ensure_dir(resolve_path(args.result_root))
    ensure_dir(resolve_path(args.new_backbone_root))
    ensure_dir(resolve_path(args.prompt_root))
    shots = parse_int_list(args.shots)
    seeds = parse_int_list(args.seeds)

    all_training_rows: List[Dict[str, Any]] = []
    all_eval_rows: List[Dict[str, Any]] = []
    all_case_rows: List[Dict[str, Any]] = []

    print(f"[DEVICE] {device}", flush=True)
    print(f"[RESULT_ROOT] {result_root}", flush=True)
    print(f"[BACKBONE_ROOT] {resolve_path(args.new_backbone_root)}", flush=True)
    print(f"[PROMPT_ROOT] {resolve_path(args.prompt_root)}", flush=True)
    print(
        f"[METHOD] {STUDENT_METHOD} style_alpha={float(args.style_alpha):.6g} "
        f"style_rho={float(args.style_rho):.6g} sadg_rho={float(args.sadg_rho):.6g} "
        f"shots={shots} seeds={seeds}",
        flush=True,
    )

    for shot in shots:
        for seed in seeds:
            train_dataset = build_train_data(args, int(shot), int(seed))
            teacher_model, teacher_info = load_or_train_teacher(
                args=args,
                device=device,
                shot=int(shot),
                seed=int(seed),
                train_dataset=train_dataset,
            )
            student_model, train_rows, output_dir, metadata = load_or_train_student(
                args=args,
                device=device,
                shot=int(shot),
                seed=int(seed),
                train_dataset=train_dataset,
                teacher_model=teacher_model,
                teacher_info=teacher_info,
            )
            prompt_dir = student_prompt_dir(args, int(shot), int(seed))
            eval_rows, case_rows = evaluate_student(
                args=args,
                model=student_model,
                device=device,
                shot=int(shot),
                seed=int(seed),
                train_case_ids=[str(x) for x in train_dataset.selected_case_ids],
                output_dir=output_dir,
                prompt_dir=prompt_dir,
                checkpoint_source=str(metadata.get("checkpoint_path", output_dir / "baseline_model_with_metadata.pt")),
                teacher_info=teacher_info,
            )
            all_training_rows.extend(train_rows)
            all_eval_rows.extend(eval_rows)
            all_case_rows.extend(case_rows)
            write_all_outputs(
                result_root,
                training_rows=all_training_rows,
                eval_rows=all_eval_rows,
                case_rows=all_case_rows,
            )
            del teacher_model
            del student_model
            if device.type == "cuda":
                torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Geometry-Preserving RadialGate with an internal SADG teacher.")
    parser.add_argument("--data_root", default=str(DATA_ROOT))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--shots", default="3,4,5")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dice_weight", type=float, default=0.5)
    parser.add_argument("--lambda_disp", type=float, default=0.05)
    parser.add_argument("--disp_margin", type=float, default=0.0)
    parser.add_argument("--sadg_rho", type=float, default=FIXED_SADG_RHO)
    parser.add_argument("--sadg_eps", type=float, default=1e-12)
    parser.add_argument("--style_alpha", type=float, default=0.08)
    parser.add_argument("--style_rho", type=float, default=0.5)
    parser.add_argument("--style_eps", type=float, default=1e-6)
    parser.add_argument("--fft_norm", default="ortho", choices=("backward", "ortho", "forward"))
    parser.add_argument("--gate_num_bins", type=int, default=8)
    parser.add_argument("--gate_rho_max", type=float, default=1.0)
    parser.add_argument("--lambda_gate_anchor", type=float, default=1e-4)
    parser.add_argument("--lambda_gate_smooth", type=float, default=1e-3)
    parser.add_argument("--lambda_gate_mono", type=float, default=1e-3)
    parser.add_argument("--lambda_geo", type=float, default=0.1)
    parser.add_argument("--geo_sdt_clip_px", type=float, default=32.0)
    parser.add_argument("--geo_eps", type=float, default=1e-6)
    parser.add_argument("--resize_hw", type=int, default=224)
    parser.add_argument("--min_fg_ratio", type=float, default=0.05)
    parser.add_argument("--filter_min_fg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--slice_policy", default="all_filtered", choices=("center9", "all", "all_filtered"))
    parser.add_argument("--num_middle_slices", type=int, default=9)
    parser.add_argument("--base_ch", type=int, default=16)
    parser.add_argument("--latent_ch", type=int, default=64)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--max_test_cases", type=int, default=0)
    parser.add_argument("--result_root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--new_backbone_root", default=str(DEFAULT_BACKBONE_ROOT))
    parser.add_argument("--prompt_root", default=str(DEFAULT_PROMPT_ROOT))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
