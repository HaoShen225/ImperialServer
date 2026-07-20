from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
# 在网络之前插一个可训练的风格调整层 实现代码

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "TrainingData" / "DisperssionWithSADG_RadialGateStyleRegularization"
DEFAULT_NEW_BACKBONE_ROOT = PROJECT_ROOT / "backbones" / "DisperssionWithSADG_RadialGateStyleRegularization"
DEFAULT_PROMPT_ROOT = PROJECT_ROOT / "backbones" / "Prompts"
SOURCE_DOMAIN = SOURCE_DOMAIN_NAME

SEG_SADG_ABLATION = "Seg-SADG"
FIXED_SADG_RHO = 0.05
ABLATIONS = (SEG_SADG_ABLATION,)
ABLATION_ALIASES = {
    "seg": SEG_SADG_ABLATION,
    "seg-sadg": SEG_SADG_ABLATION,
    "seg_sadg": SEG_SADG_ABLATION,
}

LF_BARYCENTER_RADIAL_GATE = "LF-Barycenter-RadialGate"
STYLE_METHODS = (
    LF_BARYCENTER_RADIAL_GATE,
)
DEFAULT_STYLE_METHODS = (
    LF_BARYCENTER_RADIAL_GATE,
)
STYLE_METHOD_ALIASES = {
    "lf-barycenter-radialgate": LF_BARYCENTER_RADIAL_GATE,
    "lf-barycenter-radial-gate": LF_BARYCENTER_RADIAL_GATE,
    "lf_barycenter_radialgate": LF_BARYCENTER_RADIAL_GATE,
    "lf_barycenter_radial_gate": LF_BARYCENTER_RADIAL_GATE,
    "barycenter-radialgate": LF_BARYCENTER_RADIAL_GATE,
    "barycenter-radial-gate": LF_BARYCENTER_RADIAL_GATE,
    "radialgate": LF_BARYCENTER_RADIAL_GATE,
    "radial-gate": LF_BARYCENTER_RADIAL_GATE,
}


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(name: str) -> torch.device:
    if str(name).strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(str(name))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return device


def resolve_path(path: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base / p


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_str_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def rho_tag(value: float) -> str:
    text = f"{float(value):.12g}".replace("-", "m").replace(".", "p")
    return f"rho{text}"


def value_tag(name: str, value: float) -> str:
    text = f"{float(value):.12g}".replace("-", "m").replace(".", "p")
    return f"{name}{text}"


def alpha_tag(value: float) -> str:
    return value_tag("alpha", value)


def style_rho_tag(value: float) -> str:
    return value_tag("style_rho", value)


def sadg_rho_tag(value: float) -> str:
    return value_tag("sadg_rho", value)


def gate_bins_tag(value: int) -> str:
    return f"gatebins{int(value)}"


def consistency_tag(value: float) -> str:
    return value_tag("cons", value)


def path_value_tag(name: str, value: float) -> str:
    text = f"{float(value):.12g}".replace("-", "m").replace(".", "p")
    return f"{name}{text}"


def style_method_path_name(style_method: str) -> str:
    method = normalize_style_method(style_method)
    names = {
        LF_BARYCENTER_RADIAL_GATE: "BaryRadGate",
    }
    return names[method]


def normalize_ablation(name: str) -> str:
    key = str(name).strip().lower()
    if key in ABLATION_ALIASES:
        return ABLATION_ALIASES[key]
    for ablation in ABLATIONS:
        if key == ablation.lower():
            return ablation
    raise ValueError(f"Unknown ablation {name!r}; expected one of {', '.join(ABLATIONS)}")


def parse_ablation_list(text: str) -> List[str]:
    ablations = [normalize_ablation(name) for name in parse_str_list(text)]
    if not ablations:
        raise ValueError("At least one ablation must be provided.")
    seen: set[str] = set()
    out: List[str] = []
    for ablation in ablations:
        if ablation not in seen:
            seen.add(ablation)
            out.append(ablation)
    return out


def normalize_style_method(name: str) -> str:
    key = str(name).strip()
    lowered = key.lower()
    if lowered in STYLE_METHOD_ALIASES:
        return STYLE_METHOD_ALIASES[lowered]
    for method in STYLE_METHODS:
        if lowered == method.lower():
            return method
    raise ValueError(f"Unknown style method {name!r}; expected one of {', '.join(STYLE_METHODS)}")


def parse_style_methods(text: str) -> List[str]:
    methods = [normalize_style_method(name) for name in parse_str_list(text)]
    if not methods:
        raise ValueError("At least one style method must be provided.")
    seen: set[str] = set()
    out: List[str] = []
    for method in methods:
        if method not in seen:
            seen.add(method)
            out.append(method)
    return out


def style_method_parts(style_method: str) -> tuple[str, str]:
    normalize_style_method(style_method)
    return "fixed", "cosine"


def style_gate_mode(style_method: str) -> str:
    normalize_style_method(style_method)
    return "radial"


def uses_radial_gate(style_method: str) -> bool:
    normalize_style_method(style_method)
    return True


def uses_spectral_consistency(style_method: str) -> bool:
    normalize_style_method(style_method)
    return False


def finite_values(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            f = float(value)
            if np.isfinite(f):
                out.append(f)
        except Exception:
            pass
    return out


def finite_mean(values: Iterable[Any]) -> float:
    xs = finite_values(values)
    return float(np.mean(xs)) if xs else float("nan")


def finite_std(values: Iterable[Any]) -> float:
    xs = finite_values(values)
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0 if len(xs) == 1 else float("nan")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
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
    r_h = min(r_h, max((int(H) - 1) // 2, 0))
    r_w = min(r_w, max((int(W) - 1) // 2, 0))
    h_lf = 2 * r_h + 1
    w_lf = 2 * r_w + 1
    u0 = int(H) // 2 - r_h
    v0 = int(W) // 2 - r_w
    return {
        "radius_h": int(r_h),
        "radius_w": int(r_w),
        "size_h": int(h_lf),
        "size_w": int(w_lf),
        "u0": int(u0),
        "u1": int(u0 + h_lf),
        "v0": int(v0),
        "v1": int(v0 + w_lf),
    }


def radial_cosine_mask(h_lf: int, w_lf: int, device: torch.device | None = None) -> torch.Tensor:
    radius = normalized_radius_map(h_lf, w_lf, device=device)
    mask = 0.5 * (1.0 + torch.cos(float(np.pi) * radius))
    return mask[None, None, ...]


def normalized_radius_map(h_lf: int, w_lf: int, device: torch.device | None = None) -> torch.Tensor:
    rh = max((int(h_lf) - 1) // 2, 1)
    rw = max((int(w_lf) - 1) // 2, 1)
    yy = torch.arange(int(h_lf), dtype=torch.float32, device=device) - (int(h_lf) // 2)
    xx = torch.arange(int(w_lf), dtype=torch.float32, device=device) - (int(w_lf) // 2)
    y, x = torch.meshgrid(yy / float(rh), xx / float(rw), indexing="ij")
    return torch.sqrt(y.pow(2) + x.pow(2)).clamp(0.0, 1.0)


def radial_bin_map(h_lf: int, w_lf: int, num_bins: int, device: torch.device | None = None) -> torch.Tensor:
    if int(num_bins) <= 0:
        raise ValueError("num_bins must be positive.")
    radius = normalized_radius_map(h_lf, w_lf, device=device)
    bins = torch.floor(radius * float(num_bins)).long().clamp(0, int(num_bins) - 1)
    return bins[None, None, ...]


def bounded_logit(value: float, *, eps: float = 1e-6) -> float:
    p = min(max(float(value), float(eps)), 1.0 - float(eps))
    return float(np.log(p / (1.0 - p)))


class LowFreqAmplitudeTemplateLayer(nn.Module):
    def __init__(
        self,
        *,
        H: int,
        W: int,
        alpha: float,
        style_rho: float,
        init_template: torch.Tensor,
        gate_mode: str,
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
        self.template_mode = "fixed"
        self.mask_mode = "cosine"
        self.gate_mode = str(gate_mode)
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
        self.register_buffer("T_raw", template.clone())

        self.register_buffer("cosine_mask", radial_cosine_mask(self.H_lf, self.W_lf))
        self.register_buffer("radius_map", normalized_radius_map(self.H_lf, self.W_lf)[None, None, ...])
        self.register_buffer("radial_bin_map", radial_bin_map(self.H_lf, self.W_lf, self.gate_num_bins))

        if self.gate_mode != "radial":
            raise ValueError(f"Unknown gate_mode {gate_mode!r}; expected radial.")
        init_ratio = float(self.style_rho) / float(self.gate_rho_max)
        init_logit = bounded_logit(init_ratio)
        self.gate_logits = nn.Parameter(torch.full((self.gate_num_bins,), init_logit, dtype=torch.float32))

    @property
    def trainable_template(self) -> bool:
        return False

    @property
    def trainable_gate(self) -> bool:
        return isinstance(self.gate_logits, nn.Parameter)

    @property
    def spectral_consistency_enabled(self) -> bool:
        return False

    def symmetric_template(self) -> torch.Tensor:
        return 0.5 * (self.T0 + torch.flip(self.T0, dims=(-2, -1)))

    def rho_bins(self) -> torch.Tensor:
        if self.trainable_gate:
            return float(self.gate_rho_max) * torch.sigmoid(self.gate_logits)
        ref = self.T0
        return ref.new_full((self.gate_num_bins,), float(self.style_rho))

    def rho_map(self) -> torch.Tensor:
        bins = self.rho_bins()
        idx = self.radial_bin_map.to(device=bins.device)
        return bins[idx]

    def effective_gate_map(self) -> torch.Tensor:
        return self.cosine_mask.to(device=self.T0.device, dtype=self.T0.dtype) * self.rho_map()

    def config_dict(self) -> Dict[str, Any]:
        rho = self.rho_bins().detach().cpu().tolist()
        return {
            "H": self.H,
            "W": self.W,
            "style_alpha": self.alpha,
            "style_rho": self.style_rho,
            "template_mode": self.template_mode,
            "mask_mode": self.mask_mode,
            "gate_mode": self.gate_mode,
            "gate_num_bins": self.gate_num_bins,
            "gate_rho_max": self.gate_rho_max,
            "rho_bins": [float(v) for v in rho],
            "style_eps": self.eps,
            "fft_norm": self.fft_norm,
            "lf_radius_h": self.radius_h,
            "lf_radius_w": self.radius_w,
            "lf_size_h": self.H_lf,
            "lf_size_w": self.W_lf,
            "u0": self.u0,
            "u1": self.u1,
            "v0": self.v0,
            "v1": self.v1,
        }

    def regularization_losses(
        self,
        *,
        lambda_anchor: float,
        lambda_smooth: float,
        lambda_mono: float,
    ) -> Dict[str, torch.Tensor]:
        rho = self.rho_bins()
        if not self.trainable_gate:
            zero = rho.sum() * 0.0
            return {
                "style_reg_loss": zero,
                "gate_reg_loss": zero,
                "gate_anchor_loss": zero,
                "gate_smooth_loss": zero,
                "gate_mono_loss": zero,
                "template_l2_loss": zero,
                "template_smooth_loss": zero,
            }
        target = rho.new_full(rho.shape, float(self.style_rho))
        anchor = F.mse_loss(rho, target)
        if rho.numel() > 1:
            diff = rho[1:] - rho[:-1]
            smooth = diff.pow(2).mean()
            mono = F.relu(diff).pow(2).mean()
        else:
            smooth = rho.sum() * 0.0
            mono = rho.sum() * 0.0
        total = (
            float(lambda_anchor) * anchor
            + float(lambda_smooth) * smooth
            + float(lambda_mono) * mono
        )
        return {
            "style_reg_loss": total,
            "gate_reg_loss": total,
            "gate_anchor_loss": anchor,
            "gate_smooth_loss": smooth,
            "gate_mono_loss": mono,
            "template_l2_loss": total.detach() * 0.0,
            "template_smooth_loss": total.detach() * 0.0,
        }

    def low_frequency_log_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        X = torch.fft.fftshift(torch.fft.fft2(x, norm=self.fft_norm), dim=(-2, -1))
        X_lf = X[:, :, self.u0:self.u1, self.v0:self.v1]
        return torch.log(torch.abs(X_lf) + self.eps)

    def canonicalized_log_amplitude_proxy(self, logA_lf: torch.Tensor) -> torch.Tensor:
        gate = self.effective_gate_map().to(dtype=logA_lf.dtype, device=logA_lf.device)
        T = self.symmetric_template().to(dtype=logA_lf.dtype, device=logA_lf.device)
        return logA_lf + gate * (T - logA_lf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        X = torch.fft.fft2(x, norm=self.fft_norm)
        Xs = torch.fft.fftshift(X, dim=(-2, -1))
        X_lf = Xs[:, :, self.u0:self.u1, self.v0:self.v1]

        A_lf = torch.abs(X_lf)
        phase_lf = X_lf / (A_lf + self.eps)
        T = self.symmetric_template()
        rho = self.rho_map().to(dtype=A_lf.dtype, device=A_lf.device)
        logA_cal = (1.0 - rho) * torch.log(A_lf + self.eps) + rho * T
        X_lf_cal = torch.exp(logA_cal) * phase_lf

        mask = self.cosine_mask.to(dtype=X_lf.real.dtype, device=X_lf.device)
        X_lf_new = mask * X_lf_cal + (1.0 - mask) * X_lf

        Xs_new = Xs.clone()
        Xs_new[:, :, self.u0:self.u1, self.v0:self.v1] = X_lf_new
        X_new = torch.fft.ifftshift(Xs_new, dim=(-2, -1))
        return torch.fft.ifft2(X_new, norm=self.fft_norm).real


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


def style_run_relative_dir(
    args: argparse.Namespace,
    ablation: str,
    style_method: str,
    shot: int,
    seed: int,
) -> Path:
    return (
        Path(normalize_ablation(ablation))
        / style_method_path_name(style_method)
        / path_value_tag("a", float(args.style_alpha))
        / path_value_tag("rho", float(args.style_rho))
        / f"gb{int(args.gate_num_bins)}"
        / path_value_tag("sadg", float(args.sadg_rho))
        / (
            path_value_tag("cons", float(args.lambda_spectral_consistency))
            if uses_spectral_consistency(style_method)
            else "nocons"
        )
        / f"shot{int(shot)}"
        / f"Seed{int(seed)}"
    )


def ablation_run_dir(args: argparse.Namespace, ablation: str, style_method: str, shot: int, seed: int) -> Path:
    return (
        resolve_path(args.new_backbone_root)
        / style_run_relative_dir(args, ablation, style_method, int(shot), int(seed))
    )


def prompt_run_dir(args: argparse.Namespace, ablation: str, style_method: str, shot: int, seed: int) -> Path:
    return (
        resolve_path(args.prompt_root)
        / style_run_relative_dir(args, ablation, style_method, int(shot), int(seed))
    )


def save_model_files(run_dir: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(run_dir)
    final_path = run_dir / "checkpoint_final.pt"
    baseline_path = run_dir / "baseline_model_with_metadata.pt"
    torch.save(payload, final_path)
    shutil.copyfile(final_path, baseline_path)


def build_train_data(args: argparse.Namespace, shot: int, seed: int):
    return build_train_dataset(
        SOURCE_DOMAIN,
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


def compute_source_log_template(
    train_dataset,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
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


def build_experiment_model(
    *,
    args: argparse.Namespace,
    device: torch.device,
    style_method: str,
    init_template: torch.Tensor | None,
) -> nn.Module:
    backbone = build_model(model_config(args)).to(device)
    style_method = normalize_style_method(style_method)
    template_mode, mask_mode = style_method_parts(style_method)
    if init_template is None:
        raise ValueError(f"{style_method} requires an init_template.")
    H = int(args.resize_hw)
    W = int(args.resize_hw)
    style_layer = LowFreqAmplitudeTemplateLayer(
        H=H,
        W=W,
        alpha=float(args.style_alpha),
        style_rho=float(args.style_rho),
        init_template=init_template.to(device),
        gate_mode=style_gate_mode(style_method),
        gate_num_bins=int(args.gate_num_bins),
        gate_rho_max=float(args.gate_rho_max),
        eps=float(args.style_eps),
        fft_norm=str(args.fft_norm),
    ).to(device)
    return StyleCanonicalizedModel(backbone, style_layer).to(device)


def zero_style_losses(model: nn.Module, device: torch.device) -> Dict[str, torch.Tensor]:
    ref = next(model.parameters(), None)
    zero = (ref.sum() * 0.0) if ref is not None else torch.tensor(0.0, device=device)
    return {
        "style_reg_loss": zero,
        "gate_reg_loss": zero,
        "gate_anchor_loss": zero,
        "gate_smooth_loss": zero,
        "gate_mono_loss": zero,
        "template_l2_loss": zero,
        "template_smooth_loss": zero,
    }


def style_regularization_losses(model: nn.Module, args: argparse.Namespace, device: torch.device) -> Dict[str, torch.Tensor]:
    layer = unwrap_style_layer(model)
    if layer is None or not layer.trainable_gate:
        return zero_style_losses(model, device)
    return layer.regularization_losses(
        lambda_anchor=float(args.lambda_gate_anchor),
        lambda_smooth=float(args.lambda_gate_smooth),
        lambda_mono=float(args.lambda_gate_mono),
    )


def zero_spectral_losses(model: nn.Module, device: torch.device) -> Dict[str, torch.Tensor]:
    ref = next(model.parameters(), None)
    zero = (ref.sum() * 0.0) if ref is not None else torch.tensor(0.0, device=device)
    return {
        "spectral_style_loss": zero,
        "spectral_cons_loss": zero,
        "gate_budget_loss": zero,
    }


def spectral_budget_consistency_losses(
    model: nn.Module,
    img: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    layer = unwrap_style_layer(model)
    if layer is None or not layer.spectral_consistency_enabled:
        return zero_spectral_losses(model, device)

    logA = layer.low_frequency_log_amplitude(img)
    logA_can = layer.canonicalized_log_amplitude_proxy(logA)
    bin_map = layer.radial_bin_map.to(device=logA.device)
    numerator_terms: List[torch.Tensor] = []
    denominator_terms: List[torch.Tensor] = []
    for bin_idx in range(int(layer.gate_num_bins)):
        mask = bin_map == int(bin_idx)
        if not bool(mask.any().item()):
            continue
        original_values = logA[:, :, mask[0, 0]]
        canonical_values = logA_can[:, :, mask[0, 0]]
        numerator_terms.append(canonical_values.var(unbiased=False))
        denominator_terms.append(original_values.var(unbiased=False).detach())

    if numerator_terms:
        numerator = torch.stack(numerator_terms).sum()
        denominator = torch.stack(denominator_terms).sum().detach() + float(args.spectral_cons_eps)
        spectral_cons = numerator / denominator
    else:
        spectral_cons = logA.sum() * 0.0

    rho = layer.rho_bins()
    gate_budget = (rho.mean() - float(args.style_rho)).pow(2)
    total = (
        float(args.lambda_spectral_consistency) * spectral_cons
        + float(args.lambda_gate_budget) * gate_budget
    )
    return {
        "spectral_style_loss": total,
        "spectral_cons_loss": spectral_cons,
        "gate_budget_loss": gate_budget,
    }


def style_layer_scalar_metadata(model: nn.Module) -> Dict[str, Any]:
    layer = unwrap_style_layer(model)
    if layer is None:
        return {
            "gate_mode": "none",
            "gate_num_bins": 0,
            "gate_rho_max": float("nan"),
            "rho_bin_min": float("nan"),
            "rho_bin_max": float("nan"),
            "rho_bin_mean": float("nan"),
            "rho_bins": "",
        }
    rho = layer.rho_bins().detach().cpu().float().numpy()
    return {
        "gate_mode": layer.gate_mode,
        "gate_num_bins": int(layer.gate_num_bins),
        "gate_rho_max": float(layer.gate_rho_max),
        "rho_bin_min": float(np.min(rho)) if rho.size else float("nan"),
        "rho_bin_max": float(np.max(rho)) if rho.size else float("nan"),
        "rho_bin_mean": float(np.mean(rho)) if rho.size else float("nan"),
        "rho_bins": "|".join(f"{float(v):.8g}" for v in rho),
    }


def template_payload(
    *,
    model: nn.Module,
    args: argparse.Namespace,
    ablation: str,
    style_method: str,
    shot: int,
    seed: int,
    train_case_ids: Sequence[str],
) -> Dict[str, Any] | None:
    layer = unwrap_style_layer(model)
    if layer is None:
        return None
    rho_bins = layer.rho_bins().detach().cpu()
    return {
        "T_raw": layer.T_raw.detach().cpu(),
        "T_symmetric": layer.symmetric_template().detach().cpu(),
        "T0": layer.T0.detach().cpu(),
        "cosine_mask": layer.cosine_mask.detach().cpu(),
        "radial_bin_map": layer.radial_bin_map.detach().cpu(),
        "radius_map": layer.radius_map.detach().cpu(),
        "rho_bins": rho_bins,
        "gate_logits": layer.gate_logits.detach().cpu(),
        "state_dict": {key: value.detach().cpu() for key, value in layer.state_dict().items()},
        "config": layer.config_dict(),
        "metadata": {
            "experiment": "DisperssionWithSADG_RadialGateStyleRegularization",
            "ablation": normalize_ablation(ablation),
            "style_method": normalize_style_method(style_method),
            "template_mode": layer.template_mode,
            "mask_mode": layer.mask_mode,
            "gate_mode": layer.gate_mode,
            "gate_num_bins": int(layer.gate_num_bins),
            "gate_rho_max": float(layer.gate_rho_max),
            "style_alpha": float(args.style_alpha),
            "style_rho": float(args.style_rho),
            "sadg_rho": float(args.sadg_rho),
            "lambda_gate_anchor": float(args.lambda_gate_anchor),
            "lambda_gate_smooth": float(args.lambda_gate_smooth),
            "lambda_gate_mono": float(args.lambda_gate_mono),
            "lambda_spectral_consistency": float(args.lambda_spectral_consistency),
            "lambda_gate_budget": float(args.lambda_gate_budget),
            "shot": int(shot),
            "seed": int(seed),
            "train_case_ids": list(train_case_ids),
        },
    }


def save_template_files(
    *,
    prompt_dir: Path,
    model: nn.Module,
    args: argparse.Namespace,
    ablation: str,
    style_method: str,
    shot: int,
    seed: int,
    train_case_ids: Sequence[str],
) -> None:
    payload = template_payload(
        model=model,
        args=args,
        ablation=ablation,
        style_method=style_method,
        shot=int(shot),
        seed=int(seed),
        train_case_ids=train_case_ids,
    )
    if payload is None:
        return
    ensure_dir(prompt_dir)
    layer = unwrap_style_layer(model)
    if layer is None:
        return
    torch.save(payload, prompt_dir / "style_layer_final.pt")
    torch.save(
        {
            "gate_logits": layer.gate_logits.detach().cpu(),
            "rho_bins": layer.rho_bins().detach().cpu(),
            "radial_bin_map": layer.radial_bin_map.detach().cpu(),
            "config": layer.config_dict(),
            "metadata": payload["metadata"],
        },
        prompt_dir / "gate_final.pt",
    )


def safe_torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_experiment_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    payload = safe_torch_load(checkpoint_path, map_location=device)
    backbone = unwrap_backbone(model)
    backbone.load_state_dict(extract_state_dict(payload), strict=True)
    layer = unwrap_style_layer(model)
    if layer is not None:
        if not isinstance(payload, dict) or "style_layer_state_dict" not in payload:
            raise ValueError(f"Checkpoint {checkpoint_path} does not contain style_layer_state_dict.")
        layer.load_state_dict(payload["style_layer_state_dict"], strict=True)
    metadata = extract_metadata(payload)
    metadata["checkpoint_path"] = str(checkpoint_path)
    return metadata


def epsilon_source_name(ablation: str) -> str:
    ablation = normalize_ablation(ablation)
    if ablation == SEG_SADG_ABLATION:
        return "CE + Dice"
    raise ValueError(f"Unsupported ablation: {ablation}")


def checkpoint_payload(
    model: torch.nn.Module,
    *,
    args: argparse.Namespace,
    ablation: str,
    style_method: str,
    shot: int,
    seed: int,
    train_case_ids: Sequence[str],
    n_train_slices: int,
    slice_indices_by_case: Dict[str, List[int]],
    prompt_dir: Path,
) -> Dict[str, Any]:
    cfg = model_config(args)
    ablation = normalize_ablation(ablation)
    style_method = normalize_style_method(style_method)
    template_mode, mask_mode = style_method_parts(style_method)
    layer = unwrap_style_layer(model)
    layer_meta = style_layer_scalar_metadata(model)
    payload: Dict[str, Any] = {
        "model_state_dict": unwrap_backbone(model).state_dict(),
        "metadata": {
            "experiment": "DisperssionWithSADG_RadialGateStyleRegularization",
            "loss_mode": ablation,
            "ablation": ablation,
            "sadg_method": ablation,
            "style_method": style_method,
            "template_mode": template_mode,
            "mask_mode": mask_mode,
            **layer_meta,
            "style_alpha": float(args.style_alpha),
            "style_rho": float(args.style_rho),
            "rho": float(args.style_rho),
            "rho_tag": style_rho_tag(float(args.style_rho)),
            "sadg_rho": float(args.sadg_rho),
            "sadg_rho_tag": sadg_rho_tag(float(args.sadg_rho)),
            "sadg_eps": float(args.sadg_eps),
            "lambda_disp": float(args.lambda_disp),
            "dice_weight": float(args.dice_weight),
            "lambda_gate_anchor": float(args.lambda_gate_anchor),
            "lambda_gate_smooth": float(args.lambda_gate_smooth),
            "lambda_gate_mono": float(args.lambda_gate_mono),
            "lambda_spectral_consistency": float(args.lambda_spectral_consistency),
            "lambda_gate_budget": float(args.lambda_gate_budget),
            "spectral_cons_eps": float(args.spectral_cons_eps),
            "source_domain": SOURCE_DOMAIN,
            "shot": int(shot),
            "seed": int(seed),
            "train_case_ids": list(train_case_ids),
            "n_train_slices": int(n_train_slices),
            "use_split": False,
            "slice_policy": str(args.slice_policy),
            "num_middle_slices": int(args.num_middle_slices),
            "filter_min_fg": bool(args.filter_min_fg),
            "min_fg_ratio": float(args.min_fg_ratio),
            "resize_hw": int(args.resize_hw),
            "model_type": cfg.model_type,
            "batch_norm_momentum": 0.1,
            "slice_indices_by_case": slice_indices_by_case,
            "prompt_dir": str(prompt_dir),
            "style_layer_config": layer.config_dict() if layer is not None else {},
            "model_config": {
                "in_ch": int(cfg.in_ch),
                "num_classes": int(cfg.num_classes),
                "base_ch": int(cfg.base_ch),
                "latent_ch": int(cfg.latent_ch),
                "model_type": str(cfg.model_type),
                "use_l2_norm": bool(cfg.use_l2_norm),
                "use_batch_norm": bool(cfg.use_batch_norm),
            },
            "objective": {
                "segmentation": f"CE + {float(args.dice_weight)} Dice",
                "dispersion": "L2_disperssion_loss",
                "lambda_disp": float(args.lambda_disp),
                "dice_weight": float(args.dice_weight),
                "disp_margin": float(args.disp_margin),
                "epsilon_source": epsilon_source_name(ablation),
                "update_source": "CE + Dice + Dispersion at theta + epsilon",
                "style_regularization": "fixed source spectral barycenter with learnable radial gate regularization",
                "spectral_consistency": "disabled for RadialGate-only run",
            },
            "args": vars(args),
        },
    }
    if layer is not None:
        payload["style_layer_state_dict"] = layer.state_dict()
    return payload


def normalize_training_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    ablation: str,
    style_method: str,
    shot: int,
    seed: int,
    output_dir: Path,
    checkpoint_source: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    ablation = normalize_ablation(ablation)
    style_method = normalize_style_method(style_method)
    template_mode, mask_mode = style_method_parts(style_method)
    window = lowfreq_window(int(args.resize_hw), int(args.resize_hw), float(args.style_alpha))
    out: List[Dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        row["loss_mode"] = ablation
        row["ablation"] = ablation
        row["sadg_method"] = ablation
        row["style_method"] = style_method
        row["template_mode"] = template_mode
        row["mask_mode"] = mask_mode
        row["gate_mode"] = style_gate_mode(style_method)
        row["gate_num_bins"] = int(args.gate_num_bins)
        row["gate_rho_max"] = float(args.gate_rho_max)
        row["style_alpha"] = float(args.style_alpha)
        row["style_rho"] = float(args.style_rho)
        row["sadg_rho"] = float(args.sadg_rho)
        row["lambda_disp"] = float(args.lambda_disp)
        row["dice_weight"] = float(args.dice_weight)
        row["lambda_gate_anchor"] = float(args.lambda_gate_anchor)
        row["lambda_gate_smooth"] = float(args.lambda_gate_smooth)
        row["lambda_gate_mono"] = float(args.lambda_gate_mono)
        row["lambda_spectral_consistency"] = float(args.lambda_spectral_consistency)
        row["lambda_gate_budget"] = float(args.lambda_gate_budget)
        row["spectral_cons_eps"] = float(args.spectral_cons_eps)
        row["rho"] = float(args.style_rho)
        row["rho_tag"] = style_rho_tag(float(args.style_rho))
        row["sadg_rho_tag"] = sadg_rho_tag(float(args.sadg_rho))
        row.setdefault("gate_reg_loss", 0.0)
        row.setdefault("gate_anchor_loss", 0.0)
        row.setdefault("gate_smooth_loss", 0.0)
        row.setdefault("gate_mono_loss", 0.0)
        row.setdefault("spectral_cons_loss", 0.0)
        row.setdefault("gate_budget_loss", 0.0)
        row.setdefault("spectral_style_loss", 0.0)
        row.setdefault("rho_bin_min", float("nan"))
        row.setdefault("rho_bin_max", float("nan"))
        row.setdefault("rho_bin_mean", float("nan"))
        row.setdefault("rho_bins", "")
        row["lf_radius_h"] = int(window["radius_h"])
        row["lf_radius_w"] = int(window["radius_w"])
        row["lf_size_h"] = int(window["size_h"])
        row["lf_size_w"] = int(window["size_w"])
        row["shot"] = int(shot)
        row["seed"] = int(seed)
        row["checkpoint_source"] = checkpoint_source
        row["output_dir"] = str(output_dir)
        row["prompt_dir"] = str(prompt_run_dir(args, ablation, style_method, int(shot), int(seed)))
        out.append(row)
    return out


def compute_training_losses(
    model: torch.nn.Module,
    img: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    out = model(img, return_features=True)
    return backbone_training_loss(
        out["logits"],
        mask,
        out["features"]["dec1"],
        num_classes=3,
        dice_weight=float(args.dice_weight),
        lambda_disp=float(args.lambda_disp),
        disp_margin=float(args.disp_margin),
    )


def select_epsilon_loss(losses: Dict[str, torch.Tensor], ablation: str) -> torch.Tensor:
    ablation = normalize_ablation(ablation)
    if ablation == SEG_SADG_ABLATION:
        return losses["seg_loss"]
    raise ValueError(f"{ablation} does not use SADG epsilon loss.")


def train_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    img: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
    ablation: str,
    sadg_params: Sequence[torch.nn.Parameter],
    device: torch.device,
) -> Dict[str, float]:
    ablation = normalize_ablation(ablation)

    optimizer.zero_grad(set_to_none=True)
    epsilon_losses = compute_training_losses(model, img, mask, args)
    epsilon_loss = select_epsilon_loss(epsilon_losses, ablation)
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
    style_losses: Dict[str, torch.Tensor] | None = None
    spectral_losses: Dict[str, torch.Tensor] | None = None
    optimizer.zero_grad(set_to_none=True)
    try:
        update_losses = compute_training_losses(model, img, mask, args)
        style_losses = style_regularization_losses(model, args, device)
        spectral_losses = spectral_budget_consistency_losses(model, img, args, device)
        style_total = style_losses["style_reg_loss"] + spectral_losses["spectral_style_loss"]
        total_loss = update_losses["loss"] + style_total
        total_loss.backward()
    finally:
        restore_sadg_perturbation(perturbations)

    optimizer.step()
    if update_losses is None or style_losses is None or spectral_losses is None:
        raise RuntimeError("SADG update losses were not computed.")
    style_total = style_losses["style_reg_loss"] + spectral_losses["spectral_style_loss"]
    layer_meta = style_layer_scalar_metadata(model)
    return {
        "train_loss": float((update_losses["loss"] + style_total).detach().cpu()),
        "train_seg_loss": float(update_losses["seg_loss"].detach().cpu()),
        "train_disp_loss": float(update_losses["disp_loss"].detach().cpu()),
        "style_reg_loss": float(style_total.detach().cpu()),
        "gate_reg_loss": float(style_losses["gate_reg_loss"].detach().cpu()),
        "gate_anchor_loss": float(style_losses["gate_anchor_loss"].detach().cpu()),
        "gate_smooth_loss": float(style_losses["gate_smooth_loss"].detach().cpu()),
        "gate_mono_loss": float(style_losses["gate_mono_loss"].detach().cpu()),
        "spectral_cons_loss": float(spectral_losses["spectral_cons_loss"].detach().cpu()),
        "gate_budget_loss": float(spectral_losses["gate_budget_loss"].detach().cpu()),
        "spectral_style_loss": float(spectral_losses["spectral_style_loss"].detach().cpu()),
        "template_l2_loss": float(style_losses["template_l2_loss"].detach().cpu()),
        "template_smooth_loss": float(style_losses["template_smooth_loss"].detach().cpu()),
        "epsilon_loss": float(epsilon_loss.detach().cpu()),
        "sadg_grad_norm": float(grad_norm.detach().cpu()),
        "perturb_norm": float(perturb_norm),
        **layer_meta,
    }


def train_or_load_ablation(
    *,
    args: argparse.Namespace,
    device: torch.device,
    ablation: str,
    style_method: str,
    shot: int,
    seed: int,
) -> tuple[torch.nn.Module, List[Dict[str, Any]], Dict[str, Any], Path]:
    set_seed(int(seed))
    ablation = normalize_ablation(ablation)
    style_method = normalize_style_method(style_method)
    template_mode, mask_mode = style_method_parts(style_method)
    run_dir = ablation_run_dir(args, ablation, style_method, shot, seed)
    prompt_dir = prompt_run_dir(args, ablation, style_method, shot, seed)
    train_dataset = build_train_data(args, shot, seed)
    slice_indices_by_case = train_dataset.slice_indices_by_case()
    window = lowfreq_window(int(args.resize_hw), int(args.resize_hw), float(args.style_alpha))
    init_template = compute_source_log_template(train_dataset, args=args, device=device)
    dataset_meta = {
        "loss_mode": ablation,
        "ablation": ablation,
        "sadg_method": ablation,
        "style_method": style_method,
        "template_mode": template_mode,
        "mask_mode": mask_mode,
        "gate_mode": style_gate_mode(style_method),
        "gate_num_bins": int(args.gate_num_bins),
        "gate_rho_max": float(args.gate_rho_max),
        "style_alpha": float(args.style_alpha),
        "style_rho": float(args.style_rho),
        "rho": float(args.style_rho),
        "rho_tag": style_rho_tag(float(args.style_rho)),
        "sadg_rho": float(args.sadg_rho),
        "sadg_rho_tag": sadg_rho_tag(float(args.sadg_rho)),
        "style_eps": float(args.style_eps),
        "fft_norm": str(args.fft_norm),
        "lf_radius_h": int(window["radius_h"]),
        "lf_radius_w": int(window["radius_w"]),
        "lf_size_h": int(window["size_h"]),
        "lf_size_w": int(window["size_w"]),
        "lambda_disp": float(args.lambda_disp),
        "dice_weight": float(args.dice_weight),
        "lambda_gate_anchor": float(args.lambda_gate_anchor),
        "lambda_gate_smooth": float(args.lambda_gate_smooth),
        "lambda_gate_mono": float(args.lambda_gate_mono),
        "lambda_spectral_consistency": float(args.lambda_spectral_consistency),
        "lambda_gate_budget": float(args.lambda_gate_budget),
        "spectral_cons_eps": float(args.spectral_cons_eps),
        "checkpoint_source": "new_training",
        "source_domain": SOURCE_DOMAIN,
        "shot": int(shot),
        "seed": int(seed),
        "train_case_ids": train_dataset.selected_case_ids,
        "n_train_slices": int(len(train_dataset)),
        "slice_policy": train_dataset.slice_policy,
        "num_middle_slices": int(train_dataset.num_middle_slices),
        "filter_min_fg": bool(train_dataset.filter_min_fg),
        "min_fg_ratio": float(args.min_fg_ratio),
        "resize_hw": int(args.resize_hw),
        "slice_indices_by_case": slice_indices_by_case,
        "prompt_dir": str(prompt_dir),
    }
    run_config = {
        **vars(args),
        **dataset_meta,
        "output_dir": str(run_dir),
    }
    write_json(run_dir / "dataset_metadata.json", dataset_meta)
    write_json(run_dir / "run_config.json", run_config)

    checkpoint = run_dir / "baseline_model_with_metadata.pt"
    dataset_meta["checkpoint_path"] = str(checkpoint)
    model = build_experiment_model(
        args=args,
        device=device,
        style_method=style_method,
        init_template=init_template,
    )
    if checkpoint.exists() and bool(args.resume) and not bool(args.overwrite):
        load_experiment_checkpoint(model, checkpoint, device)
        model.eval()
        save_template_files(
            prompt_dir=prompt_dir,
            model=model,
            args=args,
            ablation=ablation,
            style_method=style_method,
            shot=int(shot),
            seed=int(seed),
            train_case_ids=train_dataset.selected_case_ids,
        )
        rows = normalize_training_rows(
            read_csv(run_dir / "training_log.csv"),
            ablation=ablation,
            style_method=style_method,
            shot=int(shot),
            seed=int(seed),
            output_dir=run_dir,
            checkpoint_source="new_training",
            args=args,
        )
        print(f"[RESUME] {ablation} {style_method} shot={shot} seed={seed}: {checkpoint}")
        return model, rows, dataset_meta, run_dir

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    sadg_params = [p for p in unwrap_backbone(model).parameters() if p.requires_grad]
    loader = make_loader(train_dataset, batch_size=int(args.batch_size), shuffle=True, device=device)
    log_rows: List[Dict[str, Any]] = []

    print(
        f"[TRAIN] {ablation} {style_method} shot={shot} seed={seed} "
        f"style_rho={float(args.style_rho):.6g} sadg_rho={float(args.sadg_rho):.6g} "
        f"lambda_disp={float(args.lambda_disp):.6g} cases={train_dataset.selected_case_ids} "
        f"slices={len(train_dataset)}"
    )
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_loss: List[float] = []
        epoch_seg: List[float] = []
        epoch_disp: List[float] = []
        epoch_style_reg: List[float] = []
        epoch_gate_reg: List[float] = []
        epoch_gate_anchor: List[float] = []
        epoch_gate_smooth: List[float] = []
        epoch_gate_mono: List[float] = []
        epoch_spectral_cons: List[float] = []
        epoch_gate_budget: List[float] = []
        epoch_spectral_style: List[float] = []
        epoch_template_l2: List[float] = []
        epoch_template_smooth: List[float] = []
        epoch_epsilon: List[float] = []
        epoch_grad_norm: List[float] = []
        epoch_perturb_norm: List[float] = []
        steps = 0
        for step, (img, mask, _meta) in enumerate(loader, start=1):
            if int(args.max_train_steps) > 0 and step > int(args.max_train_steps):
                break
            img = img.to(device)
            mask = mask.to(device)
            step_row = train_step(
                model=model,
                optimizer=optimizer,
                img=img,
                mask=mask,
                args=args,
                ablation=ablation,
                sadg_params=sadg_params,
                device=device,
            )

            steps += 1
            epoch_loss.append(float(step_row["train_loss"]))
            epoch_seg.append(float(step_row["train_seg_loss"]))
            epoch_disp.append(float(step_row["train_disp_loss"]))
            epoch_style_reg.append(float(step_row["style_reg_loss"]))
            epoch_gate_reg.append(float(step_row["gate_reg_loss"]))
            epoch_gate_anchor.append(float(step_row["gate_anchor_loss"]))
            epoch_gate_smooth.append(float(step_row["gate_smooth_loss"]))
            epoch_gate_mono.append(float(step_row["gate_mono_loss"]))
            epoch_spectral_cons.append(float(step_row["spectral_cons_loss"]))
            epoch_gate_budget.append(float(step_row["gate_budget_loss"]))
            epoch_spectral_style.append(float(step_row["spectral_style_loss"]))
            epoch_template_l2.append(float(step_row["template_l2_loss"]))
            epoch_template_smooth.append(float(step_row["template_smooth_loss"]))
            epoch_epsilon.append(float(step_row["epsilon_loss"]))
            epoch_grad_norm.append(float(step_row["sadg_grad_norm"]))
            epoch_perturb_norm.append(float(step_row["perturb_norm"]))

        layer_meta = style_layer_scalar_metadata(model)
        row = {
            "loss_mode": ablation,
            "ablation": ablation,
            "sadg_method": ablation,
            "style_method": style_method,
            "template_mode": template_mode,
            "mask_mode": mask_mode,
            **layer_meta,
            "style_alpha": float(args.style_alpha),
            "style_rho": float(args.style_rho),
            "sadg_rho": float(args.sadg_rho),
            "lambda_disp": float(args.lambda_disp),
            "dice_weight": float(args.dice_weight),
            "lambda_gate_anchor": float(args.lambda_gate_anchor),
            "lambda_gate_smooth": float(args.lambda_gate_smooth),
            "lambda_gate_mono": float(args.lambda_gate_mono),
            "lambda_spectral_consistency": float(args.lambda_spectral_consistency),
            "lambda_gate_budget": float(args.lambda_gate_budget),
            "spectral_cons_eps": float(args.spectral_cons_eps),
            "rho": float(args.style_rho),
            "rho_tag": style_rho_tag(float(args.style_rho)),
            "sadg_rho_tag": sadg_rho_tag(float(args.sadg_rho)),
            "lf_radius_h": int(window["radius_h"]),
            "lf_radius_w": int(window["radius_w"]),
            "lf_size_h": int(window["size_h"]),
            "lf_size_w": int(window["size_w"]),
            "checkpoint_source": "new_training",
            "shot": int(shot),
            "seed": int(seed),
            "epoch": int(epoch),
            "train_loss": finite_mean(epoch_loss),
            "train_seg_loss": finite_mean(epoch_seg),
            "train_disp_loss": finite_mean(epoch_disp),
            "style_reg_loss": finite_mean(epoch_style_reg),
            "gate_reg_loss": finite_mean(epoch_gate_reg),
            "gate_anchor_loss": finite_mean(epoch_gate_anchor),
            "gate_smooth_loss": finite_mean(epoch_gate_smooth),
            "gate_mono_loss": finite_mean(epoch_gate_mono),
            "spectral_cons_loss": finite_mean(epoch_spectral_cons),
            "gate_budget_loss": finite_mean(epoch_gate_budget),
            "spectral_style_loss": finite_mean(epoch_spectral_style),
            "template_l2_loss": finite_mean(epoch_template_l2),
            "template_smooth_loss": finite_mean(epoch_template_smooth),
            "epsilon_loss": finite_mean(epoch_epsilon),
            "sadg_grad_norm": finite_mean(epoch_grad_norm),
            "perturb_norm": finite_mean(epoch_perturb_norm),
            "steps": int(steps),
            "train_case_ids": "|".join(train_dataset.selected_case_ids),
            "n_train_slices": int(len(train_dataset)),
            "slice_policy": str(args.slice_policy),
            "num_middle_slices": int(args.num_middle_slices),
            "filter_min_fg": bool(args.filter_min_fg),
            "output_dir": str(run_dir),
            "prompt_dir": str(prompt_dir),
        }
        log_rows.append(row)
        write_csv(run_dir / "training_log.csv", log_rows)
        print(
            f"  epoch {epoch:03d}/{int(args.epochs):03d} "
            f"loss={float(row['train_loss']):.6f} "
            f"seg={float(row['train_seg_loss']):.6f} "
            f"disp={float(row['train_disp_loss']):.6f} "
            f"style={float(row['style_reg_loss']):.6f} "
            f"gate={float(row['gate_reg_loss']):.6f} "
            f"spec={float(row['spectral_cons_loss']):.6f} "
            f"rho={float(row['rho_bin_mean']):.4f} "
            f"grad_norm={float(row['sadg_grad_norm']):.6f}"
        )

    payload = checkpoint_payload(
        model,
        args=args,
        ablation=ablation,
        style_method=style_method,
        shot=int(shot),
        seed=int(seed),
        train_case_ids=train_dataset.selected_case_ids,
        n_train_slices=len(train_dataset),
        slice_indices_by_case=slice_indices_by_case,
        prompt_dir=prompt_dir,
    )
    save_model_files(run_dir, payload)
    save_template_files(
        prompt_dir=prompt_dir,
        model=model,
        args=args,
        ablation=ablation,
        style_method=style_method,
        shot=int(shot),
        seed=int(seed),
        train_case_ids=train_dataset.selected_case_ids,
    )
    return model, log_rows, dataset_meta, run_dir


def eval_rows_to_class_rows(eval_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in eval_rows:
        for cls, class_name in FOREGROUND_CLASS_NAMES.items():
            rows.append(
                {
                    "loss_mode": row.get("loss_mode", ""),
                    "ablation": row.get("ablation", row.get("loss_mode", "")),
                    "sadg_method": row.get("sadg_method", row.get("ablation", "")),
                    "style_method": row.get("style_method", ""),
                    "template_mode": row.get("template_mode", ""),
                    "mask_mode": row.get("mask_mode", ""),
                    "gate_mode": row.get("gate_mode", ""),
                    "gate_num_bins": row.get("gate_num_bins", 0),
                    "gate_rho_max": row.get("gate_rho_max", float("nan")),
                    "rho_bin_min": row.get("rho_bin_min", float("nan")),
                    "rho_bin_max": row.get("rho_bin_max", float("nan")),
                    "rho_bin_mean": row.get("rho_bin_mean", float("nan")),
                    "rho_bins": row.get("rho_bins", ""),
                    "style_alpha": row.get("style_alpha", float("nan")),
                    "style_rho": row.get("style_rho", row.get("rho", float("nan"))),
                    "sadg_rho": row.get("sadg_rho", float("nan")),
                    "lambda_disp": row.get("lambda_disp", float("nan")),
                    "dice_weight": row.get("dice_weight", float("nan")),
                    "lambda_gate_anchor": row.get("lambda_gate_anchor", float("nan")),
                    "lambda_gate_smooth": row.get("lambda_gate_smooth", float("nan")),
                    "lambda_gate_mono": row.get("lambda_gate_mono", float("nan")),
                    "lambda_spectral_consistency": row.get("lambda_spectral_consistency", float("nan")),
                    "lambda_gate_budget": row.get("lambda_gate_budget", float("nan")),
                    "spectral_cons_eps": row.get("spectral_cons_eps", float("nan")),
                    "rho": row.get("rho", float("nan")),
                    "rho_tag": row.get("rho_tag", ""),
                    "sadg_rho_tag": row.get("sadg_rho_tag", ""),
                    "checkpoint_source": row.get("checkpoint_source", ""),
                    "shot": int(row.get("shot", 0)),
                    "seed": int(row.get("seed", 0)),
                    "domain": row.get("domain", ""),
                    "class_id": int(cls),
                    "class_name": class_name,
                    "n_cases": int(row.get("n_cases", 0)),
                    "n_slices": int(row.get("n_slices", 0)),
                    "case_dice": row.get(f"case_dice_{class_name}", float("nan")),
                    "case_hd95": row.get(f"case_hd95_{class_name}", float("nan")),
                    "slice_dice": row.get(f"slice_dice_{class_name}", float("nan")),
                    "slice_hd95": row.get(f"slice_hd95_{class_name}", float("nan")),
                    "output_dir": row.get("output_dir", ""),
                    "prompt_dir": row.get("prompt_dir", ""),
                }
            )
    return rows


def class_metric_names() -> List[str]:
    names: List[str] = []
    for prefix in ("case_dice", "case_hd95", "slice_dice", "slice_hd95"):
        for name in FOREGROUND_CLASS_NAMES.values():
            names.append(f"{prefix}_{name}")
    return names


def summarize_eval_groups(eval_rows: Sequence[Dict[str, Any]], group_fields: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in eval_rows:
        groups.setdefault(tuple(row.get(field, "") for field in group_fields), []).append(row)

    out: List[Dict[str, Any]] = []
    for key in sorted(groups, key=lambda vals: tuple(str(v) for v in vals)):
        rows = groups[key]
        item = {field: value for field, value in zip(group_fields, key)}
        item.update(
            {
                "lambda_disp": finite_mean(row.get("lambda_disp") for row in rows),
                "dice_weight": finite_mean(row.get("dice_weight") for row in rows),
                "style_alpha": finite_mean(row.get("style_alpha") for row in rows),
                "style_rho": finite_mean(row.get("style_rho", row.get("rho")) for row in rows),
                "sadg_rho": finite_mean(row.get("sadg_rho") for row in rows),
                "rho": finite_mean(row.get("rho") for row in rows),
                "gate_num_bins": finite_mean(row.get("gate_num_bins") for row in rows),
                "gate_rho_max": finite_mean(row.get("gate_rho_max") for row in rows),
                "rho_bin_min": finite_mean(row.get("rho_bin_min") for row in rows),
                "rho_bin_max": finite_mean(row.get("rho_bin_max") for row in rows),
                "rho_bin_mean": finite_mean(row.get("rho_bin_mean") for row in rows),
                "lambda_gate_anchor": finite_mean(row.get("lambda_gate_anchor") for row in rows),
                "lambda_gate_smooth": finite_mean(row.get("lambda_gate_smooth") for row in rows),
                "lambda_gate_mono": finite_mean(row.get("lambda_gate_mono") for row in rows),
                "lambda_spectral_consistency": finite_mean(row.get("lambda_spectral_consistency") for row in rows),
                "lambda_gate_budget": finite_mean(row.get("lambda_gate_budget") for row in rows),
                "n_rows": int(len(rows)),
                "n_cases_total": int(sum(int(float(row.get("n_cases", 0) or 0)) for row in rows)),
                "n_slices_total": int(sum(int(float(row.get("n_slices", 0) or 0)) for row in rows)),
                "case_dice": finite_mean(row.get("case_dice") for row in rows),
                "case_hd95": finite_mean(row.get("case_hd95") for row in rows),
                "slice_dice": finite_mean(row.get("slice_dice") for row in rows),
                "slice_hd95": finite_mean(row.get("slice_hd95") for row in rows),
            }
        )
        for metric in class_metric_names():
            item[metric] = finite_mean(row.get(metric) for row in rows)
        out.append(item)
    return out


def summarize_class_seed_average(class_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    group_fields = (
        "ablation",
        "style_method",
        "template_mode",
        "mask_mode",
        "gate_mode",
        "gate_num_bins",
        "style_alpha",
        "style_rho",
        "rho_tag",
        "sadg_rho",
        "sadg_rho_tag",
        "shot",
        "domain",
        "class_id",
        "class_name",
    )
    groups: Dict[tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in class_rows:
        groups.setdefault(tuple(row.get(field, "") for field in group_fields), []).append(row)

    out: List[Dict[str, Any]] = []
    for key in sorted(groups, key=lambda vals: tuple(str(v) for v in vals)):
        rows = groups[key]
        item = {field: value for field, value in zip(group_fields, key)}
        seeds = sorted({int(row.get("seed", 0)) for row in rows})
        item.update(
            {
                "lambda_disp": finite_mean(row.get("lambda_disp") for row in rows),
                "dice_weight": finite_mean(row.get("dice_weight") for row in rows),
                "style_alpha": finite_mean(row.get("style_alpha") for row in rows),
                "style_rho": finite_mean(row.get("style_rho", row.get("rho")) for row in rows),
                "sadg_rho": finite_mean(row.get("sadg_rho") for row in rows),
                "rho": finite_mean(row.get("rho") for row in rows),
                "gate_rho_max": finite_mean(row.get("gate_rho_max") for row in rows),
                "rho_bin_min": finite_mean(row.get("rho_bin_min") for row in rows),
                "rho_bin_max": finite_mean(row.get("rho_bin_max") for row in rows),
                "rho_bin_mean": finite_mean(row.get("rho_bin_mean") for row in rows),
                "n_seeds": int(len(seeds)),
                "seeds": "|".join(str(seed) for seed in seeds),
                "n_cases_mean": finite_mean(row.get("n_cases") for row in rows),
                "n_slices_mean": finite_mean(row.get("n_slices") for row in rows),
            }
        )
        for metric in ("case_dice", "case_hd95", "slice_dice", "slice_hd95"):
            item[f"{metric}_mean"] = finite_mean(row.get(metric) for row in rows)
            item[f"{metric}_std"] = finite_std(row.get(metric) for row in rows)
        out.append(item)
    return out


def evaluate_model(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
    ablation: str,
    style_method: str,
    shot: int,
    seed: int,
    train_case_ids: Sequence[str],
    output_dir: Path,
    prompt_dir: Path,
    checkpoint_source: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ablation = normalize_ablation(ablation)
    style_method = normalize_style_method(style_method)
    template_mode, mask_mode = style_method_parts(style_method)
    window = lowfreq_window(int(args.resize_hw), int(args.resize_hw), float(args.style_alpha))
    layer_meta = style_layer_scalar_metadata(model)
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
        row = {
            "loss_mode": ablation,
            "ablation": ablation,
            "sadg_method": ablation,
            "style_method": style_method,
            "template_mode": template_mode,
            "mask_mode": mask_mode,
            **layer_meta,
            "style_alpha": float(args.style_alpha),
            "style_rho": float(args.style_rho),
            "sadg_rho": float(args.sadg_rho),
            "lambda_disp": float(args.lambda_disp),
            "dice_weight": float(args.dice_weight),
            "lambda_gate_anchor": float(args.lambda_gate_anchor),
            "lambda_gate_smooth": float(args.lambda_gate_smooth),
            "lambda_gate_mono": float(args.lambda_gate_mono),
            "lambda_spectral_consistency": float(args.lambda_spectral_consistency),
            "lambda_gate_budget": float(args.lambda_gate_budget),
            "spectral_cons_eps": float(args.spectral_cons_eps),
            "rho": float(args.style_rho),
            "rho_tag": style_rho_tag(float(args.style_rho)),
            "sadg_rho_tag": sadg_rho_tag(float(args.sadg_rho)),
            "lf_radius_h": int(window["radius_h"]),
            "lf_radius_w": int(window["radius_w"]),
            "lf_size_h": int(window["size_h"]),
            "lf_size_w": int(window["size_w"]),
            "checkpoint_source": checkpoint_source,
            "shot": int(shot),
            "seed": int(seed),
            "domain": metrics["domain"],
            "n_cases": int(metrics["n_cases"]),
            "n_slices": int(metrics["n_slices"]),
            "slice_policy": metrics["slice_policy"],
            "num_middle_slices": int(metrics["num_middle_slices"]),
            "excluded_case_ids": "|".join(metrics["excluded_case_ids"]),
            "output_dir": str(output_dir),
            "prompt_dir": str(prompt_dir),
        }
        row.update(metrics["summary"])
        eval_rows.append(row)
        for case_row in metrics["case_rows"]:
            case_rows.append(
                {
                    "loss_mode": ablation,
                    "ablation": ablation,
                    "sadg_method": ablation,
                    "style_method": style_method,
                    "template_mode": template_mode,
                    "mask_mode": mask_mode,
                    **layer_meta,
                    "style_alpha": float(args.style_alpha),
                    "style_rho": float(args.style_rho),
                    "sadg_rho": float(args.sadg_rho),
                    "lambda_disp": float(args.lambda_disp),
                    "dice_weight": float(args.dice_weight),
                    "lambda_gate_anchor": float(args.lambda_gate_anchor),
                    "lambda_gate_smooth": float(args.lambda_gate_smooth),
                    "lambda_gate_mono": float(args.lambda_gate_mono),
                    "lambda_spectral_consistency": float(args.lambda_spectral_consistency),
                    "lambda_gate_budget": float(args.lambda_gate_budget),
                    "spectral_cons_eps": float(args.spectral_cons_eps),
                    "rho": float(args.style_rho),
                    "rho_tag": style_rho_tag(float(args.style_rho)),
                    "sadg_rho_tag": sadg_rho_tag(float(args.sadg_rho)),
                    "checkpoint_source": checkpoint_source,
                    "shot": int(shot),
                    "seed": int(seed),
                    "domain": metrics["domain"],
                    "output_dir": str(output_dir),
                    "prompt_dir": str(prompt_dir),
                    **case_row,
                }
            )
        print(
            f"[EVAL] {ablation} {style_method} shot={shot} seed={seed} domain={metrics['domain']} "
            f"case_dice={float(row['case_dice']):.6f} case_hd95={float(row['case_hd95']):.6f} "
            f"slice_dice={float(row['slice_dice']):.6f} slice_hd95={float(row['slice_hd95']):.6f}"
        )
        for class_name in FOREGROUND_CLASS_NAMES.values():
            print(
                f"  class={class_name} "
                f"case_dice={float(row[f'case_dice_{class_name}']):.6f} "
                f"case_hd95={float(row[f'case_hd95_{class_name}']):.6f} "
                f"slice_dice={float(row[f'slice_dice_{class_name}']):.6f} "
                f"slice_hd95={float(row[f'slice_hd95_{class_name}']):.6f}"
            )
    return eval_rows, case_rows


def summarize_experiment(
    *,
    ablation: str,
    shot: int,
    seed: int,
    dataset_meta: Dict[str, Any],
    train_rows: Sequence[Dict[str, Any]],
    eval_rows: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    ablation = normalize_ablation(ablation)
    last_train = train_rows[-1] if train_rows else {}
    target_rows = [row for row in eval_rows if row.get("domain", "") != SOURCE_DOMAIN]
    return {
        "loss_mode": ablation,
        "ablation": ablation,
        "sadg_method": ablation,
        "style_method": dataset_meta.get("style_method", ""),
        "template_mode": dataset_meta.get("template_mode", ""),
        "mask_mode": dataset_meta.get("mask_mode", ""),
        "gate_mode": dataset_meta.get("gate_mode", ""),
        "gate_num_bins": int(dataset_meta.get("gate_num_bins", 0)),
        "gate_rho_max": float(dataset_meta.get("gate_rho_max", float("nan"))),
        "style_alpha": float(dataset_meta.get("style_alpha", float("nan"))),
        "style_rho": float(dataset_meta.get("style_rho", dataset_meta.get("rho", float("nan")))),
        "sadg_rho": float(dataset_meta.get("sadg_rho", float("nan"))),
        "lambda_disp": float(dataset_meta.get("lambda_disp", float("nan"))),
        "dice_weight": float(dataset_meta.get("dice_weight", float("nan"))),
        "lambda_gate_anchor": float(dataset_meta.get("lambda_gate_anchor", float("nan"))),
        "lambda_gate_smooth": float(dataset_meta.get("lambda_gate_smooth", float("nan"))),
        "lambda_gate_mono": float(dataset_meta.get("lambda_gate_mono", float("nan"))),
        "lambda_spectral_consistency": float(dataset_meta.get("lambda_spectral_consistency", float("nan"))),
        "lambda_gate_budget": float(dataset_meta.get("lambda_gate_budget", float("nan"))),
        "rho": float(dataset_meta.get("rho", float("nan"))),
        "rho_tag": dataset_meta.get("rho_tag", ""),
        "sadg_rho_tag": dataset_meta.get("sadg_rho_tag", ""),
        "checkpoint_source": dataset_meta.get("checkpoint_source", ""),
        "checkpoint_path": dataset_meta.get("checkpoint_path", ""),
        "shot": int(shot),
        "seed": int(seed),
        "train_case_ids": "|".join(str(x) for x in dataset_meta.get("train_case_ids", [])),
        "n_train_slices": int(dataset_meta.get("n_train_slices", 0)),
        "slice_policy": dataset_meta.get("slice_policy", ""),
        "filter_min_fg": bool(dataset_meta.get("filter_min_fg", False)),
        "final_train_loss": last_train.get("train_loss", float("nan")),
        "final_train_seg_loss": last_train.get("train_seg_loss", float("nan")),
        "final_train_disp_loss": last_train.get("train_disp_loss", float("nan")),
        "final_style_reg_loss": last_train.get("style_reg_loss", float("nan")),
        "final_gate_reg_loss": last_train.get("gate_reg_loss", float("nan")),
        "final_gate_anchor_loss": last_train.get("gate_anchor_loss", float("nan")),
        "final_gate_smooth_loss": last_train.get("gate_smooth_loss", float("nan")),
        "final_gate_mono_loss": last_train.get("gate_mono_loss", float("nan")),
        "final_spectral_cons_loss": last_train.get("spectral_cons_loss", float("nan")),
        "final_gate_budget_loss": last_train.get("gate_budget_loss", float("nan")),
        "final_spectral_style_loss": last_train.get("spectral_style_loss", float("nan")),
        "final_template_l2_loss": last_train.get("template_l2_loss", float("nan")),
        "final_template_smooth_loss": last_train.get("template_smooth_loss", float("nan")),
        "rho_bin_min": last_train.get("rho_bin_min", float("nan")),
        "rho_bin_max": last_train.get("rho_bin_max", float("nan")),
        "rho_bin_mean": last_train.get("rho_bin_mean", float("nan")),
        "rho_bins": last_train.get("rho_bins", ""),
        "mean_6domain_case_dice": finite_mean(row.get("case_dice") for row in eval_rows),
        "mean_6domain_case_hd95": finite_mean(row.get("case_hd95") for row in eval_rows),
        "mean_6domain_slice_dice": finite_mean(row.get("slice_dice") for row in eval_rows),
        "mean_6domain_slice_hd95": finite_mean(row.get("slice_hd95") for row in eval_rows),
        "mean_5target_case_dice": finite_mean(row.get("case_dice") for row in target_rows),
        "mean_5target_case_hd95": finite_mean(row.get("case_hd95") for row in target_rows),
        "mean_5target_slice_dice": finite_mean(row.get("slice_dice") for row in target_rows),
        "mean_5target_slice_hd95": finite_mean(row.get("slice_hd95") for row in target_rows),
        "output_dir": str(output_dir),
        "prompt_dir": dataset_meta.get("prompt_dir", ""),
    }


def write_all_summaries(
    result_root: Path,
    *,
    training_rows: Sequence[Dict[str, Any]],
    eval_rows: Sequence[Dict[str, Any]],
    case_rows: Sequence[Dict[str, Any]],
    experiment_rows: Sequence[Dict[str, Any]],
) -> None:
    class_rows = eval_rows_to_class_rows(eval_rows)
    write_csv(result_root / "training_curves.csv", training_rows)
    write_csv(result_root / "eval_metrics.csv", eval_rows)
    write_csv(result_root / "eval_case_metrics.csv", case_rows)
    write_csv(result_root / "eval_domain_class_metrics.csv", class_rows)
    write_csv(result_root / "seed_avg_domain_class_metrics.csv", summarize_class_seed_average(class_rows))
    base_fields = (
        "ablation",
        "style_method",
        "template_mode",
        "mask_mode",
        "gate_mode",
        "gate_num_bins",
        "style_alpha",
        "style_rho",
        "rho_tag",
        "sadg_rho",
        "sadg_rho_tag",
    )
    write_csv(result_root / "ablation_summary.csv", summarize_eval_groups(eval_rows, base_fields))
    write_csv(result_root / "shot_summary.csv", summarize_eval_groups(eval_rows, (*base_fields, "shot")))
    write_csv(result_root / "domain_summary.csv", summarize_eval_groups(eval_rows, (*base_fields, "domain")))
    write_csv(result_root / "experiment_summary.csv", experiment_rows)


def run(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    result_root = ensure_dir(resolve_path(args.result_root))
    ensure_dir(resolve_path(args.new_backbone_root))
    ensure_dir(resolve_path(args.prompt_root))
    shots = parse_int_list(args.shots)
    seeds = parse_int_list(args.seeds)
    style_methods = parse_style_methods(args.style_methods)
    ablation = SEG_SADG_ABLATION

    all_training_rows: List[Dict[str, Any]] = []
    all_eval_rows: List[Dict[str, Any]] = []
    all_case_rows: List[Dict[str, Any]] = []
    all_experiment_rows: List[Dict[str, Any]] = []

    print(f"[DEVICE] {device}")
    print(f"[RESULT_ROOT] {result_root}")
    print(f"[NEW_BACKBONE_ROOT] {resolve_path(args.new_backbone_root)}")
    print(f"[PROMPT_ROOT] {resolve_path(args.prompt_root)}")
    print(
        f"[MATRIX] ablation={ablation} style_methods={style_methods} "
        f"style_alpha={float(args.style_alpha):.6g} style_rho={float(args.style_rho):.6g} "
        f"sadg_rho={float(args.sadg_rho):.6g} shots={shots} seeds={seeds} BS={args.batch_size}"
    )
    print(
        f"[OBJECTIVE] CE + {float(args.dice_weight)} Dice + {float(args.lambda_disp)} Dispersion "
        f"+ gate regularization anchor={float(args.lambda_gate_anchor):.6g} "
        f"smooth={float(args.lambda_gate_smooth):.6g} mono={float(args.lambda_gate_mono):.6g}"
    )
    print(f"[SLICE] policy={args.slice_policy} filter_min_fg={args.filter_min_fg} min_fg_ratio={args.min_fg_ratio}")

    for style_method in style_methods:
        run_args = argparse.Namespace(
            **{
                **vars(args),
                "style_method": style_method,
                "rho": float(args.style_rho),
                "rho_tag": style_rho_tag(float(args.style_rho)),
                "sadg_rho_tag": sadg_rho_tag(float(args.sadg_rho)),
            }
        )
        for shot in shots:
            for seed in seeds:
                model, train_rows, dataset_meta, output_dir = train_or_load_ablation(
                    args=run_args,
                    device=device,
                    ablation=ablation,
                    style_method=style_method,
                    shot=int(shot),
                    seed=int(seed),
                )
                prompt_dir = prompt_run_dir(run_args, ablation, style_method, int(shot), int(seed))

                eval_rows, case_rows = evaluate_model(
                    args=run_args,
                    model=model,
                    device=device,
                    ablation=ablation,
                    style_method=style_method,
                    shot=int(shot),
                    seed=int(seed),
                    train_case_ids=[str(x) for x in dataset_meta["train_case_ids"]],
                    output_dir=output_dir,
                    prompt_dir=prompt_dir,
                    checkpoint_source=str(dataset_meta.get("checkpoint_source", "")),
                )
                all_training_rows.extend(train_rows)
                all_eval_rows.extend(eval_rows)
                all_case_rows.extend(case_rows)
                all_experiment_rows.append(
                    summarize_experiment(
                        ablation=ablation,
                        shot=int(shot),
                        seed=int(seed),
                        dataset_meta=dataset_meta,
                        train_rows=train_rows,
                        eval_rows=eval_rows,
                        output_dir=output_dir,
                    )
                )

                write_all_summaries(
                    result_root,
                    training_rows=all_training_rows,
                    eval_rows=all_eval_rows,
                    case_rows=all_case_rows,
                    experiment_rows=all_experiment_rows,
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Seg-SADG with the source spectral barycenter RadialGate style regularizer."
    )
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
    parser.add_argument("--style_methods", default=",".join(DEFAULT_STYLE_METHODS))
    parser.add_argument("--style_alpha", type=float, default=0.08)
    parser.add_argument("--style_rho", type=float, default=0.5)
    parser.add_argument("--style_eps", type=float, default=1e-6)
    parser.add_argument("--fft_norm", default="ortho", choices=("backward", "ortho", "forward"))
    parser.add_argument("--sadg_rho", type=float, default=FIXED_SADG_RHO)
    parser.add_argument("--sadg_eps", type=float, default=1e-12)
    parser.add_argument("--gate_num_bins", type=int, default=8)
    parser.add_argument("--gate_rho_max", type=float, default=1.0)
    parser.add_argument("--lambda_gate_anchor", type=float, default=1e-4)
    parser.add_argument("--lambda_gate_smooth", type=float, default=1e-3)
    parser.add_argument("--lambda_gate_mono", type=float, default=1e-3)
    parser.add_argument("--lambda_spectral_consistency", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--lambda_gate_budget", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--spectral_cons_eps", type=float, default=1e-6, help=argparse.SUPPRESS)
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
    parser.add_argument("--new_backbone_root", default=str(DEFAULT_NEW_BACKBONE_ROOT))
    parser.add_argument("--prompt_root", default=str(DEFAULT_PROMPT_ROOT))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
