from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk
import torch

from helper.dataloaders import (
    DATA_ROOT,
    TARGET_DOMAIN_NAMES,
    normalize_slice_to_01,
    resize_np_2d,
    resolve_domain,
)


TARGET_STYLE_DOMAINS = tuple(TARGET_DOMAIN_NAMES)


def _as_resize_hw(resize_hw: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(resize_hw, int):
        return resize_hw, resize_hw
    if len(resize_hw) != 2:
        raise ValueError(f"resize_hw must be int or length-2 tuple, got {resize_hw!r}")
    return int(resize_hw[0]), int(resize_hw[1])


def fda_radius(height: int, width: int, beta: float) -> int:
    if beta <= 0:
        return 0
    return max(0, int(np.floor(min(height, width) * float(beta))))


def _center_crop_bounds(height: int, width: int, beta: float) -> tuple[int, int, int, int, int]:
    radius = fda_radius(height, width, beta)
    center_h = height // 2
    center_w = width // 2
    h1 = max(0, center_h - radius)
    h2 = min(height, center_h + radius + 1)
    w1 = max(0, center_w - radius)
    w2 = min(width, center_w + radius + 1)
    return h1, h2, w1, w2, radius


def _low_frequency_amplitude(
    x: torch.Tensor,
    beta: float,
    fft_norm: str | None = "backward",
) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"FDA expects a 4D tensor [B,C,H,W], got shape {tuple(x.shape)}")
    _, _, height, width = x.shape
    h1, h2, w1, w2, _ = _center_crop_bounds(height, width, beta)
    fft_x = torch.fft.fft2(x, dim=(-2, -1), norm=fft_norm)
    amp_shift = torch.fft.fftshift(torch.abs(fft_x), dim=(-2, -1))
    return amp_shift[..., h1:h2, w1:w2]


def fourier_domain_adapt(
    src: torch.Tensor,
    target_lowfreq_amp: torch.Tensor,
    beta: float = 0.01,
    eps: float = 1e-6,
    fft_norm: str | None = "backward",
    clamp: bool = True,
) -> torch.Tensor:
    """Apply FDA by replacing source low-frequency amplitude with target amplitude."""
    if beta <= 0:
        return src.clone()
    if src.ndim != 4:
        raise ValueError(f"FDA expects a 4D tensor [B,C,H,W], got shape {tuple(src.shape)}")

    batch_size, channels, height, width = src.shape
    h1, h2, w1, w2, _ = _center_crop_bounds(height, width, beta)
    expected_hw = (h2 - h1, w2 - w1)

    style = target_lowfreq_amp.to(device=src.device, dtype=src.dtype)
    if style.ndim == 3:
        style = style.unsqueeze(0)
    if style.ndim != 4:
        raise ValueError(f"target_lowfreq_amp must have shape [B,C,h,w] or [C,h,w], got {tuple(style.shape)}")
    if style.shape[-2:] != expected_hw:
        raise ValueError(
            f"target_lowfreq_amp window {tuple(style.shape[-2:])} does not match FDA window {expected_hw}"
        )
    if style.shape[1] != channels:
        raise ValueError(f"target style channels {style.shape[1]} do not match source channels {channels}")
    if style.shape[0] == 1 and batch_size != 1:
        style = style.expand(batch_size, -1, -1, -1)
    elif style.shape[0] != batch_size:
        raise ValueError(f"target style batch {style.shape[0]} does not match source batch {batch_size}")

    fft_src = torch.fft.fft2(src, dim=(-2, -1), norm=fft_norm)
    amp_src = torch.abs(fft_src)
    phase_src = fft_src / (amp_src + eps)

    amp_shift = torch.fft.fftshift(amp_src, dim=(-2, -1)).clone()
    amp_shift[..., h1:h2, w1:w2] = style
    amp_new = torch.fft.ifftshift(amp_shift, dim=(-2, -1))

    out = torch.fft.ifft2(amp_new * phase_src, dim=(-2, -1), norm=fft_norm).real
    if clamp:
        out = out.clamp_(0.0, 1.0)
    return out


@dataclass
class FDAStyleBank:
    global_lowfreq_amp: torch.Tensor
    domain_lowfreq_amp: torch.Tensor
    domain_names: tuple[str, ...]
    domain_slice_counts: dict[str, int]
    beta: float
    resize_hw: tuple[int, int]
    fft_norm: str | None = "backward"

    @property
    def radius(self) -> int:
        return fda_radius(self.resize_hw[0], self.resize_hw[1], self.beta)

    @property
    def window_size(self) -> int:
        return self.radius * 2 + 1

    @property
    def style_slices_total(self) -> int:
        return int(sum(self.domain_slice_counts.values()))

    def to(self, device: torch.device | str) -> "FDAStyleBank":
        self.global_lowfreq_amp = self.global_lowfreq_amp.to(device)
        self.domain_lowfreq_amp = self.domain_lowfreq_amp.to(device)
        return self

    def styles_for_batch(
        self,
        variant: str,
        batch_size: int,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        style_mode = normalize_fda_style_mode(variant)
        if style_mode == "global_mean":
            return self.global_lowfreq_amp.to(device).expand(batch_size, -1, -1, -1)
        if style_mode == "domain_style_pool":
            idx = torch.randint(
                low=0,
                high=self.domain_lowfreq_amp.shape[0],
                size=(batch_size,),
                generator=generator,
                device=torch.device("cpu"),
            )
            return self.domain_lowfreq_amp[idx].to(device)
        raise ValueError(f"Unknown FDA style mode: {style_mode}")

    def summary(self) -> dict[str, object]:
        return {
            "fda_beta": float(self.beta),
            "fda_radius": int(self.radius),
            "fda_window_size": int(self.window_size),
            "fda_fft_norm": self.fft_norm,
            "fda_style_domains": list(self.domain_names),
            "fda_style_slices_total": int(self.style_slices_total),
            "fda_style_slices_by_domain": dict(self.domain_slice_counts),
            "resize_hw": list(self.resize_hw),
        }


def normalize_fda_style_mode(variant: str) -> str:
    token = variant.strip().lower().replace("_", "-")
    if token in {"fda-globalmean", "globalmean", "global-mean", "global"}:
        return "global_mean"
    if token in {"fda-domainstylepool", "domainstylepool", "domain-style-pool", "pool"}:
        return "domain_style_pool"
    raise ValueError(f"Unknown FDA variant/style mode: {variant}")


def _iter_domain_image_paths(domain_name: str, data_root: Path) -> list[Path]:
    domain_path = resolve_domain(domain_name, data_root=data_root)
    image_dir = domain_path / "images"
    paths = sorted(image_dir.glob("*.mha"))
    if not paths:
        raise FileNotFoundError(f"No .mha target images found under {image_dir}")
    return paths


def _accumulate_lowfreq_sum(
    image_paths: Iterable[Path],
    resize_hw: tuple[int, int],
    beta: float,
    fft_norm: str | None,
    clip_min: float,
    clip_max: float,
    style_batch_size: int,
) -> tuple[torch.Tensor, int]:
    pending: list[torch.Tensor] = []
    amp_sum: torch.Tensor | None = None
    slice_count = 0

    def flush() -> None:
        nonlocal amp_sum, slice_count, pending
        if not pending:
            return
        batch = torch.stack(pending, dim=0)
        lowfreq = _low_frequency_amplitude(batch, beta=beta, fft_norm=fft_norm)
        batch_sum = lowfreq.sum(dim=0)
        amp_sum = batch_sum if amp_sum is None else amp_sum + batch_sum
        slice_count += int(batch.shape[0])
        pending = []

    for image_path in image_paths:
        image_zyx = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path))).astype(np.float32)
        for x_idx in range(image_zyx.shape[2]):
            image_2d = normalize_slice_to_01(image_zyx[:, :, x_idx], clip_min=clip_min, clip_max=clip_max)
            image_2d = resize_np_2d(image_2d, resize_hw, is_mask=False)
            pending.append(torch.from_numpy(image_2d.astype(np.float32)).unsqueeze(0))
            if len(pending) >= style_batch_size:
                flush()
    flush()

    if amp_sum is None or slice_count <= 0:
        raise RuntimeError("No slices were available to build the FDA style bank.")
    return amp_sum, slice_count


def build_target_style_bank(
    domains: Iterable[str] = TARGET_STYLE_DOMAINS,
    data_root: str | Path = DATA_ROOT,
    resize_hw: int | tuple[int, int] = 224,
    beta: float = 0.01,
    fft_norm: str | None = "backward",
    clip_min: float = -3.0,
    clip_max: float = 3.0,
    style_batch_size: int = 32,
) -> FDAStyleBank:
    if beta < 0:
        raise ValueError(f"fda beta must be non-negative, got {beta}")
    resize_hw_tuple = _as_resize_hw(resize_hw)
    domain_names = tuple(str(name) for name in domains)
    if not domain_names:
        raise ValueError("At least one target style domain is required.")

    data_root = Path(data_root)
    domain_means: list[torch.Tensor] = []
    domain_counts: dict[str, int] = {}
    global_sum: torch.Tensor | None = None
    global_count = 0

    for domain_name in domain_names:
        image_paths = _iter_domain_image_paths(domain_name, data_root=data_root)
        amp_sum, slice_count = _accumulate_lowfreq_sum(
            image_paths=image_paths,
            resize_hw=resize_hw_tuple,
            beta=beta,
            fft_norm=fft_norm,
            clip_min=clip_min,
            clip_max=clip_max,
            style_batch_size=style_batch_size,
        )
        domain_means.append(amp_sum / float(slice_count))
        domain_counts[domain_name] = int(slice_count)
        global_sum = amp_sum if global_sum is None else global_sum + amp_sum
        global_count += int(slice_count)

    if global_sum is None or global_count <= 0:
        raise RuntimeError("No target slices were available to build the FDA style bank.")

    domain_lowfreq_amp = torch.stack(domain_means, dim=0).contiguous()
    global_lowfreq_amp = (global_sum / float(global_count)).unsqueeze(0).contiguous()
    return FDAStyleBank(
        global_lowfreq_amp=global_lowfreq_amp.float(),
        domain_lowfreq_amp=domain_lowfreq_amp.float(),
        domain_names=domain_names,
        domain_slice_counts=domain_counts,
        beta=float(beta),
        resize_hw=resize_hw_tuple,
        fft_norm=fft_norm,
    )
