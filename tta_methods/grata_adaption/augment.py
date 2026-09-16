"""Deterministic method-local weak sampling and GraTA style augmentation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch.nn import functional as F


WEAK_VIEWS = (
    "identity",
    "horizontal_flip",
    "vertical_flip",
    "rotate_90",
    "rotate_180",
    "rotate_270",
)


def apply_weak_view(value: torch.Tensor, view: str) -> torch.Tensor:
    if view == "identity":
        return value
    if view == "horizontal_flip":
        return value.flip(-1)
    if view == "vertical_flip":
        return value.flip(-2)
    if view == "rotate_90":
        return value.flip(-1).transpose(-2, -1)
    if view == "rotate_180":
        return value.flip(-1).flip(-2)
    if view == "rotate_270":
        return value.transpose(-2, -1).flip(-1)
    raise ValueError(f"Unknown GraTA-Adaption weak view: {view!r}")


def apply_per_sample_weak_views(
    value: torch.Tensor, views: Sequence[str]
) -> torch.Tensor:
    """Apply one geometry per batch item without mixing the batch dimension."""
    if value.shape[0] != len(views):
        raise ValueError("The number of weak views must equal the batch size")
    return torch.cat(
        [apply_weak_view(value[index : index + 1], view) for index, view in enumerate(views)],
        dim=0,
    )


def sample_weak_views(
    batch_size: int,
    samples_per_image: int,
    views: Sequence[str],
    generator: torch.Generator,
) -> list[list[str]]:
    """Sample N distinct views independently for every image."""
    if not views:
        raise ValueError("GraTA-Adaption requires at least one weak view")
    if samples_per_image < 1 or samples_per_image > len(views):
        raise ValueError(
            f"weak_view_samples must be in [1, {len(views)}], got {samples_per_image}"
        )
    sampled: list[list[str]] = []
    for _ in range(batch_size):
        indices = torch.randperm(len(views), generator=generator)[:samples_per_image]
        sampled.append([views[int(index)] for index in indices])
    return sampled


def _uniform(generator: torch.Generator, lower: float, upper: float) -> float:
    return float(torch.empty((), device="cpu").uniform_(lower, upper, generator=generator))


def _bernoulli(generator: torch.Generator, probability: float) -> bool:
    return _uniform(generator, 0.0, 1.0) < float(probability)


def _balanced_range_sample(
    generator: torch.Generator, lower: float, upper: float
) -> float:
    if lower < 1.0 and _bernoulli(generator, 0.5):
        return _uniform(generator, lower, 1.0)
    return _uniform(generator, max(lower, 1.0), upper)


def _gaussian_blur(channel: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(3.0 * float(sigma) + 0.5))
    coordinates = torch.arange(
        -radius, radius + 1, device=channel.device, dtype=channel.dtype
    )
    kernel = torch.exp(-0.5 * (coordinates / float(sigma)).square())
    kernel = kernel / kernel.sum()
    value = channel.unsqueeze(0).unsqueeze(0)
    horizontal = kernel.view(1, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1)
    value = F.pad(value, (radius, radius, 0, 0), mode="reflect")
    value = F.conv2d(value, horizontal)
    value = F.pad(value, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(value, vertical).squeeze(0).squeeze(0)


@torch.no_grad()
def strong_style_augmentation(
    images: torch.Tensor,
    cfg: dict[str, Any],
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    augmented = images.detach().clone().float()
    counts = {name: 0 for name in ("brightness", "contrast", "gamma", "noise", "blur")}

    for sample_index in range(augmented.shape[0]):
        sample = augmented[sample_index]
        if _bernoulli(generator, float(cfg["brightness_probability"])):
            for channel_index in range(sample.shape[0]):
                lower, upper = cfg["brightness_range"]
                sample[channel_index].mul_(_uniform(generator, lower, upper))
            counts["brightness"] += 1

        if _bernoulli(generator, float(cfg["contrast_probability"])):
            for channel_index in range(sample.shape[0]):
                channel = sample[channel_index]
                minimum, maximum = channel.min(), channel.max()
                lower, upper = cfg["contrast_range"]
                factor = _balanced_range_sample(generator, lower, upper)
                channel.copy_(
                    ((channel - channel.mean()) * factor + channel.mean()).clamp(minimum, maximum)
                )
            counts["contrast"] += 1

        if _bernoulli(generator, float(cfg["gamma_probability"])):
            inverted = -sample
            for channel_index in range(inverted.shape[0]):
                channel = inverted[channel_index]
                minimum = channel.min()
                value_range = channel.max() - minimum
                lower, upper = cfg["gamma_range"]
                gamma = _balanced_range_sample(generator, lower, upper)
                channel.copy_(
                    ((channel - minimum) / (value_range + 1e-7)).clamp_min(0.0).pow(gamma)
                    * (value_range + 1e-7)
                    + minimum
                )
            sample.copy_(-inverted)
            counts["gamma"] += 1

        if _bernoulli(generator, float(cfg["noise_probability"])):
            lower, upper = cfg["noise_std_range"]
            standard_deviation = _uniform(generator, lower, upper)
            noise = torch.randn(
                sample.shape,
                generator=generator,
                device="cpu",
                dtype=torch.float32,
            ).to(sample.device)
            sample.add_(noise, alpha=standard_deviation)
            counts["noise"] += 1

        if _bernoulli(generator, float(cfg["blur_probability"])):
            blurred_channels = []
            for channel_index in range(sample.shape[0]):
                if _bernoulli(generator, float(cfg["blur_channel_probability"])):
                    lower, upper = cfg["blur_sigma_range"]
                    blurred_channels.append(
                        _gaussian_blur(
                            sample[channel_index], _uniform(generator, lower, upper)
                        )
                    )
                else:
                    blurred_channels.append(sample[channel_index])
            sample.copy_(torch.stack(blurred_channels))
            counts["blur"] += 1

        minimum, maximum = sample.min(), sample.max()
        value_range = maximum - minimum
        if float(value_range) > 0.0:
            sample.sub_(minimum).div_(value_range)
        else:
            sample.zero_()

    diagnostics = {
        f"strong_{name}_coverage": float(count / max(1, augmented.shape[0]))
        for name, count in counts.items()
    }
    return augmented.to(dtype=images.dtype), diagnostics
