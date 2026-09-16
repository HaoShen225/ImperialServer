"""Confidence-separable pixel reliability for GraTA-Adaption."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class CSLSelection:
    labels: torch.Tensor
    hard_mask: torch.Tensor
    smooth_weights: torch.Tensor
    weights: torch.Tensor
    global_fallbacks: int
    classwise_groups: int
    classwise_fallbacks: int


def weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    epsilon: float,
) -> torch.Tensor | None:
    """Weight every pixel and normalize by the realized sum of weights."""
    if labels.shape != logits.shape[:1] + logits.shape[-2:] or weights.shape != labels.shape:
        raise ValueError("Weighted consistency tensors have incompatible shapes")
    denominator = weights.sum()
    if float(denominator.detach().cpu()) <= float(epsilon):
        return None
    pixel_loss = F.cross_entropy(logits, labels, reduction="none")
    return (pixel_loss * weights).sum() / denominator


@torch.no_grad()
def confidence_features(
    probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return hard labels, p_max and non-maximum-class residual variance."""
    if probabilities.ndim != 4 or probabilities.shape[1] < 2:
        raise ValueError("CSL expects probabilities shaped [B, C>=2, H, W]")
    max_confidence, labels = probabilities.max(dim=1)
    residual_mean = (1.0 - max_confidence) / float(probabilities.shape[1] - 1)
    residual_difference = probabilities - residual_mean.unsqueeze(1)
    maximum = F.one_hot(labels, num_classes=probabilities.shape[1]).permute(0, 3, 1, 2)
    residual_difference = residual_difference.masked_fill(maximum.to(dtype=torch.bool), 0.0)
    residual_variance = residual_difference.square().sum(dim=1) / float(
        probabilities.shape[1] - 1
    )
    return labels, max_confidence, residual_variance


@torch.no_grad()
def _pcos_group(
    confidence: torch.Tensor,
    residual_variance: torch.Tensor,
    alpha: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run the official two-way PCOS separation on one flattened pixel group."""
    features = torch.stack((confidence.float(), residual_variance.float()), dim=1)
    if features.shape[0] < 2 or not torch.isfinite(features).all():
        return None
    try:
        _, singular_values, right_vectors = torch.linalg.svd(
            features.transpose(0, 1), full_matrices=False
        )
    except RuntimeError:
        return None
    if right_vectors.shape[0] < 2 or not torch.isfinite(singular_values).all():
        return None
    assignments = right_vectors.transpose(0, 1).abs().argmax(dim=1)
    groups = [features[assignments == group_id] for group_id in range(2)]
    if any(group.shape[0] < 2 for group in groups):
        return None
    reliable_group = max(range(2), key=lambda group_id: float(groups[group_id][:, 0].mean()))
    reliable_features = groups[reliable_group]
    center = reliable_features.mean(dim=0)
    variance = reliable_features.var(dim=0, unbiased=True)
    if not torch.isfinite(center).all() or not torch.isfinite(variance).all():
        return None

    confidence_z = (features[:, 0] - center[0]) / torch.sqrt(variance[0] + epsilon)
    residual_z = (center[1] - features[:, 1]) / torch.sqrt(variance[1] + epsilon)
    hard_mask = (confidence_z > 0.0) | (residual_z > 0.0)
    if not bool(hard_mask.any()):
        return None
    smooth_weights = torch.exp(
        -(confidence_z.square() + residual_z.square()) / float(alpha)
    )
    smooth_weights = torch.where(hard_mask, torch.ones_like(smooth_weights), smooth_weights)
    if not torch.isfinite(smooth_weights).all():
        return None
    return hard_mask, smooth_weights.clamp_(0.0, 1.0)


@torch.no_grad()
def select_csl_weights(
    probabilities: torch.Tensor,
    *,
    scope: str,
    mode: str,
    alpha: float,
    epsilon: float,
    classwise_min_pixels: int,
) -> CSLSelection:
    if scope not in {"global", "classwise"}:
        raise ValueError(f"Unknown CSL scope: {scope!r}")
    if mode not in {"hard", "smooth"}:
        raise ValueError(f"Unknown CSL weight mode: {mode!r}")
    if alpha <= 0.0 or epsilon <= 0.0 or classwise_min_pixels < 2:
        raise ValueError("CSL alpha/epsilon must be positive and min pixels must be at least 2")

    labels, max_confidence, residual_variance = confidence_features(probabilities)
    hard_output = torch.empty_like(max_confidence, dtype=torch.bool)
    smooth_output = torch.empty_like(max_confidence, dtype=torch.float32)
    global_fallbacks = 0
    classwise_groups = 0
    classwise_fallbacks = 0

    for image_index in range(probabilities.shape[0]):
        flat_confidence = max_confidence[image_index].reshape(-1)
        flat_residual = residual_variance[image_index].reshape(-1)
        global_result = _pcos_group(
            flat_confidence, flat_residual, alpha=alpha, epsilon=epsilon
        )
        if global_result is None:
            global_hard = torch.ones_like(flat_confidence, dtype=torch.bool)
            global_smooth = torch.ones_like(flat_confidence, dtype=torch.float32)
            global_fallbacks += 1
        else:
            global_hard, global_smooth = global_result

        if scope == "global":
            flat_hard, flat_smooth = global_hard, global_smooth
        else:
            flat_labels = labels[image_index].reshape(-1)
            flat_hard = global_hard.clone()
            flat_smooth = global_smooth.clone()
            for class_id in range(probabilities.shape[1]):
                class_mask = flat_labels == class_id
                class_pixels = int(class_mask.sum())
                if class_pixels == 0:
                    continue
                classwise_groups += 1
                if class_pixels < classwise_min_pixels:
                    classwise_fallbacks += 1
                    continue
                class_result = _pcos_group(
                    flat_confidence[class_mask],
                    flat_residual[class_mask],
                    alpha=alpha,
                    epsilon=epsilon,
                )
                if class_result is None:
                    classwise_fallbacks += 1
                    continue
                class_hard, class_smooth = class_result
                flat_hard[class_mask] = class_hard
                flat_smooth[class_mask] = class_smooth

        hard_output[image_index] = flat_hard.reshape_as(max_confidence[image_index])
        smooth_output[image_index] = flat_smooth.reshape_as(max_confidence[image_index])

    weights = hard_output.float() if mode == "hard" else smooth_output
    return CSLSelection(
        labels=labels,
        hard_mask=hard_output,
        smooth_weights=smooth_output,
        weights=weights,
        global_fallbacks=global_fallbacks,
        classwise_groups=classwise_groups,
        classwise_fallbacks=classwise_fallbacks,
    )
