"""InTEnt utilities for single-image test-time segmentation adaptation.

This module implements the core method from "Medical Image Segmentation with
InTEnt": estimate BatchNorm statistics from one test image, interpolate
between the stored source and estimated test statistics, and integrate the
resulting predictions using foreground/background-balanced entropy weights.

The original paper considers binary segmentation.  For the four-class MMS
setting used in this repository, entropy is categorical across
background/RV/MYO/LV, while the spatial aggregation gives equal weight to
pixels predicted as background and pixels predicted as any foreground class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


BatchNorm = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
BNLayerStatistics = Tuple[torch.Tensor, torch.Tensor]
BNStatistics = Dict[str, BNLayerStatistics]


def model_logits(output: Any) -> torch.Tensor:
    """Extract logits from a tensor, sequence, or ``{"logits": ...}`` output."""
    if isinstance(output, torch.Tensor):
        logits = output
    elif isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("The model returned an empty sequence.")
        logits = output[-1]
    elif isinstance(output, Mapping):
        if "logits" not in output:
            raise KeyError("A mapping model output must contain a 'logits' key.")
        logits = output["logits"]
    else:
        raise TypeError(f"Unsupported model output type: {type(output).__name__}")

    if not isinstance(logits, torch.Tensor):
        raise TypeError("Extracted logits must be a torch.Tensor.")
    return logits


def batch_norm_layers(model: nn.Module) -> Dict[str, nn.Module]:
    """Return all BatchNorm layers, keyed by their module names."""
    layers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, BatchNorm)
    }
    if not layers:
        raise RuntimeError("InTEnt requires at least one BatchNorm layer.")
    return layers


def capture_bn_stats(model: nn.Module) -> BNStatistics:
    """Clone the running mean and variance of every BatchNorm layer."""
    statistics: BNStatistics = {}
    for name, module in batch_norm_layers(model).items():
        if not module.track_running_stats:
            raise RuntimeError(
                f"BatchNorm layer {name!r} does not track running statistics."
            )
        if module.running_mean is None or module.running_var is None:
            raise RuntimeError(f"BatchNorm layer {name!r} has no running statistics.")
        statistics[name] = (
            module.running_mean.detach().clone(),
            module.running_var.detach().clone(),
        )
    return statistics


def _validate_statistics(
    layers: Mapping[str, nn.Module],
    statistics: Mapping[str, BNLayerStatistics],
) -> None:
    expected = set(layers)
    supplied = set(statistics)
    if supplied != expected:
        missing = sorted(expected - supplied)
        unexpected = sorted(supplied - expected)
        raise ValueError(
            "BatchNorm statistics mismatch: "
            f"missing={missing}, unexpected={unexpected}."
        )

    for name, module in layers.items():
        value = statistics[name]
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise TypeError(
                f"BatchNorm statistics for {name!r} must be a (mean, variance) pair."
            )
        mean, variance = value
        if not isinstance(mean, torch.Tensor) or not isinstance(variance, torch.Tensor):
            raise TypeError(f"BatchNorm statistics for {name!r} must be tensors.")
        if module.running_mean is None or module.running_var is None:
            raise RuntimeError(f"BatchNorm layer {name!r} has no running buffers.")
        if mean.shape != module.running_mean.shape or variance.shape != module.running_var.shape:
            raise ValueError(
                f"BatchNorm statistics shape mismatch for {name!r}: "
                f"mean {tuple(mean.shape)} vs {tuple(module.running_mean.shape)}, "
                f"variance {tuple(variance.shape)} vs {tuple(module.running_var.shape)}."
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
            raise ValueError(
                f"BatchNorm statistics for {name!r} contain non-finite values."
            )
        if torch.any(variance < 0):
            raise ValueError(
                f"BatchNorm variance for {name!r} contains negative values."
            )


@torch.no_grad()
def load_bn_stats(
    model: nn.Module,
    statistics: Mapping[str, BNLayerStatistics],
) -> None:
    """Copy named BatchNorm statistics into ``model`` in place."""
    layers = batch_norm_layers(model)
    _validate_statistics(layers, statistics)
    for name, module in layers.items():
        mean, variance = statistics[name]
        module.running_mean.copy_(mean.to(module.running_mean))
        module.running_var.copy_(variance.to(module.running_var))


def interpolate_bn_stats(
    source: Mapping[str, BNLayerStatistics],
    target: Mapping[str, BNLayerStatistics],
    test_fraction: float,
) -> BNStatistics:
    """Mix BN statistics as ``(1-a)*source + a*target``.

    ``test_fraction`` is the weight ``a`` of the single-image test statistics.
    It therefore equals ``1 - lambda`` under the notation of Eq. (5) in the
    InTEnt paper.
    """
    fraction = float(test_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"test_fraction must be in [0, 1], got {fraction}.")
    if set(source) != set(target):
        raise ValueError("Source and target BatchNorm layer names differ.")

    mixed: BNStatistics = {}
    for name in source:
        source_mean, source_var = source[name]
        target_mean, target_var = target[name]
        if source_mean.shape != target_mean.shape or source_var.shape != target_var.shape:
            raise ValueError(
                f"Source and target BatchNorm shapes differ for {name!r}."
            )
        if (
            not torch.isfinite(source_mean).all()
            or not torch.isfinite(source_var).all()
            or not torch.isfinite(target_mean).all()
            or not torch.isfinite(target_var).all()
        ):
            raise ValueError(
                f"Source or target BatchNorm statistics for {name!r} "
                "contain non-finite values."
            )
        if torch.any(source_var < 0) or torch.any(target_var < 0):
            raise ValueError(
                f"Source or target BatchNorm variance for {name!r} is negative."
            )
        mixed[name] = (
            torch.lerp(source_mean, target_mean.to(source_mean), fraction),
            torch.lerp(source_var, target_var.to(source_var), fraction),
        )
    return mixed


@torch.no_grad()
def estimate_test_bn_stats(model: nn.Module, image: torch.Tensor) -> BNStatistics:
    """Estimate per-layer BN statistics from one image and restore model state."""
    if image.ndim < 3 or image.shape[0] != 1:
        raise ValueError(
            f"InTEnt expects a single-image batch, got {tuple(image.shape)}."
        )

    layers = batch_norm_layers(model)
    saved_statistics = capture_bn_stats(model)
    training_flags = {
        module: bool(module.training)
        for module in model.modules()
    }
    bn_configuration = {
        name: (
            module.momentum,
            bool(module.track_running_stats),
            None
            if module.num_batches_tracked is None
            else module.num_batches_tracked.detach().clone(),
        )
        for name, module in layers.items()
    }

    try:
        # Only BatchNorm layers enter training mode.  This obtains statistics
        # from the current image without enabling Dropout or other stochastic
        # training-time behavior elsewhere in the model.
        model.eval()
        for module in layers.values():
            module.train()
            module.track_running_stats = True
            module.momentum = 1.0
        model_logits(model(image))
        return capture_bn_stats(model)
    finally:
        load_bn_stats(model, saved_statistics)
        for name, module in layers.items():
            momentum, track_running_stats, batches = bn_configuration[name]
            module.momentum = momentum
            module.track_running_stats = track_running_stats
            if batches is not None and module.num_batches_tracked is not None:
                module.num_batches_tracked.copy_(batches.to(module.num_batches_tracked))
        for module, training in training_flags.items():
            module.training = training


def _entropy_epsilon(probabilities: torch.Tensor, eps: float) -> float:
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if not probabilities.is_floating_point():
        raise TypeError("Probabilities must have a floating-point dtype.")
    return max(float(eps), float(torch.finfo(probabilities.dtype).tiny))


def categorical_entropy(
    probabilities: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return categorical entropy at every spatial position.

    Input must have shape ``[B, C, ...]``.  Values are normalized along the
    class dimension so that minor probability-sum drift does not affect the
    entropy calculation.
    """
    if probabilities.ndim < 3:
        raise ValueError("Probabilities must have shape [B, C, ...].")
    if probabilities.shape[1] < 2:
        raise ValueError("Probabilities must contain at least two classes.")
    epsilon = _entropy_epsilon(probabilities, eps)
    if not torch.isfinite(probabilities).all():
        raise ValueError("Probabilities contain non-finite values.")
    if torch.any(probabilities < 0):
        raise ValueError("Probabilities must be non-negative.")

    normalizer = probabilities.sum(dim=1, keepdim=True)
    if torch.any(normalizer <= 0):
        raise ValueError(
            "Probabilities must have a positive class sum at every position."
        )
    normalized = probabilities / normalizer
    return -(normalized * normalized.clamp_min(epsilon).log()).sum(dim=1)


def balanced_fg_bg_entropy(
    probabilities: torch.Tensor,
    background_index: int = 0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Equally weight entropy over predicted background and foreground pixels.

    For MMS, foreground is the union of RV, MYO, and LV.  Entropy itself is
    still categorical across all four classes, retaining uncertainty between
    the foreground anatomy classes.
    """
    if probabilities.ndim < 3:
        raise ValueError("Probabilities must have shape [B, C, ...].")
    if not 0 <= int(background_index) < probabilities.shape[1]:
        raise ValueError(f"Invalid background_index {background_index}.")

    entropy = categorical_entropy(probabilities, eps=eps)
    labels = probabilities.argmax(dim=1)
    values = []
    for sample_entropy, sample_labels in zip(entropy, labels):
        flat_entropy = sample_entropy.reshape(-1)
        flat_labels = sample_labels.reshape(-1)
        full_mean = flat_entropy.mean()
        background = flat_labels == int(background_index)
        foreground = ~background
        background_mean = (
            flat_entropy[background].mean()
            if bool(background.any())
            else full_mean
        )
        foreground_mean = (
            flat_entropy[foreground].mean()
            if bool(foreground.any())
            else full_mean
        )
        values.append(0.5 * (background_mean + foreground_mean))
    return torch.stack(values)


def entropy_integration_weights(
    entropies: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Convert ``[branches, batch]`` entropies into branch weights.

    Lower entropy gives a larger score.  Scores are range-normalized before
    softmax, matching the normalized weighting strategy in InTEnt.  If every
    branch has the same entropy, softmax receives all zeros and returns uniform
    weights.
    """
    if entropies.ndim != 2 or entropies.shape[0] < 1:
        raise ValueError("Entropies must have shape [branches, batch].")
    if not entropies.is_floating_point():
        raise TypeError("Entropies must have a floating-point dtype.")
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if not torch.isfinite(entropies).all():
        raise ValueError("Entropies contain non-finite values.")

    scores = -entropies
    minimum = scores.amin(dim=0, keepdim=True)
    span = scores.amax(dim=0, keepdim=True) - minimum
    normalized = (scores - minimum) / span.clamp_min(float(eps))
    normalized = torch.where(
        span > float(eps),
        normalized,
        torch.zeros_like(normalized),
    )
    return F.softmax(normalized, dim=0)


def _simplex_tolerance(tensor: torch.Tensor) -> float:
    if not tensor.is_floating_point():
        raise TypeError("Probability tensors must have a floating-point dtype.")
    return max(1e-5, 10.0 * float(torch.finfo(tensor.dtype).eps))


def integrate_probabilities(
    branch_probabilities: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Integrate ``[K,B,C,...]`` probabilities using ``[K,B]`` weights."""
    if branch_probabilities.ndim < 4:
        raise ValueError(
            "Branch probabilities must have shape [K, B, C, ...]."
        )
    if branch_probabilities.shape[2] < 2:
        raise ValueError("Branch probabilities must contain at least two classes.")
    if weights.shape != branch_probabilities.shape[:2]:
        raise ValueError(
            f"Weight shape {tuple(weights.shape)} does not match branch/batch "
            f"shape {tuple(branch_probabilities.shape[:2])}."
        )
    if not branch_probabilities.is_floating_point() or not weights.is_floating_point():
        raise TypeError("Branch probabilities and weights must be floating point.")
    if (
        not torch.isfinite(branch_probabilities).all()
        or not torch.isfinite(weights).all()
    ):
        raise ValueError("Branch probabilities and weights must be finite.")
    if torch.any(branch_probabilities < 0) or torch.any(weights < 0):
        raise ValueError("Branch probabilities and weights must be non-negative.")

    probability_tolerance = _simplex_tolerance(branch_probabilities)
    probability_sums = branch_probabilities.sum(dim=2)
    if not torch.allclose(
        probability_sums,
        torch.ones_like(probability_sums),
        rtol=probability_tolerance,
        atol=probability_tolerance,
    ):
        raise ValueError("Branch probabilities must sum to one over classes.")

    weight_tolerance = _simplex_tolerance(weights)
    weight_sums = weights.sum(dim=0)
    if not torch.allclose(
        weight_sums,
        torch.ones_like(weight_sums),
        rtol=weight_tolerance,
        atol=weight_tolerance,
    ):
        raise ValueError("Integration weights must sum to one over branches.")

    view_shape = (*weights.shape, *([1] * (branch_probabilities.ndim - 2)))
    integrated = (
        branch_probabilities * weights.reshape(view_shape)
    ).sum(dim=0)
    return integrated


def _statistics_on_reference_device(
    statistics: Mapping[str, BNLayerStatistics],
    reference: Mapping[str, BNLayerStatistics],
) -> BNStatistics:
    """Move statistics to match another named statistics collection."""
    if set(statistics) != set(reference):
        raise ValueError("BatchNorm statistics layer names differ.")
    aligned: BNStatistics = {}
    for name, (mean, variance) in statistics.items():
        reference_mean, reference_variance = reference[name]
        aligned[name] = (
            mean.to(reference_mean),
            variance.to(reference_variance),
        )
    return aligned


@dataclass(frozen=True)
class InTEntResult:
    """Integrated prediction and diagnostics for all BN-statistic branches."""

    probabilities: torch.Tensor
    branch_probabilities: torch.Tensor
    entropies: torch.Tensor
    weights: torch.Tensor
    test_fractions: torch.Tensor


class InTEnt(nn.Module):
    """Episodic InTEnt wrapper for a BN-based four-class segmenter."""

    def __init__(
        self,
        model: nn.Module,
        test_fractions: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        background_index: int = 0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        fractions = tuple(float(value) for value in test_fractions)
        if not fractions or any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError(
                "test_fractions must be a non-empty sequence in [0, 1]."
            )
        if len(set(fractions)) != len(fractions):
            raise ValueError("test_fractions must not contain duplicates.")
        if int(background_index) < 0:
            raise ValueError("background_index must be non-negative.")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.model = model
        self.test_fractions = fractions
        self.background_index = int(background_index)
        self.eps = float(eps)
        # Construct the wrapper after loading the source checkpoint.  Clones
        # protect the source snapshot from later episodic inference.
        self._source_statistics = capture_bn_stats(model)

    def refresh_source_stats(self) -> None:
        """Replace the stored source snapshot with the model's current BN state."""
        self._source_statistics = capture_bn_stats(self.model)

    @torch.no_grad()
    def forward_with_details(self, image: torch.Tensor) -> InTEntResult:
        """Run all statistic branches and return their integrated prediction."""
        if image.ndim < 3 or image.shape[0] != 1:
            raise ValueError(
                f"InTEnt expects a single-image batch, got {tuple(image.shape)}."
            )

        entry_statistics = capture_bn_stats(self.model)
        training_flags = {
            module: bool(module.training)
            for module in self.model.modules()
        }
        try:
            load_bn_stats(self.model, self._source_statistics)
            target_statistics = estimate_test_bn_stats(self.model, image)
            source_statistics = _statistics_on_reference_device(
                self._source_statistics,
                target_statistics,
            )

            branch_probabilities = []
            self.model.eval()
            for fraction in self.test_fractions:
                mixed = interpolate_bn_stats(
                    source_statistics,
                    target_statistics,
                    fraction,
                )
                load_bn_stats(self.model, mixed)
                logits = model_logits(self.model(image))
                if logits.ndim < 3 or logits.shape[0] != 1:
                    raise ValueError(
                        "Segmentation logits must have shape [1, C, ...], "
                        f"got {tuple(logits.shape)}."
                    )
                if logits.shape[1] < 2:
                    raise ValueError(
                        "InTEnt for MMS requires at least two output classes."
                    )
                branch_probabilities.append(F.softmax(logits, dim=1))

            branches = torch.stack(branch_probabilities, dim=0)
            entropies = torch.stack(
                [
                    balanced_fg_bg_entropy(
                        probabilities,
                        background_index=self.background_index,
                        eps=self.eps,
                    )
                    for probabilities in branches
                ],
                dim=0,
            )
            weights = entropy_integration_weights(entropies, eps=self.eps)
            probabilities = integrate_probabilities(branches, weights)
            fractions = torch.tensor(
                self.test_fractions,
                device=probabilities.device,
                dtype=probabilities.dtype,
            )
            return InTEntResult(
                probabilities=probabilities,
                branch_probabilities=branches,
                entropies=entropies,
                weights=weights,
                test_fractions=fractions,
            )
        finally:
            load_bn_stats(self.model, entry_statistics)
            for module, training in training_flags.items():
                module.training = training

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return the integrated four-class probability map for one image."""
        return self.forward_with_details(image).probabilities


__all__ = [
    "BNLayerStatistics",
    "BNStatistics",
    "InTEnt",
    "InTEntResult",
    "balanced_fg_bg_entropy",
    "batch_norm_layers",
    "capture_bn_stats",
    "categorical_entropy",
    "entropy_integration_weights",
    "estimate_test_bn_stats",
    "integrate_probabilities",
    "interpolate_bn_stats",
    "load_bn_stats",
    "model_logits",
]
