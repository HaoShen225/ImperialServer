"""Dual-tree propagation for source-model segmentation pseudo labels.

The implementation follows the minimum-spanning-tree filter used by Tree
Energy Loss, but applies the filter independently to caller-provided shallow
and deep guidance features.  Propagation is global by default; callers can
optionally restrict it to overlapping local windows.  Pseudo targets are
deliberately generated under
``torch.no_grad``: gradients for test-time adaptation should be taken against
the returned targets, not through the target-generation path.

The CUDA extension is vendored in this package and must be built before tree
propagation is used.  Weight generation policies intentionally live outside
this module.  Omitting ``pseudo_label_weights`` means uniform (all-one)
weights.  Spatial path decay is enabled by default with a 16-pixel
temperature and can be disabled by passing ``spatial_temperature=None``.
"""

from dataclasses import dataclass
from functools import lru_cache
import math
from numbers import Integral
from typing import Any, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


SizeArg = Union[int, Sequence[int]]
SpatialSize = Tuple[int, int]
DEFAULT_SPATIAL_TEMPERATURE = 16.0


@dataclass(frozen=True)
class DualTreePseudoLabels:
    """Independent pseudo targets propagated on shallow and deep trees."""

    shallow: torch.Tensor
    deep: torch.Tensor


def _as_pair(value: SizeArg, name: str) -> SpatialSize:
    if isinstance(value, Integral) and not isinstance(value, bool):
        pair = (int(value), int(value))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        if any(not isinstance(item, Integral) or isinstance(item, bool) for item in value):
            raise TypeError(f"{name} entries must be integers.")
        pair = (int(value[0]), int(value[1]))
    else:
        raise TypeError(f"{name} must be an integer or a sequence of two integers.")
    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"{name} entries must be positive, got {pair}.")
    return pair


def _positive_scale(
    value: Optional[float],
    name: str,
    *,
    allow_none: bool = False,
) -> Optional[float]:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} must be a positive number.")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        suffix = " or None" if allow_none else ""
        raise ValueError(f"{name} must be a positive number{suffix}.")
    return float(value)


def _validate_probability_shape(source_probabilities: torch.Tensor) -> None:
    if not isinstance(source_probabilities, torch.Tensor):
        raise TypeError("source_probabilities must be a torch.Tensor.")
    if source_probabilities.ndim != 4:
        raise ValueError(
            "source_probabilities must have shape [B, C, H, W], "
            f"got {tuple(source_probabilities.shape)}."
        )
    if not source_probabilities.is_floating_point():
        raise TypeError("source_probabilities must use a floating-point dtype.")
    if any(size <= 0 for size in source_probabilities.shape):
        raise ValueError("source_probabilities cannot contain empty dimensions.")


@torch.no_grad()
def _prepare_probabilities(source_probabilities: torch.Tensor) -> torch.Tensor:
    _validate_probability_shape(source_probabilities)
    probabilities = source_probabilities.detach()
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("source_probabilities contains NaN or infinity.")
    tolerance = 1e-5
    if bool((probabilities < -tolerance).any()) or bool((probabilities > 1.0 + tolerance).any()):
        raise ValueError(
            "source_probabilities must contain probabilities in [0, 1]; "
            "apply softmax to source-model logits before propagation."
        )
    probabilities_float = probabilities.to(dtype=torch.float32)
    class_sum = probabilities_float.sum(dim=1, keepdim=True)
    sum_tolerance = 2e-3 if probabilities.dtype in (torch.float16, torch.bfloat16) else 1e-4
    if not torch.allclose(
        class_sum,
        torch.ones_like(class_sum),
        rtol=sum_tolerance,
        atol=sum_tolerance,
    ):
        raise ValueError(
            "source_probabilities must sum to one along the class dimension; "
            "apply softmax to source-model logits before propagation."
        )
    probabilities = probabilities_float.clamp(0.0, 1.0)
    return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def make_pseudo_label_weights(
    source_probabilities: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return validated per-pixel pseudo-label weights shaped ``[B, 1, H, W]``.

    ``weights=None`` deliberately returns all-one weights.  Later experiments
    can pass externally computed confidence, Gaussian, or class-balancing
    weights without changing the propagation implementation.
    """

    _validate_probability_shape(source_probabilities)
    batch, _, height, width = source_probabilities.shape
    if weights is None:
        return torch.ones(
            (batch, 1, height, width),
            device=source_probabilities.device,
            dtype=torch.float32,
        )
    if not isinstance(weights, torch.Tensor):
        raise TypeError("weights must be a torch.Tensor or None.")
    if weights.ndim == 3:
        weights = weights.unsqueeze(1)
    if weights.ndim != 4 or tuple(weights.shape) != (batch, 1, height, width):
        raise ValueError(
            "weights must have shape [B, H, W] or [B, 1, H, W] matching "
            f"source_probabilities; got {tuple(weights.shape)}."
        )
    if not weights.is_floating_point():
        raise TypeError("weights must use a floating-point dtype.")
    if weights.device != source_probabilities.device:
        raise ValueError("weights and source_probabilities must be on the same device.")
    weights = weights.detach().to(dtype=torch.float32)
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("weights contains NaN or infinity.")
    if bool((weights < 0).any()):
        raise ValueError("weights must be non-negative.")
    return weights.contiguous()


def _validate_guidance(
    guidance: torch.Tensor,
    probabilities: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if not isinstance(guidance, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if guidance.ndim != 4:
        raise ValueError(f"{name} must have shape [B, F, H, W], got {tuple(guidance.shape)}.")
    if guidance.shape[0] != probabilities.shape[0]:
        raise ValueError(
            f"{name} batch size {guidance.shape[0]} does not match "
            f"source_probabilities batch size {probabilities.shape[0]}."
        )
    if guidance.shape[1] <= 0 or guidance.shape[2] <= 0 or guidance.shape[3] <= 0:
        raise ValueError(f"{name} cannot contain empty dimensions.")
    if not guidance.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype.")
    if guidance.device != probabilities.device:
        raise ValueError(f"{name} and source_probabilities must be on the same device.")
    guidance = guidance.detach().to(dtype=torch.float32)
    if not bool(torch.isfinite(guidance).all()):
        raise ValueError(f"{name} contains NaN or infinity.")
    target_size = probabilities.shape[-2:]
    if guidance.shape[-2:] != target_size:
        guidance = F.interpolate(
            guidance,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
    return guidance.contiguous()


@lru_cache(maxsize=1)
def _load_tree_filter_cuda() -> Any:
    try:
        from . import _tree_filter_cuda
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "EnergyLossTree CUDA extension is unavailable. Build it from "
            "Research/helper/EnergyLossTree with "
            "`python setup.py build_ext --inplace` in the TTA CUDA environment."
        ) from error
    return _tree_filter_cuda


def _axis_starts(length: int, requested_window: int, stride: int) -> Tuple[int, ...]:
    window = min(length, requested_window)
    if window == length:
        return (0,)
    starts = list(range(0, length - window + 1, stride))
    final_start = length - window
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


def _raised_cosine_window(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if height == 1:
        vertical = torch.ones(1, device=device, dtype=dtype)
    else:
        vertical = torch.hann_window(height, periodic=False, device=device, dtype=dtype)
    if width == 1:
        horizontal = torch.ones(1, device=device, dtype=dtype)
    else:
        horizontal = torch.hann_window(width, periodic=False, device=device, dtype=dtype)
    # A positive floor keeps image and non-overlapping window borders covered.
    return torch.outer(vertical, horizontal).clamp_min_(1e-3).view(1, height, width)


def _grid_edges(
    batch: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    vertex = torch.arange(height * width, device=device, dtype=torch.int32).view(height, width)
    vertical = torch.stack((vertex[:-1, :], vertex[1:, :]), dim=-1).reshape(-1, 2)
    horizontal = torch.stack((vertex[:, :-1], vertex[:, 1:]), dim=-1).reshape(-1, 2)
    edges = torch.cat((vertical, horizontal), dim=0)
    return edges.unsqueeze(0).expand(batch, -1, -1).contiguous()


def _tree_edge_affinities(
    flat_guidance: torch.Tensor,
    sorted_index: torch.Tensor,
    sorted_parent: torch.Tensor,
    *,
    width: int,
    sigma: float,
    spatial_temperature: Optional[float],
) -> torch.Tensor:
    gather_index = sorted_index.unsqueeze(1).expand(-1, flat_guidance.shape[1], -1).long()
    bfs_guidance = torch.gather(flat_guidance, 2, gather_index)
    parent_index = sorted_parent.unsqueeze(1).expand_as(gather_index).long()
    parent_guidance = torch.gather(bfs_guidance, 2, parent_index)
    tree_edge_distance = (bfs_guidance - parent_guidance).square().sum(dim=1)
    affinity_exponent = tree_edge_distance / float(sigma)

    if spatial_temperature is not None:
        bfs_vertex = sorted_index.long()
        bfs_y = torch.div(bfs_vertex, width, rounding_mode="floor").to(torch.float32)
        bfs_x = torch.remainder(bfs_vertex, width).to(torch.float32)
        parent_position = sorted_parent.long()
        parent_y = torch.gather(bfs_y, 1, parent_position)
        parent_x = torch.gather(bfs_x, 1, parent_position)
        spatial_edge_distance = torch.sqrt(
            (bfs_y - parent_y).square() + (bfs_x - parent_x).square()
        )
        affinity_exponent = (
            affinity_exponent + spatial_edge_distance / float(spatial_temperature)
        )

    return torch.exp(-affinity_exponent).contiguous()


def _tree_filter_window_batch(
    probabilities: torch.Tensor,
    guidance: torch.Tensor,
    weights: torch.Tensor,
    *,
    sigma: float,
    spatial_temperature: Optional[float],
    eps: float,
) -> torch.Tensor:
    extension = _load_tree_filter_cuda()
    batch, _, height, width = probabilities.shape
    vertex_count = height * width

    edge_index = _grid_edges(batch, height, width, guidance.device)
    vertical_distance = (guidance[:, :, :-1, :] - guidance[:, :, 1:, :]).square().sum(dim=1)
    horizontal_distance = (guidance[:, :, :, :-1] - guidance[:, :, :, 1:]).square().sum(dim=1)
    edge_distance = torch.cat(
        (
            vertical_distance.reshape(batch, -1),
            horizontal_distance.reshape(batch, -1),
        ),
        dim=1,
    ).contiguous()

    # Adding the same positive constant preserves MST ordering and matches the
    # reference Tree Energy Loss implementation.
    tree = extension.mst_forward(edge_index, edge_distance + 1.0, vertex_count)
    sorted_index, sorted_parent, sorted_child = extension.bfs_forward(tree, 4)

    tree_edge_affinity = _tree_edge_affinities(
        guidance.reshape(batch, guidance.shape[1], vertex_count),
        sorted_index,
        sorted_parent,
        width=width,
        sigma=sigma,
        spatial_temperature=spatial_temperature,
    )

    weighted_probabilities = probabilities * weights
    filter_input = torch.cat((weighted_probabilities, weights), dim=1)
    filtered = extension.refine_forward(
        filter_input.reshape(batch, filter_input.shape[1], vertex_count).contiguous(),
        tree_edge_affinity,
        sorted_index.contiguous(),
        sorted_parent.contiguous(),
        sorted_child.contiguous(),
    ).reshape_as(filter_input)

    numerator = filtered[:, :-1]
    denominator = filtered[:, -1:]
    propagated = numerator / denominator.clamp_min(float(eps))
    propagated = torch.where(denominator > float(eps), propagated, probabilities)
    propagated = propagated.clamp_min_(0.0)
    return propagated / propagated.sum(dim=1, keepdim=True).clamp_min_(float(eps))


@torch.no_grad()
def windowed_tree_propagation(
    source_probabilities: torch.Tensor,
    guidance: torch.Tensor,
    *,
    window_size: Optional[SizeArg] = None,
    stride: Optional[SizeArg] = None,
    pseudo_label_weights: Optional[torch.Tensor] = None,
    sigma: float = 1.0,
    spatial_temperature: Optional[float] = DEFAULT_SPATIAL_TEMPERATURE,
    window_batch_size: int = 16,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Propagate soft pseudo labels on a global or optional window-local MST.

    Args:
        source_probabilities: Softmax probabilities shaped ``[B, C, H, W]``.
        guidance: Caller-provided guidance features shaped ``[B, F, h, w]``.
        window_size: Optional spatial window size at pseudo-label resolution.
            ``None`` selects full-image propagation and requires ``stride=None``.
        stride: Window stride when ``window_size`` is provided.  It cannot
            exceed ``window_size``.
        pseudo_label_weights: Optional non-negative map.  ``None`` means ones.
        sigma: Positive scale in ``exp(-squared_distance / sigma)``.
        spatial_temperature: Positive pixel scale in the spatial path factor
            ``exp(-path_length / spatial_temperature)``.  ``None`` disables
            spatial decay.
        window_batch_size: Maximum number of windows sent to the extension.
        eps: Numerical floor for weighted normalization.

    Returns:
        Detached float32 probabilities shaped ``[B, C, H, W]``.
    """

    probabilities = _prepare_probabilities(source_probabilities)
    batch, _, height, width = probabilities.shape

    if (window_size is None) != (stride is None):
        raise ValueError("window_size and stride must either both be provided or both be None.")
    if window_size is None:
        window = (height, width)
        step = (height, width)
        use_windows = False
    else:
        assert stride is not None
        window = _as_pair(window_size, "window_size")
        step = _as_pair(stride, "stride")
        if step[0] > window[0] or step[1] > window[1]:
            raise ValueError(f"stride {step} cannot exceed window_size {window}.")
        use_windows = True

    if not isinstance(window_batch_size, Integral) or isinstance(window_batch_size, bool):
        raise TypeError("window_batch_size must be an integer.")
    if int(window_batch_size) <= 0:
        raise ValueError("window_batch_size must be positive.")
    validated_sigma = _positive_scale(sigma, "sigma")
    validated_temperature = _positive_scale(
        spatial_temperature,
        "spatial_temperature",
        allow_none=True,
    )
    validated_eps = _positive_scale(eps, "eps")
    assert validated_sigma is not None
    assert validated_eps is not None

    if probabilities.device.type != "cuda":
        raise RuntimeError("EnergyLossTree propagation requires CUDA tensors.")

    prepared_guidance = _validate_guidance(guidance, probabilities, "guidance")
    weights = make_pseudo_label_weights(probabilities, pseudo_label_weights)
    _load_tree_filter_cuda()

    window_height = min(height, window[0])
    window_width = min(width, window[1])
    y_starts = _axis_starts(height, window[0], step[0])
    x_starts = _axis_starts(width, window[1], step[1])
    locations = [
        (batch_index, y_start, x_start)
        for batch_index in range(batch)
        for y_start in y_starts
        for x_start in x_starts
    ]

    output = torch.zeros_like(probabilities)
    coverage = torch.zeros(
        (batch, 1, height, width),
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    if use_windows:
        blend = _raised_cosine_window(
            window_height,
            window_width,
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
    else:
        blend = torch.ones(
            (1, window_height, window_width),
            device=probabilities.device,
            dtype=probabilities.dtype,
        )

    chunk_size = int(window_batch_size)
    for chunk_start in range(0, len(locations), chunk_size):
        chunk = locations[chunk_start : chunk_start + chunk_size]
        probability_windows = torch.stack(
            [
                probabilities[
                    batch_index,
                    :,
                    y_start : y_start + window_height,
                    x_start : x_start + window_width,
                ]
                for batch_index, y_start, x_start in chunk
            ],
            dim=0,
        ).contiguous()
        guidance_windows = torch.stack(
            [
                prepared_guidance[
                    batch_index,
                    :,
                    y_start : y_start + window_height,
                    x_start : x_start + window_width,
                ]
                for batch_index, y_start, x_start in chunk
            ],
            dim=0,
        ).contiguous()
        weight_windows = torch.stack(
            [
                weights[
                    batch_index,
                    :,
                    y_start : y_start + window_height,
                    x_start : x_start + window_width,
                ]
                for batch_index, y_start, x_start in chunk
            ],
            dim=0,
        ).contiguous()
        propagated_windows = _tree_filter_window_batch(
            probability_windows,
            guidance_windows,
            weight_windows,
            sigma=validated_sigma,
            spatial_temperature=validated_temperature,
            eps=validated_eps,
        )

        for local_index, (batch_index, y_start, x_start) in enumerate(chunk):
            y_slice = slice(y_start, y_start + window_height)
            x_slice = slice(x_start, x_start + window_width)
            output[batch_index, :, y_slice, x_slice].add_(propagated_windows[local_index] * blend)
            coverage[batch_index, :, y_slice, x_slice].add_(blend)

    if bool((coverage <= 0).any()):
        raise RuntimeError("Internal window coverage error: at least one output pixel was not covered.")
    output = output / coverage
    output = output.clamp_min_(0.0)
    output = output / output.sum(dim=1, keepdim=True).clamp_min_(validated_eps)
    return output.detach()


@torch.no_grad()
def propagate_dual_tree_pseudo_labels(
    source_probabilities: torch.Tensor,
    shallow_guidance: torch.Tensor,
    deep_guidance: torch.Tensor,
    *,
    window_size: Optional[SizeArg] = None,
    stride: Optional[SizeArg] = None,
    pseudo_label_weights: Optional[torch.Tensor] = None,
    shallow_sigma: float = 0.02,
    deep_sigma: float = 1.0,
    spatial_temperature: Optional[float] = DEFAULT_SPATIAL_TEMPERATURE,
    window_batch_size: int = 16,
    eps: float = 1e-6,
) -> DualTreePseudoLabels:
    """Generate independent shallow-tree and deep-tree pseudo targets.

    Both trees share ``spatial_temperature`` so their difference is determined
    by the caller-provided guidance and feature-affinity scales.
    """

    shallow = windowed_tree_propagation(
        source_probabilities,
        shallow_guidance,
        window_size=window_size,
        stride=stride,
        pseudo_label_weights=pseudo_label_weights,
        sigma=shallow_sigma,
        spatial_temperature=spatial_temperature,
        window_batch_size=window_batch_size,
        eps=eps,
    )
    deep = windowed_tree_propagation(
        source_probabilities,
        deep_guidance,
        window_size=window_size,
        stride=stride,
        pseudo_label_weights=pseudo_label_weights,
        sigma=deep_sigma,
        spatial_temperature=spatial_temperature,
        window_batch_size=window_batch_size,
        eps=eps,
    )
    return DualTreePseudoLabels(shallow=shallow, deep=deep)


__all__ = [
    "DEFAULT_SPATIAL_TEMPERATURE",
    "DualTreePseudoLabels",
    "make_pseudo_label_weights",
    "propagate_dual_tree_pseudo_labels",
    "windowed_tree_propagation",
]
