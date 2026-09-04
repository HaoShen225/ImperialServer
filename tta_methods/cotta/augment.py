from __future__ import annotations

import torch
from torch.nn import functional as F


@torch.no_grad()
def teacher_augmentation_ensemble(
    teacher: torch.nn.Module,
    images: torch.Tensor,
    scales: list[float],
    flips: list[bool],
    standard_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average the official 7-scale x 2-flip segmentation views.

    Predictions are mapped back to the original image grid before averaging.
    ``standard_logits`` lets the caller reuse the already-computed 1x,
    unflipped teacher prediction without changing the 14-view definition.
    """
    original_shape = images.shape[-2:]
    total: torch.Tensor | None = None
    n_views = 0
    for scale in scales:
        output_shape = (
            max(1, int(round(original_shape[0] * float(scale)))),
            max(1, int(round(original_shape[1] * float(scale)))),
        )
        for do_flip in flips:
            if float(scale) == 1.0 and not do_flip and standard_logits is not None:
                logits = standard_logits
            else:
                augmented = images.flip(-1) if do_flip else images
                if output_shape != original_shape:
                    augmented = F.interpolate(
                        augmented, size=output_shape, mode="bilinear", align_corners=False
                    )
                logits = teacher(augmented)["logits"]
                if do_flip:
                    logits = logits.flip(-1)
                if logits.shape[-2:] != original_shape:
                    logits = F.interpolate(
                        logits, size=original_shape, mode="bilinear", align_corners=False
                    )
            total = logits if total is None else total + logits
            n_views += 1
    if total is None or n_views == 0:
        raise ValueError("CoTTA augmentation ensemble requires at least one view")
    return total / n_views
