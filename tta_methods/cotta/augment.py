from __future__ import annotations

import torch
from torch.nn import functional as F


@torch.no_grad()
def teacher_augmentation_ensemble(
    teacher: torch.nn.Module,
    images: torch.Tensor,
    scales: list[float],
    flip_probability: float,
    generator: torch.Generator,
) -> torch.Tensor:
    original_shape = images.shape[-2:]
    total = teacher(images)["logits"]
    for scale in scales:
        output_shape = (
            max(32, int(original_shape[0] * 0.5 * float(scale))),
            max(32, int(original_shape[1] * 0.5 * float(scale))),
        )
        flip_mask = torch.rand(images.shape[0], generator=generator) < flip_probability
        augmented = images.clone()
        for index, do_flip in enumerate(flip_mask.tolist()):
            if do_flip:
                augmented[index] = augmented[index].flip(-1)
        augmented = F.interpolate(augmented, size=output_shape, mode="bilinear", align_corners=True)
        logits = teacher(augmented)["logits"]
        logits = F.interpolate(logits, size=original_shape, mode="bilinear", align_corners=True)
        for index, do_flip in enumerate(flip_mask.tolist()):
            if do_flip:
                logits[index] = logits[index].flip(-1)
        total = total + logits
    return total / (len(scales) + 1)
