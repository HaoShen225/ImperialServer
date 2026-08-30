from __future__ import annotations

import torch


def patch_shuffle(images: torch.Tensor, patch_grid: int, generator: torch.Generator) -> torch.Tensor:
    batch, channels, height, width = images.shape
    if height % patch_grid or width % patch_grid:
        raise ValueError("Image dimensions must be divisible by the DeYO patch grid")
    patch_height, patch_width = height // patch_grid, width // patch_grid
    patches = images.reshape(batch, channels, patch_grid, patch_height, patch_grid, patch_width)
    patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(batch, patch_grid * patch_grid, channels, patch_height, patch_width)
    shuffled = []
    for index in range(batch):
        permutation = torch.randperm(patch_grid * patch_grid, generator=generator).to(images.device)
        shuffled.append(patches[index, permutation])
    value = torch.stack(shuffled).reshape(batch, patch_grid, patch_grid, channels, patch_height, patch_width)
    return value.permute(0, 3, 1, 4, 2, 5).reshape(batch, channels, height, width)
