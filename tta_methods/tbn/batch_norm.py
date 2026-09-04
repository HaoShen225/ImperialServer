"""BatchNorm surgery specific to the TBN baseline."""

from __future__ import annotations

from torch import nn

from ..common import remove_bn_running_buffers


def configure_tbn_model(model: nn.Module) -> list[str]:
    """Freeze the model and make every BatchNorm use arrival-batch statistics."""
    model.eval()
    model.requires_grad_(False)
    batch_norm_names = [
        name for name, module in model.named_modules()
        if isinstance(module, nn.BatchNorm2d)
    ]
    if not batch_norm_names:
        raise ValueError("TBN requires at least one BatchNorm2d layer")
    remove_bn_running_buffers(model)
    return batch_norm_names
