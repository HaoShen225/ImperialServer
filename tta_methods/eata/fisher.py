from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
from torch.nn import functional as F

from ..common import collect_bn_affine, configure_bn_for_batch_stats, cpu_parameter_dict


def estimate_fisher(
    model: torch.nn.Module,
    source_loader: Iterable[dict[str, Any]],
    cfg: dict[str, Any],
    device: torch.device,
    output: str | Path,
) -> Path:
    configure_bn_for_batch_stats(model)
    parameters, names = collect_bn_affine(model)
    named = dict(model.named_parameters())
    accumulators = {name: torch.zeros_like(named[name], device=device) for name in names}
    maximum = int(cfg["fisher_samples"])
    seen = 0
    for batch in source_loader:
        if seen >= maximum:
            break
        images = batch["image"].to(device)
        remaining = maximum - seen
        images = images[:remaining]
        model.zero_grad(set_to_none=True)
        logits = model(images)["logits"]
        pseudo_labels = logits.detach().argmax(dim=1)
        loss = F.cross_entropy(logits, pseudo_labels)
        loss.backward()
        batch_count = int(images.shape[0])
        for name in names:
            if named[name].grad is not None:
                accumulators[name] += named[name].grad.detach().square() * batch_count
        seen += batch_count
    if seen == 0:
        raise RuntimeError("No Vendor-A samples were available for Fisher estimation")
    artifact = {
        "fisher": {name: value.detach().cpu() / seen for name, value in accumulators.items()},
        "source_parameters": {name: value for name, value in cpu_parameter_dict(model).items() if name in accumulators},
        "parameter_names": names,
        "samples": seen,
        "objective": "pseudo_label_pixel_cross_entropy",
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, destination)
    return destination
