from __future__ import annotations

from typing import Any

import torch

from .base import BaseTTA
from .cotta.method import CoTTA
from .deyo.method import DeYO
from .eata.method import EATA
from .roid.method import RoID
from .rotta.method import RoTTA
from .sar.method import SAR
from .source import Source
from .tent.method import TENT


METHODS = {
    "source": Source,
    "tent": TENT,
    "eata": EATA,
    "sar": SAR,
    "cotta": CoTTA,
    "rotta": RoTTA,
    "roid": RoID,
    "deyo": DeYO,
}


def build_method(
    name: str,
    model: torch.nn.Module,
    cfg: dict[str, Any],
    protocol_cfg: dict[str, Any],
    device: torch.device,
) -> BaseTTA:
    try:
        method_class = METHODS[name]
    except KeyError as error:
        raise ValueError(f"Unknown method {name!r}; available methods: {sorted(METHODS)}") from error
    if not bool(cfg.get("profile_verified", False)):
        raise ValueError(f"Method profile {name!r} is not verified")
    return method_class(model=model, cfg=cfg, protocol_cfg=protocol_cfg, device=device)


__all__ = ["METHODS", "build_method", "BaseTTA"]
