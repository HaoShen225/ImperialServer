from __future__ import annotations

import torch

from .base import AdaptationResult, BaseTTA


class Source(BaseTTA):
    prediction_source = "source_model"

    def setup(self) -> None:
        self.model.eval()
        self.model.requires_grad_(False)

    def adapt(self, images: torch.Tensor) -> AdaptationResult:
        return AdaptationResult(n_seen=int(images.shape[0]), n_selected=0, updated=False)
