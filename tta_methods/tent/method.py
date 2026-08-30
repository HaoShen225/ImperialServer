from __future__ import annotations

import torch

from ..base import AdaptationResult, BaseTTA
from ..common import build_optimizer, collect_bn_affine, configure_bn_for_batch_stats, slice_entropy


class TENT(BaseTTA):
    def setup(self) -> None:
        configure_bn_for_batch_stats(self.model)
        self.parameters, self.parameter_names = collect_bn_affine(self.model)
        if not self.parameters:
            raise ValueError("TENT requires BatchNorm affine parameters")
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return build_optimizer(self.parameters, self.cfg)

    def adapt(self, images: torch.Tensor) -> AdaptationResult:
        assert self.optimizer is not None
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(images)["logits"]
        loss = slice_entropy(logits).mean()
        loss.backward()
        self.optimizer.step()
        return AdaptationResult(loss=float(loss.detach()), n_seen=int(images.shape[0]), n_selected=int(images.shape[0]), updated=True)
