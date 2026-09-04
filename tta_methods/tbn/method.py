from __future__ import annotations

import torch

from ..base import AdaptationResult, BaseTTA
from .batch_norm import configure_tbn_model


class TBN(BaseTTA):
    """Use current arrival-batch BN statistics without changing model parameters."""

    prediction_source = "tbn_model"

    def setup(self) -> None:
        self.batch_norm_names = configure_tbn_model(self.model)
        self.optimizer = None

    def adapt(self, images: torch.Tensor) -> AdaptationResult:
        return AdaptationResult(
            n_seen=int(images.shape[0]),
            n_selected=0,
            updated=False,
        )
