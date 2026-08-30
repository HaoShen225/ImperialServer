"""Public lifecycle shared by every adaptation method."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import time
from typing import Any

import torch

from .common import cpu_parameter_dict, cpu_state_dict, parameter_drift


@dataclass
class AdaptationResult:
    loss: float | None = None
    n_seen: int = 0
    n_selected: int = 0
    updated: bool = False
    extras: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseTTA(ABC):
    prediction_source = "student"

    def __init__(self, model: torch.nn.Module, cfg: dict[str, Any], protocol_cfg: dict[str, Any], device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.protocol_cfg = protocol_cfg
        self.device = device
        self.optimizer: torch.optim.Optimizer | None = None
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(cfg.get("method_seed", 0)))
        self.source_parameter_state = cpu_parameter_dict(self.model)
        self.setup()
        self._initial_state = self.capture_state()

    @abstractmethod
    def setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def adapt(self, images: torch.Tensor) -> AdaptationResult:
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)["logits"]

    def process_batch(self, images: torch.Tensor) -> tuple[torch.Tensor, AdaptationResult]:
        timing = self.protocol_cfg["timing"]
        if timing == "adapt_then_predict":
            self._synchronize(images)
            started = time.perf_counter()
            info = self.adapt(images)
            self._synchronize(images)
            adaptation_seconds = time.perf_counter() - started
            started = time.perf_counter()
            logits = self.predict(images)
            self._synchronize(images)
            prediction_seconds = time.perf_counter() - started
        elif timing == "predict_then_adapt":
            self._synchronize(images)
            started = time.perf_counter()
            logits = self.predict(images)
            self._synchronize(images)
            prediction_seconds = time.perf_counter() - started
            started = time.perf_counter()
            info = self.adapt(images)
            self._synchronize(images)
            adaptation_seconds = time.perf_counter() - started
        else:
            raise ValueError(f"Unknown timing: {timing}")
        info.extras.setdefault("adaptation_seconds", adaptation_seconds)
        info.extras.setdefault("prediction_seconds", prediction_seconds)
        info.extras.setdefault("parameter_drift", parameter_drift(self.model, self.source_parameter_state))
        return logits, info

    @staticmethod
    def _synchronize(images: torch.Tensor) -> None:
        if images.device.type == "cuda":
            torch.cuda.synchronize(images.device)

    def capture_state(self) -> dict[str, Any]:
        return {"model": cpu_state_dict(self.model), "generator": self.generator.get_state().clone()}

    def reset(self) -> None:
        self.model.load_state_dict(self._initial_state["model"], strict=True)
        self.generator.set_state(self._initial_state["generator"])
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self) -> torch.optim.Optimizer | None:
        return None

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]
