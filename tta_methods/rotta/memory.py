from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


@dataclass
class MemoryItem:
    image: torch.Tensor
    uncertainty: float
    age: int = 0


class CSTUMemory:
    def __init__(self, capacity: int, num_categories: int, lambda_timeliness: float, lambda_uncertainty: float):
        self.capacity = int(capacity)
        self.num_categories = int(num_categories)
        self.lambda_timeliness = float(lambda_timeliness)
        self.lambda_uncertainty = float(lambda_uncertainty)
        self.data: list[list[MemoryItem]] = [[] for _ in range(num_categories)]

    def __len__(self) -> int:
        return sum(len(items) for items in self.data)

    def _score(self, item: MemoryItem) -> float:
        return self.lambda_timeliness / (1.0 + math.exp(-item.age / self.capacity)) + self.lambda_uncertainty * item.uncertainty / math.log(self.num_categories)

    def _majority_categories(self) -> list[int]:
        sizes = [len(items) for items in self.data]
        maximum = max(sizes)
        return [index for index, size in enumerate(sizes) if size == maximum]

    def _try_evict(self, categories: list[int], new_score: float) -> bool:
        candidate: tuple[float, int, int] | None = None
        for category in categories:
            for index, item in enumerate(self.data[category]):
                score = self._score(item)
                if candidate is None or score >= candidate[0]:
                    candidate = (score, category, index)
        if candidate is None:
            return True
        if candidate[0] > new_score:
            self.data[candidate[1]].pop(candidate[2])
            return True
        return False

    def add(self, image: torch.Tensor, category: int, uncertainty: float) -> bool:
        item = MemoryItem(image.detach().cpu().clone(), float(uncertainty), 0)
        per_category = self.capacity / self.num_categories
        if len(self.data[category]) < per_category:
            accept = len(self) < self.capacity or self._try_evict(self._majority_categories(), self._score(item))
        else:
            accept = self._try_evict([category], self._score(item))
        if accept:
            self.data[category].append(item)
        for items in self.data:
            for existing in items:
                existing.age += 1
        return accept

    def get(self) -> tuple[list[torch.Tensor], list[float]]:
        images, ages = [], []
        for items in self.data:
            for item in items:
                images.append(item.image)
                ages.append(item.age / self.capacity)
        return images, ages

    def state_dict(self) -> dict[str, Any]:
        return {"data": [[{"image": item.image.clone(), "uncertainty": item.uncertainty, "age": item.age} for item in items] for items in self.data]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.data = [[MemoryItem(entry["image"].clone(), float(entry["uncertainty"]), int(entry["age"])) for entry in items] for items in state["data"]]
