from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import load_config


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.layer4 = nn.Sequential(
            nn.Conv2d(4, 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )

    def forward(self, value):
        return self.layer4(torch.relu(self.bn(self.conv(value))))


class TinySegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyEncoder()
        self.decoder_bn = nn.BatchNorm2d(4)
        self.segmentation_head = nn.Conv2d(4, 4, 1)

    def forward(self, value, return_features=False):
        features = torch.relu(self.decoder_bn(self.encoder(value)))
        return {"logits": self.segmentation_head(features), "features": features if return_features else None}


@pytest.fixture
def config():
    return load_config(ROOT / "config.yaml")


@pytest.fixture
def tiny_model():
    torch.manual_seed(17)
    return TinySegmentationModel()


@pytest.fixture
def images():
    generator = torch.Generator().manual_seed(23)
    return torch.rand(4, 1, 16, 16, generator=generator)


def method_config(config, name):
    value = deepcopy(config["methods"][name])
    if name == "eata":
        value["fisher_path"] = None
    if name == "rotta":
        value["update_frequency"] = 4
        value["memory_capacity"] = 4
    if name == "cotta":
        value["augmentation_scales"] = [1.0]
    return value
