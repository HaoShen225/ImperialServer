"""ResUNet-34 with BatchNorm throughout the encoder and decoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet34_Weights, resnet34


class ResNet34Encoder(nn.Module):
    def __init__(self, pretrained: bool, in_channels: int = 1):
        super().__init__()
        backbone = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        if in_channels != 3:
            conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                if in_channels == 1:
                    conv.weight.copy_(backbone.conv1.weight.mean(dim=1, keepdim=True))
                else:
                    averaged = backbone.conv1.weight.mean(dim=1, keepdim=True)
                    conv.weight.copy_(averaged.repeat(1, in_channels, 1, 1))
            backbone.conv1 = conv
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        stem = self.relu(self.bn1(self.conv1(images)))
        layer1 = self.layer1(self.maxpool(stem))
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return stem, layer1, layer2, layer3, layer4


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        value = torch.cat([value, skip], dim=1)
        return self.conv2(self.conv1(value))


class ResUNet34(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 4, pretrained_encoder: bool = True):
        super().__init__()
        self.encoder = ResNet34Encoder(pretrained_encoder, in_channels)
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)
        self.final_decoder = ConvBNReLU(64, 32)
        self.segmentation_head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, images: torch.Tensor, return_features: bool = False) -> dict[str, torch.Tensor | None]:
        stem, layer1, layer2, layer3, layer4 = self.encoder(images)
        value = self.decoder4(layer4, layer3)
        value = self.decoder3(value, layer2)
        value = self.decoder2(value, layer1)
        value = self.decoder1(value, stem)
        value = F.interpolate(value, size=images.shape[-2:], mode="bilinear", align_corners=False)
        features = self.final_decoder(value)
        logits = self.segmentation_head(features)
        return {"logits": logits, "features": features if return_features else None}


def build_model(cfg: dict[str, Any], pretrained_override: bool | None = None) -> ResUNet34:
    model_cfg = cfg["model"]
    if model_cfg["name"] != "resunet34":
        raise ValueError(f"Unknown model: {model_cfg['name']}")
    pretrained = bool(model_cfg["pretrained_encoder"]) if pretrained_override is None else pretrained_override
    return ResUNet34(
        in_channels=int(model_cfg["in_channels"]),
        num_classes=int(model_cfg["num_classes"]),
        pretrained_encoder=pretrained,
    )


def load_source_checkpoint(model: nn.Module, path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    return checkpoint
