from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    in_ch: int = 1
    num_classes: int = 3
    base_ch: int = 16
    latent_ch: int = 64
    model_type: str = "d_l2_disp_bn"
    use_l2_norm: bool = True
    use_batch_norm: bool = True


class ConvBlock(nn.Module):
    """Two 3x3 Conv + BatchNorm2d + ReLU layers."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, kernel_size=3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet3Level(nn.Module):
    """TTA_SPIDER three-level lightweight U-Net with BatchNorm2d."""

    def __init__(self, in_ch: int = 1, base_ch: int = 16, latent_ch: int = 64, out_ch: int = 3):
        super().__init__()
        c1, c2, c3 = int(base_ch), int(base_ch) * 2, int(base_ch) * 4
        self.enc1 = ConvBlock(in_ch, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(c2, c3)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c3, latent_ch)
        self.up3 = nn.ConvTranspose2d(latent_ch, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3 + c3, c2)
        self.up2 = nn.ConvTranspose2d(c2, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c1)
        self.up1 = nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1)
        self.seg_head = nn.Conv2d(c1, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))
        up3 = self.up3(bottleneck)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))
        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        logits = self.seg_head(dec1)
        if not return_features:
            return logits
        return {
            "logits": logits,
            "features": {
                "enc1": enc1,
                "enc1_skip": enc1,
                "enc2": enc2,
                "enc3": enc3,
                "bottleneck": bottleneck,
                "z": bottleneck,
                "dec3": dec3,
                "dec2": dec2,
                "dec1": dec1,
            },
        }


class Dec1L2NormUNet(nn.Module):
    """Apply optional channel L2 normalization to dec1 before seg_head."""

    def __init__(self, base_model: nn.Module, use_l2_norm: bool = True):
        super().__init__()
        self.base = base_model
        self.use_l2_norm = bool(use_l2_norm)

    def attention_parameters(self) -> Iterator[nn.Parameter]:
        return iter(())

    def _maybe_l2_normalize(self, dec1: torch.Tensor) -> torch.Tensor:
        if not self.use_l2_norm:
            return dec1
        return F.normalize(dec1, p=2, dim=1, eps=1e-8) * math.sqrt(float(dec1.shape[1]))

    def forward(self, x: torch.Tensor, return_features: bool = False):
        m = self.base
        enc1 = m.enc1(x)
        enc2 = m.enc2(m.pool1(enc1))
        enc3 = m.enc3(m.pool2(enc2))
        bottleneck = m.bottleneck(m.pool3(enc3))
        up3 = m.up3(bottleneck)
        dec3 = m.dec3(torch.cat([up3, enc3], dim=1))
        up2 = m.up2(dec3)
        dec2 = m.dec2(torch.cat([up2, enc2], dim=1))
        up1 = m.up1(dec2)
        dec1 = m.dec1(torch.cat([up1, enc1], dim=1))
        dec1_head = self._maybe_l2_normalize(dec1)
        logits = m.seg_head(dec1_head)
        if not return_features:
            return logits
        return {
            "logits": logits,
            "features": {
                "enc1": enc1,
                "enc1_skip": enc1,
                "enc2": enc2,
                "enc3": enc3,
                "bottleneck": bottleneck,
                "z": bottleneck,
                "dec3": dec3,
                "dec2": dec2,
                "dec1": dec1,
                "dec1_head": dec1_head,
            },
        }

    @property
    def dec1(self):
        return self.base.dec1

    @property
    def seg_head(self):
        return self.base.seg_head


def _canonical_model_type(model_type: str) -> str:
    aliases = {
        "dec1_l2_norm_unet": "dec1_l2_norm_unet_bn",
        "d_l2_disp": "d_l2_disp_bn",
        "ccsc_iic_disp": "ccsc_iic_disp_bn",
        "unet": "unet3level_bn",
        "unet3level": "unet3level_bn",
        "plain_unet": "unet3level_bn",
        "unet_bn": "unet3level_bn",
        "plain_unet_bn": "unet3level_bn",
    }
    return aliases.get(str(model_type).lower(), str(model_type).lower())


def build_unet(config: ModelConfig | None = None) -> UNet3Level:
    cfg = config or ModelConfig()
    return UNet3Level(cfg.in_ch, cfg.base_ch, cfg.latent_ch, cfg.num_classes)


def build_model(config: ModelConfig | None = None) -> nn.Module:
    cfg = config or ModelConfig()
    model_type = _canonical_model_type(cfg.model_type)
    base = build_unet(cfg)
    if model_type in {"dec1_l2_norm_unet_bn", "d_l2_disp_bn", "ccsc_iic_disp_bn"}:
        return Dec1L2NormUNet(base, use_l2_norm=cfg.use_l2_norm)
    if model_type == "unet3level_bn":
        return base
    raise ValueError(f"Unsupported model_type: {cfg.model_type}")
