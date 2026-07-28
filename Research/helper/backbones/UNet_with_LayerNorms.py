"""U-Net backbone with batch-independent LayerNorm.

The architecture and public interface mirror :mod:`helper.backbones.UNet`.
Every ``BatchNorm2d`` in the original double-convolution blocks is replaced by
``LayerNorm2d``, which applies ``nn.LayerNorm`` to the channel vector at each
spatial location.  This is the convolutional analogue of feature-dimension
LayerNorm used by vision transformers and does not depend on batch statistics
or a fixed input resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.LayerNorm):
    """Apply LayerNorm over channels independently at every NCHW pixel."""

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        if int(num_channels) <= 0:
            raise ValueError(f"num_channels must be positive, got {num_channels}.")
        super().__init__(
            normalized_shape=int(num_channels),
            eps=float(eps),
            elementwise_affine=bool(elementwise_affine),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 4:
            raise ValueError(
                f"LayerNorm2d expects an NCHW tensor, got shape {tuple(tensor.shape)}."
            )
        expected_channels = int(self.normalized_shape[0])
        if int(tensor.shape[1]) != expected_channels:
            raise ValueError(
                f"LayerNorm2d expected {expected_channels} channels, "
                f"got {int(tensor.shape[1])}."
            )
        channels_last = tensor.permute(0, 2, 3, 1)
        normalized = super().forward(channels_last)
        return normalized.permute(0, 3, 1, 2).contiguous()


class FeaturesSegmenter(nn.Module):
    def __init__(self, in_channels: int = 64, out_channels: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = F.relu(self.conv1(tensor))
        tensor = F.relu(self.conv2(tensor))
        return self.conv3(tensor)


class DoubleConv(nn.Module):
    """Two ``Conv2d -> LayerNorm2d -> ReLU`` blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
    ) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.double_conv(tensor)


class Down(nn.Module):
    """Downscale with max pooling followed by a double convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(tensor)


class Up(nn.Module):
    """Upscale, concatenate the encoder skip, and apply a double convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )
            self.conv = DoubleConv(
                in_channels,
                out_channels,
                in_channels // 2,
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size=2,
                stride=2,
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(
        self,
        decoder_tensor: torch.Tensor,
        encoder_tensor: torch.Tensor,
    ) -> torch.Tensor:
        decoder_tensor = self.up(decoder_tensor)
        difference_y = encoder_tensor.size(2) - decoder_tensor.size(2)
        difference_x = encoder_tensor.size(3) - decoder_tensor.size(3)
        decoder_tensor = F.pad(
            decoder_tensor,
            [
                difference_x // 2,
                difference_x - difference_x // 2,
                difference_y // 2,
                difference_y - difference_y // 2,
            ],
        )
        tensor = torch.cat([encoder_tensor, decoder_tensor], dim=1)
        return self.conv(tensor)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.conv(tensor)


class UNet(nn.Module):
    """Drop-in LayerNorm replacement for ``helper.backbones.UNet.UNet``."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        only_feature: bool = True,
        bilinear: bool = False,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.only_feature = only_feature
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        if not self.only_feature:
            self.outc = OutConv(64, n_classes)

    def forward(
        self,
        tensor: torch.Tensor,
        only_feature: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del only_feature  # Retained for compatibility with the original API.
        encoder_1 = self.inc(tensor)
        encoder_2 = self.down1(encoder_1)
        encoder_3 = self.down2(encoder_2)
        encoder_4 = self.down3(encoder_3)
        bottleneck = self.down4(encoder_4)
        decoder = self.up1(bottleneck, encoder_4)
        decoder = self.up2(decoder, encoder_3)
        decoder = self.up3(decoder, encoder_2)
        features = self.up4(decoder, encoder_1)
        if self.only_feature:
            return features
        return features, self.outc(features)


UNetWithLayerNorm = UNet


__all__ = (
    "DoubleConv",
    "Down",
    "FeaturesSegmenter",
    "LayerNorm2d",
    "OutConv",
    "UNet",
    "UNetWithLayerNorm",
    "Up",
)
