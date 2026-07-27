"""GraTA ResUNet-34 backbone for ACDC/MMS segmentation.

The encoder and decoder follow the ResUNet implementation released with
GraTA.  The unused reconstruction, rotation, denoising, and super-resolution
heads are intentionally omitted.  An optional per-pixel L2 projection can be
applied immediately before the segmentation head.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class L2SphereProjection(nn.Module):
    """Project every spatial feature vector onto the unit L2 sphere."""

    def __init__(self, enabled: bool = False, eps: float = 1e-12) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")
        self.enabled = bool(enabled)
        self.eps = float(eps)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return features
        return F.normalize(features, p=2.0, dim=1, eps=self.eps)


def _conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    """ResNet-34 basic residual block used by the GraTA encoder."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs

        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))

        if self.downsample is not None:
            residual = self.downsample(inputs)

        return self.relu(outputs + residual)


class ResNet34Encoder(nn.Module):
    """ResNet-34 encoder returning the GraTA skip-feature hierarchy."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")

        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks=3, stride=1)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=2)

    def _make_layer(
        self,
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        downsample: Optional[nn.Module] = None
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers = [
            BasicBlock(
                self.inplanes,
                out_channels,
                stride=stride,
                downsample=downsample,
            )
        ]
        self.inplanes = out_channels
        layers.extend(
            BasicBlock(self.inplanes, out_channels) for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        skip0 = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.maxpool(skip0)

        skip1 = self.layer1(outputs)
        skip2 = self.layer2(skip1)
        skip3 = self.layer3(skip2)
        bottleneck = self.layer4(skip3)

        return bottleneck, (skip0, skip1, skip2, skip3, bottleneck)


class UNetBlock(nn.Module):
    """GraTA decoder block combining a transposed convolution and a skip."""

    def __init__(
        self,
        up_channels: int,
        skip_channels: int,
        out_channels: int = 256,
    ) -> None:
        super().__init__()
        if out_channels % 2:
            raise ValueError("out_channels must be even.")
        branch_channels = out_channels // 2
        self.skip_projection = nn.Conv2d(
            skip_channels,
            branch_channels,
            kernel_size=1,
        )
        self.upsample = nn.ConvTranspose2d(
            up_channels,
            branch_channels,
            kernel_size=2,
            stride=2,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(
        self,
        decoder_features: torch.Tensor,
        skip_features: torch.Tensor,
    ) -> torch.Tensor:
        upsampled = self.upsample(decoder_features)
        projected_skip = self.skip_projection(skip_features)
        if upsampled.shape[-2:] != projected_skip.shape[-2:]:
            raise ValueError(
                "Decoder and skip spatial shapes differ: "
                f"{tuple(upsampled.shape[-2:])} vs "
                f"{tuple(projected_skip.shape[-2:])}."
            )
        combined = torch.cat((upsampled, projected_skip), dim=1)
        return self.bn(F.relu(combined, inplace=True))


class ResUNet34(nn.Module):
    """GraTA ResUNet-34 adapted to single-channel, four-class segmentation.

    ``forward`` mirrors the existing Research U-Net contract: by default it
    returns ``(head_features, logits)``.  With ``only_feature=True`` it returns
    only the actual feature tensor consumed by the segmentation head.
    """

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 4,
        only_feature: bool = False,
        use_l2_projection: bool = False,
        projection_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if n_channels < 1:
            raise ValueError(f"n_channels must be positive, got {n_channels}.")
        if n_classes < 2:
            raise ValueError(f"n_classes must be at least 2, got {n_classes}.")

        self.n_channels = int(n_channels)
        self.n_classes = int(n_classes)
        self.only_feature = bool(only_feature)

        self.encoder = ResNet34Encoder(in_channels=self.n_channels)
        self.up1 = UNetBlock(512, 256, 256)
        self.up2 = UNetBlock(256, 128, 256)
        self.up3 = UNetBlock(256, 64, 256)
        self.up4 = UNetBlock(256, 64, 256)
        self.up5 = nn.ConvTranspose2d(256, 32, kernel_size=2, stride=2)
        self.bnout = nn.BatchNorm2d(32)
        self.l2_projection = L2SphereProjection(
            enabled=use_l2_projection,
            eps=projection_eps,
        )
        self.seg_head = nn.Conv2d(32, self.n_classes, kernel_size=1)

        self._initialize_weights()

    @property
    def use_l2_projection(self) -> bool:
        return self.l2_projection.enabled

    def set_l2_projection(self, enabled: bool) -> None:
        """Enable or disable the projection without changing model weights."""
        self.l2_projection.enabled = bool(enabled)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _validate_input(self, inputs: torch.Tensor) -> None:
        if inputs.ndim != 4:
            raise ValueError(
                "ResUNet34 expects [B, C, H, W] input, "
                f"got shape {tuple(inputs.shape)}."
            )
        if inputs.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} input channels, "
                f"got {inputs.shape[1]}."
            )
        height, width = inputs.shape[-2:]
        if height % 32 or width % 32:
            raise ValueError(
                "Input height and width must be divisible by 32, "
                f"got {(height, width)}."
            )

    def forward(
        self,
        inputs: torch.Tensor,
        only_feature: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        self._validate_input(inputs)

        bottleneck, skips = self.encoder(inputs)
        outputs = F.relu(bottleneck, inplace=False)
        outputs = self.up1(outputs, skips[3])
        outputs = self.up2(outputs, skips[2])
        outputs = self.up3(outputs, skips[1])
        outputs = self.up4(outputs, skips[0])
        outputs = self.up5(outputs)

        head_features = F.relu(self.bnout(outputs), inplace=True)
        head_features = self.l2_projection(head_features)
        logits = self.seg_head(head_features)

        return_features_only = (
            self.only_feature if only_feature is None else bool(only_feature)
        )
        if return_features_only:
            return head_features
        return head_features, logits


# The official GraTA repository names the class ``ResUnet``.  Keep the alias
# for straightforward migration while exposing the explicit ResUNet34 name.
ResUnet = ResUNet34


__all__ = [
    "BasicBlock",
    "L2SphereProjection",
    "ResNet34Encoder",
    "ResUNet34",
    "ResUnet",
    "UNetBlock",
]
