"""Shared stride 4/8/16/32 pyramid and padding-mask propagation."""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .dit import DiTBase


FEATURE_NAMES = ("p2", "p3", "p4", "p5")
FEATURE_STRIDES = (4, 8, 16, 32)


class DiTFeaturePyramid(nn.Module):
    """Turn four stride-16 DiT taps into a common 4-level pyramid."""

    size_divisibility = 32

    def __init__(self, backbone: DiTBase | None = None) -> None:
        super().__init__()
        self.backbone = backbone or DiTBase()
        dim = self.backbone.embed_dim
        self.transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(dim, dim, 2, 2),
                    nn.BatchNorm2d(dim),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim, dim, 2, 2),
                ),
                nn.ConvTranspose2d(dim, dim, 2, 2),
                nn.Identity(),
                nn.MaxPool2d(2, 2),
            ]
        )
        self.num_channels = [dim] * 4
        self.strides = list(FEATURE_STRIDES)

    @staticmethod
    def pad(images: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        height, width = images.shape[-2:]
        padded_height = math.ceil(height / 32) * 32
        padded_width = math.ceil(width / 32) * 32
        pad_height, pad_width = padded_height - height, padded_width - width
        images = F.pad(images, (0, pad_width, 0, pad_height))
        if mask is None:
            mask = torch.zeros(
                (images.shape[0], height, width), dtype=torch.bool, device=images.device
            )
        mask = F.pad(mask, (0, pad_width, 0, pad_height), value=True)
        return images, mask

    def forward(
        self, images: Tensor, mask: Tensor | None = None
    ) -> tuple[OrderedDict[str, Tensor], OrderedDict[str, Tensor]]:
        images, mask = self.pad(images, mask)
        taps = self.backbone(images)
        features: OrderedDict[str, Tensor] = OrderedDict()
        masks: OrderedDict[str, Tensor] = OrderedDict()
        for name, tap, transform, stride in zip(
            FEATURE_NAMES, taps.values(), self.transforms, FEATURE_STRIDES
        ):
            feature = transform(tap)
            expected = (images.shape[-2] // stride, images.shape[-1] // stride)
            if feature.shape[-2:] != expected:
                raise RuntimeError(
                    f"{name} has shape {feature.shape[-2:]}, expected {expected} for stride {stride}"
                )
            features[name] = feature
            masks[name] = F.interpolate(mask[:, None].float(), size=expected, mode="nearest")[:, 0].bool()
        return features, masks

