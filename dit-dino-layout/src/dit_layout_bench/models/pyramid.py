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

    def __init__(
        self, backbone: DiTBase | None = None, *, out_channels: int = 256
    ) -> None:
        super().__init__()
        if out_channels < 1:
            raise ValueError("out_channels must be positive")
        self.backbone = backbone or DiTBase()
        dim = self.backbone.embed_dim
        # Reduce channels while features are still stride-16. The legacy
        # pyramid upsampled all 768 DiT channels and only then let DINO project
        # them to 256, which was especially expensive for P2.
        self.projections = nn.ModuleList(
            [nn.Conv2d(dim, out_channels, 1) for _ in FEATURE_NAMES]
        )
        self.transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(out_channels, out_channels, 2, 2),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                    nn.ConvTranspose2d(out_channels, out_channels, 2, 2),
                ),
                nn.ConvTranspose2d(out_channels, out_channels, 2, 2),
                nn.Identity(),
                nn.MaxPool2d(2, 2),
            ]
        )
        self.num_channels = [out_channels] * 4
        self.strides = list(FEATURE_STRIDES)

        # TODO: Compare PubLayNet COCO AP-small against the legacy 768-channel
        # pyramid before treating this architecture as the final baseline.

    @staticmethod
    def pad(images: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if images.ndim != 4:
            raise ValueError(f"images must have shape [B,C,H,W], got {tuple(images.shape)}")
        height, width = images.shape[-2:]
        padded_height = math.ceil(height / 32) * 32
        padded_width = math.ceil(width / 32) * 32
        pad_height, pad_width = padded_height - height, padded_width - width
        if mask is None:
            mask = torch.zeros(
                (images.shape[0], height, width), dtype=torch.bool, device=images.device
            )
        elif tuple(mask.shape) != (images.shape[0], height, width):
            raise ValueError(
                "mask must have shape [B,H,W] matching images; "
                f"got {tuple(mask.shape)}"
            )
        else:
            mask = mask.to(device=images.device, dtype=torch.bool)
        images = F.pad(images, (0, pad_width, 0, pad_height))
        mask = F.pad(mask, (0, pad_width, 0, pad_height), value=True)
        return images, mask

    def forward(
        self, images: Tensor, mask: Tensor | None = None
    ) -> tuple[OrderedDict[str, Tensor], OrderedDict[str, Tensor]]:
        images, mask = self.pad(images, mask)
        taps = self.backbone(images)
        features: OrderedDict[str, Tensor] = OrderedDict()
        masks: OrderedDict[str, Tensor] = OrderedDict()
        for name, tap, projection, transform, stride in zip(
            FEATURE_NAMES,
            taps.values(),
            self.projections,
            self.transforms,
            FEATURE_STRIDES,
        ):
            feature = transform(projection(tap))
            expected = (images.shape[-2] // stride, images.shape[-1] // stride)
            if feature.shape[-2:] != expected:
                raise RuntimeError(
                    f"{name} has shape {feature.shape[-2:]}, "
                    f"expected {expected} for stride {stride}"
                )
            features[name] = feature
            resized_mask = F.interpolate(
                mask[:, None].float(), size=expected, mode="nearest"
            )
            masks[name] = resized_mask[:, 0].bool()
        return features, masks
