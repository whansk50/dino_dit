"""Clean-room DiT-base encoder with UniLM checkpoint-compatible parameter names.

This module implements the architecture described by DiT/BEiT using standard
PyTorch operations. It does not import the MPViT-derived Detectron2 wrapper in
the upstream object_detection directory.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _drop_path(x: Tensor, probability: float, training: bool) -> Tensor:
    if not 0.0 <= probability < 1.0:
        raise ValueError(f"drop path probability must be in [0, 1), got {probability}")
    if probability == 0.0 or not training:
        return x
    keep = 1.0 - probability
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
    return x.div(keep) * random_tensor


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        return _drop_path(x, self.probability, self.training)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(0.0)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("embedding dimension must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, channels = x.shape
        bias = torch.cat((self.q_bias, torch.zeros_like(self.q_bias), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, bias)
        qkv = qkv.reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        attention = self.attn_drop(attention)
        output = (attention @ v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(output))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, drop_path: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.gamma_1 = nn.Parameter(torch.full((dim,), 0.1))
        self.gamma_2 = nn.Parameter(torch.full((dim,), 0.1))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        return x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, image_size: tuple[int, int], patch_size: int, dim: int) -> None:
        super().__init__()
        if any(size % patch_size for size in image_size):
            raise ValueError("pretraining image dimensions must be divisible by patch size")
        self.img_size = image_size
        self.patch_size = (patch_size, patch_size)
        self.patch_shape = (image_size[0] // patch_size, image_size[1] // patch_size)
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]
        self.num_patches_h, self.num_patches_w = self.patch_shape
        self.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, images: Tensor) -> tuple[Tensor, tuple[int, int]]:
        features = self.proj(images)
        height, width = features.shape[-2:]
        return features.flatten(2).transpose(1, 2), (height, width)


class DiTBase(nn.Module):
    """DiT-base/16 encoder that exposes four intermediate transformer layers."""

    def __init__(
        self,
        image_size: tuple[int, int] = (224, 224),
        out_indices: Sequence[int] = (3, 5, 7, 11),
        drop_path_rate: float = 0.1,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")
        if not out_indices or len(set(out_indices)) != len(out_indices):
            raise ValueError("out_indices must be non-empty and unique")
        if min(out_indices) < 0 or max(out_indices) >= 12:
            raise ValueError("out_indices must refer to transformer layers 0..11")
        self.embed_dim = 768
        self.out_indices = tuple(out_indices)
        self._out_index_set = set(out_indices)
        self.patch_embed = PatchEmbed(image_size, 16, self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, self.embed_dim)
        )
        self.pos_drop = nn.Dropout(0.0)
        rates = torch.linspace(0, drop_path_rate, 12).tolist()
        self.blocks = nn.ModuleList(
            [Block(self.embed_dim, 12, 4.0, rates[index]) for index in range(12)]
        )
        self.use_checkpoint = use_checkpoint
        self._initialize()

    def _initialize(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for index, block in enumerate(self.blocks, start=1):
            block.attn.proj.weight.data.div_(math.sqrt(2.0 * index))
            block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * index))

    def _position_tokens(self, height: int, width: int) -> Tensor:
        cls_position = self.pos_embed[:, :1]
        patch_position = self.pos_embed[:, 1:].reshape(
            1,
            self.patch_embed.num_patches_h,
            self.patch_embed.num_patches_w,
            self.embed_dim,
        )
        patch_position = patch_position.permute(0, 3, 1, 2)
        patch_position = F.interpolate(
            patch_position, size=(height, width), mode="bicubic", align_corners=False
        )
        return torch.cat((cls_position, patch_position.flatten(2).transpose(1, 2)), dim=1)

    def forward_features(self, images: Tensor) -> OrderedDict[str, Tensor]:
        tokens, (height, width) = self.patch_embed(images)
        cls = self.cls_token.expand(images.shape[0], -1, -1)
        positions = self._position_tokens(height, width)
        tokens = self.pos_drop(torch.cat((cls, tokens), dim=1) + positions)
        outputs: OrderedDict[str, Tensor] = OrderedDict()
        for index, block in enumerate(self.blocks):
            if self.use_checkpoint and self.training and tokens.requires_grad:
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
            if index in self._out_index_set:
                feature = tokens[:, 1:].transpose(1, 2).reshape(
                    images.shape[0], self.embed_dim, height, width
                )
                outputs[f"layer{index}"] = feature.contiguous()
        if len(outputs) != len(self.out_indices):
            raise RuntimeError(
                f"Expected {len(self.out_indices)} DiT features, got {len(outputs)}"
            )
        return outputs

    def forward(self, images: Tensor) -> OrderedDict[str, Tensor]:
        return self.forward_features(images)
