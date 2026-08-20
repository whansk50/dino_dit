"""Strict, auditable loading of self-supervised DiT checkpoints."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LoadReport:
    path: str
    sha256: str
    loaded_keys: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]

    def summary(self) -> str:
        return (
            f"path={self.path}, sha256={self.sha256}, loaded={self.loaded_keys}, "
            f"missing={len(self.missing_keys)}, unexpected={len(self.unexpected_keys)}, "
            f"shape_mismatches={len(self.shape_mismatches)}"
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unwrap(checkpoint: object) -> dict[str, Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must contain a mapping")
    state = checkpoint.get("model", checkpoint.get("module", checkpoint))
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint state_dict must be a mapping")
    result: dict[str, Tensor] = {}
    prefixes = ("module.", "backbone.bottom_up.backbone.", "backbone.")
    for key, value in state.items():
        if not isinstance(key, str):
            continue
        if not isinstance(value, Tensor):
            continue
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
                    break
        if key in result:
            raise ValueError(f"Checkpoint keys collide after prefix removal: {key}")
        result[key] = value
    return result


def _resize_absolute_position(position: Tensor, target: Tensor) -> Tensor:
    if position.shape == target.shape:
        return position
    if position.ndim != 3 or target.ndim != 3:
        raise ValueError("Absolute position embeddings must have shape [1,tokens,channels]")
    if position.shape[0] != 1 or target.shape[0] != 1:
        raise ValueError("Absolute position embeddings must have batch dimension 1")
    if position.shape[-1] != target.shape[-1]:
        raise ValueError("Absolute position embedding channel dimensions do not match")
    extra = 1
    source_tokens = position.shape[1] - extra
    target_tokens = target.shape[1] - extra
    source_size = int(source_tokens**0.5)
    target_size = int(target_tokens**0.5)
    if source_size**2 != source_tokens or target_size**2 != target_tokens:
        raise ValueError("Absolute position embedding has a non-square patch grid")
    grid = position[:, extra:].reshape(1, source_size, source_size, position.shape[-1])
    grid = F.interpolate(
        grid.permute(0, 3, 1, 2),
        size=(target_size, target_size),
        mode="bicubic",
        align_corners=False,
    )
    return torch.cat((position[:, :extra], grid.flatten(2).transpose(1, 2)), dim=1)


def load_dit_pretrained(model: nn.Module, path: str | Path) -> LoadReport:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"DiT checkpoint not found: {path}")
    raw = torch.load(path, map_location="cpu")
    state = _unwrap(raw)
    model_state = model.state_dict()
    if "pos_embed" in state and "pos_embed" in model_state:
        state["pos_embed"] = _resize_absolute_position(
            state["pos_embed"], model_state["pos_embed"]
        )

    shape_mismatches = tuple(
        sorted(
            key
            for key in state
            if key in model_state and state[key].shape != model_state[key].shape
        )
    )
    for key in shape_mismatches:
        del state[key]
    incompatible = model.load_state_dict(state, strict=False)
    loaded = set(state).intersection(model_state)
    expected_blocks = {key for key in model_state if key.startswith("blocks.")}
    missing_blocks = expected_blocks.difference(loaded)
    if missing_blocks:
        sample = ", ".join(sorted(missing_blocks)[:5])
        raise RuntimeError(f"DiT transformer checkpoint is incomplete; missing {sample}")
    return LoadReport(
        path=str(path.resolve()),
        sha256=sha256_file(path),
        loaded_keys=len(loaded),
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
        shape_mismatches=shape_mismatches,
    )
