# DiT -> DINO 백본 어댑터.
#
# beit.py의 BEiT는 patch_size에 따라 self.fpn1..fpn4에 서로 다른 up/down-sample 연산을 채워
# 4개 tap(layer3/5/7/11, 전부 stride16)을 서로 다른 stride로 바꾼다. patch_size=16 분기는
# 4/8/16/32를 만들고, patch_size=8 분기는 4/8/16/32를 "다른 조합"으로 만든다(beit.py:485-498).
# patch_size=8 분기의 연산(2x ConvTranspose / Identity / MaxPool2 / MaxPool4)을 patch_size=16
# 그대로의 backbone(사전학습 conv 유지)에 붙이면, 입력 stride가 16이므로 결과는 8/16/32/64가
# 된다 — 새 연산을 발명하지 않고 beit.py 안의 기존 조합을 재배치한 것뿐이다.
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.misc import NestedTensor

from .beit import BEiT

# beit.py forward_features가 사용하는 tap 이름 -> 목표 stride
_LAYER_TO_STRIDE = {"layer3": 8, "layer5": 16, "layer7": 32, "layer11": 64}
_MAX_STRIDE = 64


class DiTMultiScale(BEiT):
    """BEiT(patch16)을 그대로 쓰되 fpn1..fpn4만 8/16/32/64용 연산으로 교체한다.

    patch_embed(conv kernel=16, stride=16)은 그대로 둬야 사전학습 conv 가중치와 shape이
    맞는다. forward_features는 BEiT 원본을 그대로 상속해 쓴다 — cls token 결합, rel_pos_bias,
    gradient checkpointing 로직에 손대지 않는다.
    """

    def __init__(self, *args, **kwargs):
        kwargs.pop("patch_size", None)
        super().__init__(*args, patch_size=16, **kwargs)
        embed_dim = self.embed_dim
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )  # stride16 -> 8
        self.fpn2 = nn.Identity()  # stride16
        self.fpn3 = nn.MaxPool2d(kernel_size=2, stride=2)  # stride16 -> 32
        self.fpn4 = nn.MaxPool2d(kernel_size=4, stride=4)  # stride16 -> 64


class DiTBackbone(nn.Module):
    """DINO의 backbone 인터페이스(NestedTensor -> Dict[str, NestedTensor])를 따르는 DiT 래퍼."""

    def __init__(self, dit: DiTMultiScale, strides: List[int]):
        super().__init__()
        self.dit = dit
        self.strides = strides
        self.num_channels = [dit.embed_dim] * len(strides)
        self._keep_layers = [name for name, s in _LAYER_TO_STRIDE.items() if s in strides]

    def forward(self, tensor_list: NestedTensor) -> Dict[str, NestedTensor]:
        x = tensor_list.tensors
        mask = tensor_list.mask
        b, c, h, w = x.shape

        pad_h = (math.ceil(h / _MAX_STRIDE) * _MAX_STRIDE) - h
        pad_w = (math.ceil(w / _MAX_STRIDE) * _MAX_STRIDE) - w
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            mask = F.pad(mask, (0, pad_w, 0, pad_h), value=True)

        feat_out = self.dit.forward_features(x)

        out: Dict[str, NestedTensor] = {}
        for name in self._keep_layers:
            feat = feat_out[name]
            feat_mask = F.interpolate(mask[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(feat, feat_mask)
        return out


def build_dit_backbone(args) -> DiTBackbone:
    strides = list(getattr(args, "dit_pyramid_strides", [8, 16, 32, 64]))
    unknown = [s for s in strides if s not in _LAYER_TO_STRIDE.values()]
    assert not unknown, f"dit_pyramid_strides {strides} 중 {unknown}는 지원하지 않음 (8/16/32/64만 가능)"

    img_size = getattr(args, "dit_img_size", [224, 224])
    pos_type = getattr(args, "dit_pos_type", "abs")
    drop_path = getattr(args, "dit_drop_path", 0.1)
    use_checkpoint = getattr(args, "dit_use_checkpoint", True)
    out_features = getattr(args, "dit_out_layers", ["layer3", "layer5", "layer7", "layer11"])

    pos_kwargs = {
        "abs": dict(use_abs_pos_emb=True),
        "shared_rel": dict(use_shared_rel_pos_bias=True),
        "rel": dict(use_rel_pos_bias=True),
    }[pos_type]

    dit = DiTMultiScale(
        img_size=img_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=drop_path,
        init_values=0.1,
        use_checkpoint=use_checkpoint,
        out_features=out_features,
        **pos_kwargs,
    )

    pretrained_path = getattr(args, "dit_pretrained_path", None)
    if pretrained_path:
        from .load_dit_weights import load_dit_pretrained
        load_dit_pretrained(dit, pretrained_path)

    return DiTBackbone(dit, strides)
