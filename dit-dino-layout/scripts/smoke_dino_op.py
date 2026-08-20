"""Small forward/backward check for the vendored CUDA extension."""

import torch

from dit_layout_bench.backends.dino import _activate_dino


_activate_dino()
from models.dino.ops.functions.ms_deform_attn_func import MSDeformAttnFunction


spatial_shapes = torch.tensor([[2, 2]], device="cuda", dtype=torch.long)
level_start_index = torch.tensor([0], device="cuda", dtype=torch.long)
value = torch.randn(1, 4, 1, 4, device="cuda", requires_grad=True)
locations = torch.rand(1, 2, 1, 1, 2, 2, device="cuda", requires_grad=True)
weights = torch.rand(1, 2, 1, 1, 2, device="cuda", requires_grad=True)
output = MSDeformAttnFunction.apply(
    value, spatial_shapes, level_start_index, locations, weights, 1
)
output.sum().backward()
print(f"DINO op OK: output={tuple(output.shape)}, backward={value.grad is not None}")
