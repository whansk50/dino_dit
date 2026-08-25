from collections import OrderedDict
import unittest

import torch
from torch import nn
import torch.nn.functional as F

from dit_layout_bench.checkpoint import validate_reduced_pyramid_checkpoint
from dit_layout_bench.models.dit import Attention, _canonical_gradient_layout
from dit_layout_bench.models.pyramid import DiTFeaturePyramid


class TinyBackbone(nn.Module):
    embed_dim = 8

    def forward(self, images):
        height, width = images.shape[-2] // 16, images.shape[-1] // 16
        feature = torch.ones(images.shape[0], self.embed_dim, height, width)
        return OrderedDict((f"layer{index}", feature) for index in range(4))


class OptimizationTests(unittest.TestCase):
    def test_singleton_gradient_uses_canonical_ddp_bucket_strides(self):
        gradient = torch.empty_strided((1, 1, 8), (64, 8, 1))
        canonical = _canonical_gradient_layout(gradient)
        self.assertEqual(canonical.stride(), (8, 8, 1))

    def test_sdpa_attention_matches_materialized_reference_on_cpu(self):
        torch.manual_seed(7)
        module = Attention(dim=32, num_heads=4).eval()
        inputs = torch.randn(2, 11, 32)

        actual = module(inputs)
        bias = torch.cat(
            (module.q_bias, torch.zeros_like(module.q_bias), module.v_bias)
        )
        qkv = F.linear(inputs, module.qkv.weight, bias)
        qkv = qkv.reshape(2, 11, 3, 4, 8)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        weights = ((query * module.scale) @ key.transpose(-2, -1)).softmax(-1)
        reference = (weights @ value).transpose(1, 2).reshape(2, 11, 32)
        reference = module.proj(reference)

        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)

    def test_reduced_pyramid_projects_before_upsampling(self):
        pyramid = DiTFeaturePyramid(TinyBackbone(), out_channels=4).eval()
        features, masks = pyramid(torch.zeros(1, 3, 64, 96))

        self.assertEqual(
            {name: tuple(value.shape) for name, value in features.items()},
            {
                "p2": (1, 4, 16, 24),
                "p3": (1, 4, 8, 12),
                "p4": (1, 4, 4, 6),
                "p5": (1, 4, 2, 3),
            },
        )
        self.assertEqual(list(masks), ["p2", "p3", "p4", "p5"])
        self.assertEqual(pyramid.projections[0].in_channels, 8)
        self.assertEqual(pyramid.projections[0].out_channels, 4)

    def test_legacy_pyramid_checkpoint_is_rejected_with_clear_error(self):
        legacy = {
            "model": {
                "backbone.0.pyramid.transforms.0.0.weight": torch.empty(768, 768, 2, 2)
            }
        }
        with self.assertRaisesRegex(RuntimeError, "legacy 768-channel FPN"):
            validate_reduced_pyramid_checkpoint(legacy, expected_channels=256)

    def test_reduced_pyramid_checkpoint_width_is_validated(self):
        checkpoint = {
            "model": {
                "backbone.0.pyramid.projections.0.weight": torch.empty(256, 768, 1, 1)
            }
        }
        validate_reduced_pyramid_checkpoint(checkpoint, expected_channels=256)
        with self.assertRaisesRegex(RuntimeError, "checkpoint=256, config=128"):
            validate_reduced_pyramid_checkpoint(checkpoint, expected_channels=128)


if __name__ == "__main__":
    unittest.main()
