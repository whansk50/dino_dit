"""
DiT -> DINO 백본 어댑터 스모크 테스트 (계획서 §7 Step 1).

확인하는 것:
  1. 4개 feature level(P3/P4/P5/P6)이 실제로 stride 8/16/32/64, 768채널로 나오는가.
     - 64의 배수인 입력(패딩이 발생하지 않는 경우)과, 64의 배수가 아닌 입력(내부 패딩이
       발생하는 경우) 둘 다 검증한다 - 후자가 dit_backbone.py의 패딩 로직을 실제로 건드린다.
  2. 각 level의 mask shape이 feature의 공간 크기와 일치하는가.
  3. (--weights 지정 시) 사전학습 weight 로드 후 missing/unexpected 키 요약이 정상적으로
     나오고, transformer block 파라미터가 실제로 로드됐는지.

GPU 없이도(CPU) 동작한다. 실제 학습 설정과 동일하게 depth=12 전체를 그대로 쓴다 - 블록 수를
줄이면 검증 대상 자체가 실제 설정과 달라져 스모크 테스트의 의미가 없어진다. 배치 1, 224 기반
patch_embed라 CPU에서도 수 초~수십 초 내에 끝난다.
"""
import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # test/dit_dino를 루트로

# 일부러 models 패키지를 거치지 않는다 (models/__init__.py는 MSDeformAttn CUDA 확장을 요구하는
# DINO 트랜스포머까지 import한다) - 이 스모크 테스트는 DiT 백본 어댑터만 검증하면 되므로,
# 그 빌드가 아직 안 됐거나 실패했어도 이 테스트는 독립적으로 돌아가야 한다.
from dit.dit_backbone import DiTBackbone, DiTMultiScale, _LAYER_TO_STRIDE
from util.misc import NestedTensor

ALL_STRIDES = [8, 16, 32, 64]


def expected_shapes(h, w):
    pad_h = math.ceil(h / 64) * 64
    pad_w = math.ceil(w / 64) * 64
    hp, wp = pad_h // 16, pad_w // 16
    return {
        8: (hp * 2, wp * 2),
        16: (hp, wp),
        32: (hp // 2, wp // 2),
        64: (hp // 4, wp // 4),
    }


def build_test_backbone():
    dit = DiTMultiScale(
        img_size=[224, 224], embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, drop_path_rate=0.1, init_values=0.1, use_checkpoint=False,
        use_abs_pos_emb=True, out_features=["layer3", "layer5", "layer7", "layer11"],
    )
    return DiTBackbone(dit, strides=ALL_STRIDES)


def check_shapes(backbone, h, w):
    x = torch.randn(1, 3, h, w)
    mask = torch.zeros(1, h, w, dtype=torch.bool)  # 패딩 없는 실제 이미지 영역 = 전부 False
    nested = NestedTensor(x, mask)

    expected = expected_shapes(h, w)
    with torch.no_grad():
        out = backbone(nested)

    assert len(out) == 4, f"4개 level이 나와야 하는데 {len(out)}개"
    print(f"\ninput=({h},{w}) -> padded to multiple of 64 -> feature shapes:")
    for name, nt in out.items():
        stride = _LAYER_TO_STRIDE[name]
        feat_h, feat_w = nt.tensors.shape[-2:]
        exp_h, exp_w = expected[stride]
        print(f"  {name} (stride {stride}): feat={tuple(nt.tensors.shape)} mask={tuple(nt.mask.shape)}")
        assert nt.tensors.shape[1] == 768, f"{name}: 채널 수가 768이 아니라 {nt.tensors.shape[1]}"
        assert (feat_h, feat_w) == (exp_h, exp_w), (
            f"{name}(stride {stride}): 기대 크기 {(exp_h, exp_w)}, 실제 {(feat_h, feat_w)}"
        )
        assert nt.mask.shape[-2:] == nt.tensors.shape[-2:], f"{name}: mask shape이 feature와 다르다"


def check_weights(weights_path):
    from dit.load_dit_weights import load_dit_pretrained

    dit = DiTMultiScale(
        img_size=[224, 224], embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, drop_path_rate=0.1, init_values=0.1, use_checkpoint=False,
        use_abs_pos_emb=True, out_features=["layer3", "layer5", "layer7", "layer11"],
    )
    print(f"\nLoading pretrained weights from {weights_path} ...")
    load_dit_pretrained(dit, weights_path)
    print("OK: block 파라미터가 정상적으로 로드됐다 (missing_keys에 blocks.*가 없음이 위에서 assert됨).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=None, help="dit-base-224-p16-500k-62d53a.pth 경로 (생략 가능)")
    args = parser.parse_args()

    backbone = build_test_backbone()
    backbone.eval()

    print("=== shape/stride 확인: 64의 배수 입력 (패딩 없음) ===")
    check_shapes(backbone, 832, 1344)

    print("\n=== shape/stride 확인: 64의 배수가 아닌 입력 (내부 패딩 발생) ===")
    check_shapes(backbone, 800, 1333)

    if args.weights:
        print("\n=== 사전학습 weight 로드 확인 ===")
        check_weights(args.weights)
    else:
        print("\n(--weights 미지정 - weight 로드 검증은 건너뜀)")

    print("\n모든 확인 통과.")


if __name__ == "__main__":
    main()
