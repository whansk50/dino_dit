# DiT(BEiT-family) 사전학습 weight 로더.
#
# dit/object_detection/ditod/mycheckpointer.py는 재사용하지 않는다 — detectron2의
# DetectionCheckpointer를 상속하고 fvcore 내부 API(_IncompatibleKeys, _strip_prefix_if_present)에
# 묶여 있으며, 키 경로도 "backbone.bottom_up.backbone." prefix로 detectron2 모듈 트리에
# 하드코딩되어 있어 여기서는 동작하지 않는다. 여기서 필요한 pos_embed bicubic 보간 로직은
# BEiT/DeiT(microsoft/unilm)에서 표준화된 방식을 참고해 새로 작성했다.
import torch
import torch.nn.functional as F


def _unwrap_state_dict(ckpt):
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "module" in ckpt:
        state_dict = ckpt["module"]
    else:
        state_dict = ckpt
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def _interpolate_pos_embed(state_dict, model):
    if "pos_embed" not in state_dict or model.pos_embed is None:
        return
    ckpt_pos_embed = state_dict["pos_embed"]
    embedding_size = ckpt_pos_embed.shape[-1]
    num_extra_tokens = model.pos_embed.shape[1] - model.patch_embed.num_patches
    orig_size = int((ckpt_pos_embed.shape[1] - num_extra_tokens) ** 0.5)
    new_size_h = model.patch_embed.num_patches_h
    new_size_w = model.patch_embed.num_patches_w

    if orig_size == new_size_h and orig_size == new_size_w:
        return

    extra_tokens = ckpt_pos_embed[:, :num_extra_tokens]
    pos_tokens = ckpt_pos_embed[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = F.interpolate(
        pos_tokens, size=(new_size_h, new_size_w), mode="bicubic", align_corners=False
    )
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    state_dict["pos_embed"] = torch.cat((extra_tokens, pos_tokens), dim=1)


def load_dit_pretrained(model, ckpt_path: str, logger=None):
    log = logger.info if logger is not None else print

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = _unwrap_state_dict(ckpt)
    state_dict = {k: v for k, v in state_dict.items() if "relative_position_index" not in k}

    _interpolate_pos_embed(state_dict, model)

    incompatible = model.load_state_dict(state_dict, strict=False)

    loaded_block_keys = [k for k in state_dict if k.startswith("blocks.")]
    assert loaded_block_keys, (
        f"'{ckpt_path}'에서 transformer block 파라미터를 하나도 찾지 못했다 - "
        "체크포인트 형식이 예상과 다르거나 경로가 잘못됐을 가능성이 높다."
    )
    missing_blocks = [k for k in incompatible.missing_keys if k.startswith("blocks.")]
    assert not missing_blocks, (
        f"blocks.* 파라미터 {len(missing_blocks)}개가 로드되지 않고 랜덤 초기화로 남았다: "
        f"{missing_blocks[:5]}..."
    )

    log(f"[load_dit_pretrained] loaded from {ckpt_path}")
    log(f"[load_dit_pretrained] missing_keys ({len(incompatible.missing_keys)}): {incompatible.missing_keys}")
    log(f"[load_dit_pretrained] unexpected_keys ({len(incompatible.unexpected_keys)}): {incompatible.unexpected_keys}")
    return incompatible
