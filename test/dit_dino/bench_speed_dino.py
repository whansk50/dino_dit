"""
DiT + DINO 추론 속도/VRAM/파라미터 수 측정.

측정 구간의 정의 (baseline_dit_cascade/bench_speed_d2.py와 동일한 기준으로 나눈다):
  - preprocess_ms  : 연속된 두 배치를 받아오는 사이의 시간 (Dataset.__getitem__의 resize/normalize/
                      to-tensor와 collate_fn의 NestedTensor 생성). 정확한 측정을 위해 num_workers=0으로
                      강제한다.
  - inference_ms   : model(samples) 호출 1회의 wall time
                      (DiT backbone -> input_proj -> deformable encoder/decoder -> class/bbox head).
  - postprocess_ms : postprocessors['bbox'](outputs, orig_target_sizes) + 결과를 CPU로 옮기는 시간.

baseline_dit_cascade/bench_speed_d2.py와 동일한 스키마의 json을 출력하므로
common/collect_results.py가 두 실험을 동일하게 처리한다.
"""
import argparse
import json
import time

import torch
from torch.utils.data import DataLoader, SequentialSampler

import util.misc as utils
from datasets import build_dataset
from main import build_model_main, get_args_parser
from util.slconfig import SLConfig


def count_params(model):
    # "backbone"은 Joiner(DiTBackbone, position_embedding)까지, "detector"는 input_proj +
    # deformable encoder/decoder + class/bbox head 등 나머지 전부.
    # baseline_dit_cascade/bench_speed_d2.py의 count_params()와 동일한 경계 정의를 쓴다.
    backbone = sum(p.numel() for p in model.backbone.parameters())
    total = sum(p.numel() for p in model.parameters())
    return {"backbone": backbone, "detector": total - backbone, "total": total}


def _parse_options(options):
    """`KEY=VALUE` 문자열 리스트를 main.py의 --options(DictAction)와 동일한 규칙으로 dict화한다."""
    if not options:
        return None
    from util.slconfig import DictAction
    parsed = {}
    for kv in options:
        key, val = kv.split('=', maxsplit=1)
        val = [DictAction._parse_int_float_bool(v) for v in val.split(',')]
        parsed[key] = val[0] if len(val) == 1 else val
    return parsed


def build_eval_model(config_file, resume, publaynet_root, options):
    parser = get_args_parser()
    # dataset_file/coco_path는 cfg 파일이 아니라 여기서 argparse 값으로 직접 채운다 -
    # main.py의 cfg 병합 로직은 cfg 파일이 argparse 인자와 같은 키를 정의하면
    # "Key ... can used by args only"로 에러를 내므로, DINO_4scale_dit.py는
    # 이 두 값을 일부러 설정하지 않는다 (config/DINO/DINO_4scale_dit.py 주석 참고).
    args = parser.parse_args(["-c", config_file, "--dataset_file", "publaynet", "--coco_path", publaynet_root])
    args.options = _parse_options(options)

    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg_dict = cfg._cfg_dict.to_dict()
    for k, v in cfg_dict.items():
        setattr(args, k, v)
    if not getattr(args, "use_ema", None):
        args.use_ema = False

    model, criterion, postprocessors = build_model_main(args)
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    if resume:
        checkpoint = torch.load(resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    dataset_val = build_dataset(image_set="val", args=args)
    data_loader = DataLoader(
        dataset_val, 1, sampler=SequentialSampler(dataset_val),
        drop_last=False, collate_fn=utils.collate_fn, num_workers=0,
    )
    return args, model, postprocessors, data_loader, device


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-file", "-c", required=True)
    parser.add_argument("--resume", default=None, help="학습된 DINO 체크포인트 (.pth)")
    parser.add_argument("--publaynet-root", default="../data/publaynet")
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", default=None)
    parser.add_argument("--options", nargs="*", default=None, help="key=value 형태의 cfg 오버라이드")
    args = parser.parse_args()

    _, model, postprocessors, data_loader, device = build_eval_model(
        args.config_file, args.resume, args.publaynet_root, args.options
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    preprocess_ms, inference_ms, postprocess_ms = [], [], []
    data_iter = iter(data_loader)
    prev_t = time.perf_counter()

    with torch.no_grad():
        for i in range(args.warmup + args.num_images):
            samples, targets = next(data_iter)
            samples = samples.to(device)
            t0 = time.perf_counter()

            outputs = model(samples)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()

            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
            results = postprocessors["bbox"](outputs, orig_target_sizes)
            _ = [{k: v.cpu() for k, v in r.items()} for r in results]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t2 = time.perf_counter()

            if i >= args.warmup:
                preprocess_ms.append((t0 - prev_t) * 1000)
                inference_ms.append((t1 - t0) * 1000)
                postprocess_ms.append((t2 - t1) * 1000)
            prev_t = t2

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    ms_per_page = avg(preprocess_ms) + avg(inference_ms) + avg(postprocess_ms)
    result = {
        "framework": "dino",
        "preprocess_ms": avg(preprocess_ms),
        "inference_ms": avg(inference_ms),
        "postprocess_ms": avg(postprocess_ms),
        "ms_per_page": ms_per_page,
        "pages_per_sec": 1000.0 / ms_per_page if ms_per_page > 0 else 0.0,
        "inference_peak_vram_mb": (
            torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0 if device.type == "cuda" else 0.0
        ),
        "params": count_params(model),
    }

    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
