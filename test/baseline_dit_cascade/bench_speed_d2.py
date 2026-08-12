"""
DiT + Cascade R-CNN(detectron2) 추론 속도/VRAM/파라미터 수 측정.

측정 구간의 정의 (dit_dino/bench_speed_dino.py와 동일한 기준으로 나눈다):
  - preprocess_ms  : 연속된 두 배치를 받아오는 사이의 시간 (디코드/리사이즈/to-tensor/collate,
                      DataLoader worker 쪽에서 일어나는 일). 정확한 측정을 위해 NUM_WORKERS=0으로 강제한다.
  - inference_ms   : model(batch) 호출 1회의 wall time. detectron2 GeneralizedRCNN은 정규화·
                      backbone·RPN·ROI heads·box/NMS 후처리가 전부 forward() 안에서 일어나므로
                      이 구간에 함께 포함된다 (detectron2 자체의 구조적 특성 - 별도 API로 분리되지 않는다).
  - postprocess_ms : forward()가 반환한 Instances를 CPU로 옮기는 데 걸리는 시간.

dit_dino/bench_speed_dino.py와 동일한 스키마의 json을 출력하므로 common/collect_results.py가
두 실험을 동일하게 처리한다.
"""
import argparse
import json
import time

import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling import build_model

from ditod import add_vit_config


def count_params(model):
    # "backbone"은 FPN(Feature Pyramid 포함 VIT_Backbone)까지, "detector"는 RPN+ROI heads 등
    # 나머지 전부. dit_dino 쪽 count_params()와 동일한 경계 정의를 쓴다.
    backbone = sum(p.numel() for p in model.backbone.parameters())
    total = sum(p.numel() for p in model.parameters())
    return {"backbone": backbone, "detector": total - backbone, "total": total}


def build_eval_model(config_file, weights, publaynet_root, opts):
    register_coco_instances(
        "publaynet_val", {}, f"{publaynet_root}/val.json", f"{publaynet_root}/val"
    )

    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    if weights:
        cfg.MODEL.WEIGHTS = weights
    cfg.DATASETS.TEST = ("publaynet_val",)
    cfg.DATALOADER.NUM_WORKERS = 0  # 배치 간 시간차로 preprocess를 재는 동안 prefetch가 섞이지 않게
    cfg.freeze()

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    return cfg, model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", default=None, help="cfg.MODEL.WEIGHTS를 덮어쓸 체크포인트 경로")
    parser.add_argument("--publaynet-root", default="../data/publaynet")
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", default=None)
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="cfg 오버라이드, 예: MODEL.WEIGHTS x.pth")
    args = parser.parse_args()

    cfg, model = build_eval_model(args.config_file, args.weights, args.publaynet_root, args.opts)
    device = torch.device(cfg.MODEL.DEVICE)
    data_loader = build_detection_test_loader(cfg, "publaynet_val")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    preprocess_ms, inference_ms, postprocess_ms = [], [], []
    data_iter = iter(data_loader)
    prev_t = time.perf_counter()

    with torch.no_grad():
        for i in range(args.warmup + args.num_images):
            batch = next(data_iter)
            t0 = time.perf_counter()

            outputs = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()

            _ = [o["instances"].to("cpu") for o in outputs]
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
        "framework": "detectron2",
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
