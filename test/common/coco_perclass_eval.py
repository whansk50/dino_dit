"""
COCO 포맷 결과 json에 대해 전체 AP 지표와 class별 AP를 계산한다.

detectron2 COCOEvaluator와 dit_dino/datasets/coco_eval.py의 CocoEvaluator.dump()가
남기는 coco_instances_results.json을 그대로 입력으로 받는다. pycocotools만 있으면 되고
detectron2/DINO 어느 쪽도 import하지 않으므로, 한쪽 학습 환경이 깨져 있어도 독립적으로
결과를 채점할 수 있다.

사용례:
    python coco_perclass_eval.py --gt ../data/publaynet/val.json \\
        --dt ../baseline_dit_cascade/output/inference/coco_instances_results.json \\
        --name "DiT+CascadeRCNN" --out results_baseline.json
"""
import argparse
import contextlib
import io
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# pycocotools COCOeval(iouType='bbox').stats의 고정 순서
METRIC_NAMES = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]


def _run_eval(coco_gt, coco_dt, cat_ids=None):
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    if cat_ids is not None:
        coco_eval.params.catIds = cat_ids
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    return dict(zip(METRIC_NAMES, coco_eval.stats.tolist()))


def evaluate(gt_path, dt_path):
    coco_gt = COCO(gt_path)
    if len(coco_gt.getAnnIds()) == 0:
        raise ValueError(f"'{gt_path}'에 annotation이 없다 - GT 경로를 확인하라.")

    with open(dt_path) as f:
        dt = json.load(f)
    if not dt:
        raise ValueError(f"'{dt_path}'가 비어 있다 - 예측 결과가 없다.")
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(dt_path)

    overall = _run_eval(coco_gt, coco_dt)

    per_class = {}
    for cat_id, cat in zip(coco_gt.getCatIds(), coco_gt.loadCats(coco_gt.getCatIds())):
        per_class[cat["name"]] = _run_eval(coco_gt, coco_dt, cat_ids=[cat_id])

    return {"overall": overall, "per_class": per_class}


def format_table(result, name=None):
    lines = []
    header = ["model"] + METRIC_NAMES[:6]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    row = [name or "overall"] + [f"{result['overall'][m]:.4f}" for m in METRIC_NAMES[:6]]
    lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("| class | " + " | ".join(METRIC_NAMES[:3]) + " |")
    lines.append("|---|---|---|---|")
    for cls_name, stats in result["per_class"].items():
        lines.append(
            "| " + cls_name + " | " + " | ".join(f"{stats[m]:.4f}" for m in METRIC_NAMES[:3]) + " |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", required=True, help="GT annotation json (예: ../data/publaynet/val.json)")
    parser.add_argument("--dt", required=True, help="예측 결과 json (coco_instances_results.json)")
    parser.add_argument("--name", default=None, help="표/json에 쓸 모델 이름 (예: DiT+CascadeRCNN)")
    parser.add_argument("--out", default=None, help="결과를 저장할 json 경로 (생략 시 stdout만)")
    args = parser.parse_args()

    result = evaluate(args.gt, args.dt)
    if args.name:
        result["name"] = args.name

    print(format_table(result, args.name))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
