"""
baseline_dit_cascade / dit_dino 각각의 결과 json을 모아 계획서 §13의 두 markdown 표를 만든다.

입력은 전부 다른 스크립트가 이미 만들어 둔 json이다 - 이 스크립트 자체는 detectron2/DINO를
import하지 않으므로 한쪽 학습 환경이 깨져 있어도 이미 나온 결과만으로 표를 조립할 수 있다.

    eval json   : coco_perclass_eval.py --out <path> 의 출력
    speed json  : bench_speed_d2.py / bench_speed_dino.py --out <path> 의 출력
                  {"ms_per_page", "pages_per_sec", "inference_peak_vram_mb",
                   "params": {"backbone", "detector", "total"}, ...}

사용례:
    python collect_results.py \\
        --baseline-name "DiT + Cascade R-CNN" \\
        --baseline-eval  results_baseline.json --baseline-speed speed_baseline.json \\
        --exp-name "DiT + DINO" \\
        --exp-eval results_dino.json --exp-speed speed_dino.json \\
        --out benchmark_table.md
"""
import argparse
import json

OVERALL_METRICS = ["AP", "AP50", "AP75", "APs", "APm", "APl"]


def _load(path):
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def build_overall_table(rows):
    header = ["Model"] + OVERALL_METRICS + ["FPS", "VRAM(MB)"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, ev, sp in rows:
        vals = [f"{ev['overall'][m]:.4f}" if ev else "-" for m in OVERALL_METRICS]
        fps = f"{sp['pages_per_sec']:.2f}" if sp else "-"
        vram = f"{sp['inference_peak_vram_mb']:.0f}" if sp else "-"
        lines.append("| " + " | ".join([name] + vals + [fps, vram]) + " |")
    return "\n".join(lines)


def build_perclass_table(rows):
    class_names = []
    for _, ev, _ in rows:
        if ev:
            for cls_name in ev["per_class"]:
                if cls_name not in class_names:
                    class_names.append(cls_name)

    header = ["Model"] + class_names
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, ev, _ in rows:
        vals = []
        for cls_name in class_names:
            if ev and cls_name in ev["per_class"]:
                vals.append(f"{ev['per_class'][cls_name]['AP']:.4f}")
            else:
                vals.append("-")
        lines.append("| " + " | ".join([name] + vals) + " |")
    return "\n".join(lines)


def build_params_table(rows):
    header = ["Model", "Backbone params", "Detector params", "Total params"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, _, sp in rows:
        if sp and "params" in sp:
            p = sp["params"]
            vals = [f"{p['backbone']:,}", f"{p['detector']:,}", f"{p['total']:,}"]
        else:
            vals = ["-", "-", "-"]
        lines.append("| " + " | ".join([name] + vals) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-name", default="DiT + Cascade R-CNN")
    parser.add_argument("--baseline-eval", default=None)
    parser.add_argument("--baseline-speed", default=None)
    parser.add_argument("--exp-name", default="DiT + DINO")
    parser.add_argument("--exp-eval", default=None)
    parser.add_argument("--exp-speed", default=None)
    parser.add_argument("--out", default=None, help="markdown을 저장할 경로 (생략 시 stdout만)")
    args = parser.parse_args()

    rows = [
        (args.baseline_name, _load(args.baseline_eval), _load(args.baseline_speed)),
        (args.exp_name, _load(args.exp_eval), _load(args.exp_speed)),
    ]

    sections = [
        "## Benchmark 결과표 (계획서 §13)\n",
        build_overall_table(rows),
        "\n### Class별 AP\n",
        build_perclass_table(rows),
        "\n### Parameter Count\n",
        build_params_table(rows),
    ]
    output = "\n".join(sections)

    print(output)
    if args.out:
        with open(args.out, "w") as f:
            f.write(output + "\n")
        print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
