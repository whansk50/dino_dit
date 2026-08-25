"""Inference command."""

import json
import sys

from dit_layout_bench.backends import get_backend
from dit_layout_bench.arguments import parser_for, validated_config
from dit_layout_bench.prediction import collect_image_paths, save_visualization
from dit_layout_bench.tracking import MLflowTracker


def main(argv=None) -> None:
    parser = parser_for("inference")
    args = parser.parse_args(argv)
    config = validated_config(parser, args, require_data=False, require_resume=True)
    try:
        image_paths = collect_image_paths(args.image)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    threshold = (
        args.score_threshold
        if args.score_threshold is not None
        else config.score_threshold()
    )
    visualization_dir = args.visualization_dir or config.output_dir / "inference"
    backend = get_backend(config.detector)
    with MLflowTracker(config, "inference") as tracker:
        predictor = backend.build_predictor(config, score_threshold=threshold)
        predictions = []
        for index, image_path in enumerate(image_paths, start=1):
            print(
                f"[{index}/{len(image_paths)}] {image_path.name}",
                file=sys.stderr,
            )
            image_predictions = predictor(image_path)
            predictions.extend(image_predictions)
            if args.visualize:
                save_visualization(
                    image_path,
                    image_predictions,
                    visualization_dir / f"{image_path.name}.prediction.png",
                )
        tracker.log_metrics(
            {"images": len(image_paths), "predictions": len(predictions)}
        )
    payload = json.dumps(predictions, indent=2, ensure_ascii=False)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(payload + "\n", encoding="utf-8")
        print(f"Predictions: {args.json_output}", file=sys.stderr)
    else:
        print(payload)
    if args.visualize:
        print(f"Visualizations: {visualization_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
