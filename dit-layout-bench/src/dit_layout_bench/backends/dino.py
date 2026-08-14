"""Adapter from the shared DiT pyramid to the official IDEA DINO runner."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from dit_layout_bench.paths import DINO_ROOT, PROJECT_ROOT, require_path
from dit_layout_bench.spec import MAX_LONG_EDGE, PIXEL_MEAN_01, PIXEL_STD_01, SHORT_EDGE_SCALES


def _activate_dino() -> None:
    root = str(require_path(DINO_ROOT, "DINO source tree"))
    if root not in sys.path:
        sys.path.insert(0, root)


def build_dino_backbone(args):
    """Build the interface expected by DINO's detector constructor."""
    _activate_dino()
    from torch import nn
    from util.misc import NestedTensor
    from models.dino.position_encoding import build_position_encoding
    from dit_layout_bench.checkpoint import load_dit_pretrained
    from dit_layout_bench.models.dit import DiTBase
    from dit_layout_bench.models.pyramid import DiTFeaturePyramid

    class Adapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            encoder = DiTBase(
                drop_path_rate=args.dit_drop_path,
                use_checkpoint=args.dit_use_checkpoint,
            )
            if args.dit_pretrained:
                report = load_dit_pretrained(encoder, args.dit_pretrained)
                print(f"DiT checkpoint: {report}")
            self.pyramid = DiTFeaturePyramid(encoder)
            self.num_channels = self.pyramid.num_channels

        def forward(self, samples: NestedTensor):
            features, masks = self.pyramid(samples.tensors, samples.mask)
            return {
                str(index): NestedTensor(feature, masks[name])
                for index, (name, feature) in enumerate(features.items())
            }

    class Joiner(nn.Sequential):
        def __init__(self, backbone, position_embedding) -> None:
            super().__init__(backbone, position_embedding)
            self.num_channels = backbone.num_channels

        def forward(self, samples):
            features = self[0](samples)
            nested = list(features.values())
            positions = [self[1](feature).to(feature.tensors.dtype) for feature in nested]
            return nested, positions

    return Joiner(Adapter(), build_position_encoding(args))


def _build_publaynet(image_set, args):
    _activate_dino()
    from datasets.coco import CocoDetection
    import datasets.transforms as transforms

    root = Path(args.coco_path)
    split = "train" if image_set in {"train", "train_reg"} else "val"
    image_dir = root / split
    annotation = root / "annotations" / f"{split}.json"
    if not image_dir.is_dir() or not annotation.is_file():
        raise FileNotFoundError(
            f"Expected PubLayNet {split} data at {image_dir} and {annotation}"
        )
    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(list(PIXEL_MEAN_01), list(PIXEL_STD_01)),
        ]
    )
    if split == "train":
        pipeline = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResize(list(SHORT_EDGE_SCALES), max_size=MAX_LONG_EDGE),
                normalize,
            ]
        )
    else:
        pipeline = transforms.Compose(
            [transforms.RandomResize([max(SHORT_EDGE_SCALES)], max_size=MAX_LONG_EDGE), normalize]
        )
    return CocoDetection(image_dir, annotation, transforms=pipeline, return_masks=False)


def run(config, *, evaluate: bool = False) -> None:
    """Run official DINO training/evaluation after installing scoped adapters."""
    _activate_dino()
    import main as dino_main
    import models.dino.dino as dino_model
    from dit_layout_bench.evaluation import print_per_category_ap

    dino_model.build_backbone = build_dino_backbone
    dino_main.build_dataset = _build_publaynet
    from dit_layout_bench.optim import parameter_groups

    dino_main.get_param_dict = lambda args, model: parameter_groups(
        model, detector_lr=args.lr, backbone_lr=args.lr_backbone
    )
    original_evaluate = dino_main.evaluate

    def evaluate_with_classes(*args, **kwargs):
        result = original_evaluate(*args, **kwargs)
        evaluator = result[1]
        if "bbox" in evaluator.coco_eval:
            print_per_category_ap(evaluator.coco_eval["bbox"])
        return result

    dino_main.evaluate = evaluate_with_classes
    config.output_dir.mkdir(parents=True, exist_ok=True)
    parser = dino_main.get_args_parser()
    arguments = [
        "--config_file", str(PROJECT_ROOT / "configs" / "dino_publaynet.py"),
        "--dataset_file", "publaynet",
        "--coco_path", str(config.data_root),
        "--output_dir", str(config.output_dir),
        "--device", config.device,
        "--seed", str(config.seed),
        "--num_workers", str(config.num_workers),
    ]
    if config.amp:
        arguments.append("--amp")
    if config.resume:
        arguments.extend(("--resume", str(config.resume)))
    if evaluate:
        arguments.append("--eval")
    arguments.extend(
        [
            "--options",
            f"batch_size={config.batch_size}",
            f"epochs={config.epochs}",
            f"dit_pretrained={config.pretrained}" if config.pretrained else "dit_pretrained=None",
        ]
    )
    args = parser.parse_args(arguments)
    dino_main.main(args)


def train(config) -> None:
    run(config, evaluate=False)


def evaluate(config) -> None:
    if config.resume is None:
        raise ValueError("DINO evaluation requires --resume")
    run(config, evaluate=True)


def predict(config, image_path: Path, *, score_threshold: float = 0.5):
    _activate_dino()
    import torch
    import torchvision.transforms.functional as vision
    from PIL import Image
    from util.misc import nested_tensor_from_tensor_list
    from util.slconfig import SLConfig
    import models.dino.dino as dino_model
    from main import build_model_main

    raw = SLConfig.fromfile(str(PROJECT_ROOT / "configs" / "dino_publaynet.py"))
    values = raw._cfg_dict.to_dict()
    values.update(
        device=config.device,
        dit_pretrained=str(config.pretrained) if config.pretrained else None,
    )
    args = SimpleNamespace(**values)
    dino_model.build_backbone = build_dino_backbone
    model, _, postprocessors = build_model_main(args)
    checkpoint = torch.load(config.resume, map_location="cpu")
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.to(config.device).eval()

    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size
    scale = min(max(SHORT_EDGE_SCALES) / min(image.size), MAX_LONG_EDGE / max(image.size))
    resized = vision.resize(image, [round(original_height * scale), round(original_width * scale)])
    tensor = vision.to_tensor(resized)
    tensor = vision.normalize(tensor, list(PIXEL_MEAN_01), list(PIXEL_STD_01))
    samples = nested_tensor_from_tensor_list([tensor]).to(config.device)
    with torch.no_grad():
        outputs = model(samples)
        target_sizes = torch.tensor([[original_height, original_width]], device=config.device)
        result = postprocessors["bbox"](outputs, target_sizes)[0]
    keep = result["scores"] >= score_threshold
    records = []
    for box, score, label in zip(
        result["boxes"][keep].cpu(), result["scores"][keep].cpu(), result["labels"][keep].cpu()
    ):
        category_id = int(label)
        if 1 <= category_id <= 5:
            records.append(
                {
                    "image_id": image_path.stem,
                    "box_xyxy": box.tolist(),
                    "score": float(score),
                    "category_id": category_id,
                }
            )
    return records
