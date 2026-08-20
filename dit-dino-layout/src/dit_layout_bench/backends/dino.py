"""Adapter from the shared DiT pyramid to the official IDEA DINO runner."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from importlib import util as importlib_util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

from dit_layout_bench.paths import CONFIG_ROOT, DINO_ROOT, require_path


DINO_CONFIG = CONFIG_ROOT / "dino_publaynet.py"
DINO_MAIN_MODULE = "_dit_layout_bench_dino_main"


def _activate_dino() -> None:
    root = str(require_path(DINO_ROOT, "DINO source tree"))
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_dino_main():
    """Load DINO's generic ``main.py`` under an unambiguous module name."""
    _activate_dino()
    if DINO_MAIN_MODULE in sys.modules:
        return sys.modules[DINO_MAIN_MODULE]
    main_path = require_path(DINO_ROOT / "main.py", "DINO entrypoint")
    spec = importlib_util.spec_from_file_location(DINO_MAIN_MODULE, main_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load DINO entrypoint: {main_path}")
    module = importlib_util.module_from_spec(spec)
    sys.modules[DINO_MAIN_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(DINO_MAIN_MODULE, None)
        raise
    return module


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
                print(f"DiT checkpoint: {report.summary()}")
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
            transforms.Normalize(list(args.data_norm_mean), list(args.data_norm_std)),
        ]
    )
    if split == "train":
        augmentations = []
        if args.data_random_flip == "horizontal":
            augmentations.append(transforms.RandomHorizontalFlip())
        augmentations.extend(
            [
                transforms.RandomResize(
                    list(args.data_aug_scales), max_size=args.data_aug_max_size
                ),
                normalize,
            ]
        )
        pipeline = transforms.Compose(augmentations)
    else:
        pipeline = transforms.Compose(
            [
                transforms.RandomResize(
                    [max(args.data_aug_scales)], max_size=args.data_aug_max_size
                ),
                normalize,
            ]
        )
    return CocoDetection(image_dir, annotation, transforms=pipeline, return_masks=False)


def _effective_dino_values(config) -> dict[str, Any]:
    """Translate the common YAML schema to DINO's SLConfig names."""
    input_settings = config.input
    training = config.training
    detector = config.detector_settings
    return {
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "lr": training["detector_lr"],
        "lr_backbone": training["backbone_lr"],
        "weight_decay": training["weight_decay"],
        "dit_pretrained": str(config.pretrained) if config.pretrained else None,
        "dit_drop_path": config.dit["drop_path"],
        "dit_use_checkpoint": config.dit["use_checkpoint"],
        "data_aug_scales": list(input_settings["short_edge_scales"]),
        "data_aug_max_size": input_settings["max_long_edge"],
        "data_norm_mean": list(input_settings["mean"]),
        "data_norm_std": list(input_settings["std"]),
        "data_random_flip": input_settings["random_flip"],
        "hidden_dim": detector["hidden_dim"],
        "num_feature_levels": detector["num_feature_levels"],
        "enc_layers": detector["enc_layers"],
        "dec_layers": detector["dec_layers"],
        "nheads": detector["nheads"],
        "dim_feedforward": detector["dim_feedforward"],
        "dropout": detector["dropout"],
        "enc_n_points": detector["enc_n_points"],
        "dec_n_points": detector["dec_n_points"],
        "num_queries": detector["num_queries"],
        "num_select": detector["num_select"],
        "use_dn": detector["use_dn"],
        "dn_number": detector["dn_number"],
        "lr_drop": detector["lr_drop_epoch"],
        "clip_max_norm": detector["clip_max_norm"],
        "cls_loss_coef": detector["cls_loss_coef"],
        "bbox_loss_coef": detector["bbox_loss_coef"],
        "giou_loss_coef": detector["giou_loss_coef"],
        "focal_alpha": detector["focal_alpha"],
        "save_checkpoint_interval": training["checkpoint_every_epochs"],
    }


def _option(name: str, value: Any) -> str:
    if value is None:
        serialized = "None"
    elif isinstance(value, (list, tuple)):
        serialized = ",".join(map(str, value))
    else:
        serialized = str(value)
    return f"{name}={serialized}"


def _runner_arguments(config, *, evaluate: bool) -> list[str]:
    arguments = [
        "--config_file",
        str(DINO_CONFIG),
        "--dataset_file",
        "publaynet",
        "--coco_path",
        str(config.data_root),
        "--output_dir",
        str(config.output_dir),
        "--device",
        config.device,
        "--seed",
        str(config.seed),
        "--num_workers",
        str(config.num_workers),
    ]
    if config.amp:
        arguments.append("--amp")
    if config.resume:
        arguments.extend(("--resume", str(config.resume)))
    if evaluate:
        arguments.append("--eval")
    arguments.append("--options")
    arguments.extend(
        _option(name, value)
        for name, value in _effective_dino_values(config).items()
    )
    return arguments


@contextmanager
def _patched_dino_runtime(dino_main, dino_model) -> Iterator[None]:
    """Install DINO integration hooks and restore upstream globals afterwards."""
    from dit_layout_bench.evaluation import print_per_category_ap
    from dit_layout_bench.optim import parameter_groups
    from dit_layout_bench.tracking import active_tracker
    import engine as dino_engine

    originals = {
        "build_backbone": dino_model.build_backbone,
        "build_dataset": dino_main.build_dataset,
        "get_param_dict": dino_main.get_param_dict,
        "evaluate": dino_main.evaluate,
        "train_step_callback": dino_engine.TRAIN_STEP_CALLBACK,
    }

    def optimizer_groups(args, model):
        return parameter_groups(
            model, detector_lr=args.lr, backbone_lr=args.lr_backbone
        )

    def evaluate_with_classes(*args, **kwargs):
        result = originals["evaluate"](*args, **kwargs)
        evaluator = result[1]
        tracker = active_tracker()
        if tracker is not None:
            tracker.log_metrics({f"eval/{key}": value for key, value in result[0].items()})
        if "bbox" in evaluator.coco_eval:
            categories = print_per_category_ap(evaluator.coco_eval["bbox"])
            if tracker is not None:
                tracker.log_metrics({f"eval/AP_{key}": value for key, value in categories.items()})
        return result

    def train_step_callback(metrics, step):
        tracker = active_tracker()
        if tracker is None or step % config_tracking_interval() != 0:
            return
        tracker.log_metrics({f"train/{key}": value for key, value in metrics.items()}, step)

    def config_tracking_interval():
        tracker = active_tracker()
        return int(tracker.config.tracking["log_every_steps"]) if tracker else 1

    dino_model.build_backbone = build_dino_backbone
    dino_main.build_dataset = _build_publaynet
    dino_main.get_param_dict = optimizer_groups
    dino_main.evaluate = evaluate_with_classes
    dino_engine.TRAIN_STEP_CALLBACK = train_step_callback
    try:
        yield
    finally:
        dino_model.build_backbone = originals["build_backbone"]
        dino_main.build_dataset = originals["build_dataset"]
        dino_main.get_param_dict = originals["get_param_dict"]
        dino_main.evaluate = originals["evaluate"]
        dino_engine.TRAIN_STEP_CALLBACK = originals["train_step_callback"]


def run(config, *, evaluate: bool = False) -> None:
    """Run official DINO training/evaluation after installing scoped adapters."""
    _activate_dino()
    dino_main = _load_dino_main()
    import models.dino.dino as dino_model

    config.output_dir.mkdir(parents=True, exist_ok=True)
    parser = dino_main.get_args_parser()
    args = parser.parse_args(_runner_arguments(config, evaluate=evaluate))
    with _patched_dino_runtime(dino_main, dino_model):
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
    from dit_layout_bench.prediction import prediction_record

    raw = SLConfig.fromfile(str(DINO_CONFIG))
    values = raw._cfg_dict.to_dict()
    values.update(_effective_dino_values(config), device=config.device)
    args = SimpleNamespace(**values)
    dino_main = _load_dino_main()
    original_backbone = dino_model.build_backbone
    dino_model.build_backbone = build_dino_backbone
    try:
        model, _, postprocessors = dino_main.build_model_main(args)
    finally:
        dino_model.build_backbone = original_backbone
    checkpoint = torch.load(config.resume, map_location="cpu")
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.to(config.device).eval()

    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size
    scale = min(
        max(config.input["short_edge_scales"]) / min(image.size),
        config.input["max_long_edge"] / max(image.size),
    )
    resized = vision.resize(image, [round(original_height * scale), round(original_width * scale)])
    tensor = vision.to_tensor(resized)
    tensor = vision.normalize(
        tensor, list(config.input["mean"]), list(config.input["std"])
    )
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
            records.append(prediction_record(image_path.stem, box, score, category_id))
    return records
