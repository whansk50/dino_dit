"""Adapter from the shared DiT pyramid to the official IDEA DINO runner."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dit_layout_bench.paths import CONFIG_ROOT, DINO_ROOT, require_path
from dit_layout_bench.config import RunConfig, per_process_batch_size
from dit_layout_bench.tracking import process_world_size


DINO_CONFIG = CONFIG_ROOT / "dino_publaynet.py"
DINO_MAIN_MODULE = "_dit_layout_bench_dino_main"

_COCO_BBOX_METRIC_NAMES = (
    "mAP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
)


def _evaluation_metrics(
    stats: dict[str, Any], *, prefix: str = "eval"
) -> dict[str, Any]:
    """Flatten DINO's COCO bbox summary into MLflow scalar metrics."""
    metrics = {
        f"{prefix}/{name}": value
        for name, value in stats.items()
        if name != "coco_eval_bbox"
    }
    bbox_summary = stats.get("coco_eval_bbox")
    if bbox_summary is not None:
        for name, value in zip(_COCO_BBOX_METRIC_NAMES, bbox_summary):
            # COCOeval reports ratios while per-category AP below is a percentage.
            metrics[f"{prefix}/bbox_{name}"] = float(value) * 100.0
    return metrics


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
            self.pyramid = DiTFeaturePyramid(
                encoder, out_channels=args.dit_pyramid_channels
            )
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


def _effective_dino_values(
    config: RunConfig, *, for_training: bool = True
) -> dict[str, Any]:
    """Translate the common YAML schema to DINO's SLConfig names."""
    input_settings = config.input
    training = config.training
    detector = config.detector_settings
    return {
        "batch_size": (
            per_process_batch_size(config.batch_size, process_world_size())
            if for_training
            else config.batch_size
        ),
        "epochs": config.epochs,
        "lr": training["detector_lr"],
        "lr_backbone": training["backbone_lr"],
        "weight_decay": training["weight_decay"],
        "warmup_iters": training["warmup_iters"],
        "warmup_factor": training["warmup_factor"],
        "evaluate_every_epochs": training["evaluate_every_epochs"],
        "dit_pretrained": (
            str(config.pretrained) if config.pretrained and not config.resume else None
        ),
        "dit_drop_path": config.dit["drop_path"],
        "dit_use_checkpoint": config.dit["use_checkpoint"],
        "dit_pyramid_channels": config.dit["pyramid_channels"],
        "data_aug_scales": list(input_settings["short_edge_scales"]),
        "data_aug_max_size": input_settings["max_long_edge"],
        "data_norm_mean": list(input_settings["mean"]),
        "data_norm_std": list(input_settings["std"]),
        "data_random_flip": input_settings["random_flip"],
        "optimizer": detector["optimizer"],
        "fused_optimizer": detector["fused_optimizer"],
        "adam_betas": list(detector["adam_betas"]),
        "scheduler": detector["scheduler"],
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
        "dn_box_noise_scale": detector["dn_box_noise_scale"],
        "dn_label_noise_ratio": detector["dn_label_noise_ratio"],
        "aux_loss": detector["aux_loss"],
        "lr_drop": detector["lr_drop_epoch"],
        "lr_drop_list": list(detector["lr_drop_epochs"]),
        "clip_max_norm": detector["clip_max_norm"],
        "set_cost_class": detector["set_cost_class"],
        "set_cost_bbox": detector["set_cost_bbox"],
        "set_cost_giou": detector["set_cost_giou"],
        "cls_loss_coef": detector["cls_loss_coef"],
        "bbox_loss_coef": detector["bbox_loss_coef"],
        "giou_loss_coef": detector["giou_loss_coef"],
        "focal_alpha": detector["focal_alpha"],
        "use_ema": detector["use_ema"],
        "ema_decay": detector["ema_decay"],
        "ema_epoch": detector["ema_epoch"],
        "weights_dir": str(config.weights_dir),
        "data_pin_memory": training["pin_memory"],
        "data_non_blocking": training["non_blocking"],
        "data_persistent_workers": training["persistent_workers"],
        "data_prefetch_factor": training["prefetch_factor"],
        "tracking_log_every_steps": config.tracking["log_every_steps"],
        "ddp_gradient_as_bucket_view": detector["ddp_gradient_as_bucket_view"],
        "ddp_static_graph": detector["ddp_static_graph"],
    }


def _option(name: str, value: Any) -> str:
    if value is None:
        serialized = "None"
    elif isinstance(value, (list, tuple)):
        # DINO's legacy DictAction collapses a one-item comma-separated value
        # to a scalar, so retain brackets for empty and singleton sequences.
        serialized = repr(list(value)) if len(value) <= 1 else ",".join(map(str, value))
    else:
        serialized = str(value)
    return f"{name}={serialized}"


def _runner_arguments(config: RunConfig) -> list[str]:
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
    arguments.append("--options")
    arguments.extend(
        _option(name, value)
        for name, value in _effective_dino_values(config).items()
    )
    return arguments


@dataclass(frozen=True)
class DinoIntegration:
    """Explicit project callbacks consumed by the vendored DINO runner."""

    build_backbone: Callable[..., Any]
    build_dataset: Callable[..., Any]
    build_parameter_groups: Callable[..., Any]
    on_evaluation: Callable[..., Any]
    on_train_step: Callable[..., Any]


def _build_dino_integration(config: RunConfig) -> DinoIntegration:
    from dit_layout_bench.evaluation import print_per_category_ap
    from dit_layout_bench.optim import parameter_groups
    from dit_layout_bench.tracking import active_tracker

    def optimizer_groups(args, model):
        return parameter_groups(
            model, detector_lr=args.lr, backbone_lr=args.lr_backbone
        )

    def evaluation_callback(stats, evaluator, runner_args):
        tracker = active_tracker()
        epoch = getattr(runner_args, "_tracking_eval_epoch", 0)
        prefix = getattr(runner_args, "_tracking_eval_prefix", "eval")
        if tracker is not None:
            tracker.log_metrics(
                _evaluation_metrics(stats, prefix=prefix), step=epoch
            )
        if "bbox" in evaluator.coco_eval:
            categories = print_per_category_ap(evaluator.coco_eval["bbox"])
            if tracker is not None:
                tracker.log_metrics(
                    {f"{prefix}/AP_{key}": value for key, value in categories.items()},
                    step=epoch,
                )

    def train_step_callback(metrics, step):
        tracker = active_tracker()
        interval = int(config.tracking["log_every_steps"])
        if tracker is None or step % interval != 0:
            return
        tracker.log_metrics({f"train/{key}": value for key, value in metrics.items()}, step)

    return DinoIntegration(
        build_backbone=build_dino_backbone,
        build_dataset=_build_publaynet,
        build_parameter_groups=optimizer_groups,
        on_evaluation=evaluation_callback,
        on_train_step=train_step_callback,
    )


def train(config: RunConfig) -> None:
    """Run DINO with explicit project integration callbacks."""
    dino_main = _load_dino_main()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.weights_dir.mkdir(parents=True, exist_ok=True)
    parser = dino_main.get_args_parser()
    args = parser.parse_args(_runner_arguments(config))
    try:
        dino_main.main(args, integration=_build_dino_integration(config))
    finally:
        # This runner is imported by the shared CLI rather than executed as a
        # standalone script, so explicitly release the DDP process group.
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def build_predictor(config: RunConfig, *, score_threshold: float = 0.5):
    """Load a DINO checkpoint once and return an image prediction callable."""
    _activate_dino()
    import torch
    import torchvision.transforms.functional as vision
    from PIL import Image
    from util.misc import nested_tensor_from_tensor_list
    from util.slconfig import SLConfig
    from dit_layout_bench.prediction import prediction_record

    raw = SLConfig.fromfile(str(DINO_CONFIG))
    values = raw._cfg_dict.to_dict()
    values.update(
        _effective_dino_values(config, for_training=False), device=config.device
    )
    args = SimpleNamespace(**values)
    dino_main = _load_dino_main()
    model, _, postprocessors = dino_main.build_model_main(
        args, backbone_builder=build_dino_backbone
    )
    from dit_layout_bench.checkpoint import load_detector_checkpoint

    checkpoint = load_detector_checkpoint(
        config.resume,
        expected_pyramid_channels=config.dit["pyramid_channels"],
    )
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.to(config.device).eval()

    def predict_image(image_path: Path):
        image_path = Path(image_path)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        original_width, original_height = image.size
        scale = min(
            max(config.input["short_edge_scales"]) / min(image.size),
            config.input["max_long_edge"] / max(image.size),
        )
        resized = vision.resize(
            image,
            [round(original_height * scale), round(original_width * scale)],
        )
        tensor = vision.to_tensor(resized)
        tensor = vision.normalize(
            tensor, list(config.input["mean"]), list(config.input["std"])
        )
        samples = nested_tensor_from_tensor_list([tensor]).to(config.device)
        with torch.inference_mode():
            outputs = model(samples)
            target_sizes = torch.tensor(
                [[original_height, original_width]], device=config.device
            )
            result = postprocessors["bbox"](outputs, target_sizes)[0]
        keep = result["scores"] >= score_threshold
        records = []
        for box, score, label in zip(
            result["boxes"][keep].cpu(),
            result["scores"][keep].cpu(),
            result["labels"][keep].cpu(),
        ):
            category_id = int(label)
            if 1 <= category_id <= 5:
                records.append(
                    prediction_record(image_path.stem, box, score, category_id)
                )
        return records

    return predict_image
