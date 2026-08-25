"""Detectron2 Cascade R-CNN backend using the shared DiT feature pyramid."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dit_layout_bench.config import RunConfig


@dataclass(frozen=True)
class CascadeAPI:
    torch: Any
    CfgNode: Any
    DetectionCheckpointer: Any
    get_cfg: Any
    register_coco_instances: Any
    DefaultTrainer: Any
    COCOEvaluator: Any
    ShapeSpec: Any
    BACKBONE_REGISTRY: Any
    Backbone: Any


def _load_api() -> CascadeAPI:
    try:
        import torch
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import CfgNode, get_cfg
        from detectron2.data.datasets import register_coco_instances
        from detectron2.engine import DefaultTrainer
        from detectron2.evaluation import COCOEvaluator
        from detectron2.layers import ShapeSpec
        from detectron2.modeling import BACKBONE_REGISTRY, Backbone
    except ImportError as error:
        raise RuntimeError(
            "Cascade R-CNN requires a PyTorch-compatible Detectron2 installation"
        ) from error
    return CascadeAPI(
        torch=torch,
        CfgNode=CfgNode,
        DetectionCheckpointer=DetectionCheckpointer,
        get_cfg=get_cfg,
        register_coco_instances=register_coco_instances,
        DefaultTrainer=DefaultTrainer,
        COCOEvaluator=COCOEvaluator,
        ShapeSpec=ShapeSpec,
        BACKBONE_REGISTRY=BACKBONE_REGISTRY,
        Backbone=Backbone,
    )


def _register_backbone(api: CascadeAPI) -> None:
    from dit_layout_bench.models.dit import DiTBase
    from dit_layout_bench.checkpoint import load_dit_pretrained
    from dit_layout_bench.models.pyramid import (
        FEATURE_NAMES,
        FEATURE_STRIDES,
        DiTFeaturePyramid,
    )

    class DetectronDiT(api.Backbone):
        def __init__(self, cfg) -> None:
            super().__init__()
            encoder = DiTBase(
                drop_path_rate=cfg.MODEL.DIT.DROP_PATH,
                use_checkpoint=cfg.MODEL.DIT.USE_CHECKPOINT,
            )
            if cfg.MODEL.DIT.PRETRAINED:
                report = load_dit_pretrained(encoder, cfg.MODEL.DIT.PRETRAINED)
                print(f"DiT checkpoint: {report.summary()}")
            self.pyramid = DiTFeaturePyramid(
                encoder, out_channels=cfg.MODEL.DIT.PYRAMID_CHANNELS
            )
            self._size_divisibility = self.pyramid.size_divisibility
            self._out_features = list(FEATURE_NAMES)
            self._out_feature_channels = dict.fromkeys(
                FEATURE_NAMES, cfg.MODEL.DIT.PYRAMID_CHANNELS
            )
            self._out_feature_strides = dict(zip(FEATURE_NAMES, FEATURE_STRIDES))

        def forward(self, images):
            features, _ = self.pyramid(images)
            return features

        def output_shape(self):
            return {
                name: api.ShapeSpec(
                    channels=self._out_feature_channels[name],
                    stride=self._out_feature_strides[name],
                )
                for name in self._out_features
            }

    def build_clean_dit_backbone(cfg, input_shape):
        return DetectronDiT(cfg)

    if "build_clean_dit_backbone" not in api.BACKBONE_REGISTRY:
        api.BACKBONE_REGISTRY.register(build_clean_dit_backbone)


def _iterations_per_epoch(config: RunConfig) -> int:
    annotation = config.data_root / "annotations" / "train.json"
    image_count = len(json.loads(annotation.read_text(encoding="utf-8"))["images"])
    return max(1, math.ceil(image_count / config.batch_size))


def _configure_model(cfg: Any, config: RunConfig, cfg_node_type: Any) -> None:
    input_settings = config.input
    cascade = config.detector_settings
    cfg.MODEL.DIT = cfg_node_type()
    cfg.MODEL.DIT.PRETRAINED = (
        str(config.pretrained) if config.pretrained and not config.resume else ""
    )
    cfg.MODEL.DIT.DROP_PATH = config.dit["drop_path"]
    cfg.MODEL.DIT.USE_CHECKPOINT = config.dit["use_checkpoint"]
    cfg.MODEL.DIT.PYRAMID_CHANNELS = config.dit["pyramid_channels"]
    cfg.MODEL.WEIGHTS = str(config.resume) if config.resume else ""
    cfg.MODEL.BACKBONE.NAME = "build_clean_dit_backbone"
    cfg.MODEL.PIXEL_MEAN = [value * 255 for value in input_settings["mean"]]
    cfg.MODEL.PIXEL_STD = [value * 255 for value in input_settings["std"]]
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.ROI_HEADS.NAME = "CascadeROIHeads"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = cascade["roi_batch_size_per_image"]
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = cascade["nms_threshold"]
    cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
    cfg.MODEL.ROI_BOX_HEAD.NUM_CONV = 0
    cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 2
    cfg.MODEL.ROI_BOX_HEAD.FC_DIM = 1024
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = cascade["class_agnostic_bbox"]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[value] for value in cascade["anchor_sizes"]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [cascade["aspect_ratios"]]
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = cascade["rpn_batch_size_per_image"]


def _configure_input(cfg: Any, config: RunConfig) -> None:
    input_settings = config.input
    cfg.INPUT.FORMAT = "RGB"
    cfg.INPUT.MIN_SIZE_TRAIN = tuple(input_settings["short_edge_scales"])
    cfg.INPUT.MAX_SIZE_TRAIN = input_settings["max_long_edge"]
    cfg.INPUT.MIN_SIZE_TEST = max(input_settings["short_edge_scales"])
    cfg.INPUT.MAX_SIZE_TEST = input_settings["max_long_edge"]
    cfg.INPUT.RANDOM_FLIP = input_settings["random_flip"]


def _configure_training(cfg: Any, config: RunConfig) -> None:
    training = config.training
    cascade = config.detector_settings
    iterations_per_epoch = _iterations_per_epoch(config)
    cfg.DATASETS.TRAIN = ("publaynet_train",)
    cfg.DATASETS.TEST = ("publaynet_val",)
    cfg.DATALOADER.NUM_WORKERS = config.num_workers
    cfg.SOLVER.IMS_PER_BATCH = config.batch_size
    cfg.SOLVER.BASE_LR = training["detector_lr"]
    cfg.SOLVER.WEIGHT_DECAY = training["weight_decay"]
    cfg.SOLVER.MAX_ITER = iterations_per_epoch * config.epochs
    cfg.SOLVER.STEPS = tuple(
        max(1, round(cfg.SOLVER.MAX_ITER * fraction))
        for fraction in cascade["lr_steps"]
    )
    cfg.SOLVER.WARMUP_ITERS = min(
        training["warmup_iters"], max(0, iterations_per_epoch - 1)
    )
    cfg.SOLVER.WARMUP_FACTOR = training["warmup_factor"]
    cfg.SOLVER.AMP.ENABLED = config.amp
    cfg.TEST.EVAL_PERIOD = (
        iterations_per_epoch * training["evaluate_every_epochs"]
    )


def _build_cfg(api: CascadeAPI, config: RunConfig, *, for_training: bool):
    cfg = api.get_cfg()
    _configure_model(cfg, config, api.CfgNode)
    _configure_input(cfg, config)
    if for_training:
        _configure_training(cfg, config)
    else:
        cfg.DATASETS.TRAIN = ()
        cfg.DATASETS.TEST = ()
    cfg.OUTPUT_DIR = str(config.output_dir)
    cfg.SEED = config.seed
    cfg.MODEL.DEVICE = config.device
    cfg.freeze()
    return cfg


def _validate_resume_checkpoint(config: RunConfig) -> None:
    if config.resume is None:
        raise ValueError("Cascade R-CNN inference requires --resume")

    from dit_layout_bench.checkpoint import load_detector_checkpoint

    load_detector_checkpoint(
        config.resume,
        expected_pyramid_channels=config.dit["pyramid_channels"],
    )


def _register_data(api: CascadeAPI, root: Path) -> None:
    from detectron2.data import DatasetCatalog

    for split in ("train", "val"):
        name = f"publaynet_{split}"
        if name not in DatasetCatalog.list():
            api.register_coco_instances(
                name,
                {},
                str(root / "annotations" / f"{split}.json"),
                str(root / split),
            )


def _build_trainer(api: CascadeAPI, config: RunConfig):
    import weakref

    training_settings = config.training

    from detectron2.engine import hooks
    from detectron2.engine.train_loop import HookBase
    from dit_layout_bench.paths import RECENT_CHECKPOINT_NAME
    from detectron2.utils.events import EventWriter, get_event_storage
    from dit_layout_bench.tracking import active_tracker

    class MLflowWriter(EventWriter):
        def write(self):
            storage = get_event_storage()
            interval = int(config.tracking["log_every_steps"])
            if storage.iter % interval:
                return
            tracker = active_tracker()
            if tracker is None:
                return
            latest = storage.latest_with_smoothing_hint(window_size=20)
            tracker.log_metrics(
                {f"train/{name}": value for name, (value, _) in latest.items()},
                storage.iter,
            )

        def close(self):
            pass

    class RecentCheckpointHook(HookBase):
        """Publish recent.pth only after the matching validation succeeds."""

        def after_step(self):
            next_iteration = self.trainer.iter + 1
            period = self.trainer.cfg.TEST.EVAL_PERIOD
            should_validate = next_iteration == self.trainer.max_iter or (
                period > 0 and next_iteration % period == 0
            )
            if should_validate:
                self.trainer.checkpointer.save(Path(RECENT_CHECKPOINT_NAME).stem)

    class Trainer(api.DefaultTrainer):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.checkpointer = api.DetectionCheckpointer(
                self.model,
                str(config.weights_dir),
                trainer=weakref.proxy(self),
            )

        def build_hooks(self):
            default_hooks = super().build_hooks()
            default_hooks = [
                hook
                for hook in default_hooks
                if not isinstance(hook, hooks.PeriodicCheckpointer)
            ]
            return default_hooks + [RecentCheckpointHook()]

        def build_writers(self):
            return super().build_writers() + [MLflowWriter()]

        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            return api.COCOEvaluator(
                dataset_name, output_dir=output_folder or cfg.OUTPUT_DIR
            )

        @classmethod
        def build_optimizer(cls, cfg, model):
            from dit_layout_bench.optim import parameter_groups

            return api.torch.optim.AdamW(
                parameter_groups(
                    model,
                    detector_lr=training_settings["detector_lr"],
                    backbone_lr=training_settings["backbone_lr"],
                ),
                weight_decay=training_settings["weight_decay"],
            )

    return Trainer


def train(config: RunConfig) -> None:
    api = _load_api()
    _register_backbone(api)
    _register_data(api, config.data_root)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.weights_dir.mkdir(parents=True, exist_ok=True)
    if config.resume is not None:
        _validate_resume_checkpoint(config)
    cfg = _build_cfg(api, config, for_training=True)
    Trainer = _build_trainer(api, config)
    trainer = Trainer(cfg)
    if config.resume is not None:
        checkpoint = trainer.checkpointer.load(str(config.resume))
        trainer.start_iter = int(checkpoint.get("iteration", -1)) + 1
    trainer.train()


def build_predictor(config: RunConfig, *, score_threshold: float = 0.5):
    """Load a Cascade R-CNN checkpoint once and return a prediction callable."""
    api = _load_api()
    _register_backbone(api)
    _validate_resume_checkpoint(config)
    cfg = _build_cfg(api, config, for_training=False)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = str(config.resume)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.freeze()
    from detectron2.engine import DefaultPredictor
    import numpy as np
    from PIL import Image
    from dit_layout_bench.prediction import prediction_record

    predictor = DefaultPredictor(cfg)

    def predict_image(image_path: Path):
        image_path = Path(image_path)
        with Image.open(image_path) as source:
            rgb = np.asarray(source.convert("RGB"))
        output = predictor(rgb[:, :, ::-1])["instances"].to("cpu")
        return [
            prediction_record(image_path.stem, box, score, int(label) + 1)
            for box, score, label in zip(
                output.pred_boxes.tensor, output.scores, output.pred_classes
            )
        ]

    return predict_image
