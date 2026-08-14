"""Detectron2 Cascade R-CNN backend using the shared DiT feature pyramid."""

from __future__ import annotations

from pathlib import Path

from dit_layout_bench.spec import MAX_LONG_EDGE, PIXEL_MEAN_255, PIXEL_STD_255, SHORT_EDGE_SCALES


def _imports():
    try:
        import torch
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import get_cfg
        from detectron2.data import build_detection_train_loader
        from detectron2.data.datasets import register_coco_instances
        from detectron2.engine import DefaultTrainer
        from detectron2.evaluation import COCOEvaluator
        from detectron2.layers import ShapeSpec
        from detectron2.modeling import BACKBONE_REGISTRY, Backbone
    except ImportError as error:
        raise RuntimeError(
            "Cascade R-CNN requires a PyTorch-compatible Detectron2 installation"
        ) from error
    return locals()


def _register_backbone(api) -> None:
    torch = api["torch"]
    Backbone = api["Backbone"]
    ShapeSpec = api["ShapeSpec"]
    registry = api["BACKBONE_REGISTRY"]
    from dit_layout_bench.models.dit import DiTBase
    from dit_layout_bench.models.pyramid import DiTFeaturePyramid, FEATURE_NAMES, FEATURE_STRIDES
    from dit_layout_bench.checkpoint import load_dit_pretrained

    class DetectronDiT(Backbone):
        def __init__(self, cfg) -> None:
            super().__init__()
            encoder = DiTBase(
                drop_path_rate=cfg.MODEL.DIT.DROP_PATH,
                use_checkpoint=cfg.MODEL.DIT.USE_CHECKPOINT,
            )
            if cfg.MODEL.DIT.PRETRAINED:
                report = load_dit_pretrained(encoder, cfg.MODEL.DIT.PRETRAINED)
                print(f"DiT checkpoint: {report}")
            self.pyramid = DiTFeaturePyramid(encoder)
            self._size_divisibility = self.pyramid.size_divisibility
            self._out_features = list(FEATURE_NAMES)
            self._out_feature_channels = dict.fromkeys(FEATURE_NAMES, encoder.embed_dim)
            self._out_feature_strides = dict(zip(FEATURE_NAMES, FEATURE_STRIDES))

        def forward(self, images):
            features, _ = self.pyramid(images)
            return features

        def output_shape(self):
            return {
                name: ShapeSpec(
                    channels=self._out_feature_channels[name],
                    stride=self._out_feature_strides[name],
                )
                for name in self._out_features
            }

    def build_clean_dit_backbone(cfg, input_shape):
        return DetectronDiT(cfg)

    if "build_clean_dit_backbone" not in registry:
        registry.register(build_clean_dit_backbone)


def _configuration(api, config):
    from detectron2.config import CfgNode as CN

    cfg = api["get_cfg"]()
    cfg.MODEL.DIT = CN()
    cfg.MODEL.DIT.PRETRAINED = str(config.pretrained) if config.pretrained else ""
    cfg.MODEL.DIT.DROP_PATH = 0.1
    cfg.MODEL.DIT.USE_CHECKPOINT = True
    cfg.MODEL.WEIGHTS = str(config.resume) if config.resume else ""
    cfg.MODEL.BACKBONE.NAME = "build_clean_dit_backbone"
    cfg.MODEL.PIXEL_MEAN = list(PIXEL_MEAN_255)
    cfg.MODEL.PIXEL_STD = list(PIXEL_STD_255)
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.ROI_HEADS.NAME = "CascadeROIHeads"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    cfg.INPUT.FORMAT = "RGB"
    cfg.INPUT.MIN_SIZE_TRAIN = SHORT_EDGE_SCALES
    cfg.INPUT.MAX_SIZE_TRAIN = MAX_LONG_EDGE
    cfg.INPUT.MIN_SIZE_TEST = max(SHORT_EDGE_SCALES)
    cfg.INPUT.MAX_SIZE_TEST = MAX_LONG_EDGE
    cfg.INPUT.RANDOM_FLIP = "horizontal"
    cfg.DATASETS.TRAIN = ("publaynet_train",)
    cfg.DATASETS.TEST = ("publaynet_val",)
    cfg.DATALOADER.NUM_WORKERS = config.num_workers
    cfg.SOLVER.IMS_PER_BATCH = config.batch_size
    cfg.SOLVER.BASE_LR = 1e-4
    import json
    import math

    annotation = config.data_root / "annotations" / "train.json"
    image_count = (
        len(json.loads(annotation.read_text(encoding="utf-8"))["images"])
        if annotation.is_file()
        else config.batch_size
    )
    iterations_per_epoch = math.ceil(image_count / config.batch_size)
    cfg.SOLVER.MAX_ITER = iterations_per_epoch * config.epochs
    cfg.SOLVER.CHECKPOINT_PERIOD = iterations_per_epoch
    cfg.SOLVER.STEPS = (
        max(1, round(cfg.SOLVER.MAX_ITER * 0.75)),
        max(1, round(cfg.SOLVER.MAX_ITER * 0.9)),
    )
    cfg.SOLVER.WARMUP_ITERS = min(1000, max(0, iterations_per_epoch - 1))
    cfg.SOLVER.AMP.ENABLED = config.amp
    cfg.TEST.EVAL_PERIOD = iterations_per_epoch
    cfg.OUTPUT_DIR = str(config.output_dir)
    cfg.SEED = config.seed
    cfg.MODEL.DEVICE = config.device
    cfg.freeze()
    return cfg


def _register_data(api, root: Path) -> None:
    register = api["register_coco_instances"]
    from detectron2.data import DatasetCatalog

    for split in ("train", "val"):
        name = f"publaynet_{split}"
        if name not in DatasetCatalog.list():
            register(
                name,
                {},
                str(root / "annotations" / f"{split}.json"),
                str(root / split),
            )


def _trainer(api):
    torch = api["torch"]
    DefaultTrainer = api["DefaultTrainer"]
    COCOEvaluator = api["COCOEvaluator"]

    class Trainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            return COCOEvaluator(dataset_name, output_dir=output_folder or cfg.OUTPUT_DIR)

        @classmethod
        def build_optimizer(cls, cfg, model):
            from dit_layout_bench.optim import parameter_groups

            return torch.optim.AdamW(
                parameter_groups(model, detector_lr=1e-4, backbone_lr=1e-5),
                weight_decay=0.05,
            )

    return Trainer


def run(config, *, evaluate: bool = False) -> None:
    api = _imports()
    _register_backbone(api)
    _register_data(api, config.data_root)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _configuration(api, config)
    Trainer = _trainer(api)
    if evaluate:
        if config.resume is None:
            raise ValueError("Cascade R-CNN evaluation requires --resume")
        model = Trainer.build_model(cfg)
        api["DetectionCheckpointer"](model).load(str(config.resume))
        Trainer.test(cfg, model)
        return
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=config.resume is not None)
    trainer.train()


def train(config) -> None:
    run(config, evaluate=False)


def evaluate(config) -> None:
    run(config, evaluate=True)


def predict(config, image_path: Path, *, score_threshold: float = 0.5):
    api = _imports()
    _register_backbone(api)
    cfg = _configuration(api, config)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = str(config.resume)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.freeze()
    from detectron2.engine import DefaultPredictor
    import numpy as np
    from PIL import Image

    predictor = DefaultPredictor(cfg)
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    output = predictor(rgb[:, :, ::-1])["instances"].to("cpu")
    return [
        {
            "image_id": image_path.stem,
            "box_xyxy": box.tolist(),
            "score": float(score),
            "category_id": int(label) + 1,
        }
        for box, score, label in zip(
            output.pred_boxes.tensor, output.scores, output.pred_classes
        )
    ]
