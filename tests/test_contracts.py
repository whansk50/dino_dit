import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from dit_layout_bench.config import RunConfig, load_settings
from dit_layout_bench.arguments import build_config, parser_for, validated_config
from dit_layout_bench.checkpoint import load_dit_pretrained, safe_torch_load
from dit_layout_bench.backends.dino import (
    _activate_dino,
    _build_dino_integration,
    _effective_dino_values,
    _evaluation_metrics,
    _training_epoch_metrics,
    _runner_arguments,
)
from dit_layout_bench.backends.cascade_rcnn import (
    _cascade_mlflow_metrics,
    _should_save_periodic_checkpoint,
    _solver_steps,
)
from dit_layout_bench.launcher import LocalLaunch, _run_process, parse_cuda_devices
from dit_layout_bench.runtime import distributed_session
from dit_layout_bench.data import validate_publaynet
from dit_layout_bench.optim import parameter_groups
from dit_layout_bench.prediction import (
    collect_image_paths,
    prediction_record,
    save_visualization,
)
from dit_layout_bench.spec import category_id_to_train_label, train_label_to_category_id
from dit_layout_bench.tracking import MLflowTracker, process_rank, process_world_size
from scripts.publaynet_subset import create_publaynet_subset
from scripts.validation_runtime import (
    PROJECT_ROOT,
    SOURCE_ROOT,
    execution_mode,
    expected_distributed_tag,
    project_environment,
)
from scripts.validate_cascade_training import (
    _validation_settings as cascade_validation_settings,
)
from scripts.validate_dino_training import (
    _validation_settings as dino_validation_settings,
)

FIXTURE = Path(__file__).parent / "fixtures" / "publaynet"


def _write_publaynet_fixture(root: Path) -> None:
    categories = [
        {"id": 1, "name": "Text"},
        {"id": 2, "name": "Title"},
        {"id": 3, "name": "List"},
        {"id": 4, "name": "Table"},
        {"id": 5, "name": "Figure"},
    ]
    (root / "annotations").mkdir(parents=True)
    for index, split in enumerate(("train", "val"), start=1):
        (root / split).mkdir()
        document = {
            "images": [{"id": index, "file_name": f"{split}.jpg"}],
            "annotations": [],
            "categories": categories,
        }
        (root / "annotations" / f"{split}.json").write_text(
            json.dumps(document), encoding="utf-8"
        )


class ContractTests(unittest.TestCase):
    def test_cascade_lr_steps_stay_inside_short_validation_run(self):
        self.assertEqual(_solver_steps(1, (0.75, 0.9)), ())
        self.assertEqual(_solver_steps(2, (0.75, 0.9)), (1,))
        self.assertEqual(_solver_steps(4, (0.75, 0.9)), (3,))

    def test_cascade_mlflow_uses_current_raw_metric_and_namespace(self):
        metrics = _cascade_mlflow_metrics(
            {
                "total_loss": (1.25, 19),
                "bbox/AP": (42.0, 19),
                "bbox/AP50": (61.0, 9),
            },
            current_iteration=19,
        )
        self.assertEqual(
            metrics,
            {"train/total_loss": 1.25, "eval/bbox/AP": 42.0},
        )
        evaluation_only = _cascade_mlflow_metrics(
            {"total_loss": (1.25, 19), "bbox/AP": (42.0, 19)},
            current_iteration=19,
            include_training=False,
        )
        self.assertEqual(evaluation_only, {"eval/bbox/AP": 42.0})
        self.assertTrue(_should_save_periodic_checkpoint(20, 40, 20))
        self.assertFalse(_should_save_periodic_checkpoint(40, 40, 20))
        self.assertEqual(_solver_steps(8, (0.75, 0.9)), (6, 7))
        with self.assertRaises(ValueError):
            _solver_steps(0, (0.75, 0.9))

    def test_validation_subprocess_imports_package_from_source_tree(self):
        with patch.dict(os.environ, {"PYTHONPATH": "/external/python"}, clear=True):
            environment = project_environment()
        self.assertEqual(PROJECT_ROOT, Path(__file__).parents[1])
        self.assertEqual(SOURCE_ROOT, PROJECT_ROOT / "src")
        self.assertEqual(
            environment["PYTHONPATH"],
            f"{SOURCE_ROOT}{os.pathsep}/external/python",
        )

    def test_validation_runtime_describes_single_and_distributed_execution(self):
        self.assertEqual(expected_distributed_tag(1), "false")
        self.assertEqual(expected_distributed_tag(2), "true")
        self.assertEqual(execution_mode(1), "single GPU")
        self.assertEqual(execution_mode(2), "2-process DDP")
        with self.assertRaises(ValueError):
            expected_distributed_tag(0)

    def test_legacy_checkpoint_metadata_uses_safe_loader(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy.pth"
            torch.save(
                {"value": np.float64(1.25), "args": argparse.Namespace(epoch=1)},
                checkpoint,
            )
            loaded = safe_torch_load(checkpoint)
            self.assertEqual(float(loaded["value"]), 1.25)
            self.assertEqual(loaded["args"].epoch, 1)

    def test_incomplete_encoder_checkpoint_is_rejected(self):
        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList(
                    [torch.nn.Linear(1, 1, bias=False)]
                )
                self.patch_embed = torch.nn.Linear(1, 1, bias=False)

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "incomplete.pth"
            torch.save(
                {"model": {"blocks.0.weight": torch.ones(1, 1)}}, checkpoint
            )
            with self.assertRaisesRegex(RuntimeError, "patch_embed.weight"):
                load_dit_pretrained(TinyEncoder(), checkpoint)

    def test_category_mapping_round_trip(self):
        for category_id in range(1, 6):
            self.assertEqual(
                train_label_to_category_id(category_id_to_train_label(category_id)),
                category_id,
            )

    def test_unknown_category_rejected(self):
        with self.assertRaises(ValueError):
            category_id_to_train_label(0)

    def test_dataset_layout_and_config(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_publaynet_fixture(root)
            summaries = validate_publaynet(root)
            self.assertEqual([item.images for item in summaries], [1, 1])
            RunConfig("dino", root, root / "out", None).validate()

    def test_dataset_category_names_must_match_publaynet_mapping(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_publaynet_fixture(root)
            annotation = root / "annotations" / "train.json"
            document = json.loads(annotation.read_text(encoding="utf-8"))
            document["categories"][0]["name"] = "Figure"
            annotation.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "category mapping must be exactly"):
                validate_publaynet(root)

    def test_runtime_validator_creates_symlinked_publaynet_subset(self):
        from PIL import Image

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "subset"
            _write_publaynet_fixture(source)
            Image.new("RGB", (64, 64), "white").save(source / "train" / "train.jpg")
            annotation = source / "annotations" / "train.json"
            document = json.loads(annotation.read_text(encoding="utf-8"))
            document["annotations"] = [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [4, 4, 16, 16],
                    "area": 256,
                    "iscrowd": 0,
                }
            ]
            annotation.write_text(json.dumps(document), encoding="utf-8")

            create_publaynet_subset(
                source, destination, split="train", image_count=1, seed=7
            )
            subset_image = destination / "train" / "train.jpg"
            subset = json.loads(
                (destination / "annotations" / "train.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(subset_image.is_symlink())
            self.assertEqual(len(subset["images"]), 1)
            self.assertEqual(len(subset["annotations"]), 1)

    def test_runtime_validator_enables_production_execution_features(self):
        args = SimpleNamespace(
            config=Path(__file__).parents[1] / "configs" / "dino_train.yaml",
            seed=42,
            pretrained=Path(__file__),
            image_size=128,
            full_detector=False,
        )
        with TemporaryDirectory() as directory:
            settings = dino_validation_settings(
                args, work_dir=Path(directory), world_size=2, epochs=1
            )
        self.assertIs(settings["run"]["amp"], True)
        self.assertIs(settings["dit"]["use_checkpoint"], True)
        self.assertIs(settings["dino"]["ddp_static_graph"], True)
        self.assertIs(settings["dino"]["fused_optimizer"], True)
        self.assertIs(settings["tracking"]["enabled"], True)
        self.assertEqual(settings["training"]["batch_size"], 2)

        with TemporaryDirectory() as directory:
            single_settings = dino_validation_settings(
                args, work_dir=Path(directory), world_size=1, epochs=1
            )
        self.assertEqual(single_settings["training"]["batch_size"], 1)

    def test_cascade_runtime_validator_enables_production_execution_features(self):
        args = SimpleNamespace(
            config=Path(__file__).parents[1]
            / "configs"
            / "cascade_rcnn_train.yaml",
            seed=42,
            pretrained=Path(__file__),
            image_size=128,
        )
        with TemporaryDirectory() as directory:
            settings = cascade_validation_settings(
                args, work_dir=Path(directory), world_size=2, epochs=1
            )
        self.assertIs(settings["run"]["amp"], True)
        self.assertIs(settings["dit"]["use_checkpoint"], True)
        self.assertIs(settings["tracking"]["enabled"], True)
        self.assertEqual(settings["training"]["batch_size"], 2)
        self.assertEqual(settings["cascade_rcnn"]["roi_batch_size_per_image"], 32)

        with TemporaryDirectory() as directory:
            single_settings = cascade_validation_settings(
                args, work_dir=Path(directory), world_size=1, epochs=1
            )
        self.assertEqual(single_settings["training"]["batch_size"], 1)

    def test_runtime_validators_accept_one_selected_device(self):
        import scripts.validate_cascade_training as cascade_validator
        import scripts.validate_dino_training as dino_validator

        validators = (
            (dino_validator, "dino_train.yaml"),
            (cascade_validator, "cascade_rcnn_train.yaml"),
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for validator, config_name in validators:
                with self.subTest(validator=validator.__name__):
                    work_dir = root / config_name
                    argv = [
                        "--data-root",
                        str(FIXTURE),
                        "--pretrained",
                        str(Path(__file__)),
                        "--config",
                        str(Path(__file__).parents[1] / "configs" / config_name),
                        "--devices",
                        "2",
                        "--train-images",
                        "2",
                        "--val-images",
                        "1",
                        "--image-size",
                        "32",
                        "--work-dir",
                        str(work_dir),
                    ]
                    with patch.dict(
                        "sys.modules", {"detectron2": Mock()}
                    ), patch.object(
                        validator, "validate_cuda_devices"
                    ) as validate_devices, patch.object(
                        validator, "create_publaynet_subset"
                    ), patch.object(
                        validator, "_run_training"
                    ) as run_training, patch.object(
                        validator, "_verify_checkpoint"
                    ), patch.object(
                        validator, "_verify_mlflow"
                    ):
                        validator.main(argv)

                    validate_devices.assert_called_once_with((2,))
                    self.assertEqual(run_training.call_count, 2)
                    self.assertEqual(
                        [item.args[1] for item in run_training.call_args_list],
                        [(2,), (2,)],
                    )

    def test_detector_validation(self):
        with self.assertRaises(ValueError):
            RunConfig("unknown", FIXTURE, FIXTURE / "out", None).validate()

    def test_cascade_detector_uses_cascade_yaml_section(self):
        config = RunConfig(
            "cascade_rcnn", FIXTURE, FIXTURE / "out", None,
            settings=load_settings(detector="cascade_rcnn"),
        )
        self.assertEqual(config.detector_settings["anchor_sizes"], [32, 64, 128, 256])

    def test_detector_configs_have_no_other_backend_section(self):
        dino = load_settings(detector="dino")
        cascade = load_settings(detector="cascade_rcnn")
        self.assertIn("dino", dino)
        self.assertNotIn("cascade_rcnn", dino)
        self.assertIn("cascade_rcnn", cascade)
        self.assertNotIn("dino", cascade)
        self.assertIn("prefetch_factor", dino["training"])
        self.assertNotIn("prefetch_factor", cascade["training"])

    def test_example_configs_select_matching_schema(self):
        root = Path(__file__).parents[1] / "configs"
        dino = load_settings(root / "dino_train.yaml")
        cascade = load_settings(root / "cascade_rcnn_train.yaml")
        self.assertEqual(dino["run"]["detector"], "dino")
        self.assertEqual(cascade["run"]["detector"], "cascade_rcnn")

    def test_cli_detector_cannot_conflict_with_config(self):
        parser = parser_for("train")
        config_path = Path(__file__).parents[1] / "configs" / "dino_train.yaml"
        args = parser.parse_args(
            ["--detector", "cascade_rcnn", "--config", str(config_path)]
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            build_config(args, require_data=False)

    def test_runtime_detector_must_match_settings(self):
        config = RunConfig(
            "cascade_rcnn",
            FIXTURE,
            FIXTURE / "out",
            None,
            settings=load_settings(detector="dino"),
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            config.validate(require_data=False)

    def test_optimizer_split_keeps_pyramid_at_detector_lr(self):
        class Parameter:
            requires_grad = True

            def numel(self):
                return 1

        encoder = Parameter()
        pyramid = Parameter()
        head = Parameter()

        class Model:
            def named_parameters(self):
                return iter(
                    [
                        ("backbone.0.pyramid.backbone.blocks.0.weight", encoder),
                        ("backbone.0.pyramid.transforms.0.weight", pyramid),
                        ("class_embed.weight", head),
                    ]
                )

        groups = parameter_groups(Model(), detector_lr=1e-4, backbone_lr=1e-5)
        self.assertEqual(groups[0]["params"], [pyramid, head])
        self.assertEqual(groups[1]["params"], [encoder])

    def test_dotted_and_alias_overrides(self):
        settings = load_settings(
            options=["training.epochs=24", "num_queries=500", "run.amp=true"]
        )
        self.assertEqual(settings["training"]["epochs"], 24)
        self.assertEqual(settings["dino"]["num_queries"], 500)
        self.assertIs(settings["run"]["amp"], True)

    def test_unknown_override_rejected(self):
        with self.assertRaises(ValueError):
            load_settings(options=["dino.not_a_setting=1"])

    def test_override_type_is_checked(self):
        with self.assertRaisesRegex(ValueError, "training.epochs must be an integer"):
            load_settings(options=["training.epochs=wrong"])

    def test_detector_numeric_ranges_are_validated(self):
        cases = (
            ("dino", "dino.score_threshold=1.1", "dino.score_threshold"),
            (
                "cascade_rcnn",
                "cascade_rcnn.score_threshold=-0.1",
                "cascade_rcnn.score_threshold",
            ),
            (
                "cascade_rcnn",
                "cascade_rcnn.anchor_sizes=[32,64,128,0]",
                "cascade_rcnn.anchor_sizes",
            ),
            (
                "cascade_rcnn",
                "cascade_rcnn.aspect_ratios=[0.5,0,2.0]",
                "cascade_rcnn.aspect_ratios",
            ),
            (
                "cascade_rcnn",
                "cascade_rcnn.roi_batch_size_per_image=0",
                "roi_batch_size_per_image",
            ),
            (
                "cascade_rcnn",
                "cascade_rcnn.rpn_batch_size_per_image=0",
                "rpn_batch_size_per_image",
            ),
        )
        for detector, option, message in cases:
            with self.subTest(option=option):
                settings = load_settings(options=[option], detector=detector)
                config = RunConfig(
                    detector, FIXTURE, FIXTURE / "out", None, settings=settings
                )
                with self.assertRaisesRegex(ValueError, message):
                    config.validate(require_data=False)

    def test_inference_cli_rejects_invalid_score_threshold(self):
        parser = parser_for("inference")
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["--image", str(__file__), "--score-threshold", "nan"]
            )

    def test_partial_config_merges_with_defaults(self):
        settings = load_settings(Path(__file__).parent / "fixtures" / "custom.yaml")
        self.assertEqual(settings["training"]["epochs"], 20)
        self.assertEqual(settings["dino"]["num_queries"], 450)
        self.assertEqual(settings["input"]["max_long_edge"], 1333)

    def test_dedicated_cli_flag_wins_and_updates_effective_settings(self):
        args = parser_for("train").parse_args(
            ["--options", "training.epochs=5", "--epochs", "7"]
        )
        config = build_config(args, require_data=False)
        self.assertEqual(config.epochs, 7)
        self.assertEqual(config.training["epochs"], 7)

    def test_prediction_contract(self):
        class Box:
            def tolist(self):
                return [1.0, 2.0, 3.0, 4.0]

        record = prediction_record("page", Box(), 0.75, 3)
        self.assertEqual(record["category_id"], 3)
        self.assertEqual(record["box_xyxy"], [1.0, 2.0, 3.0, 4.0])
        with self.assertRaises(ValueError):
            prediction_record("page", Box(), 0.75, 0)

    def test_inference_collects_folder_images_in_stable_order(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("b.PNG", "a.jpg", "notes.txt"):
                (root / name).touch()
            self.assertEqual(
                [path.name for path in collect_image_paths(root)],
                ["a.jpg", "b.PNG"],
            )

    def test_inference_visualization_is_saved(self):
        from PIL import Image

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "page.png"
            output = root / "visualized" / "page.prediction.png"
            Image.new("RGB", (100, 80), "white").save(source)
            saved = save_visualization(
                source,
                [
                    {
                        "image_id": "page",
                        "box_xyxy": [10.0, 10.0, 60.0, 50.0],
                        "score": 0.9,
                        "category_id": 1,
                    }
                ],
                output,
            )
            self.assertEqual(saved, output)
            self.assertTrue(output.is_file())

    def test_folder_inference_loads_model_once(self):
        from PIL import Image
        import inference as inference_cli

        class Tracker:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def log_metrics(self, metrics):
                self.metrics = metrics

        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("one.jpg", "two.png"):
                Image.new("RGB", (8, 8), "white").save(root / name)
            config = SimpleNamespace(
                detector="dino",
                output_dir=root / "output",
                score_threshold=lambda: 0.5,
            )
            predictor = Mock(return_value=[])
            backend = SimpleNamespace(build_predictor=Mock(return_value=predictor))
            with patch.object(
                inference_cli, "validated_config", return_value=config
            ), patch.object(
                inference_cli, "MLflowTracker", return_value=Tracker()
            ), patch.object(
                inference_cli, "get_backend", return_value=backend
            ), redirect_stdout(io.StringIO()):
                inference_cli.main(
                    ["--image", str(root), "--resume", "--no-visualize"]
                )
            backend.build_predictor.assert_called_once_with(
                config, score_threshold=0.5
            )
            self.assertEqual(predictor.call_count, 2)
            self.assertTrue(
                all(
                    len(call.args) == 1 and not call.kwargs
                    for call in predictor.call_args_list
                )
            )

    def test_dino_runner_receives_effective_common_config(self):
        settings = load_settings(
            options=["training.epochs=3", "dino.num_queries=123"]
        )
        config = RunConfig(
            "dino",
            FIXTURE,
            FIXTURE / "out",
            None,
            epochs=3,
            settings=settings,
        )
        arguments = _runner_arguments(config)
        self.assertIn("batch_size=2", arguments)
        self.assertIn("epochs=3", arguments)
        self.assertIn("num_queries=123", arguments)
        self.assertIn("enc_layers=6", arguments)
        self.assertIn("data_norm_mean=0.5,0.5,0.5", arguments)
        self.assertIn("warmup_iters=1000", arguments)
        self.assertIn("evaluate_every_epochs=1", arguments)
        self.assertIn("weights_dir=weights", arguments)
        self.assertIn("optimizer=adamw", arguments)
        self.assertIn("fused_optimizer=True", arguments)
        self.assertIn("adam_betas=0.9,0.999", arguments)
        self.assertIn("set_cost_class=2.0", arguments)
        self.assertIn("dn_box_noise_scale=0.4", arguments)
        self.assertIn("use_ema=False", arguments)
        self.assertIn("lr_drop_list=[11]", arguments)
        self.assertIn("dit_pyramid_channels=256", arguments)
        self.assertIn("data_pin_memory=True", arguments)
        self.assertIn("data_non_blocking=True", arguments)
        self.assertIn("data_persistent_workers=True", arguments)
        self.assertIn("data_prefetch_factor=2", arguments)
        self.assertIn("ddp_gradient_as_bucket_view=True", arguments)
        self.assertIn("ddp_static_graph=True", arguments)

    def test_dino_encoder_can_be_disabled_without_changing_its_depth_setting(self):
        settings = load_settings(options=["dino.encoder_enabled=false"])
        config = RunConfig(
            "dino",
            FIXTURE,
            FIXTURE / "out",
            None,
            settings=settings,
        )

        self.assertEqual(settings["dino"]["enc_layers"], 6)
        self.assertEqual(_effective_dino_values(config)["enc_layers"], 0)
        self.assertIn("enc_layers=0", _runner_arguments(config))

    def test_dino_option_parser_preserves_singleton_lists(self):
        _activate_dino()
        from util.slconfig import DictAction

        parser = argparse.ArgumentParser()
        parser.add_argument("--options", nargs="+", action=DictAction)
        args = parser.parse_args(["--options", "lr_drop_list=[11]"])
        self.assertEqual(args.options["lr_drop_list"], [11])

    def test_dino_denoising_tensors_follow_target_device(self):
        _activate_dino()
        from models.dino.dn_components import prepare_for_cdn

        label_encoder = torch.nn.Embedding(6, 8)
        targets = [
            {
                "labels": torch.tensor([1], dtype=torch.long),
                "boxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
            }
        ]
        query_label, query_box, attention_mask, metadata = prepare_for_cdn(
            (targets, 2, 0.5, 0.4),
            training=True,
            num_queries=10,
            num_classes=6,
            hidden_dim=8,
            label_enc=label_encoder,
        )
        self.assertEqual(query_label.device, targets[0]["labels"].device)
        self.assertEqual(query_box.device, targets[0]["boxes"].device)
        self.assertEqual(attention_mask.device, targets[0]["boxes"].device)
        self.assertGreater(metadata["pad_size"], 0)

    def test_dino_resume_is_forwarded_only_when_explicit(self):
        settings = load_settings()
        without_resume = RunConfig(
            "dino",
            FIXTURE,
            FIXTURE / "out",
            None,
            settings=settings,
        )
        self.assertNotIn("--resume", _runner_arguments(without_resume))

        checkpoint = FIXTURE / "recent.pth"
        with_resume = RunConfig(
            "dino",
            FIXTURE,
            FIXTURE / "out",
            None,
            weights_dir=FIXTURE,
            resume=checkpoint,
            settings=settings,
        )
        arguments = _runner_arguments(with_resume)
        resume_index = arguments.index("--resume")
        self.assertEqual(arguments[resume_index + 1], str(checkpoint))

    def test_resume_training_does_not_require_redundant_pretrained_path(self):
        with TemporaryDirectory() as directory:
            weights_dir = Path(directory) / "weights"
            weights_dir.mkdir()
            checkpoint = weights_dir / "recent.pth"
            checkpoint.touch()
            missing_pretrained = Path(directory) / "missing-pretrained.pth"
            args = parser_for("train").parse_args(
                [
                    "--weights-dir",
                    str(weights_dir),
                    "--pretrained",
                    str(missing_pretrained),
                    "--resume",
                ]
            )
            config = validated_config(
                parser_for("train"),
                args,
                require_data=False,
                require_pretrained=True,
            )
            self.assertEqual(config.pretrained, missing_pretrained)
            self.assertEqual(config.weights_dir, weights_dir)
            self.assertEqual(config.resume, checkpoint)

    def test_resume_does_not_accept_an_external_checkpoint_path(self):
        parser = parser_for("train")
        with self.assertRaises(SystemExit):
            parser.parse_args(["--resume", "outside.pth"])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside.pth"
            outside.touch()
            config = RunConfig(
                "dino",
                root,
                root / "output",
                None,
                resume=outside,
                weights_dir=root / "weights",
                settings=load_settings(),
            )
            with self.assertRaisesRegex(ValueError, "weights_dir/recent.pth"):
                config.validate(require_data=False)

    def test_mlflow_settings_are_versioned_in_yaml(self):
        settings = load_settings()
        self.assertIs(settings["tracking"]["enabled"], True)
        self.assertEqual(settings["tracking"]["experiment_name"], "dit-dino-publaynet")

    def test_dino_coco_summary_is_flattened_for_mlflow(self):
        metrics = _evaluation_metrics(
            {
                "loss": 0.25,
                "coco_eval_bbox": [0.42, 0.61, 0.45, 0.1, 0.4, 0.5],
            }
        )
        self.assertEqual(metrics["eval/loss"], 0.25)
        self.assertEqual(metrics["eval/bbox_mAP"], 42.0)
        self.assertEqual(metrics["eval/bbox_AP50"], 61.0)
        self.assertNotIn("eval/coco_eval_bbox", metrics)

    def test_dino_epoch_losses_distinguish_raw_and_scaled_values(self):
        metrics = _training_epoch_metrics(
            {
                "loss": 4.0,
                "loss_bbox": 2.0,
                "loss_bbox_unscaled": 0.4,
                "class_error": 3.0,
                "class_error_unscaled": 3.0,
            }
        )
        self.assertEqual(metrics["loss"], 4.0)
        self.assertEqual(metrics["loss_bbox"], 0.4)
        self.assertEqual(metrics["loss_bbox_scaled"], 2.0)
        self.assertEqual(metrics["class_error"], 3.0)
        self.assertNotIn("class_error_scaled", metrics)

    def test_dino_evaluations_reach_mlflow_with_epoch_steps(self):
        import mlflow

        with TemporaryDirectory() as directory:
            tracking_uri = (Path(directory) / "mlruns").as_uri()
            settings = {
                "tracking": {
                    "enabled": True,
                    "tracking_uri": tracking_uri,
                    "experiment_name": "integration-test",
                    "run_name": "eval-steps",
                }
            }
            config = SimpleNamespace(
                tracking=settings["tracking"],
                settings=settings,
                detector="dino",
                batch_size=1,
            )
            previous_uri = mlflow.get_tracking_uri()
            try:
                with patch.dict(
                    os.environ, {"MLFLOW_ALLOW_FILE_STORE": "true"}
                ), patch(
                    "dit_layout_bench.evaluation.print_per_category_ap",
                    return_value={"Text": 91.0},
                ):
                    with MLflowTracker(config, "train"):
                        integration = _build_dino_integration(config)
                        for epoch in (1, 2):
                            for prefix in ("eval", "eval_ema"):
                                loss = float(
                                    epoch if prefix == "eval" else epoch + 10
                                )
                                integration.on_evaluation(
                                    {
                                        "loss": loss,
                                        "coco_eval_bbox": [0.42, 0.61, 0.45],
                                    },
                                    SimpleNamespace(coco_eval={"bbox": object()}),
                                    SimpleNamespace(
                                        _tracking_eval_epoch=epoch,
                                        _tracking_eval_prefix=prefix,
                                    ),
                                )
            finally:
                mlflow.set_tracking_uri(previous_uri)

            with patch.dict(os.environ, {"MLFLOW_ALLOW_FILE_STORE": "true"}):
                client = mlflow.MlflowClient(tracking_uri=tracking_uri)
                experiment = client.get_experiment_by_name("integration-test")
                run = client.search_runs([experiment.experiment_id])[0]
                keys = (
                    "eval/loss",
                    "eval/bbox_mAP",
                    "eval/AP_Text",
                    "eval_ema/loss",
                    "eval_ema/bbox_mAP",
                    "eval_ema/AP_Text",
                )
                rows = sorted(
                    (metric.key, metric.step, metric.value)
                    for key in keys
                    for metric in client.get_metric_history(run.info.run_id, key)
                )
            self.assertEqual(
                rows,
                [
                    ("eval/AP_Text", 1, 91.0),
                    ("eval/AP_Text", 2, 91.0),
                    ("eval/bbox_mAP", 1, 42.0),
                    ("eval/bbox_mAP", 2, 42.0),
                    ("eval/loss", 1, 1.0),
                    ("eval/loss", 2, 2.0),
                    ("eval_ema/AP_Text", 1, 91.0),
                    ("eval_ema/AP_Text", 2, 91.0),
                    ("eval_ema/bbox_mAP", 1, 42.0),
                    ("eval_ema/bbox_mAP", 2, 42.0),
                    ("eval_ema/loss", 1, 11.0),
                    ("eval_ema/loss", 2, 12.0),
                ],
            )

    def test_resumed_training_continues_the_same_mlflow_run(self):
        import mlflow

        with TemporaryDirectory() as directory:
            root = Path(directory)
            tracking_uri = (root / "mlruns").as_uri()
            weights_dir = root / "weights"
            settings = {
                "tracking": {
                    "enabled": True,
                    "tracking_uri": tracking_uri,
                    "experiment_name": "resume-test",
                    "run_name": "continuous-run",
                }
            }
            common = {
                "tracking": settings["tracking"],
                "settings": settings,
                "detector": "dino",
                "batch_size": 1,
                "weights_dir": weights_dir,
            }
            previous_uri = mlflow.get_tracking_uri()
            try:
                with patch.dict(os.environ, {"MLFLOW_ALLOW_FILE_STORE": "true"}):
                    with MLflowTracker(SimpleNamespace(**common, resume=None), "train") as tracker:
                        tracker.log_metrics({"train/loss": 2.0}, step=1)
                    resume = weights_dir / "recent.pth"
                    resume.touch()
                    with MLflowTracker(
                        SimpleNamespace(**common, resume=resume), "train"
                    ) as tracker:
                        tracker.log_metrics({"train/loss": 1.0}, step=2)
            finally:
                mlflow.set_tracking_uri(previous_uri)

            with patch.dict(os.environ, {"MLFLOW_ALLOW_FILE_STORE": "true"}):
                client = mlflow.MlflowClient(tracking_uri=tracking_uri)
                experiment = client.get_experiment_by_name("resume-test")
                runs = client.search_runs([experiment.experiment_id])
                history = client.get_metric_history(runs[0].info.run_id, "train/loss")
                resume_artifacts = client.list_artifacts(
                    runs[0].info.run_id, "resume-configs/resume-1"
                )
            self.assertEqual(len(runs), 1)
            self.assertEqual(
                [(metric.step, metric.value) for metric in history],
                [(1, 2.0), (2, 1.0)],
            )
            self.assertEqual(
                {Path(item.path).name for item in resume_artifacts},
                {"effective-config.yaml", "runtime.yaml"},
            )

    def test_failed_mlflow_resume_starts_a_linked_attempt(self):
        import mlflow

        with TemporaryDirectory() as directory:
            root = Path(directory)
            tracking_uri = (root / "mlruns").as_uri()
            weights_dir = root / "weights"
            settings = {
                "tracking": {
                    "enabled": True,
                    "tracking_uri": tracking_uri,
                    "experiment_name": "failed-resume-test",
                    "run_name": "retry",
                }
            }
            common = {
                "tracking": settings["tracking"],
                "settings": settings,
                "detector": "dino",
                "batch_size": 1,
                "weights_dir": weights_dir,
            }
            resume = weights_dir / "recent.pth"
            previous_uri = mlflow.get_tracking_uri()
            try:
                with patch.dict(os.environ, {"MLFLOW_ALLOW_FILE_STORE": "true"}):
                    with MLflowTracker(
                        SimpleNamespace(**common, resume=None), "train"
                    ) as tracker:
                        tracker.log_metrics({"train/loss": 3.0}, step=1)
                    resume.touch()
                    with self.assertRaisesRegex(RuntimeError, "interrupted"):
                        with MLflowTracker(
                            SimpleNamespace(**common, resume=resume), "train"
                        ) as tracker:
                            tracker.log_metrics({"train/loss": 2.0}, step=2)
                            raise RuntimeError("interrupted")
                    failed_run_id = (
                        weights_dir / "mlflow-run-id.txt"
                    ).read_text(encoding="utf-8")
                    with MLflowTracker(
                        SimpleNamespace(**common, resume=resume), "train"
                    ) as tracker:
                        tracker.log_metrics({"train/loss": 1.5}, step=2)
                    retry_run_id = (
                        weights_dir / "mlflow-run-id.txt"
                    ).read_text(encoding="utf-8")
            finally:
                mlflow.set_tracking_uri(previous_uri)

            client = mlflow.MlflowClient(tracking_uri=tracking_uri)
            retry_run = client.get_run(retry_run_id)
            retry_history = client.get_metric_history(retry_run_id, "train/loss")
            self.assertNotEqual(failed_run_id, retry_run_id)
            self.assertEqual(retry_run.data.tags["resume_of"], failed_run_id)
            self.assertEqual(retry_run.data.tags["resume_source_status"], "FAILED")
            self.assertEqual(
                [(metric.step, metric.value) for metric in retry_history],
                [(2, 1.5)],
            )

    def test_tracking_reads_active_torch_process_group(self):
        with patch.object(
            torch.distributed, "is_available", return_value=True
        ), patch.object(
            torch.distributed, "is_initialized", return_value=True
        ), patch.object(
            torch.distributed, "get_rank", return_value=3
        ), patch.object(
            torch.distributed, "get_world_size", return_value=8
        ):
            self.assertEqual(process_rank(), 3)
            self.assertEqual(process_world_size(), 8)

    def test_tracking_defaults_to_single_process(self):
        with patch.object(torch.distributed, "is_initialized", return_value=False):
            self.assertEqual(process_rank(), 0)
            self.assertEqual(process_world_size(), 1)

    def test_internal_ddp_device_list_is_explicit_and_validated(self):
        self.assertEqual(parse_cuda_devices("2,3"), (2, 3))
        for value in ("", "0,0", "0,cuda", "-1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_cuda_devices(value)

    def test_internal_ddp_worker_receives_explicit_torch_topology(self):
        launch = LocalLaunch((2, 3), master_port=1234)
        received = {}

        def worker(rank, topology, marker):
            received.update(rank=rank, topology=topology, marker=marker)

        _run_process(1, launch, worker, ("ready",))
        self.assertEqual(received["rank"], 1)
        self.assertIs(received["topology"], launch)
        self.assertEqual(received["marker"], "ready")
        self.assertEqual(launch.world_size, 2)
        self.assertEqual(launch.init_method, "tcp://127.0.0.1:1234")

    def test_distributed_session_initializes_and_releases_owned_group(self):
        with patch(
            "dit_layout_bench.runtime.activate_device",
            return_value=torch.device("cuda:1"),
        ) as activate, patch.object(
            torch.distributed, "is_available", return_value=True
        ), patch.object(
            torch.distributed, "is_initialized", side_effect=[False, True]
        ), patch.object(
            torch.distributed, "init_process_group"
        ) as initialize, patch.object(
            torch.distributed, "destroy_process_group"
        ) as destroy:
            with distributed_session(
                "cuda:1",
                rank=1,
                world_size=2,
                init_method="tcp://127.0.0.1:1234",
            ) as device:
                self.assertEqual(device, torch.device("cuda:1"))
        activate.assert_called_once_with("cuda:1")
        initialize.assert_called_once_with(
            backend="nccl",
            init_method="tcp://127.0.0.1:1234",
            world_size=2,
            rank=1,
        )
        destroy.assert_called_once_with()

    def test_global_batch_is_divided_per_gpu(self):
        settings = load_settings(options=["training.batch_size=6"])
        config = RunConfig(
            "dino",
            FIXTURE,
            FIXTURE / "out",
            None,
            batch_size=6,
            settings=settings,
        )
        with patch(
            "dit_layout_bench.backends.dino.process_world_size", return_value=2
        ):
            self.assertEqual(_effective_dino_values(config)["batch_size"], 3)

    def test_inference_does_not_apply_training_batch_partition(self):
        settings = load_settings(options=["training.batch_size=5"])
        config = RunConfig(
            "dino",
            FIXTURE,
            FIXTURE / "out",
            None,
            batch_size=5,
            settings=settings,
        )
        values = _effective_dino_values(config, for_training=False)
        self.assertEqual(values["batch_size"], 5)

    def test_non_divisible_global_batch_is_rejected_before_training(self):
        import train as train_entry

        config = SimpleNamespace(detector="dino", device="cuda", batch_size=5)
        with patch.object(
            train_entry, "validated_config", return_value=config
        ), patch.object(
            train_entry, "validate_cuda_devices"
        ), patch.object(
            train_entry, "_launch_distributed"
        ) as launch, self.assertRaises(SystemExit):
            train_entry.main(["--devices", "0,1"])
        launch.assert_not_called()

    def test_train_entrypoint_launches_internal_ddp(self):
        import train as train_entry

        config = SimpleNamespace(detector="dino", device="cuda", batch_size=4)
        with patch.object(
            train_entry, "validated_config", return_value=config
        ), patch.object(
            train_entry, "validate_cuda_devices"
        ), patch.object(train_entry, "_launch_distributed") as launch:
            train_entry.main(["--devices", "0,1"])
        launch.assert_called_once_with((0, 1), config)

    def test_train_entrypoint_uses_one_selected_device_without_ddp(self):
        import train as train_entry

        for detector in ("dino", "cascade_rcnn"):
            with self.subTest(detector=detector):
                config = SimpleNamespace(
                    detector=detector, device="cuda:2", batch_size=1
                )

                def capture_config(_parser, args, **_kwargs):
                    self.assertEqual(args.device, "cuda:2")
                    return config

                with patch.object(
                    train_entry, "validated_config", side_effect=capture_config
                ), patch.object(
                    train_entry, "validate_cuda_devices"
                ) as validate_devices, patch.object(
                    train_entry, "_launch_distributed"
                ) as launch, patch.object(train_entry, "_train") as train_worker:
                    train_entry.main(["--devices", "2"])

                validate_devices.assert_called_once_with((2,))
                launch.assert_not_called()
                train_worker.assert_called_once_with(config, rank=0, world_size=1)

    def test_cascade_config_is_independent_of_distributed_topology(self):
        parser = parser_for("train")
        args = parser.parse_args(
            [
                "--detector",
                "cascade_rcnn",
                "--data-root",
                str(FIXTURE),
                "--pretrained",
                str(Path(__file__)),
            ]
        )
        config = validated_config(
            parser,
            args,
            require_data=False,
            training=True,
            require_pretrained=True,
        )
        self.assertEqual(config.detector, "cascade_rcnn")


if __name__ == "__main__":
    unittest.main()
