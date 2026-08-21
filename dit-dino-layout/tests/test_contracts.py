import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dit_layout_bench.config import RunConfig, load_settings
from dit_layout_bench.cli import _config, _parser, _validated_config
from dit_layout_bench.checkpoint import _safe_torch_load
from dit_layout_bench.backends.dino import _activate_dino, _runner_arguments
from dit_layout_bench.data import validate_publaynet
from dit_layout_bench.optim import parameter_groups
from dit_layout_bench.prediction import prediction_record
from dit_layout_bench.spec import category_id_to_train_label, train_label_to_category_id
from dit_layout_bench.tracking import process_rank, process_world_size

FIXTURE = Path(__file__).parent / "fixtures" / "publaynet"


class ContractTests(unittest.TestCase):
    def test_legacy_checkpoint_metadata_uses_safe_loader(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy.pth"
            torch.save(
                {"value": np.float64(1.25), "args": argparse.Namespace(epoch=1)},
                checkpoint,
            )
            loaded = _safe_torch_load(checkpoint)
            self.assertEqual(float(loaded["value"]), 1.25)
            self.assertEqual(loaded["args"].epoch, 1)

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
        summaries = validate_publaynet(FIXTURE)
        self.assertEqual([item.images for item in summaries], [1, 1])
        RunConfig("dino", FIXTURE, FIXTURE / "out", None).validate()

    def test_detector_validation(self):
        with self.assertRaises(ValueError):
            RunConfig("unknown", FIXTURE, FIXTURE / "out", None).validate()

    def test_cascade_detector_uses_cascade_yaml_section(self):
        config = RunConfig(
            "cascade_rcnn", FIXTURE, FIXTURE / "out", None,
            settings=load_settings(options=["run.detector=cascade_rcnn"]),
        )
        self.assertEqual(config.detector_settings["anchor_sizes"], [32, 64, 128, 256])

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

    def test_partial_config_merges_with_defaults(self):
        settings = load_settings(Path(__file__).parent / "fixtures" / "custom.yaml")
        self.assertEqual(settings["training"]["epochs"], 20)
        self.assertEqual(settings["dino"]["num_queries"], 450)
        self.assertEqual(settings["input"]["max_long_edge"], 1333)

    def test_dedicated_cli_flag_wins_and_updates_effective_settings(self):
        args = _parser("train").parse_args(
            ["--options", "training.epochs=5", "--epochs", "7"]
        )
        config = _config(args, require_data=False)
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
        arguments = _runner_arguments(config, evaluate=False)
        self.assertIn("epochs=3", arguments)
        self.assertIn("num_queries=123", arguments)
        self.assertIn("data_norm_mean=0.5,0.5,0.5", arguments)
        self.assertIn("warmup_iters=1000", arguments)
        self.assertIn("evaluate_every_epochs=1", arguments)
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

    def test_dino_option_parser_preserves_singleton_lists(self):
        _activate_dino()
        from util.slconfig import DictAction

        parser = argparse.ArgumentParser()
        parser.add_argument("--options", nargs="+", action=DictAction)
        args = parser.parse_args(["--options", "lr_drop_list=[11]"])
        self.assertEqual(args.options["lr_drop_list"], [11])

    def test_dino_resume_is_forwarded_only_when_explicit(self):
        settings = load_settings()
        without_resume = RunConfig(
            "dino",
            FIXTURE,
            FIXTURE / "out",
            None,
            settings=settings,
        )
        self.assertNotIn("--resume", _runner_arguments(without_resume, evaluate=False))

        checkpoint = FIXTURE / "detector.pth"
        with_resume = RunConfig(
            "dino",
            FIXTURE,
            FIXTURE / "out",
            None,
            resume=checkpoint,
            settings=settings,
        )
        arguments = _runner_arguments(with_resume, evaluate=False)
        resume_index = arguments.index("--resume")
        self.assertEqual(arguments[resume_index + 1], str(checkpoint))

    def test_resume_training_does_not_require_redundant_pretrained_path(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pth"
            checkpoint.touch()
            args = _parser("train").parse_args(
                [
                    "--data-root",
                    str(FIXTURE),
                    "--resume",
                    str(checkpoint),
                ]
            )
            config = _validated_config(
                _parser("train"),
                args,
                require_data=True,
                require_pretrained=True,
            )
            self.assertIsNone(config.pretrained)
            self.assertEqual(config.resume, checkpoint)

    def test_mlflow_settings_are_versioned_in_yaml(self):
        settings = load_settings()
        self.assertIs(settings["tracking"]["enabled"], True)
        self.assertEqual(settings["tracking"]["experiment_name"], "dit-dino-publaynet")

    def test_tracking_reads_torchrun_topology_before_process_group_init(self):
        environment = {"RANK": "3", "WORLD_SIZE": "8", "LOCAL_RANK": "1"}
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(process_rank(), 3)
            self.assertEqual(process_world_size(), 8)

    def test_tracking_defaults_to_single_process(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(process_rank(), 0)
            self.assertEqual(process_world_size(), 1)

    def test_torchrun_rejects_backend_without_ddp_integration(self):
        environment = {"RANK": "0", "WORLD_SIZE": "2", "LOCAL_RANK": "0"}
        parser = _parser("train")
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
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaises(SystemExit):
                _validated_config(
                    parser,
                    args,
                    require_data=True,
                    require_pretrained=True,
                )


if __name__ == "__main__":
    unittest.main()
