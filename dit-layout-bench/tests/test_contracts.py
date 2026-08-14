from pathlib import Path
import unittest

from dit_layout_bench.config import RunConfig
from dit_layout_bench.data import validate_publaynet
from dit_layout_bench.spec import category_id_to_train_label, train_label_to_category_id
from dit_layout_bench.optim import parameter_groups

FIXTURE = Path(__file__).parent / "fixtures" / "publaynet"


class ContractTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
