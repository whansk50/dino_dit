import ast
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).parents[1]


class SourceTests(unittest.TestCase):
    def test_python_sources_parse(self):
        for path in (ROOT / "train.py", ROOT / "inference.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for source_root in (ROOT / "src", ROOT / "scripts", ROOT / "tests"):
            for path in source_root.rglob("*.py"):
                if "build" in path.parts:
                    continue
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_runtime_does_not_import_upstream_dit_wrapper(self):
        sources = "\n".join(
            path.read_text(encoding="utf-8") for path in (ROOT / "src").rglob("*.py")
        )
        self.assertNotIn("dit.object_detection", sources)
        self.assertNotIn("ditod.backbone", sources)

    def test_runtime_uses_only_vendored_dino(self):
        from dit_layout_bench.paths import DINO_ROOT

        self.assertEqual(DINO_ROOT, ROOT / "src" / "dit_layout_bench" / "_vendor" / "dino")
        self.assertTrue((DINO_ROOT / "LICENSE").is_file())

    def test_vendored_dino_config_dumps_with_installed_yapf(self):
        from dit_layout_bench.backends.dino import _activate_dino, DINO_CONFIG

        _activate_dino()
        from util.slconfig import SLConfig

        with TemporaryDirectory() as directory:
            output = Path(directory) / "config_cfg.py"
            SLConfig.fromfile(str(DINO_CONFIG)).dump(output)
            compile(output.read_text(encoding="utf-8"), str(output), "exec")

    def test_dino_does_not_implicitly_resume_from_output_directory(self):
        source = (
            ROOT / "src" / "dit_layout_bench" / "_vendor" / "dino" / "main.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn(
            "args.resume = os.path.join(args.output_dir, 'checkpoint.pth')",
            source,
        )

    def test_dino_integration_is_explicit_instead_of_monkey_patched(self):
        backend = (
            ROOT / "src" / "dit_layout_bench" / "backends" / "dino.py"
        ).read_text(encoding="utf-8")
        dispatcher = (
            ROOT / "src" / "dit_layout_bench" / "backends" / "__init__.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("dino_model.build_backbone =", backend)
        self.assertNotIn("dino_main.build_dataset =", backend)
        self.assertIn("integration=_build_dino_integration(config)", backend)
        self.assertNotIn("importlib", dispatcher)

    def test_cascade_owns_ddp_lifecycle_and_restores_nested_trainer_state(self):
        backend = (
            ROOT / "src" / "dit_layout_bench" / "backends" / "cascade_rcnn.py"
        ).read_text(encoding="utf-8")
        self.assertIn("with distributed_session(config.device) as device:", backend)
        self.assertIn("api.comm.create_local_process_group", backend)
        self.assertIn("trainer.start_iter = trainer.iter + 1", backend)
        self.assertNotIn('checkpoint.get("iteration"', backend)


if __name__ == "__main__":
    unittest.main()
