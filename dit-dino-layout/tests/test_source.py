import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]


class SourceTests(unittest.TestCase):
    def test_python_sources_parse(self):
        for path in ROOT.rglob("*.py"):
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


if __name__ == "__main__":
    unittest.main()
