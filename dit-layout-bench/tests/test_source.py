import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]


class SourceTests(unittest.TestCase):
    def test_python_sources_parse(self):
        for path in ROOT.rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_runtime_does_not_import_upstream_dit_wrapper(self):
        sources = "\n".join(
            path.read_text(encoding="utf-8") for path in (ROOT / "src").rglob("*.py")
        )
        self.assertNotIn("dit.object_detection", sources)
        self.assertNotIn("ditod.backbone", sources)


if __name__ == "__main__":
    unittest.main()
