from __future__ import annotations

import ast
import importlib
import unittest
from pathlib import Path


MLDB_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = MLDB_ROOT / "src"
FEATURE_PACKAGES = (
    "api",
    "catalog",
    "common",
    "evaluation",
    "model",
    "orchestration",
    "repository",
    "runtime",
    "study",
    "training",
    "verification",
)


class BootstrapImportTests(unittest.TestCase):
    def test_implementation_package_imports(self) -> None:
        module = importlib.import_module("mldb.src")
        self.assertEqual("mldb.src", module.__name__)

        for feature in FEATURE_PACKAGES:
            imported = importlib.import_module(f"mldb.src.{feature}")
            self.assertEqual(f"mldb.src.{feature}", imported.__name__)

    def test_runtime_source_does_not_import_skeleton(self) -> None:
        violations: list[str] = []
        for path in sorted(SRC_ROOT.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "mldb.skeleton" or alias.name.startswith(
                            "mldb.skeleton."
                        ):
                            violations.append(f"{path}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module == "mldb.skeleton" or module.startswith("mldb.skeleton."):
                        violations.append(f"{path}: from {module}")
                    elif node.level and (
                        module == "skeleton" or module.startswith("skeleton.")
                    ):
                        violations.append(f"{path}: relative import from {module}")

        self.assertEqual([], violations)


if __name__ == "__main__":
    unittest.main()
