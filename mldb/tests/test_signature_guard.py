from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mldb.tests.signature_guard import (
    collect_signature_mismatches,
    compare_module_signatures,
)


MLDB_ROOT = Path(__file__).resolve().parents[1]


class RepositorySignatureGuardTests(unittest.TestCase):
    def test_implemented_modules_match_frozen_skeleton(self) -> None:
        mismatches = collect_signature_mismatches(
            MLDB_ROOT / "skeleton",
            MLDB_ROOT / "src",
        )
        self.assertEqual((), mismatches, "\n".join(mismatches))


class SignatureGuardSelfTests(unittest.TestCase):
    def test_matching_public_surface_passes(self) -> None:
        skeleton_source = """\
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Result:
    value: str
    count: int

    def render(self, prefix: str = \"\") -> str:
        ...

def load(path: str, *, strict: bool = False) -> Result:
    ...
"""
        implementation_source = """\
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Result:
    value: str
    count: int

    def render(self, prefix: str = \"\") -> str:
        return prefix + self.value

def load(path: str, *, strict: bool = False) -> Result:
    return Result(path, int(strict))
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            skeleton_file = root / "skeleton.py"
            implementation_file = root / "implementation.py"
            skeleton_file.write_text(skeleton_source, encoding="utf-8")
            implementation_file.write_text(implementation_source, encoding="utf-8")

            self.assertEqual(
                (),
                compare_module_signatures(skeleton_file, implementation_file),
            )

    def test_intentional_function_signature_mismatch_is_detected(self) -> None:
        skeleton_source = """\
def load(path: str, *, strict: bool = False) -> str:
    ...
"""
        implementation_source = """\
def load(path: str, strict: bool = False) -> str:
    return path
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            skeleton_file = root / "skeleton.py"
            implementation_file = root / "implementation.py"
            skeleton_file.write_text(skeleton_source, encoding="utf-8")
            implementation_file.write_text(implementation_source, encoding="utf-8")

            mismatches = compare_module_signatures(skeleton_file, implementation_file)

        self.assertEqual(1, len(mismatches))
        self.assertIn("public function load signature differs", mismatches[0])

    def test_intentional_method_signature_mismatch_is_detected(self) -> None:
        skeleton_source = """\
class Reader:
    def read(self, key: str, *, cached: bool = True) -> str:
        ...
"""
        implementation_source = """\
class Reader:
    def read(self, key: str, cached: bool = True) -> str:
        return key
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            skeleton_file = root / "skeleton.py"
            implementation_file = root / "implementation.py"
            skeleton_file.write_text(skeleton_source, encoding="utf-8")
            implementation_file.write_text(implementation_source, encoding="utf-8")

            mismatches = compare_module_signatures(skeleton_file, implementation_file)

        self.assertEqual(1, len(mismatches))
        self.assertIn("public class Reader signature differs", mismatches[0])

    def test_intentional_dataclass_field_order_mismatch_is_detected(self) -> None:
        skeleton_source = """\
from dataclasses import dataclass

@dataclass
class Record:
    first: str
    second: int
"""
        implementation_source = """\
from dataclasses import dataclass

@dataclass
class Record:
    second: int
    first: str
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            skeleton_file = root / "skeleton.py"
            implementation_file = root / "implementation.py"
            skeleton_file.write_text(skeleton_source, encoding="utf-8")
            implementation_file.write_text(implementation_source, encoding="utf-8")

            mismatches = compare_module_signatures(skeleton_file, implementation_file)

        self.assertEqual(1, len(mismatches))
        self.assertIn("public class Record signature differs", mismatches[0])

    def test_missing_public_binding_is_detected_without_comparing_value(self) -> None:
        skeleton_source = """\
TaskId = object()
TaskProblemType = str
"""
        implementation_source = """\
TaskId = str
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            skeleton_file = root / "skeleton.py"
            implementation_file = root / "implementation.py"
            skeleton_file.write_text(skeleton_source, encoding="utf-8")
            implementation_file.write_text(implementation_source, encoding="utf-8")

            mismatches = compare_module_signatures(skeleton_file, implementation_file)

        self.assertEqual(1, len(mismatches))
        self.assertIn("missing public binding TaskProblemType", mismatches[0])

    def test_enum_member_order_mismatch_is_detected(self) -> None:
        skeleton_source = """\
from enum import Enum, auto

class Kind(Enum):
    FIRST = auto()
    SECOND = auto()
"""
        implementation_source = """\
from enum import Enum, auto

class Kind(Enum):
    SECOND = auto()
    FIRST = auto()
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            skeleton_file = root / "skeleton.py"
            implementation_file = root / "implementation.py"
            skeleton_file.write_text(skeleton_source, encoding="utf-8")
            implementation_file.write_text(implementation_source, encoding="utf-8")

            mismatches = compare_module_signatures(skeleton_file, implementation_file)

        self.assertEqual(1, len(mismatches))
        self.assertIn("public class Kind signature differs", mismatches[0])

    def test_missing_implementation_module_is_not_a_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            skeleton_root = root / "skeleton"
            implementation_root = root / "src"
            skeleton_root.mkdir()
            implementation_root.mkdir()
            (skeleton_root / "future.py").write_text(
                "def future(value: str) -> str:\n    ...\n",
                encoding="utf-8",
            )

            self.assertEqual(
                (),
                collect_signature_mismatches(skeleton_root, implementation_root),
            )

    def test_implementation_only_public_module_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            skeleton_root = root / "skeleton"
            implementation_root = root / "src"
            skeleton_root.mkdir()
            implementation_root.mkdir()
            (implementation_root / "extra.py").write_text(
                "def public_helper() -> None:\n    return None\n",
                encoding="utf-8",
            )

            mismatches = collect_signature_mismatches(
                skeleton_root,
                implementation_root,
            )

        self.assertEqual(1, len(mismatches))
        self.assertIn("implementation-only module exposes public surface", mismatches[0])

    def test_private_implementation_only_module_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            skeleton_root = root / "skeleton"
            implementation_root = root / "src"
            skeleton_root.mkdir()
            implementation_root.mkdir()
            (implementation_root / "_helpers.py").write_text(
                "def helper() -> None:\n    return None\n",
                encoding="utf-8",
            )

            self.assertEqual(
                (),
                collect_signature_mismatches(skeleton_root, implementation_root),
            )


if __name__ == "__main__":
    unittest.main()
