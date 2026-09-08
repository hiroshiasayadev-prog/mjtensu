from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mldb.src.api.controller import seal_definition
from mldb.src.common.ids import EntityKind
from mldb.src.repository._local_filesystem import LocalFilesystem
from mldb.src.repository.layout import RepositoryLayout
from mldb.src.verification.sealing import PytestRunResult


class _PytestSummaryPlugin:
    def __init__(self) -> None:
        self.collected = 0
        self.passed = 0
        self.failed = 0
        self.errors = 0

    def pytest_collection_finish(self, session) -> None:
        self.collected = len(session.items)

    def pytest_collectreport(self, report) -> None:
        if report.failed:
            self.errors += 1

    def pytest_runtest_logreport(self, report) -> None:
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1
        elif report.failed:
            self.errors += 1


class _PytestRunner:
    def run(self, test_dir: Path) -> PytestRunResult:
        plugin = _PytestSummaryPlugin()
        try:
            exit_code = pytest.main(["-q", str(test_dir)], plugins=[plugin])
        except BaseException as error:
            return PytestRunResult(
                invocation_completed=False,
                collected_count=plugin.collected,
                passed_count=plugin.passed,
                failed_count=plugin.failed,
                error_count=plugin.errors + 1,
                diagnostics=(f"pytest invocation failed: {type(error).__name__}: {error}",),
            )
        return PytestRunResult(
            invocation_completed=True,
            collected_count=plugin.collected,
            passed_count=plugin.passed,
            failed_count=plugin.failed,
            error_count=plugin.errors,
            diagnostics=(f"pytest exit code: {int(exit_code)}",),
        )


_KIND_BY_NAME = {
    "architecture": EntityKind.ARCHITECTURE,
    "train-protocol": EntityKind.TRAIN_PROTOCOL,
    "evaluation-protocol": EntityKind.EVALUATION_PROTOCOL,
    "study": EntityKind.STUDY,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seal one authored MLDB definition through the public Controller gate.")
    parser.add_argument("--kind", choices=tuple(_KIND_BY_NAME), required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = RepositoryLayout(args.repo_root.resolve())
    filesystem = LocalFilesystem()
    kind = _KIND_BY_NAME[args.kind]
    runner = None if kind is EntityKind.STUDY else _PytestRunner()
    response = seal_definition(kind, args.id, layout, filesystem, runner)
    print(f"{response.kind.value} {response.id}: {response.result}")


if __name__ == "__main__":
    main()
