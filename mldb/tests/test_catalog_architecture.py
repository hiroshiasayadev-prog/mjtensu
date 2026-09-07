from __future__ import annotations

import unittest

from mldb.src.catalog.architecture import (
    Architecture,
    ArchitectureImplementation,
    ArchitectureInterface,
    ArchitectureStatus,
    ArchitectureStructure,
    validate_architecture_metadata,
)
from mldb.src.common.ids import ArchitectureId, TaskId


class CatalogArchitectureTests(unittest.TestCase):
    def test_draft_may_omit_implementation_hash(self) -> None:
        architecture = self._architecture(status=ArchitectureStatus.DRAFT, sha256=None)

        report = validate_architecture_metadata(architecture)

        self.assertTrue(report.valid)

    def test_sealed_requires_valid_implementation_hash(self) -> None:
        missing = self._architecture(status=ArchitectureStatus.SEALED, sha256=None)
        malformed = self._architecture(status=ArchitectureStatus.SEALED, sha256="bad")
        valid = self._architecture(status=ArchitectureStatus.SEALED, sha256="f" * 64)

        self.assertEqual(
            ["implementation.sha256"],
            [issue.path for issue in validate_architecture_metadata(missing).issues],
        )
        self.assertEqual(
            ["implementation.sha256"],
            [issue.path for issue in validate_architecture_metadata(malformed).issues],
        )
        self.assertTrue(validate_architecture_metadata(valid).valid)

    def test_versioned_identity_and_fixed_build_metadata_are_validated(self) -> None:
        architecture = self._architecture(
            architecture_id="plain-cnn-v0",
            framework="other",
            entrypoint="create",
        )

        report = validate_architecture_metadata(architecture)

        self.assertEqual(
            ["id", "implementation.framework", "implementation.entrypoint"],
            [issue.path for issue in report.issues],
        )

    def test_metadata_validation_does_not_invoke_or_load_build(self) -> None:
        architecture = self._architecture()

        self.assertTrue(validate_architecture_metadata(architecture).valid)

    def _architecture(
        self,
        *,
        architecture_id: str = "plain-cnn-v1",
        status: ArchitectureStatus = ArchitectureStatus.DRAFT,
        sha256: str | None = None,
        framework: str = "pytorch",
        entrypoint: str = "build",
    ) -> Architecture:
        return Architecture(
            schema="mjtensu.mldb/architecture/v1",
            id=ArchitectureId(architecture_id),
            status=status,
            task=TaskId("test-task-v1"),
            name="Plain CNN",
            family="plain-cnn",
            description="Test architecture",
            implementation=ArchitectureImplementation(
                framework=framework,
                entrypoint=entrypoint,
                sha256=sha256,
            ),
            interface=ArchitectureInterface(
                input={"kind": "image-tensor"},
                output={"kind": "classification-logits"},
            ),
            structure=ArchitectureStructure(summary="A small classifier."),
            parameters={"channels": [16, 32]},
        )


if __name__ == "__main__":
    unittest.main()
