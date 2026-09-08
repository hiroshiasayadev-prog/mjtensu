from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from mldb.src.catalog.architecture import (
    Architecture,
    ArchitectureImplementation,
    ArchitectureInterface,
    ArchitectureStatus,
    ArchitectureStructure,
)
from mldb.src.catalog.corpus import Corpus, CorpusArtifact, CorpusDataSpec
from mldb.src.catalog.task import (
    CategoricalTarget,
    Task,
    TaskInput,
    TaskScope,
)
from mldb.src.common.errors import LifecycleConflictError, MldbError
from mldb.src.common.ids import (
    ArchitectureId,
    CorpusId,
    ModelId,
    TaskId,
    TrainingRunId,
    TrainProtocolId,
)
from mldb.src.common.parameters import PublicParameterDeclaration
from mldb.src.model.identity import (
    Model,
    model_for_completed_training_run,
    model_id_for_training_run,
    validate_model_metadata,
)
from mldb.src.model.loading import ModelHandle, load_model
from mldb.src.model.persistence import ensure_model_for_completed_training_run
from mldb.src.repository._local_filesystem import LocalFilesystem
from mldb.src.repository.layout import RepositoryLayout
from mldb.src.runtime.catalog_handles import (
    ArchitectureHandle,
    CorpusHandle,
    TaskHandle,
)
from mldb.src.runtime.definition_handles import TrainProtocolHandle
from mldb.src.training.preflight import preflight_training
from mldb.src.training.protocol import (
    TrainProtocol,
    TrainProtocolImplementation,
    TrainProtocolStatus,
)
from mldb.src.training.run import (
    TrainingRun,
    TrainingRunExecution,
    TrainingRunResult,
    TrainingRunStatus,
)
from mldb.src.training.weights import CanonicalWeightsArtifact


class _CountingFilesystem(LocalFilesystem):
    def __init__(self) -> None:
        self.text_replacements = 0

    def replace_text(self, path: Path, text: str, *, encoding: str) -> None:
        self.text_replacements += 1
        super().replace_text(path, text, encoding=encoding)


def _completed_run(
    *,
    artifact: CanonicalWeightsArtifact | None = None,
    status: TrainingRunStatus = TrainingRunStatus.COMPLETED,
) -> TrainingRun:
    if artifact is None:
        artifact = CanonicalWeightsArtifact(
            format="pytorch-state-dict",
            path="artifacts/weights.pt",
            sha256="a" * 64,
            bytes=123,
        )
    return TrainingRun(
        schema="mjtensu.mldb/training-run/v1",
        id=TrainingRunId("tr-20260908-001"),
        status=status,
        corpus=CorpusId("gray-v1"),
        architecture=ArchitectureId("plain-v1"),
        train_protocol=TrainProtocolId("standard-v1"),
        parameters={"epochs": 10},
        execution=TrainingRunExecution(
            seed=42,
            started_at="start",
            finished_at="finish" if status is not TrainingRunStatus.RUNNING else None,
        ),
        result=(
            TrainingRunResult(weights=artifact)
            if status is TrainingRunStatus.COMPLETED
            else None
        ),
    )


def _write_canonical_training_run(
    layout: RepositoryLayout,
    run: TrainingRun,
) -> None:
    assert run.result is not None
    path = layout.training_run_paths(run.id).metadata_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": run.schema,
                "id": run.id,
                "status": run.status.value,
                "corpus": run.corpus,
                "architecture": run.architecture,
                "train_protocol": run.train_protocol,
                "parameters": dict(run.parameters),
                "execution": {
                    "seed": run.execution.seed,
                    "started_at": run.execution.started_at,
                    "finished_at": run.execution.finished_at,
                },
                "result": {
                    "weights": {
                        "format": run.result.weights.format,
                        "path": run.result.weights.path,
                        "sha256": run.result.weights.sha256,
                        "bytes": run.result.weights.bytes,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


class TrainingPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.layout = RepositoryLayout(self.root)
        self.filesystem = LocalFilesystem()
        self.task_id = TaskId("tile-task-v1")
        self.architecture_path = self.root / "architecture.py"
        self.protocol_path = self.root / "protocol.py"
        self.architecture_bytes = b"raise RuntimeError('must not import during preflight')\n"
        self.protocol_bytes = b"raise RuntimeError('must not import during preflight')\n"
        self.architecture_path.write_bytes(self.architecture_bytes)
        self.protocol_path.write_bytes(self.protocol_bytes)

        task = Task(
            schema="mjtensu.mldb/task/v1",
            id=self.task_id,
            name="tiles",
            problem_type="classification",
            description="tiles",
            input=TaskInput(semantic_unit="tile"),
            target=CategoricalTarget(type="categorical", labels=("a", "b")),
            semantics={},
            scope=TaskScope(includes=("tiles",), excludes=()),
        )
        self.task = TaskHandle(metadata=task, metadata_path=self.root / "task.yaml")
        corpus = Corpus(
            schema="mjtensu.mldb/corpus/v1",
            id=CorpusId("gray-v1"),
            task=self.task_id,
            artifact=CorpusArtifact(format="sqlite", sha256="0" * 64),
            data=CorpusDataSpec(schema="image-v1", table="samples"),
            representation={},
            builder_parameters={},
            splits={},
        )
        self.corpus = CorpusHandle(
            metadata=corpus,
            metadata_path=self.root / "corpus.yaml",
            artifact_path=self.root / "corpus.sqlite",
            builder_path=self.root / "corpus.py",
        )
        architecture = Architecture(
            schema="mjtensu.mldb/architecture/v1",
            id=ArchitectureId("plain-v1"),
            status=ArchitectureStatus.SEALED,
            task=self.task_id,
            name="plain",
            family="plain",
            description="plain",
            implementation=ArchitectureImplementation(
                framework="pytorch",
                entrypoint="build",
                sha256=hashlib.sha256(self.architecture_bytes).hexdigest(),
            ),
            interface=ArchitectureInterface(input={}, output={"kind": "logits"}),
            structure=ArchitectureStructure(summary="linear"),
        )
        self.architecture = ArchitectureHandle(
            metadata=architecture,
            metadata_path=self.root / "architecture.yaml",
            implementation_path=self.architecture_path,
        )
        protocol = TrainProtocol(
            schema="mjtensu.mldb/train-protocol/v1",
            id=TrainProtocolId("standard-v1"),
            status=TrainProtocolStatus.SEALED,
            task=self.task_id,
            name="standard",
            description="standard",
            implementation=TrainProtocolImplementation(
                entrypoint="train",
                sha256=hashlib.sha256(self.protocol_bytes).hexdigest(),
            ),
            parameters={
                "epochs": PublicParameterDeclaration(default=10),
                "amp": PublicParameterDeclaration(default=False),
            },
        )
        self.protocol = TrainProtocolHandle(
            metadata=protocol,
            metadata_path=self.root / "protocol.yaml",
            implementation_path=self.protocol_path,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _preflight(
        self,
        *,
        seed: object = 42,
        overrides: dict[str, object] | None = None,
        architecture: ArchitectureHandle | None = None,
        protocol: TrainProtocolHandle | None = None,
    ):
        with (
            patch(
                "mldb.src.training.preflight.resolve_corpus",
                return_value=self.corpus,
            ),
            patch(
                "mldb.src.training.preflight.resolve_architecture",
                return_value=architecture or self.architecture,
            ),
            patch(
                "mldb.src.training.preflight.resolve_train_protocol",
                return_value=protocol or self.protocol,
            ),
            patch(
                "mldb.src.training.preflight.resolve_task",
                return_value=self.task,
            ) as resolve_task,
        ):
            result = preflight_training(
                self.corpus.metadata.id,
                (architecture or self.architecture).metadata.id,
                (protocol or self.protocol).metadata.id,
                seed,  # type: ignore[arg-type]
                overrides or {},
                self.layout,
                self.filesystem,
            )
        return result, resolve_task

    def test_resolves_exact_task_compatibility_and_complete_parameters(self) -> None:
        result, resolve_task = self._preflight(overrides={"epochs": 20})

        self.assertEqual(self.task_id, result.task.metadata.id)
        self.assertEqual({"epochs": 20, "amp": False}, result.parameters)
        self.assertEqual(42, result.seed)
        self.assertNotIn("seed", result.parameters)
        resolve_task.assert_called_once_with(
            self.task_id,
            self.layout,
            self.filesystem,
        )

    def test_rejects_task_incompatibility_and_unknown_parameter(self) -> None:
        incompatible_metadata = replace(
            self.architecture.metadata,
            task=TaskId("other-task-v1"),
        )
        incompatible = replace(
            self.architecture,
            metadata=incompatible_metadata,
        )
        with self.assertRaises(ValueError):
            self._preflight(architecture=incompatible)

        with self.assertRaisesRegex(ValueError, "unknown public parameter"):
            self._preflight(overrides={"unknown": 1})

    def test_rejects_draft_execution_assets(self) -> None:
        draft_architecture = replace(
            self.architecture,
            metadata=replace(
                self.architecture.metadata,
                status=ArchitectureStatus.DRAFT,
                implementation=replace(
                    self.architecture.metadata.implementation,
                    sha256=None,
                ),
            ),
        )
        with self.assertRaises(ValueError):
            self._preflight(architecture=draft_architecture)

        draft_protocol = replace(
            self.protocol,
            metadata=replace(
                self.protocol.metadata,
                status=TrainProtocolStatus.DRAFT,
                implementation=replace(
                    self.protocol.metadata.implementation,
                    sha256=None,
                ),
            ),
        )
        with self.assertRaises(ValueError):
            self._preflight(protocol=draft_protocol)

    def test_rejects_static_integrity_mismatch(self) -> None:
        bad_architecture = replace(
            self.architecture,
            metadata=replace(
                self.architecture.metadata,
                implementation=replace(
                    self.architecture.metadata.implementation,
                    sha256="0" * 64,
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self._preflight(architecture=bad_architecture)

    def test_seed_contract_preserves_integers_and_rejects_bool_or_float(self) -> None:
        large_seed = 2**100
        result, _ = self._preflight(seed=large_seed)
        self.assertEqual(large_seed, result.seed)
        self.assertIs(type(result.seed), int)

        for invalid in (True, False, 1.0, "1"):
            with self.subTest(seed=invalid):
                with self.assertRaises(ValueError):
                    self._preflight(seed=invalid)


class ModelIdentityAndPersistenceTests(unittest.TestCase):
    def test_deterministic_model_identity_and_validation(self) -> None:
        run_id = TrainingRunId("tr-20260908-017")
        self.assertEqual(
            ModelId("mdl-20260908-017"),
            model_id_for_training_run(run_id),
        )
        model = Model(
            schema="mjtensu.mldb/model/v1",
            id=ModelId("mdl-20260908-017"),
            training_run=run_id,
        )
        self.assertTrue(validate_model_metadata(model).valid)

        mismatch = replace(
            model,
            training_run=TrainingRunId("tr-20260908-018"),
        )
        self.assertFalse(validate_model_metadata(mismatch).valid)

    def test_model_construction_is_completed_only(self) -> None:
        expected = model_for_completed_training_run(_completed_run())
        self.assertEqual(ModelId("mdl-20260908-001"), expected.id)

        with self.assertRaises(LifecycleConflictError):
            model_for_completed_training_run(
                _completed_run(status=TrainingRunStatus.RUNNING)
            )

    def test_ensure_is_idempotent_and_does_not_copy_weights(self) -> None:
        run = _completed_run()
        with tempfile.TemporaryDirectory() as temporary_directory:
            layout = RepositoryLayout(Path(temporary_directory))
            filesystem = _CountingFilesystem()
            _write_canonical_training_run(layout, run)
            first = ensure_model_for_completed_training_run(
                run,
                layout,
                filesystem,
            )
            second = ensure_model_for_completed_training_run(
                run,
                layout,
                filesystem,
            )

            self.assertEqual(first, second)
            self.assertEqual(1, filesystem.text_replacements)
            self.assertEqual(
                (
                    "schema: mjtensu.mldb/model/v1\n"
                    "id: mdl-20260908-001\n"
                    "training_run: tr-20260908-001\n"
                ),
                layout.model_metadata_path(first.id).read_text(encoding="utf-8"),
            )
            self.assertFalse(
                (layout.data_root / "models" / "weights.pt").exists()
            )

    def test_ensure_rejects_non_completed_and_conflicting_model(self) -> None:
        running = _completed_run(status=TrainingRunStatus.RUNNING)
        with tempfile.TemporaryDirectory() as temporary_directory:
            layout = RepositoryLayout(Path(temporary_directory))
            filesystem = LocalFilesystem()
            with (
                patch(
                    "mldb.src.model.persistence._read_canonical_training_run",
                    return_value=running,
                ),
                self.assertRaises(LifecycleConflictError),
            ):
                ensure_model_for_completed_training_run(
                    running,
                    layout,
                    filesystem,
                )

            completed = _completed_run()
            destination = layout.model_metadata_path(
                ModelId("mdl-20260908-001")
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                "schema: mjtensu.mldb/model/v1\n"
                "id: mdl-20260908-001\n"
                "training_run: tr-20260908-002\n",
                encoding="utf-8",
            )
            with (
                patch(
                    "mldb.src.model.persistence._read_canonical_training_run",
                    return_value=completed,
                ),
                self.assertRaises(LifecycleConflictError),
            ):
                ensure_model_for_completed_training_run(
                    completed,
                    layout,
                    filesystem,
                )

    def test_ensure_rejects_duplicate_lineage_under_another_identity(self) -> None:
        run = _completed_run()
        with tempfile.TemporaryDirectory() as temporary_directory:
            layout = RepositoryLayout(Path(temporary_directory))
            filesystem = LocalFilesystem()
            models = layout.entity_directory.__self__.data_root / "models"
            models.mkdir(parents=True)
            (models / "mdl-20260908-002.yaml").write_text(
                "schema: mjtensu.mldb/model/v1\n"
                "id: mdl-20260908-002\n"
                "training_run: tr-20260908-001\n",
                encoding="utf-8",
            )
            with (
                patch(
                    "mldb.src.model.persistence._read_canonical_training_run",
                    return_value=run,
                ),
                self.assertRaises(LifecycleConflictError),
            ):
                ensure_model_for_completed_training_run(
                    run,
                    layout,
                    filesystem,
                )


class ModelLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.architecture_path = self.root / "plain-v1.py"
        architecture_source = (
            "import torch\n\n"
            "def build():\n"
            "    return torch.nn.Linear(2, 2)\n"
        )
        self.architecture_path.write_text(architecture_source, encoding="utf-8")
        architecture = Architecture(
            schema="mjtensu.mldb/architecture/v1",
            id=ArchitectureId("plain-v1"),
            status=ArchitectureStatus.SEALED,
            task=TaskId("tile-task-v1"),
            name="plain",
            family="plain",
            description="plain",
            implementation=ArchitectureImplementation(
                framework="pytorch",
                entrypoint="build",
                sha256=hashlib.sha256(
                    self.architecture_path.read_bytes()
                ).hexdigest(),
            ),
            interface=ArchitectureInterface(input={}, output={"kind": "logits"}),
            structure=ArchitectureStructure(summary="linear"),
        )
        self.architecture = ArchitectureHandle(
            metadata=architecture,
            metadata_path=None,
            implementation_path=self.architecture_path,
        )
        self.weights_path = self.root / "weights.pt"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _handle_for_state(
        self,
        state: dict[str, torch.Tensor],
        *,
        sha256: str | None = None,
        byte_count: int | None = None,
    ) -> ModelHandle:
        torch.save(state, self.weights_path)
        data = self.weights_path.read_bytes()
        artifact = CanonicalWeightsArtifact(
            format="pytorch-state-dict",
            path="artifacts/weights.pt",
            sha256=sha256 or hashlib.sha256(data).hexdigest(),
            bytes=len(data) if byte_count is None else byte_count,
        )
        run = _completed_run(artifact=artifact)
        model = Model(
            schema="mjtensu.mldb/model/v1",
            id=ModelId("mdl-20260908-001"),
            training_run=run.id,
        )
        return ModelHandle(
            metadata=model,
            metadata_path=None,
            training_run=run,
            training_run_metadata_path=None,
            weights_path=self.weights_path,
            architecture=self.architecture,
        )

    def test_load_model_rejects_sha_and_byte_mismatch(self) -> None:
        state = torch.nn.Linear(2, 2).state_dict()
        with self.assertRaisesRegex(MldbError, "SHA-256"):
            load_model(self._handle_for_state(state, sha256="0" * 64))
        with self.assertRaisesRegex(MldbError, "byte count"):
            load_model(self._handle_for_state(state, byte_count=1))

    def test_load_model_rejects_strict_topology_mismatch(self) -> None:
        incompatible = torch.nn.Linear(3, 3).state_dict()
        with self.assertRaises(RuntimeError):
            load_model(self._handle_for_state(incompatible))

    def test_load_model_successfully_builds_fresh_architecture(self) -> None:
        torch.manual_seed(17)
        source = torch.nn.Linear(2, 2)
        handle = self._handle_for_state(source.state_dict())

        loaded = load_model(handle)

        self.assertIsInstance(loaded, torch.nn.Linear)
        for key, expected in source.state_dict().items():
            self.assertTrue(torch.equal(expected, loaded.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
