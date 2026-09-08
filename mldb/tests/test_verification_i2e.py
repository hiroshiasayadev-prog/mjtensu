from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import sys
import types
import unittest
from unittest.mock import patch

from mldb.src.catalog.architecture import (
    Architecture,
    ArchitectureImplementation,
    ArchitectureInterface,
    ArchitectureStatus,
    ArchitectureStructure,
)
from mldb.src.common.errors import LifecycleConflictError, NotFoundError
from mldb.src.common.ids import (
    ArchitectureId,
    EvaluationProtocolId,
    StudyId,
    TaskId,
    TrainProtocolId,
)
from mldb.src.common.parameters import PublicParameterDeclaration
from mldb.src.repository._local_filesystem import LocalFilesystem
from mldb.src.repository.layout import RepositoryLayout
from mldb.src.runtime.catalog_handles import ArchitectureHandle
from mldb.src.runtime.definition_handles import (
    EvaluationProtocolHandle,
    StudyHandle,
    TrainProtocolHandle,
)
from mldb.src.runtime._yaml import _load_yaml
from mldb.src.evaluation.protocol import (
    EvaluationOutputs,
    EvaluationProtocol,
    EvaluationProtocolImplementation,
    EvaluationProtocolStatus,
)
from mldb.src.study.definition import (
    Study,
    StudyEvaluationStage,
    StudyStatus,
    StudyTrainingModelSource,
)
from mldb.src.verification.definition_validation import (
    validate_architecture_definition,
    validate_study_definition,
    validate_task_definition,
)
from mldb.src.training.protocol import (
    TrainProtocol,
    TrainProtocolImplementation,
    TrainProtocolStatus,
)
from mldb.src.verification.lifecycle import (
    seal_architecture,
    seal_evaluation_protocol,
    seal_study,
    seal_train_protocol,
)
from mldb.src.verification.sealing import (
    PytestRunResult,
    verify_architecture_tests,
)


class FakeFilesystem:
    def __init__(self) -> None:
        self.text: dict[Path, str] = {}
        self.data: dict[Path, bytes] = {}
        self.directories: set[Path] = set()
        self.file_exists_error: Exception | None = None
        self.read_text_error: Exception | None = None
        self.replace_calls: list[Path] = []

    def file_exists(self, path: Path) -> bool:
        if self.file_exists_error is not None:
            raise self.file_exists_error
        return path in self.text or path in self.data

    def directory_exists(self, path: Path) -> bool:
        return path in self.directories

    def read_text(self, path: Path, *, encoding: str) -> str:
        if self.read_text_error is not None:
            raise self.read_text_error
        return self.text[path]

    def read_bytes(self, path: Path) -> bytes:
        return self.data[path]

    def replace_text(self, path: Path, text: str, *, encoding: str) -> None:
        self.replace_calls.append(path)
        self.text[path] = text


class FakeRunner:
    def __init__(self, result: PytestRunResult, callback=None) -> None:
        self.result = result
        self.callback = callback
        self.calls: list[Path] = []

    def run(self, test_dir: Path) -> PytestRunResult:
        self.calls.append(test_dir)
        if self.callback is not None:
            self.callback()
        return self.result


PASS = PytestRunResult(True, 1, 1, 0, 0)


def architecture(status: ArchitectureStatus, sha256: str | None = None) -> Architecture:
    return Architecture(
        schema="mjtensu.mldb/architecture/v1",
        id=ArchitectureId("arch-v1"),
        status=status,
        task=TaskId("task-v1"),
        name="Arch",
        family="plain",
        description="desc",
        implementation=ArchitectureImplementation("pytorch", "build", sha256),
        interface=ArchitectureInterface(input={"kind": "image"}, output={"kind": "class"}),
        structure=ArchitectureStructure(summary="plain"),
    )


def train_protocol(
    status: TrainProtocolStatus,
    sha256: str | None = None,
) -> TrainProtocol:
    return TrainProtocol(
        schema="mjtensu.mldb/train-protocol/v1",
        id=TrainProtocolId("train-v1"),
        status=status,
        task=TaskId("task-v1"),
        name="Train",
        description="desc",
        implementation=TrainProtocolImplementation("train", sha256),
        parameters={"lr": PublicParameterDeclaration(default=0.1)},
    )


def evaluation_protocol(
    status: EvaluationProtocolStatus,
    sha256: str | None = None,
) -> EvaluationProtocol:
    return EvaluationProtocol(
        schema="mjtensu.mldb/evaluation-protocol/v1",
        id=EvaluationProtocolId("eval-v1"),
        status=status,
        task=TaskId("task-v1"),
        name="Eval",
        description="desc",
        implementation=EvaluationProtocolImplementation("evaluate", sha256),
        parameters={"threshold": PublicParameterDeclaration(default=0.5)},
        outputs=EvaluationOutputs(metrics={}, artifacts={}),
    )


def resolver_module(**functions):
    module = types.ModuleType("mldb.src.runtime.resolution")
    for name, value in functions.items():
        setattr(module, name, value)
    return patch.dict(sys.modules, {"mldb.src.runtime.resolution": module})


class VerificationGateTests(unittest.TestCase):
    def test_gate_matrix(self) -> None:
        root = Path("repo")
        test_dir = root / "mldb_tests/architectures/arch-v1"
        fs = FakeFilesystem()
        cases = [
            (False, PASS, False, 0),
            (True, PytestRunResult(False, 1, 1, 0, 0), False, 1),
            (True, PytestRunResult(True, 0, 0, 0, 0), False, 1),
            (True, PytestRunResult(True, 1, 0, 0, 0), False, 1),
            (True, PytestRunResult(True, 1, 1, 1, 0), False, 1),
            (True, PytestRunResult(True, 1, 1, 0, 1), False, 1),
            (True, PASS, True, 1),
        ]
        for exists, pytest_result, expected, calls in cases:
            with self.subTest(pytest_result=pytest_result, exists=exists):
                fs.directories = {test_dir} if exists else set()
                runner = FakeRunner(pytest_result)
                result = verify_architecture_tests(
                    ArchitectureId("arch-v1"), test_dir, fs, runner
                )
                self.assertIs(result.successful, expected)
                self.assertEqual(len(runner.calls), calls)


class LifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = RepositoryLayout(Path("repo"))
        self.fs = FakeFilesystem()
        self.metadata_path = self.layout.architecture_metadata_path(ArchitectureId("arch-v1"))
        self.impl_path = self.layout.architecture_implementation_path(ArchitectureId("arch-v1"))
        self.test_dir = self.layout.architecture_test_dir(ArchitectureId("arch-v1"))
        self.fs.data[self.impl_path] = b"print('stable')\n"
        self.fs.directories.add(self.test_dir)

    def _draft_yaml(self) -> str:
        return """schema: mjtensu.mldb/architecture/v1
id: arch-v1
status: draft
task: task-v1
name: Arch
family: plain
description: desc
supplemental:
  owner: alice
implementation:
  framework: pytorch
  entrypoint: build
  sha256: null
interface:
  input:
    kind: image
  output:
    kind: class
structure:
  summary: plain
"""

    def test_draft_seal_and_preserve_supplemental_metadata(self) -> None:
        self.fs.text[self.metadata_path] = self._draft_yaml()
        handle = ArchitectureHandle(architecture(ArchitectureStatus.DRAFT), self.metadata_path, self.impl_path)
        module = resolver_module(resolve_architecture=lambda *_: handle)
        with module:
            result = seal_architecture(ArchitectureId("arch-v1"), self.layout, self.fs, FakeRunner(PASS))
        self.assertEqual(result, "sealed")
        persisted = _load_yaml(self.fs.text[self.metadata_path])
        self.assertEqual(persisted["status"], "sealed")
        self.assertEqual(persisted["supplemental"], {"owner": "alice"})
        self.assertEqual(
            persisted["implementation"]["sha256"],
            hashlib.sha256(self.fs.data[self.impl_path]).hexdigest(),
        )

    def test_real_resolver_and_filesystem_draft_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = RepositoryLayout(Path(temporary))
            fs = LocalFilesystem()
            task_id = TaskId("task-v1")
            task_path = layout.task_metadata_path(task_id)
            task_path.parent.mkdir(parents=True, exist_ok=True)
            task_path.write_text(json.dumps({
                "schema": "mjtensu.mldb/task/v1", "id": str(task_id), "name": "task",
                "problem_type": "classification", "description": "task",
                "input": {"semantic_unit": "image"},
                "target": {"type": "categorical", "labels": ["a"]},
                "semantics": {}, "scope": {"includes": ["a"], "excludes": []},
            }), encoding="utf-8")
            architecture_id = ArchitectureId("real-v1")
            metadata_path = layout.architecture_metadata_path(architecture_id)
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            implementation_path = layout.architecture_implementation_path(architecture_id)
            implementation_path.write_bytes(b"def build():\n    return None\n")
            metadata_path.write_text(json.dumps({
                "schema": "mjtensu.mldb/architecture/v1", "id": str(architecture_id),
                "status": "draft", "task": str(task_id), "name": "real",
                "family": "plain", "description": "real",
                "implementation": {"framework": "pytorch", "entrypoint": "build"},
                "interface": {"input": {"kind": "image"}, "output": {"kind": "class"}},
                "structure": {"summary": "plain"}, "supplemental": {"owner": "alice"},
            }), encoding="utf-8")
            test_dir = layout.architecture_test_dir(architecture_id)
            test_dir.mkdir(parents=True, exist_ok=True)
            self.assertEqual("sealed", seal_architecture(architecture_id, layout, fs, FakeRunner(PASS)))
            persisted = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual("sealed", persisted["status"])
            self.assertEqual({"owner": "alice"}, persisted["supplemental"])
            self.assertEqual(hashlib.sha256(implementation_path.read_bytes()).hexdigest(), persisted["implementation"]["sha256"])

    def test_mutation_during_pytest_refuses_seal(self) -> None:
        self.fs.text[self.metadata_path] = self._draft_yaml()
        handle = ArchitectureHandle(architecture(ArchitectureStatus.DRAFT), self.metadata_path, self.impl_path)
        runner = FakeRunner(PASS, callback=lambda: self.fs.data.__setitem__(self.impl_path, b"changed"))
        with resolver_module(resolve_architecture=lambda *_: handle):
            with self.assertRaises(LifecycleConflictError):
                seal_architecture(ArchitectureId("arch-v1"), self.layout, self.fs, runner)
        self.assertIn("status: draft", self.fs.text[self.metadata_path])

    def test_already_sealed_does_not_run_pytest_and_checks_hash(self) -> None:
        digest = hashlib.sha256(self.fs.data[self.impl_path]).hexdigest()
        self.fs.text[self.metadata_path] = self._draft_yaml().replace(
            "status: draft", "status : sealed"
        ).replace("sha256: null", f"sha256 : {digest}")
        handle = ArchitectureHandle(architecture(ArchitectureStatus.SEALED, digest), self.metadata_path, self.impl_path)
        runner = FakeRunner(PASS)
        with resolver_module(resolve_architecture=lambda *_: handle):
            result = seal_architecture(ArchitectureId("arch-v1"), self.layout, self.fs, runner)
        self.assertEqual(result, "already_sealed")
        self.assertEqual(runner.calls, [])
        self.assertEqual(self.fs.replace_calls, [])

        self.fs.data[self.impl_path] = b"mutated"
        with resolver_module(resolve_architecture=lambda *_: handle):
            with self.assertRaises(LifecycleConflictError):
                seal_architecture(ArchitectureId("arch-v1"), self.layout, self.fs, runner)
        self.assertEqual(runner.calls, [])

    def test_study_seal_runs_no_pytest(self) -> None:
        study_id = StudyId("study-v1")
        metadata_path = self.layout.study_metadata_path(study_id)
        self.fs.text[metadata_path] = """schema: mjtensu.mldb/study/v1
id: study-v1
status: draft
name: Study
description: desc
supplemental: keep-me
model:
  train:
    corpus: corpus-v1
    protocol: train-v1
    architectures: [arch-v1]
    parameters: {}
    seeds: [1]
evaluations: []
"""
        study = Study(
            schema="mjtensu.mldb/study/v1",
            id=study_id,
            status=StudyStatus.DRAFT,
            name="Study",
            description="desc",
            model=StudyTrainingModelSource(
                corpus="corpus-v1",
                protocol="train-v1",
                architectures=("arch-v1",),
                parameters={},
                seeds=(1,),
            ),
            evaluations=(StudyEvaluationStage("eval", "eval-corpus-v1", "eval-v1", {}),),
        )
        handle = StudyHandle(study, metadata_path)
        with resolver_module(resolve_study=lambda *_: handle):
            result = seal_study(study_id, self.layout, self.fs)
        self.assertEqual(result, "sealed")
        persisted = _load_yaml(self.fs.text[metadata_path])
        self.assertEqual(persisted["supplemental"], "keep-me")
        self.assertEqual(persisted["status"], "sealed")


    def test_semantic_patch_preserves_full_authored_mapping_for_all_kinds(self) -> None:
        architecture_id = ArchitectureId("arch-v1")
        architecture_authored = '''schema : mjtensu.mldb/architecture/v1
id : arch-v1
status : draft
task : task-v1
name : Arch
family : plain
description : desc
implementation :
  framework : pytorch
  entrypoint : build
interface :
  input : {kind: image}
  output : {kind: class}
structure :
  summary : plain
supplemental :
  deep :
    unicode : "?: ? #1"
    nested : [{a: {b: 1}, c: [2, 3]}, null, true]
'''
        self.fs.text[self.metadata_path] = architecture_authored
        architecture_handle = ArchitectureHandle(
            architecture(ArchitectureStatus.DRAFT),
            self.metadata_path,
            self.impl_path,
        )
        architecture_expected = _load_yaml(architecture_authored)
        architecture_expected["status"] = "sealed"
        architecture_expected["implementation"]["sha256"] = hashlib.sha256(
            self.fs.data[self.impl_path]
        ).hexdigest()
        with resolver_module(resolve_architecture=lambda *_: architecture_handle):
            self.assertEqual(
                "sealed",
                seal_architecture(architecture_id, self.layout, self.fs, FakeRunner(PASS)),
            )
        self.assertEqual(
            architecture_expected,
            _load_yaml(self.fs.text[self.metadata_path]),
        )

        train_id = TrainProtocolId("train-v1")
        train_metadata_path = self.layout.train_protocol_metadata_path(train_id)
        train_impl_path = self.layout.train_protocol_implementation_path(train_id)
        train_test_dir = self.layout.train_protocol_test_dir(train_id)
        self.fs.data[train_impl_path] = b"def train(context):\n    return None\n"
        self.fs.directories.add(train_test_dir)
        train_authored = '''schema: mjtensu.mldb/train-protocol/v1
id: train-v1
status : draft
task: task-v1
name: Train
description: desc
implementation :
  entrypoint : train
parameters:
  lr:
    default: 0.1
    description: "learning rate"
    type: number
    minimum: 0.0001
    maximum: 1.0
    suggested: [0.01, 0.1]
supplemental:
  owner:
    team: ml
    tags: [fast, stable]
'''
        self.fs.text[train_metadata_path] = train_authored
        train_handle = TrainProtocolHandle(
            train_protocol(TrainProtocolStatus.DRAFT),
            train_metadata_path,
            train_impl_path,
        )
        train_expected = _load_yaml(train_authored)
        train_expected["status"] = "sealed"
        train_expected["implementation"]["sha256"] = hashlib.sha256(
            self.fs.data[train_impl_path]
        ).hexdigest()
        train_runner = FakeRunner(PASS)
        with resolver_module(resolve_train_protocol=lambda *_: train_handle):
            self.assertEqual(
                "sealed",
                seal_train_protocol(train_id, self.layout, self.fs, train_runner),
            )
        self.assertEqual([train_test_dir], train_runner.calls)
        self.assertEqual(
            train_expected,
            _load_yaml(self.fs.text[train_metadata_path]),
        )

        evaluation_id = EvaluationProtocolId("eval-v1")
        evaluation_metadata_path = self.layout.evaluation_protocol_metadata_path(evaluation_id)
        evaluation_impl_path = self.layout.evaluation_protocol_implementation_path(evaluation_id)
        evaluation_test_dir = self.layout.evaluation_protocol_test_dir(evaluation_id)
        self.fs.data[evaluation_impl_path] = b"def evaluate(context):\n    return None\n"
        self.fs.directories.add(evaluation_test_dir)
        evaluation_authored = '''schema: mjtensu.mldb/evaluation-protocol/v1
id: eval-v1
status : draft
task: task-v1
name: Eval
description: desc
implementation:
  entrypoint: evaluate
parameters:
  threshold:
    default: 0.5
    description: "decision threshold"
    type: number
    minimum: 0.0
    maximum: 1.0
    suggested: [0.25, 0.5, 0.75]
outputs:
  metrics: {}
  artifacts: {}
supplemental:
  presentation:
    labels: ["??", "recall"]
    detail: {enabled: true, note: "keep: punctuation # literal"}
'''
        self.fs.text[evaluation_metadata_path] = evaluation_authored
        evaluation_handle = EvaluationProtocolHandle(
            evaluation_protocol(EvaluationProtocolStatus.DRAFT),
            evaluation_metadata_path,
            evaluation_impl_path,
        )
        evaluation_expected = _load_yaml(evaluation_authored)
        evaluation_expected["status"] = "sealed"
        evaluation_expected["implementation"]["sha256"] = hashlib.sha256(
            self.fs.data[evaluation_impl_path]
        ).hexdigest()
        evaluation_runner = FakeRunner(PASS)
        with resolver_module(resolve_evaluation_protocol=lambda *_: evaluation_handle):
            self.assertEqual(
                "sealed",
                seal_evaluation_protocol(
                    evaluation_id, self.layout, self.fs, evaluation_runner
                ),
            )
        self.assertEqual([evaluation_test_dir], evaluation_runner.calls)
        self.assertEqual(
            evaluation_expected,
            _load_yaml(self.fs.text[evaluation_metadata_path]),
        )

        study_id = StudyId("study-v1")
        study_metadata_path = self.layout.study_metadata_path(study_id)
        study_authored = '''schema: mjtensu.mldb/study/v1
id : study-v1
status : draft
name: Study
description: desc
model:
  train:
    corpus: corpus-v1
    protocol: train-v1
    architectures: [arch-v1]
    parameters: {}
    seeds: [1]
evaluations: []
supplemental:
  deep:
    one:
      two:
        three: {message: "preserve me", values: [1, 2, null, false]}
'''
        self.fs.text[study_metadata_path] = study_authored
        study = Study(
            schema="mjtensu.mldb/study/v1",
            id=study_id,
            status=StudyStatus.DRAFT,
            name="Study",
            description="desc",
            model=StudyTrainingModelSource(
                corpus="corpus-v1",
                protocol="train-v1",
                architectures=("arch-v1",),
                parameters={},
                seeds=(1,),
            ),
            evaluations=(StudyEvaluationStage("eval", "eval-corpus-v1", "eval-v1", {}),),
        )
        study_expected = _load_yaml(study_authored)
        study_expected["status"] = "sealed"
        with resolver_module(
            resolve_study=lambda *_: StudyHandle(study, study_metadata_path)
        ):
            self.assertEqual(
                "sealed",
                seal_study(study_id, self.layout, self.fs),
            )
        self.assertEqual(study_expected, _load_yaml(self.fs.text[study_metadata_path]))


class DefinitionValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = RepositoryLayout(Path("repo"))
        self.fs = FakeFilesystem()

    def test_missing_requested_definition_is_not_found(self) -> None:
        with self.assertRaises(NotFoundError):
            validate_task_definition(TaskId("task-v1"), self.layout, self.fs)

    def test_dependency_not_found_is_validation_issue(self) -> None:
        path = self.layout.task_metadata_path(TaskId("task-v1"))
        self.fs.text[path] = "exists"
        def missing(*_):
            raise NotFoundError("upstream missing")
        with resolver_module(resolve_task=missing):
            result = validate_task_definition(TaskId("task-v1"), self.layout, self.fs)
        self.assertFalse(result.valid)
        self.assertEqual(result.issues[0].code, "definition.dependency.invalid")

    def test_resolver_swallowed_filesystem_exception_is_repropagated(self) -> None:
        path = self.layout.task_metadata_path(TaskId("task-v1"))
        self.fs.text[path] = "exists"
        self.fs.read_text_error = OSError("read failed")
        with self.assertRaisesRegex(OSError, "read failed"):
            validate_task_definition(TaskId("task-v1"), self.layout, self.fs)

    def test_sealed_architecture_sha_validation(self) -> None:
        architecture_id = ArchitectureId("arch-v1")
        metadata_path = self.layout.architecture_metadata_path(architecture_id)
        implementation_path = self.layout.architecture_implementation_path(architecture_id)
        self.fs.text[metadata_path] = "exists"
        self.fs.data[implementation_path] = b"implementation"
        digest = hashlib.sha256(self.fs.data[implementation_path]).hexdigest()
        handle = ArchitectureHandle(
            architecture(ArchitectureStatus.SEALED, digest),
            metadata_path,
            implementation_path,
        )
        with resolver_module(resolve_architecture=lambda *_: handle):
            self.assertTrue(validate_architecture_definition(architecture_id, self.layout, self.fs).valid)
        self.fs.data[implementation_path] = b"changed"
        with resolver_module(resolve_architecture=lambda *_: handle):
            result = validate_architecture_definition(architecture_id, self.layout, self.fs)
        self.assertFalse(result.valid)
        self.assertEqual(result.issues[-1].code, "definition.implementation.sha256_mismatch")

    def test_infrastructure_exception_propagates(self) -> None:
        self.fs.file_exists_error = OSError("disk offline")
        with self.assertRaisesRegex(OSError, "disk offline"):
            validate_task_definition(TaskId("task-v1"), self.layout, self.fs)

    def test_study_task_mismatch_and_unknown_parameter_are_issues(self) -> None:
        study_id = StudyId("study-v1")
        self.fs.text[self.layout.study_metadata_path(study_id)] = "exists"
        study = Study(
            schema="mjtensu.mldb/study/v1",
            id=study_id,
            status=StudyStatus.DRAFT,
            name="Study",
            description="desc",
            model=StudyTrainingModelSource(
                corpus="train-corpus-v1",
                protocol="train-v1",
                architectures=("arch-v1",),
                parameters={"unknown": types.SimpleNamespace(values=(1,))},
                seeds=(1,),
            ),
            evaluations=(StudyEvaluationStage("eval", "eval-corpus-v1", "eval-v1", {"bad": 1}),),
        )
        study_handle = StudyHandle(study, self.layout.study_metadata_path(study_id))
        task_a = TaskId("task-a-v1")
        task_b = TaskId("task-b-v1")
        module = resolver_module(
            resolve_study=lambda *_: study_handle,
            resolve_corpus=lambda corpus_id, *_: types.SimpleNamespace(
                metadata=types.SimpleNamespace(task=task_a if corpus_id == "train-corpus-v1" else task_b)
            ),
            resolve_train_protocol=lambda *_: types.SimpleNamespace(
                metadata=types.SimpleNamespace(task=task_a, parameters={"lr": object()})
            ),
            resolve_architecture=lambda *_: types.SimpleNamespace(
                metadata=types.SimpleNamespace(task=task_b)
            ),
            resolve_evaluation_protocol=lambda *_: types.SimpleNamespace(
                metadata=types.SimpleNamespace(task=task_a, parameters={"known": object()})
            ),
            resolve_model=lambda *_: None,
        )
        with module:
            result = validate_study_definition(study_id, self.layout, self.fs)
        self.assertFalse(result.valid)
        codes = [issue.code for issue in result.issues]
        self.assertIn("study.task_mismatch", codes)
        self.assertIn("study.parameter.unknown", codes)


if __name__ == "__main__":
    unittest.main()
