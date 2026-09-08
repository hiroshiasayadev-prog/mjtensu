from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from mldb.src.common.errors import MldbError, NotFoundError, ValidationFailedError
from mldb.src.common.ids import (
    ArchitectureId, CorpusId, EvaluationProtocolId, StudyId, TaskId, TrainProtocolId,
)
from mldb.src.repository._local_filesystem import LocalFilesystem
from mldb.src.repository.layout import RepositoryLayout
from mldb.src.runtime.executable_loader import load_architecture_build
from mldb.src.runtime._yaml import _load_yaml
from mldb.src.runtime.resolution import (
    resolve_architecture, resolve_corpus, resolve_evaluation_protocol, resolve_model,
    resolve_study, resolve_task, resolve_train_protocol,
)


class _NoScanFilesystem(LocalFilesystem):
    def list_directory(self, path: Path) -> tuple[Path, ...]:
        raise AssertionError(f"resolver must not scan directories: {path}")


class RuntimeYamlTests(unittest.TestCase):
    def test_bare_string_flow_sequence(self) -> None:
        value = _load_yaml("architectures: [plain-v1, mobile-v1]\n")
        self.assertEqual({"architectures": ["plain-v1", "mobile-v1"]}, value)

    def test_numeric_flow_sequence(self) -> None:
        value = _load_yaml("values: [1, -2, 3.5, 1e3, true, false, null]\n")
        self.assertEqual({"values": [1, -2, 3.5, 1000.0, True, False, None]}, value)

    def test_nested_json_compatible_flow_and_block_values(self) -> None:
        value = _load_yaml(
            "parameters:\n"
            "  optimizer:\n"
            "    default: {name: adamw, betas: [0.9, 0.999], flags: {amsgrad: false, note: null}}\n"
            "  labels: ['east', south, \"west\"]\n"
        )
        self.assertEqual(
            {
                "parameters": {
                    "optimizer": {
                        "default": {
                            "name": "adamw",
                            "betas": [0.9, 0.999],
                            "flags": {"amsgrad": False, "note": None},
                        }
                    },
                    "labels": ["east", "south", "west"],
                }
            },
            value,
        )


class RuntimeResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.layout = RepositoryLayout(self.root)
        self.fs = _NoScanFilesystem()
        for directory in ("tasks", "corpora", "architectures", "train_protocols", "evaluation_protocols", "studies"):
            (self.root / "mldb_data" / directory).mkdir(parents=True, exist_ok=True)
        self.task_id = TaskId("tile-task-v1")
        self._write_json(self.layout.task_metadata_path(self.task_id), self._task())

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_json(self, path: Path, value: object) -> None:
        path.write_text(json.dumps(value), encoding="utf-8")

    def _task(self) -> dict[str, object]:
        return {
            "schema": "mjtensu.mldb/task/v1", "id": str(self.task_id), "name": "tiles",
            "problem_type": "classification", "description": "tile classification",
            "input": {"semantic_unit": "single-tile-image"},
            "target": {"type": "categorical", "labels": ["a", "b"]},
            "semantics": {}, "scope": {"includes": ["tiles"], "excludes": []},
        }

    def _make_corpus(self, *, target: str = "a", class_index: int = 0,
                     payload: bytes = b"\x01\x02\x03\x04",
                     representation: dict[str, object] | None = None) -> CorpusId:
        corpus_id = CorpusId("gray-v1")
        artifact = self.layout.corpus_artifact_path(corpus_id)
        connection = sqlite3.connect(artifact)
        connection.execute("CREATE TABLE samples (sample_id TEXT, split TEXT, target TEXT, class_index INTEGER, pixels BLOB)")
        connection.execute("INSERT INTO samples VALUES (?, ?, ?, ?, ?)", ("s1", "train", target, class_index, payload))
        connection.commit(); connection.close()
        data = artifact.read_bytes()
        metadata = {
            "schema": "mjtensu.mldb/corpus/v1", "id": str(corpus_id), "task": str(self.task_id),
            "artifact": {"format": "sqlite", "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)},
            "data": {"schema": "mjtensu.mldb/image-classification-corpus/v1", "table": "samples"},
            "representation": representation or {"kind": "image", "dtype": "uint8", "shape": [2, 2], "payload_column": "pixels"},
            "builder": {"parameters": {}}, "splits": {"train": 1},
        }
        self._write_json(self.layout.corpus_metadata_path(corpus_id), metadata)
        self.layout.corpus_builder_path(corpus_id).write_text("# builder\n", encoding="utf-8")
        return corpus_id

    def test_task_resolves_exact_canonical_path_without_scan(self) -> None:
        handle = resolve_task(self.task_id, self.layout, self.fs)
        self.assertEqual(self.task_id, handle.metadata.id)
        self.assertEqual(self.layout.task_metadata_path(self.task_id), handle.metadata_path)

    def test_wrong_id_and_wrong_schema_fail(self) -> None:
        wrong = self._task(); wrong["id"] = "other-v1"
        self._write_json(self.layout.task_metadata_path(self.task_id), wrong)
        with self.assertRaises(ValidationFailedError):
            resolve_task(self.task_id, self.layout, self.fs)
        wrong = self._task(); wrong["schema"] = "wrong/v1"
        self._write_json(self.layout.task_metadata_path(self.task_id), wrong)
        with self.assertRaises(ValidationFailedError):
            resolve_task(self.task_id, self.layout, self.fs)

    def test_corpus_requires_all_canonical_siblings(self) -> None:
        corpus_id = self._make_corpus()
        self.layout.corpus_builder_path(corpus_id).unlink()
        with self.assertRaises(NotFoundError):
            resolve_corpus(corpus_id, self.layout, self.fs)

    def test_corpus_resolves_and_validates_semantics(self) -> None:
        corpus_id = self._make_corpus()
        handle = resolve_corpus(corpus_id, self.layout, self.fs)
        self.assertEqual(self.layout.corpus_artifact_path(corpus_id), handle.artifact_path)
        self.assertEqual(self.layout.corpus_builder_path(corpus_id), handle.builder_path)

    def test_corpus_sha_mismatch_fails(self) -> None:
        corpus_id = self._make_corpus()
        raw = json.loads(self.layout.corpus_metadata_path(corpus_id).read_text(encoding="utf-8"))
        raw["artifact"]["sha256"] = "0" * 64
        self._write_json(self.layout.corpus_metadata_path(corpus_id), raw)
        with self.assertRaises(ValidationFailedError):
            resolve_corpus(corpus_id, self.layout, self.fs)

    def test_corpus_target_class_index_mismatch_fails(self) -> None:
        corpus_id = self._make_corpus(target="b", class_index=0)
        with self.assertRaises(ValidationFailedError):
            resolve_corpus(corpus_id, self.layout, self.fs)

    def test_corpus_representation_payload_mismatch_fails(self) -> None:
        corpus_id = self._make_corpus(payload=b"\x01\x02")
        with self.assertRaises(ValidationFailedError):
            resolve_corpus(corpus_id, self.layout, self.fs)

    def test_architecture_resolution_and_executable_hash(self) -> None:
        architecture_id = ArchitectureId("plain-v1")
        implementation = self.layout.architecture_implementation_path(architecture_id)
        implementation.write_text("def build():\n    return object()\n", encoding="utf-8")
        digest = hashlib.sha256(implementation.read_bytes()).hexdigest()
        metadata = {
            "schema": "mjtensu.mldb/architecture/v1", "id": str(architecture_id), "status": "sealed",
            "task": str(self.task_id), "name": "plain", "family": "plain", "description": "plain",
            "implementation": {"framework": "pytorch", "entrypoint": "build", "sha256": digest},
            "interface": {"input": {"kind": "image"}, "output": {"kind": "logits"}},
            "structure": {"summary": "plain"},
        }
        self._write_json(self.layout.architecture_metadata_path(architecture_id), metadata)
        handle = resolve_architecture(architecture_id, self.layout, self.fs)
        self.assertTrue(callable(load_architecture_build(handle)))
        implementation.write_text("def build():\n    return 2\n", encoding="utf-8")
        with self.assertRaises(MldbError):
            load_architecture_build(handle)

    def test_study_format_yaml_with_flow_values_resolves(self) -> None:
        corpus_id = self._make_corpus()
        for architecture_name in ("plain-v1", "mobile-v1"):
            architecture_id = ArchitectureId(architecture_name)
            self.layout.architecture_implementation_path(architecture_id).write_text(
                "def build():\n    return object()\n", encoding="utf-8"
            )
            self.layout.architecture_metadata_path(architecture_id).write_text(
                f"""schema: mjtensu.mldb/architecture/v1
id: {architecture_name}
status: draft
task: {self.task_id}
name: {architecture_name}
family: classifier
description: classifier architecture
implementation: {{framework: pytorch, entrypoint: build}}
interface:
  input: {{kind: image, shape: [1, 2, 2]}}
  output: {{kind: logits}}
structure:
  summary: test architecture
  traits: [small, grayscale]
""",
                encoding="utf-8",
            )

        train_id = TrainProtocolId("train-v1")
        self.layout.train_protocol_implementation_path(train_id).write_text(
            "def train(context):\n    return object()\n", encoding="utf-8"
        )
        self.layout.train_protocol_metadata_path(train_id).write_text(
            f"""schema: mjtensu.mldb/train-protocol/v1
id: {train_id}
status: draft
task: {self.task_id}
name: train
description: train protocol
implementation: {{entrypoint: train}}
parameters:
  optimizer:
    default: {{name: adamw, betas: [0.9, 0.999], enabled: true, note: null}}
""",
            encoding="utf-8",
        )

        eval_id = EvaluationProtocolId("eval-v1")
        self.layout.evaluation_protocol_implementation_path(eval_id).write_text(
            "def evaluate(context):\n    return None\n", encoding="utf-8"
        )
        self.layout.evaluation_protocol_metadata_path(eval_id).write_text(
            f"""schema: mjtensu.mldb/evaluation-protocol/v1
id: {eval_id}
status: draft
task: {self.task_id}
name: eval
description: evaluation protocol
implementation: {{entrypoint: evaluate}}
parameters:
  batch_size: {{default: 32}}
outputs:
  metrics: {{}}
  artifacts: {{}}
""",
            encoding="utf-8",
        )

        study_id = StudyId("study-flow-v1")
        self.layout.study_metadata_path(study_id).write_text(
            f"""schema: mjtensu.mldb/study/v1
id: {study_id}
status: draft
name: flow study
description: Study YAML using block and flow collections.
model:
  train:
    corpus: {corpus_id}
    protocol: {train_id}
    architectures: [plain-v1, mobile-v1]
    parameters:
      optimizer:
        values: [{{name: adamw, betas: [0.9, 0.999], enabled: true, note: null}}]
    seeds: [42, 43]
evaluations:
  - stage: final-holdout
    corpus: {corpus_id}
    protocol: {eval_id}
    parameters: {{batch_size: 32, tag: fast}}
""",
            encoding="utf-8",
        )

        handle = resolve_study(study_id, self.layout, self.fs)
        self.assertEqual(["plain-v1", "mobile-v1"], list(handle.metadata.model.architectures))
        self.assertEqual([42, 43], list(handle.metadata.model.seeds))
        optimizer_value = handle.metadata.model.parameters["optimizer"].values[0]
        self.assertEqual("adamw", optimizer_value["name"])
        self.assertEqual([0.9, 0.999], optimizer_value["betas"])
        self.assertIsNone(optimizer_value["note"])

    def test_protocols_and_training_study_resolve_direct_typed_references(self) -> None:
        corpus_id = self._make_corpus()
        architecture_id = ArchitectureId("plain-v1")
        architecture_py = self.layout.architecture_implementation_path(architecture_id)
        architecture_py.write_text("def build():\n    return object()\n", encoding="utf-8")
        self._write_json(self.layout.architecture_metadata_path(architecture_id), {
            "schema": "mjtensu.mldb/architecture/v1", "id": str(architecture_id), "status": "draft",
            "task": str(self.task_id), "name": "plain", "family": "plain", "description": "plain",
            "implementation": {"framework": "pytorch", "entrypoint": "build"},
            "interface": {"input": {"kind": "image"}, "output": {"kind": "logits"}},
            "structure": {"summary": "plain"},
        })
        train_id = TrainProtocolId("train-v1")
        self.layout.train_protocol_implementation_path(train_id).write_text("def train(context):\n    return object()\n", encoding="utf-8")
        self._write_json(self.layout.train_protocol_metadata_path(train_id), {
            "schema": "mjtensu.mldb/train-protocol/v1", "id": str(train_id), "status": "draft",
            "task": str(self.task_id), "name": "train", "description": "train",
            "implementation": {"entrypoint": "train"}, "parameters": {},
        })
        eval_id = EvaluationProtocolId("eval-v1")
        self.layout.evaluation_protocol_implementation_path(eval_id).write_text("def evaluate(context):\n    return None\n", encoding="utf-8")
        self._write_json(self.layout.evaluation_protocol_metadata_path(eval_id), {
            "schema": "mjtensu.mldb/evaluation-protocol/v1", "id": str(eval_id), "status": "draft",
            "task": str(self.task_id), "name": "eval", "description": "eval",
            "implementation": {"entrypoint": "evaluate"}, "parameters": {},
            "outputs": {"metrics": {}, "artifacts": {}},
        })
        self.assertEqual(train_id, resolve_train_protocol(train_id, self.layout, self.fs).metadata.id)
        self.assertEqual(eval_id, resolve_evaluation_protocol(eval_id, self.layout, self.fs).metadata.id)
        study_id = StudyId("study-v1")
        self._write_json(self.layout.study_metadata_path(study_id), {
            "schema": "mjtensu.mldb/study/v1", "id": str(study_id), "status": "draft",
            "name": "study", "description": "study",
            "model": {"train": {"corpus": str(corpus_id), "protocol": str(train_id),
                                  "architectures": [str(architecture_id)], "parameters": {}, "seeds": [42]}},
            "evaluations": [{"stage": "eval", "corpus": str(corpus_id), "protocol": str(eval_id), "parameters": {}}],
        })
        self.assertEqual(study_id, resolve_study(study_id, self.layout, self.fs).metadata.id)


class RuntimeModelDependencyTests(unittest.TestCase):
    def test_model_completed_lineage_when_i2b_is_available(self) -> None:
        try:
            import mldb.src.model.identity  # noqa: F401
            import mldb.src.model.loading  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("Wave I2-B model identity/loading implementation not present")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            layout = RepositoryLayout(root)
            fs = _NoScanFilesystem()
            for directory in ("tasks", "architectures", "models", "training_runs"):
                (root / "mldb_data" / directory).mkdir(parents=True, exist_ok=True)

            task_id = TaskId("tile-task-v1")
            layout.task_metadata_path(task_id).write_text(json.dumps({
                "schema": "mjtensu.mldb/task/v1", "id": str(task_id), "name": "tiles",
                "problem_type": "classification", "description": "tiles",
                "input": {"semantic_unit": "single-tile-image"},
                "target": {"type": "categorical", "labels": ["a", "b"]},
                "semantics": {}, "scope": {"includes": ["tiles"], "excludes": []},
            }), encoding="utf-8")

            architecture_id = ArchitectureId("plain-v1")
            implementation = layout.architecture_implementation_path(architecture_id)
            implementation.write_text("def build():\n    return object()\n", encoding="utf-8")
            layout.architecture_metadata_path(architecture_id).write_text(json.dumps({
                "schema": "mjtensu.mldb/architecture/v1", "id": str(architecture_id), "status": "draft",
                "task": str(task_id), "name": "plain", "family": "plain", "description": "plain",
                "implementation": {"framework": "pytorch", "entrypoint": "build"},
                "interface": {"input": {"kind": "image"}, "output": {"kind": "logits"}},
                "structure": {"summary": "plain"},
            }), encoding="utf-8")

            run_id = "tr-20260908-001"
            run_paths = layout.training_run_paths(run_id)
            run_paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
            run_paths.weights_path.write_bytes(b"weights")
            run_paths.metadata_path.write_text(json.dumps({
                "schema": "mjtensu.mldb/training-run/v1", "id": run_id, "status": "completed",
                "corpus": "unused-corpus-v1", "architecture": str(architecture_id),
                "train_protocol": "unused-protocol-v1", "parameters": {},
                "execution": {"seed": 42, "started_at": "2026-09-08T00:00:00+09:00", "finished_at": "2026-09-08T00:01:00+09:00"},
                "result": {"weights": {"format": "pytorch-state-dict", "path": "artifacts/weights.pt",
                                          "sha256": hashlib.sha256(b"weights").hexdigest(), "bytes": 7}},
            }), encoding="utf-8")
            model_id = "mdl-20260908-001"
            layout.model_metadata_path(model_id).write_text(json.dumps({
                "schema": "mjtensu.mldb/model/v1", "id": model_id, "training_run": run_id,
            }), encoding="utf-8")

            handle = resolve_model(model_id, layout, fs)
            self.assertEqual(run_paths.weights_path, handle.weights_path)
            self.assertEqual(architecture_id, handle.architecture.metadata.id)


if __name__ == "__main__":
    unittest.main()
