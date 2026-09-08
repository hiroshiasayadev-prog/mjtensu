from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mldb.src.catalog.architecture import ArchitectureStatus
from mldb.src.common.ids import (
    ArchitectureId,
    CorpusId,
    EvaluationProtocolId,
    EvaluationRunId,
    ModelId,
    TrainingRunId,
    TrainProtocolId,
)
from mldb.src.evaluation.interface import UnavailableOutput
from mldb.src.evaluation.protocol import (
    EvaluationArtifactDeclaration,
    EvaluationMetricDeclaration,
    EvaluationOutputs,
)
from mldb.src.evaluation.run import (
    EvaluationRun,
    EvaluationRunExecution,
    EvaluationRunStatus,
)
from mldb.src.repository._local_filesystem import LocalFilesystem
from mldb.src.repository.layout import RepositoryLayout
from mldb.src.runtime import result_acceptance as acceptance
from mldb.src.training.run import (
    TrainingRun,
    TrainingRunExecution,
    TrainingRunStatus,
)


@dataclass(frozen=True)
class _Ref:
    key: str = "candidate"
    content_identity: str = "0" * 64
    bytes: int = 0


@dataclass(frozen=True)
class _TrainingCandidate:
    weights: object


@dataclass(frozen=True)
class _EvaluationCandidate:
    metrics: dict[str, int | float]
    artifacts: dict[str, object]
    unavailable_outputs: tuple[UnavailableOutput, ...] = ()


class _TrackingFilesystem(LocalFilesystem):
    def __init__(self) -> None:
        self.byte_replacements: list[Path] = []

    def replace_bytes(self, path: Path, data: bytes) -> None:
        self.byte_replacements.append(path)
        super().replace_bytes(path, data)


class ResultAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.layout = RepositoryLayout(self.root)
        self.fs = _TrackingFilesystem()
        self.payload = b"candidate-weights"
        self.training_run = TrainingRun(
            schema="mjtensu.mldb/training-run/v1",
            id=TrainingRunId("tr-20260908-001"),
            status=TrainingRunStatus.RUNNING,
            corpus=CorpusId("train-corpus-v1"),
            architecture=ArchitectureId("plain-v1"),
            train_protocol=TrainProtocolId("train-v1"),
            parameters={"epochs": 10},
            execution=TrainingRunExecution(seed=42, started_at="start"),
        )
        self.evaluation_run = EvaluationRun(
            schema="mjtensu.mldb/evaluation-run/v1",
            id=EvaluationRunId("ev-20260908-001"),
            status=EvaluationRunStatus.RUNNING,
            model=ModelId("mdl-20260908-001"),
            corpus=CorpusId("eval-corpus-v1"),
            evaluation_protocol=EvaluationProtocolId("eval-v1"),
            parameters={"threshold": 0.5},
            execution=EvaluationRunExecution(started_at="start"),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _architecture(self):
        metadata = SimpleNamespace(
            status=ArchitectureStatus.SEALED,
            implementation=SimpleNamespace(sha256="a" * 64),
        )
        return SimpleNamespace(metadata=metadata)

    def _preflight(self, outputs: EvaluationOutputs, *, parameters=None):
        return SimpleNamespace(
            parameters=(
                dict(self.evaluation_run.parameters)
                if parameters is None
                else parameters
            ),
            protocol=SimpleNamespace(metadata=SimpleNamespace(outputs=outputs)),
            task=object(),
        )

    def _accept_training(self, *, run=None, payload=None):
        run = self.training_run if run is None else run
        payload = self.payload if payload is None else payload
        candidate = _TrainingCandidate(_Ref(bytes=len(payload)))
        with (
            patch.object(acceptance, "read_training_run", return_value=run),
            patch.object(
                acceptance,
                "_read_verified_candidate_bytes",
                return_value=payload,
            ),
            patch.object(
                acceptance,
                "resolve_architecture",
                return_value=self._architecture(),
            ),
            patch.object(acceptance, "load_architecture_build", return_value=lambda: None),
            patch.object(acceptance, "load_canonical_weights"),
            patch.object(acceptance, "persist_training_run_transition") as persist,
            patch.object(
                acceptance,
                "ensure_model_for_completed_training_run",
            ) as ensure,
        ):
            result = acceptance.accept_training_success(
                run, 7, candidate, "finish", self.layout, self.fs, object()
            )
        return result, persist, ensure

    def test_training_valid_candidate(self) -> None:
        result, persist, ensure = self._accept_training()
        weights_path = self.layout.training_run_paths(result.id).weights_path
        self.assertEqual(TrainingRunStatus.COMPLETED, result.status)
        self.assertEqual(self.payload, weights_path.read_bytes())
        self.assertEqual(hashlib.sha256(self.payload).hexdigest(), result.result.weights.sha256)
        self.assertEqual(len(self.payload), result.result.weights.bytes)
        persist.assert_called_once_with(result, self.layout, self.fs)
        ensure.assert_called_once_with(result, self.layout, self.fs)

    def test_training_candidate_sha_or_bytes_failure(self) -> None:
        candidate = _TrainingCandidate(_Ref(bytes=len(self.payload)))
        for message in ("candidate SHA-256 mismatch", "candidate byte-count mismatch"):
            with self.subTest(message=message):
                with (
                    patch.object(acceptance, "read_training_run", return_value=self.training_run),
                    patch.object(
                        acceptance,
                        "_read_verified_candidate_bytes",
                        side_effect=ValueError(message),
                    ),
                    patch.object(acceptance, "resolve_architecture") as resolve_architecture,
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        acceptance.accept_training_success(
                            self.training_run,
                            7,
                            candidate,
                            "finish",
                            self.layout,
                            self.fs,
                            object(),
                        )
                    resolve_architecture.assert_not_called()

    def test_training_rejects_stale_or_wrong_running_run(self) -> None:
        wrong = replace(self.training_run, parameters={"epochs": 11})
        with (
            patch.object(acceptance, "read_training_run", return_value=wrong),
            patch.object(acceptance, "_read_verified_candidate_bytes") as verify,
        ):
            with self.assertRaisesRegex(ValueError, "exact current canonical RUNNING"):
                acceptance.accept_training_success(
                    self.training_run, 7, _TrainingCandidate(_Ref()), "finish",
                    self.layout, self.fs, object(),
                )
            verify.assert_not_called()

    def test_training_rejects_architecture_strict_compatibility_failure(self) -> None:
        candidate = _TrainingCandidate(_Ref(bytes=len(self.payload)))
        with (
            patch.object(acceptance, "read_training_run", return_value=self.training_run),
            patch.object(acceptance, "_read_verified_candidate_bytes", return_value=self.payload),
            patch.object(acceptance, "resolve_architecture", return_value=self._architecture()),
            patch.object(acceptance, "load_architecture_build", return_value=lambda: None),
            patch.object(
                acceptance,
                "load_canonical_weights",
                side_effect=RuntimeError("strict load failure"),
            ),
            patch.object(acceptance, "persist_training_run_transition") as persist,
            patch.object(acceptance, "ensure_model_for_completed_training_run") as ensure,
        ):
            with self.assertRaisesRegex(RuntimeError, "strict load failure"):
                acceptance.accept_training_success(
                    self.training_run, 7, candidate, "finish",
                    self.layout, self.fs, object(),
                )
        self.assertFalse(self.layout.training_run_paths(self.training_run.id).weights_path.exists())
        persist.assert_not_called()
        ensure.assert_not_called()

    def test_training_commits_exact_bytes_before_completed_persistence(self) -> None:
        candidate = _TrainingCandidate(_Ref(bytes=len(self.payload)))
        weights_path = self.layout.training_run_paths(self.training_run.id).weights_path
        observed: list[str] = []

        def persist(run, layout, filesystem):
            self.assertEqual(self.payload, weights_path.read_bytes())
            self.assertEqual(TrainingRunStatus.COMPLETED, run.status)
            observed.append("persist")

        with (
            patch.object(acceptance, "read_training_run", return_value=self.training_run),
            patch.object(acceptance, "_read_verified_candidate_bytes", return_value=self.payload),
            patch.object(acceptance, "resolve_architecture", return_value=self._architecture()),
            patch.object(acceptance, "load_architecture_build", return_value=lambda: None),
            patch.object(acceptance, "load_canonical_weights"),
            patch.object(acceptance, "persist_training_run_transition", side_effect=persist),
            patch.object(acceptance, "ensure_model_for_completed_training_run"),
        ):
            acceptance.accept_training_success(
                self.training_run, 7, candidate, "finish", self.layout, self.fs, object()
            )
        self.assertEqual(["persist"], observed)

    def test_training_completes_run_before_model_ensure(self) -> None:
        candidate = _TrainingCandidate(_Ref(bytes=len(self.payload)))
        observed: list[str] = []

        def persist(run, layout, filesystem):
            observed.append(f"persist:{run.status.value}")

        def ensure(run, layout, filesystem):
            observed.append(f"model:{run.status.value}")

        with (
            patch.object(acceptance, "read_training_run", return_value=self.training_run),
            patch.object(acceptance, "_read_verified_candidate_bytes", return_value=self.payload),
            patch.object(acceptance, "resolve_architecture", return_value=self._architecture()),
            patch.object(acceptance, "load_architecture_build", return_value=lambda: None),
            patch.object(acceptance, "load_canonical_weights"),
            patch.object(acceptance, "persist_training_run_transition", side_effect=persist),
            patch.object(acceptance, "ensure_model_for_completed_training_run", side_effect=ensure),
        ):
            acceptance.accept_training_success(
                self.training_run, 7, candidate, "finish", self.layout, self.fs, object()
            )
        self.assertEqual(["persist:completed", "model:completed"], observed)

    def test_training_same_candidate_accepts_preexisting_identical_canonical_bytes(self) -> None:
        paths = self.layout.training_run_paths(self.training_run.id)
        paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
        paths.weights_path.write_bytes(self.payload)
        result, _, _ = self._accept_training()
        canonical_replacements = [
            path for path in self.fs.byte_replacements if path == paths.weights_path
        ]
        self.assertEqual([], canonical_replacements)
        self.assertEqual(self.payload, paths.weights_path.read_bytes())
        self.assertEqual(TrainingRunStatus.COMPLETED, result.status)

    def test_training_rejects_conflicting_canonical_bytes(self) -> None:
        paths = self.layout.training_run_paths(self.training_run.id)
        paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
        paths.weights_path.write_bytes(b"other")
        with self.assertRaisesRegex(ValueError, "conflict"):
            self._accept_training()
        self.assertEqual(b"other", paths.weights_path.read_bytes())

    def test_training_terminal_replay_is_rejected(self) -> None:
        terminal = replace(
            self.training_run,
            status=TrainingRunStatus.COMPLETED,
            execution=replace(self.training_run.execution, finished_at="finish"),
        )
        with (
            patch.object(acceptance, "read_training_run", return_value=terminal),
            patch.object(acceptance, "_read_verified_candidate_bytes") as verify,
        ):
            with self.assertRaisesRegex(ValueError, "RUNNING"):
                acceptance.accept_training_success(
                    terminal, 7, _TrainingCandidate(_Ref()), "finish",
                    self.layout, self.fs, object(),
                )
            verify.assert_not_called()

    def _accept_evaluation(
        self,
        outputs: EvaluationOutputs,
        candidate: _EvaluationCandidate,
        *,
        preflight=None,
    ):
        prepared = preflight or self._preflight(outputs)
        with (
            patch.object(acceptance, "read_evaluation_run", return_value=self.evaluation_run),
            patch.object(acceptance, "preflight_evaluation", return_value=prepared),
            patch.object(acceptance, "_read_verified_candidate_bytes", return_value=b"artifact"),
            patch.object(acceptance, "persist_evaluation_run_transition") as persist,
        ):
            result = acceptance.accept_evaluation_success(
                self.evaluation_run,
                9,
                candidate,
                "finish",
                self.layout,
                self.fs,
                object(),
            )
        return result, persist

    def test_evaluation_completed(self) -> None:
        outputs = EvaluationOutputs(
            metrics={"accuracy": EvaluationMetricDeclaration(type="number")},
            artifacts={},
        )
        result, persist = self._accept_evaluation(
            outputs,
            _EvaluationCandidate(metrics={"accuracy": 0.9}, artifacts={}),
        )
        self.assertEqual(EvaluationRunStatus.COMPLETED, result.status)
        self.assertEqual({"accuracy": 0.9}, result.result.metrics)
        self.assertEqual((), result.unavailable_outputs)
        self.assertEqual((), result.validation_issues)
        persist.assert_called_once_with(result, self.layout, self.fs)

    def test_evaluation_completed_partial(self) -> None:
        outputs = EvaluationOutputs(
            metrics={
                "accuracy": EvaluationMetricDeclaration(type="number"),
                "loss": EvaluationMetricDeclaration(type="number"),
            },
            artifacts={},
        )
        unavailable = UnavailableOutput(
            output="metrics.loss", type="not-produced", message="not available"
        )
        result, _ = self._accept_evaluation(
            outputs,
            _EvaluationCandidate(
                metrics={"accuracy": 0.9}, artifacts={}, unavailable_outputs=(unavailable,)
            ),
        )
        self.assertEqual(EvaluationRunStatus.COMPLETED_PARTIAL, result.status)
        self.assertEqual((unavailable,), tuple(result.unavailable_outputs))

    def test_evaluation_required_output_invalidity_is_not_partial(self) -> None:
        outputs = EvaluationOutputs(
            metrics={"accuracy": EvaluationMetricDeclaration(type="number")},
            artifacts={
                "report": EvaluationArtifactDeclaration(
                    format="json",
                    schema="mjtensu.mldb/eval-artifact/report/v1",
                    required=True,
                )
            },
        )
        with (
            patch.object(acceptance, "read_evaluation_run", return_value=self.evaluation_run),
            patch.object(
                acceptance,
                "preflight_evaluation",
                return_value=self._preflight(outputs),
            ),
            patch.object(acceptance, "persist_evaluation_run_transition") as persist,
        ):
            with self.assertRaisesRegex(ValueError, "Required artifact 'report' was not returned"):
                acceptance.accept_evaluation_success(
                    self.evaluation_run,
                    9,
                    _EvaluationCandidate(metrics={"accuracy": 0.9}, artifacts={}),
                    "finish",
                    self.layout,
                    self.fs,
                    object(),
                )
            persist.assert_not_called()

    def test_evaluation_candidate_artifact_verification_failure(self) -> None:
        outputs = EvaluationOutputs(
            metrics={"accuracy": EvaluationMetricDeclaration(type="number")},
            artifacts={
                "report": EvaluationArtifactDeclaration(
                    format="json",
                    schema="mjtensu.mldb/eval-artifact/report/v1",
                    required=False,
                )
            },
        )
        candidate = _EvaluationCandidate(
            metrics={"accuracy": 0.9}, artifacts={"report": _Ref()}
        )
        with (
            patch.object(acceptance, "read_evaluation_run", return_value=self.evaluation_run),
            patch.object(acceptance, "preflight_evaluation", return_value=self._preflight(outputs)),
            patch.object(
                acceptance,
                "_read_verified_candidate_bytes",
                side_effect=ValueError("candidate artifact integrity failure"),
            ),
            patch.object(acceptance, "accept_evaluation_result") as validate_result,
            patch.object(acceptance, "persist_evaluation_run_transition") as persist,
        ):
            with self.assertRaisesRegex(ValueError, "integrity failure"):
                acceptance.accept_evaluation_success(
                    self.evaluation_run, 9, candidate, "finish",
                    self.layout, self.fs, object(),
                )
            validate_result.assert_not_called()
            persist.assert_not_called()

    def test_evaluation_rejects_stale_or_wrong_run(self) -> None:
        wrong = replace(self.evaluation_run, parameters={"threshold": 0.7})
        with (
            patch.object(acceptance, "read_evaluation_run", return_value=wrong),
            patch.object(acceptance, "preflight_evaluation") as preflight,
        ):
            with self.assertRaisesRegex(ValueError, "exact current canonical RUNNING"):
                acceptance.accept_evaluation_success(
                    self.evaluation_run,
                    9,
                    _EvaluationCandidate(metrics={}, artifacts={}),
                    "finish",
                    self.layout,
                    self.fs,
                    object(),
                )
            preflight.assert_not_called()

    def test_evaluation_rejects_fresh_preflight_parameter_disagreement(self) -> None:
        outputs = EvaluationOutputs(metrics={}, artifacts={})
        prepared = self._preflight(outputs, parameters={"threshold": 0.7})
        with (
            patch.object(acceptance, "read_evaluation_run", return_value=self.evaluation_run),
            patch.object(acceptance, "preflight_evaluation", return_value=prepared),
            patch.object(acceptance, "_read_verified_candidate_bytes") as verify,
        ):
            with self.assertRaisesRegex(ValueError, "parameters disagree"):
                acceptance.accept_evaluation_success(
                    self.evaluation_run,
                    9,
                    _EvaluationCandidate(metrics={}, artifacts={}),
                    "finish",
                    self.layout,
                    self.fs,
                    object(),
                )
            verify.assert_not_called()

    def test_evaluation_terminal_replay_is_rejected(self) -> None:
        terminal = replace(
            self.evaluation_run,
            status=EvaluationRunStatus.COMPLETED,
            execution=replace(self.evaluation_run.execution, finished_at="finish"),
        )
        with (
            patch.object(acceptance, "read_evaluation_run", return_value=terminal),
            patch.object(acceptance, "preflight_evaluation") as preflight,
        ):
            with self.assertRaisesRegex(ValueError, "RUNNING"):
                acceptance.accept_evaluation_success(
                    terminal,
                    9,
                    _EvaluationCandidate(metrics={}, artifacts={}),
                    "finish",
                    self.layout,
                    self.fs,
                    object(),
                )
            preflight.assert_not_called()


if __name__ == "__main__":
    unittest.main()
