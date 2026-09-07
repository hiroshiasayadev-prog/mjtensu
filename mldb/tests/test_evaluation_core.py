from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from mldb.src.evaluation.interface import EvaluationResult, UnavailableOutput
from mldb.src.common.parameters import PublicParameterDeclaration
from mldb.src.evaluation.protocol import (
    EvaluationArtifactDeclaration,
    EvaluationMetricDeclaration,
    EvaluationOutputs,
    EvaluationProtocol,
    EvaluationProtocolImplementation,
    EvaluationProtocolStatus,
    validate_evaluation_protocol_metadata,
)
from mldb.src.evaluation.result_validation import accept_evaluation_result
from mldb.src.evaluation.run import (
    EvaluationRun,
    EvaluationRunExecution,
    EvaluationRunResult,
    EvaluationRunStatus,
    EvaluationRunStudyLineage,
    validate_evaluation_run,
    validate_evaluation_run_transition,
)


_PREDICTIONS_SCHEMA = "mjtensu.mldb/eval-artifact/categorical-predictions/v1"
_CONFUSION_SCHEMA = "mjtensu.mldb/eval-artifact/confusion-matrix/v1"


def _task_handle() -> object:
    return SimpleNamespace(
        metadata=SimpleNamespace(
            target=SimpleNamespace(labels=("a", "b")),
        )
    )


def _predictions_declaration(*, required: bool = True) -> EvaluationArtifactDeclaration:
    return EvaluationArtifactDeclaration(
        format="jsonl",
        schema=_PREDICTIONS_SCHEMA,
        required=required,
    )


def _confusion_declaration(*, required: bool = False) -> EvaluationArtifactDeclaration:
    return EvaluationArtifactDeclaration(
        format="csv",
        schema=_CONFUSION_SCHEMA,
        required=required,
    )


class EvaluationProtocolMetadataTests(unittest.TestCase):
    def test_sealed_protocol_metadata_validates_without_repository_io(self) -> None:
        protocol = EvaluationProtocol(
            schema="mjtensu.mldb/evaluation-protocol/v1",
            id="classifier-eval-v1",
            status=EvaluationProtocolStatus.SEALED,
            task="task-v1",
            name="Classifier evaluation",
            description="Evaluate categorical predictions.",
            implementation=EvaluationProtocolImplementation(
                entrypoint="evaluate",
                sha256="a" * 64,
            ),
            parameters={"batch_size": PublicParameterDeclaration(default=32)},
            outputs=EvaluationOutputs(
                metrics={"accuracy": EvaluationMetricDeclaration(type="number")},
                artifacts={"predictions": _predictions_declaration()},
            ),
        )

        self.assertTrue(validate_evaluation_protocol_metadata(protocol).valid)


class EvaluationResultAcceptanceTests(unittest.TestCase):
    def test_full_valid_result_is_accepted_and_materialized(self) -> None:
        outputs = EvaluationOutputs(
            metrics={"accuracy": EvaluationMetricDeclaration(type="number")},
            artifacts={"predictions": _predictions_declaration()},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            work = root / "work"
            artifacts = root / "artifacts"
            work.mkdir()
            predictions = work / "predictions.jsonl"
            predictions.write_text(
                '{"sample_id":"1","target":"a","prediction":"a"}\n'
                '{"sample_id":"2","target":"b","prediction":"a"}\n',
                encoding="utf-8",
            )

            accepted = accept_evaluation_result(
                outputs,
                EvaluationResult(
                    metrics={"accuracy": 0.5},
                    artifacts={"predictions": predictions},
                    unavailable_outputs=(),
                ),
                _task_handle(),
                work,
                artifacts,
            )

            self.assertEqual({"accuracy": 0.5}, accepted.metrics)
            self.assertEqual((), accepted.unavailable_outputs)
            self.assertEqual((), accepted.validation_issues)
            metadata = accepted.artifacts["predictions"]
            self.assertEqual("artifacts/predictions.jsonl", metadata.path)
            self.assertEqual("jsonl", metadata.format)
            self.assertEqual(_PREDICTIONS_SCHEMA, metadata.schema)
            self.assertEqual(64, len(metadata.sha256))
            self.assertEqual(predictions.stat().st_size, metadata.bytes)
            self.assertEqual(
                predictions.read_bytes(),
                (artifacts / "predictions.jsonl").read_bytes(),
            )

    def test_silently_omitted_optional_artifact_does_not_make_result_partial(self) -> None:
        outputs = EvaluationOutputs(
            metrics={"accuracy": EvaluationMetricDeclaration(type="number")},
            artifacts={"confusion": _confusion_declaration()},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            work = root / "work"
            work.mkdir()

            accepted = accept_evaluation_result(
                outputs,
                EvaluationResult(
                    metrics={"accuracy": 1.0},
                    artifacts={},
                    unavailable_outputs=(),
                ),
                _task_handle(),
                work,
                root / "artifacts",
            )

            self.assertEqual((), accepted.unavailable_outputs)
            self.assertEqual((), accepted.validation_issues)

    def test_unavailable_optional_artifact_is_partial_reason(self) -> None:
        outputs = EvaluationOutputs(
            metrics={"accuracy": EvaluationMetricDeclaration(type="number")},
            artifacts={"confusion": _confusion_declaration()},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            work = root / "work"
            work.mkdir()

            accepted = accept_evaluation_result(
                outputs,
                EvaluationResult(
                    metrics={"accuracy": 1.0},
                    artifacts={},
                    unavailable_outputs=(
                        UnavailableOutput(
                            output="artifacts.confusion",
                            type="not-produced",
                            message="No diagnostic table was produced.",
                        ),
                    ),
                ),
                _task_handle(),
                work,
                root / "artifacts",
            )

            self.assertEqual(1, len(accepted.unavailable_outputs))
            self.assertEqual({}, accepted.artifacts)
            self.assertEqual((), accepted.validation_issues)

    def test_invalid_optional_artifact_is_omitted_with_issue(self) -> None:
        outputs = EvaluationOutputs(
            metrics={"accuracy": EvaluationMetricDeclaration(type="number")},
            artifacts={"confusion": _confusion_declaration()},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            work = root / "work"
            work.mkdir()
            confusion = work / "confusion.csv"
            confusion.write_text(
                "target,prediction,count\n"
                "a,a,1\n"
                "a,a,2\n",
                encoding="utf-8",
            )

            accepted = accept_evaluation_result(
                outputs,
                EvaluationResult(
                    metrics={"accuracy": 1.0},
                    artifacts={"confusion": confusion},
                    unavailable_outputs=(),
                ),
                _task_handle(),
                work,
                root / "artifacts",
            )

            self.assertEqual({}, accepted.artifacts)
            self.assertEqual(1, len(accepted.validation_issues))
            self.assertEqual("artifacts.confusion", accepted.validation_issues[0].output)
            self.assertEqual("schema-validation-failed", accepted.validation_issues[0].type)

    def test_required_metric_missing_fails(self) -> None:
        outputs = EvaluationOutputs(
            metrics={"accuracy": EvaluationMetricDeclaration(type="number")},
            artifacts={},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            work = Path(temporary_directory) / "work"
            work.mkdir()
            with self.assertRaises(ValueError):
                accept_evaluation_result(
                    outputs,
                    EvaluationResult(metrics={}, artifacts={}, unavailable_outputs=()),
                    _task_handle(),
                    work,
                    work.parent / "artifacts",
                )

    def test_invalid_metric_scalars_fail(self) -> None:
        outputs = EvaluationOutputs(
            metrics={"accuracy": EvaluationMetricDeclaration(type="number")},
            artifacts={},
        )
        for value in (True, float("nan"), float("inf"), "1.0"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary_directory:
                work = Path(temporary_directory) / "work"
                work.mkdir()
                with self.assertRaises(ValueError):
                    accept_evaluation_result(
                        outputs,
                        EvaluationResult(
                            metrics={"accuracy": value},  # type: ignore[dict-item]
                            artifacts={},
                            unavailable_outputs=(),
                        ),
                        _task_handle(),
                        work,
                        work.parent / "artifacts",
                    )

    def test_required_artifact_unavailable_fails(self) -> None:
        outputs = EvaluationOutputs(
            metrics={"accuracy": EvaluationMetricDeclaration(type="number")},
            artifacts={"predictions": _predictions_declaration()},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            work = Path(temporary_directory) / "work"
            work.mkdir()
            with self.assertRaises(ValueError):
                accept_evaluation_result(
                    outputs,
                    EvaluationResult(
                        metrics={"accuracy": 1.0},
                        artifacts={},
                        unavailable_outputs=(
                            UnavailableOutput(
                                output="artifacts.predictions",
                                type="not-produced",
                                message="Required predictions were not produced.",
                            ),
                        ),
                    ),
                    _task_handle(),
                    work,
                    work.parent / "artifacts",
                )

    def test_required_artifact_invalid_fails(self) -> None:
        outputs = EvaluationOutputs(
            metrics={"accuracy": EvaluationMetricDeclaration(type="number")},
            artifacts={"predictions": _predictions_declaration()},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            work = root / "work"
            work.mkdir()
            predictions = work / "predictions.jsonl"
            predictions.write_text(
                '{"sample_id":"1","target":"outside","prediction":"a"}\n',
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                accept_evaluation_result(
                    outputs,
                    EvaluationResult(
                        metrics={"accuracy": 1.0},
                        artifacts={"predictions": predictions},
                        unavailable_outputs=(),
                    ),
                    _task_handle(),
                    work,
                    root / "artifacts",
                )

    def test_partial_acceptance_retains_trusted_artifact(self) -> None:
        outputs = EvaluationOutputs(
            metrics={"accuracy": EvaluationMetricDeclaration(type="number")},
            artifacts={"predictions": _predictions_declaration()},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            work = root / "work"
            work.mkdir()
            predictions = work / "predictions.jsonl"
            predictions.write_text(
                '{"sample_id":"1","target":"a","prediction":"a"}\n',
                encoding="utf-8",
            )

            accepted = accept_evaluation_result(
                outputs,
                EvaluationResult(
                    metrics={},
                    artifacts={"predictions": predictions},
                    unavailable_outputs=(
                        UnavailableOutput(
                            output="metrics.accuracy",
                            type="undefined",
                            message="Accuracy is unavailable for this population.",
                        ),
                    ),
                ),
                _task_handle(),
                work,
                root / "artifacts",
            )

            self.assertEqual({}, accepted.metrics)
            self.assertIn("predictions", accepted.artifacts)
            self.assertEqual(1, len(accepted.unavailable_outputs))

    def test_empty_partial_result_fails(self) -> None:
        outputs = EvaluationOutputs(
            metrics={"accuracy": EvaluationMetricDeclaration(type="number")},
            artifacts={},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            work = Path(temporary_directory) / "work"
            work.mkdir()
            with self.assertRaises(ValueError):
                accept_evaluation_result(
                    outputs,
                    EvaluationResult(
                        metrics={},
                        artifacts={},
                        unavailable_outputs=(
                            UnavailableOutput(
                                output="metrics.accuracy",
                                type="undefined",
                                message="Metric is unavailable.",
                            ),
                        ),
                    ),
                    _task_handle(),
                    work,
                    work.parent / "artifacts",
                )

    def test_declared_output_mismatch_fails(self) -> None:
        outputs = EvaluationOutputs(metrics={}, artifacts={})
        with tempfile.TemporaryDirectory() as temporary_directory:
            work = Path(temporary_directory) / "work"
            work.mkdir()
            with self.assertRaises(ValueError):
                accept_evaluation_result(
                    outputs,
                    EvaluationResult(
                        metrics={"undeclared": 1.0},
                        artifacts={},
                        unavailable_outputs=(),
                    ),
                    _task_handle(),
                    work,
                    work.parent / "artifacts",
                )

    def test_undeclared_artifact_key_fails(self) -> None:
        outputs = EvaluationOutputs(metrics={}, artifacts={})
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            work = root / "work"
            work.mkdir()
            candidate = work / "extra.json"
            candidate.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                accept_evaluation_result(
                    outputs,
                    EvaluationResult(
                        metrics={},
                        artifacts={"extra": candidate},
                        unavailable_outputs=(),
                    ),
                    _task_handle(),
                    work,
                    root / "artifacts",
                )
            self.assertFalse((root / "artifacts").exists())

    def test_required_artifact_path_escape_fails(self) -> None:
        outputs = EvaluationOutputs(
            metrics={},
            artifacts={"predictions": _predictions_declaration()},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            work = root / "work"
            work.mkdir()
            outside = root / "outside.jsonl"
            outside.write_text(
                '{"sample_id":"1","target":"a","prediction":"a"}\n',
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                accept_evaluation_result(
                    outputs,
                    EvaluationResult(
                        metrics={},
                        artifacts={"predictions": Path("../outside.jsonl")},
                        unavailable_outputs=(),
                    ),
                    _task_handle(),
                    work,
                    root / "artifacts",
                )


class EvaluationRunLifecycleTests(unittest.TestCase):
    def test_only_running_to_terminal_transitions_are_valid(self) -> None:
        for target in (
            EvaluationRunStatus.COMPLETED,
            EvaluationRunStatus.COMPLETED_PARTIAL,
            EvaluationRunStatus.FAILED,
            EvaluationRunStatus.CANCELLED,
        ):
            self.assertTrue(
                validate_evaluation_run_transition(
                    EvaluationRunStatus.RUNNING,
                    target,
                ).valid
            )

        self.assertFalse(
            validate_evaluation_run_transition(
                EvaluationRunStatus.RUNNING,
                EvaluationRunStatus.RUNNING,
            ).valid
        )
        self.assertFalse(
            validate_evaluation_run_transition(
                EvaluationRunStatus.COMPLETED,
                EvaluationRunStatus.FAILED,
            ).valid
        )

    def test_completed_partial_requires_output_and_reason_and_preserves_lineage(self) -> None:
        run = EvaluationRun(
            schema="mjtensu.mldb/evaluation-run/v1",
            id="ev-20260908-001",
            status=EvaluationRunStatus.COMPLETED_PARTIAL,
            model="mdl-20260908-001",
            corpus="corpus-v1",
            evaluation_protocol="eval-v1",
            parameters={},
            execution=EvaluationRunExecution(started_at=object(), finished_at=object()),
            result=EvaluationRunResult(metrics={"accuracy": 1.0}, artifacts={}),
            unavailable_outputs=(
                UnavailableOutput(
                    output="artifacts.confusion",
                    type="not-produced",
                    message="Optional diagnostic unavailable.",
                ),
            ),
            study=EvaluationRunStudyLineage(
                run="sr-20260908-001",
                trial="trial-0001",
                stage="holdout",
            ),
        )

        self.assertTrue(validate_evaluation_run(run).valid)
        self.assertEqual("sr-20260908-001", run.study.run)
        self.assertEqual("trial-0001", run.study.trial)
        self.assertEqual("holdout", run.study.stage)


if __name__ == "__main__":
    unittest.main()
