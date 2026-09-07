from __future__ import annotations

from dataclasses import replace
import unittest

from mldb.src.common.ids import (
    ArchitectureId,
    CorpusId,
    EvaluationProtocolId,
    ModelId,
    StudyId,
    StudyRunId,
    TaskId,
    TrainProtocolId,
)
from mldb.src.common.parameters import PublicParameterDeclaration
from mldb.src.evaluation.protocol import (
    EvaluationOutputs,
    EvaluationProtocol,
    EvaluationProtocolImplementation,
    EvaluationProtocolStatus,
)
from mldb.src.study.definition import (
    Study,
    StudyEvaluationStage,
    StudyExistingModelSource,
    StudyStatus,
    StudyTrainingModelSource,
    StudyTrainingParameterAxis,
    validate_study_metadata,
)
from mldb.src.study.expansion import materialize_study_plan
from mldb.src.study.plan import (
    StudyPlanEvaluation,
    StudyPlanRow,
    StudyPlanTraining,
    validate_study_plan,
)
from mldb.src.study.run import (
    StudyRun,
    StudyRunEvaluationSummary,
    StudyRunExecution,
    StudyRunPlan,
    StudyRunStatus,
    StudyRunSummary,
    StudyRunTrainingSummary,
    validate_study_run,
    validate_study_run_transition,
)
from mldb.src.training.protocol import (
    TrainProtocol,
    TrainProtocolImplementation,
    TrainProtocolStatus,
)


class StudyDefinitionTests(unittest.TestCase):
    def test_valid_training_study_accepts_type_distinct_axis_values(self) -> None:
        study = _training_study(
            parameters={
                "typed": StudyTrainingParameterAxis(values=[True, 1]),
            }
        )

        self.assertTrue(validate_study_metadata(study).valid)

    def test_bool_seed_is_rejected(self) -> None:
        study = replace(
            _training_study(),
            model=replace(_training_study().model, seeds=[42, True]),
        )

        report = validate_study_metadata(study)

        self.assertFalse(report.valid)
        self.assertIn("study.training.seed", {issue.code for issue in report.issues})

    def test_recursive_type_sensitive_duplicate_axis_values_are_rejected(self) -> None:
        first = {"items": [1, True], "mode": "x"}
        same_value_different_mapping_order = {"mode": "x", "items": [1, True]}
        study = _training_study(
            parameters={
                "structured": StudyTrainingParameterAxis(
                    values=[first, same_value_different_mapping_order]
                )
            }
        )

        report = validate_study_metadata(study)

        self.assertFalse(report.valid)
        self.assertIn(
            "study.training.parameter_value_duplicate",
            {issue.code for issue in report.issues},
        )

    def test_training_architectures_and_existing_models_must_be_unique(self) -> None:
        training = _training_study(architectures=["arch-a-v1", "arch-a-v1"])
        existing = _existing_study(models=["mdl-20260908-001", "mdl-20260908-001"])

        self.assertFalse(validate_study_metadata(training).valid)
        self.assertFalse(validate_study_metadata(existing).valid)

    def test_evaluations_are_non_empty_and_stage_unique(self) -> None:
        empty = replace(_training_study(), evaluations=[])
        duplicate = replace(
            _training_study(),
            evaluations=[_stage("eval-a", "eval-v1"), _stage("eval-a", "eval-v2")],
        )

        self.assertFalse(validate_study_metadata(empty).valid)
        self.assertFalse(validate_study_metadata(duplicate).valid)


class StudyExpansionOrderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train_protocol = _train_protocol()
        self.evaluation_protocols = {
            EvaluationProtocolId("eval-v1"): _evaluation_protocol("eval-v1"),
            EvaluationProtocolId("unused-v1"): _evaluation_protocol("unused-v1"),
        }

    def test_parameter_mapping_insertion_order_does_not_change_trials(self) -> None:
        first = _training_study(
            parameters={
                "beta": StudyTrainingParameterAxis(values=["B2", "B1"]),
                "alpha": StudyTrainingParameterAxis(values=[2, 1]),
            },
            architectures=["arch-b-v1", "arch-a-v1"],
            seeds=[9, 7],
        )
        second = replace(
            first,
            model=replace(
                first.model,
                parameters={
                    "alpha": StudyTrainingParameterAxis(values=[2, 1]),
                    "beta": StudyTrainingParameterAxis(values=["B2", "B1"]),
                },
            ),
        )

        first_plan = materialize_study_plan(
            first, self.train_protocol, self.evaluation_protocols
        )
        second_plan = materialize_study_plan(
            second, self.train_protocol, self.evaluation_protocols
        )

        self.assertEqual(first_plan, second_plan)

    def test_authored_architecture_values_and_seed_order_are_preserved(self) -> None:
        study = _training_study(
            parameters={
                "beta": StudyTrainingParameterAxis(values=["B2", "B1"]),
                "alpha": StudyTrainingParameterAxis(values=[2, 1]),
            },
            architectures=["arch-b-v1", "arch-a-v1"],
            seeds=[9, 7],
        )

        plan = materialize_study_plan(
            study, self.train_protocol, self.evaluation_protocols
        )
        coordinates = [
            (
                row.training.architecture,
                row.training.parameters["alpha"],
                row.training.parameters["beta"],
                row.training.seed,
            )
            for row in plan
            if row.training is not None
        ]

        self.assertEqual(
            [
                ("arch-b-v1", 2, "B2", 9),
                ("arch-b-v1", 2, "B2", 7),
                ("arch-b-v1", 2, "B1", 9),
                ("arch-b-v1", 2, "B1", 7),
                ("arch-b-v1", 1, "B2", 9),
                ("arch-b-v1", 1, "B2", 7),
                ("arch-b-v1", 1, "B1", 9),
                ("arch-b-v1", 1, "B1", 7),
            ],
            coordinates[:8],
        )
        self.assertTrue(all(item[0] == "arch-a-v1" for item in coordinates[8:]))

    def test_materialization_resolves_protocol_defaults_and_preserves_stage_order(self) -> None:
        study = replace(
            _training_study(),
            evaluations=[
                StudyEvaluationStage(
                    stage="second-authored",
                    corpus=CorpusId("eval-corpus-v1"),
                    protocol=EvaluationProtocolId("eval-v1"),
                    parameters={"threshold": 0.7},
                ),
                StudyEvaluationStage(
                    stage="first-name-but-later",
                    corpus=CorpusId("eval-corpus-v2"),
                    protocol=EvaluationProtocolId("eval-v1"),
                    parameters={},
                ),
            ],
        )

        plan = materialize_study_plan(
            study, self.train_protocol, self.evaluation_protocols
        )
        first = plan[0]
        assert first.training is not None

        self.assertEqual(
            {"alpha": 10, "beta": "default", "default_only": "kept"},
            first.training.parameters,
        )
        self.assertEqual(
            ["second-authored", "first-name-but-later"],
            [evaluation.stage for evaluation in first.evaluations],
        )
        self.assertEqual(
            {"threshold": 0.7, "batch_size": 64},
            first.evaluations[0].parameters,
        )
        self.assertEqual(
            {"threshold": 0.5, "batch_size": 64},
            first.evaluations[1].parameters,
        )

    def test_existing_model_order_and_trial_numbering_are_preserved(self) -> None:
        study = _existing_study(
            models=[
                "mdl-20260908-003",
                "mdl-20260908-001",
                "mdl-20260908-002",
            ]
        )

        plan = materialize_study_plan(study, None, self.evaluation_protocols)

        self.assertEqual(
            ["trial-0001", "trial-0002", "trial-0003"],
            [row.trial for row in plan],
        )
        self.assertEqual(
            ["mdl-20260908-003", "mdl-20260908-001", "mdl-20260908-002"],
            [row.model for row in plan],
        )
        self.assertTrue(all(row.training is None for row in plan))


class StudyPlanValidationTests(unittest.TestCase):
    def test_trial_numbering_must_start_at_one_without_gaps(self) -> None:
        rows = [
            _plan_row("trial-0001", {"x": 1}),
            _plan_row("trial-0003", {"x": 2}),
        ]

        self.assertFalse(validate_study_plan(rows).valid)

    def test_exactly_one_model_source_and_non_empty_evaluations_are_required(self) -> None:
        both = replace(
            _plan_row("trial-0001", {"x": 1}),
            model=ModelId("mdl-20260908-001"),
        )
        none = StudyPlanRow(trial="trial-0001", evaluations=[_plan_evaluation()])
        no_evaluations = replace(_plan_row("trial-0001", {"x": 1}), evaluations=[])

        self.assertFalse(validate_study_plan([both]).valid)
        self.assertFalse(validate_study_plan([none]).valid)
        self.assertFalse(validate_study_plan([no_evaluations]).valid)

    def test_type_distinct_parameter_coordinates_are_not_duplicates(self) -> None:
        plan = [
            _plan_row("trial-0001", {"x": True}),
            _plan_row("trial-0002", {"x": 1}),
        ]

        self.assertTrue(validate_study_plan(plan).valid)

    def test_recursive_mapping_order_does_not_hide_duplicate_coordinate(self) -> None:
        plan = [
            _plan_row("trial-0001", {"x": {"a": [1, True], "b": "x"}}),
            _plan_row("trial-0002", {"x": {"b": "x", "a": [1, True]}}),
        ]

        report = validate_study_plan(plan)

        self.assertFalse(report.valid)
        self.assertIn(
            "study_plan.coordinate_duplicate",
            {issue.code for issue in report.issues},
        )


class StudyRunDomainTests(unittest.TestCase):
    def test_running_may_omit_plan_but_completed_requires_plan_and_finished_at(self) -> None:
        running = _study_run(StudyRunStatus.RUNNING, plan=None, finished_at=None)
        completed_without_plan = _study_run(
            StudyRunStatus.COMPLETED, plan=None, finished_at="done"
        )
        completed_without_finished_at = _study_run(
            StudyRunStatus.COMPLETED, plan=_run_plan(), finished_at=None
        )

        self.assertTrue(validate_study_run(running).valid)
        self.assertFalse(validate_study_run(completed_without_plan).valid)
        self.assertFalse(validate_study_run(completed_without_finished_at).valid)

    def test_plan_and_summary_counts_reject_bool_and_negative_values(self) -> None:
        invalid_plan = replace(_run_plan(), trials=True)
        invalid_summary = StudyRunSummary(
            training=StudyRunTrainingSummary(completed=1, failed=-1, cancelled=0),
            evaluation=StudyRunEvaluationSummary(
                completed=1,
                completed_partial=0,
                failed=0,
                cancelled=False,
                blocked=0,
            ),
        )
        run = replace(
            _study_run(StudyRunStatus.FAILED, plan=invalid_plan, finished_at="done"),
            summary=invalid_summary,
        )

        self.assertFalse(validate_study_run(run).valid)

    def test_only_running_to_terminal_transitions_are_valid(self) -> None:
        terminals = (
            StudyRunStatus.COMPLETED,
            StudyRunStatus.COMPLETED_WITH_FAILURES,
            StudyRunStatus.FAILED,
            StudyRunStatus.CANCELLED,
        )

        for terminal in terminals:
            with self.subTest(target=terminal):
                self.assertTrue(
                    validate_study_run_transition(
                        StudyRunStatus.RUNNING, terminal
                    ).valid
                )
                self.assertFalse(
                    validate_study_run_transition(
                        terminal, StudyRunStatus.RUNNING
                    ).valid
                )

        self.assertFalse(
            validate_study_run_transition(
                StudyRunStatus.RUNNING, StudyRunStatus.RUNNING
            ).valid
        )


def _stage(stage: str, protocol: str) -> StudyEvaluationStage:
    return StudyEvaluationStage(
        stage=stage,
        corpus=CorpusId("eval-corpus-v1"),
        protocol=EvaluationProtocolId(protocol),
        parameters={},
    )


def _training_study(
    *,
    parameters: dict[str, StudyTrainingParameterAxis] | None = None,
    architectures: list[str] | None = None,
    seeds: list[int] | None = None,
) -> Study:
    return Study(
        schema="mjtensu.mldb/study/v1",
        id=StudyId("study-core-v1"),
        status=StudyStatus.SEALED,
        name="Study core",
        description="Study core tests",
        model=StudyTrainingModelSource(
            corpus=CorpusId("train-corpus-v1"),
            protocol=TrainProtocolId("train-v1"),
            architectures=[
                ArchitectureId(item)
                for item in (architectures if architectures is not None else ["arch-a-v1"])
            ],
            parameters=parameters if parameters is not None else {},
            seeds=seeds if seeds is not None else [42],
        ),
        evaluations=[_stage("eval", "eval-v1")],
    )


def _existing_study(*, models: list[str]) -> Study:
    return Study(
        schema="mjtensu.mldb/study/v1",
        id=StudyId("study-existing-v1"),
        status=StudyStatus.SEALED,
        name="Existing models",
        description="Existing model ordering",
        model=StudyExistingModelSource(models=[ModelId(item) for item in models]),
        evaluations=[_stage("eval", "eval-v1")],
    )


def _train_protocol() -> TrainProtocol:
    return TrainProtocol(
        schema="mjtensu.mldb/train-protocol/v1",
        id=TrainProtocolId("train-v1"),
        status=TrainProtocolStatus.SEALED,
        task=TaskId("task-v1"),
        name="Train",
        description="Train",
        implementation=TrainProtocolImplementation(entrypoint="train", sha256="a" * 64),
        parameters={
            "alpha": PublicParameterDeclaration(default=10),
            "beta": PublicParameterDeclaration(default="default"),
            "default_only": PublicParameterDeclaration(default="kept"),
        },
    )


def _evaluation_protocol(protocol_id: str) -> EvaluationProtocol:
    return EvaluationProtocol(
        schema="mjtensu.mldb/evaluation-protocol/v1",
        id=EvaluationProtocolId(protocol_id),
        status=EvaluationProtocolStatus.SEALED,
        task=TaskId("task-v1"),
        name="Evaluation",
        description="Evaluation",
        implementation=EvaluationProtocolImplementation(
            entrypoint="evaluate", sha256="b" * 64
        ),
        parameters={
            "threshold": PublicParameterDeclaration(default=0.5),
            "batch_size": PublicParameterDeclaration(default=64),
        },
        outputs=EvaluationOutputs(metrics={}, artifacts={}),
    )


def _plan_evaluation() -> StudyPlanEvaluation:
    return StudyPlanEvaluation(
        stage="eval",
        corpus=CorpusId("eval-corpus-v1"),
        protocol=EvaluationProtocolId("eval-v1"),
        parameters={"threshold": 0.5},
    )


def _plan_row(trial: str, parameters: dict[str, object]) -> StudyPlanRow:
    return StudyPlanRow(
        trial=trial,
        training=StudyPlanTraining(
            architecture=ArchitectureId("arch-v1"),
            corpus=CorpusId("train-corpus-v1"),
            protocol=TrainProtocolId("train-v1"),
            seed=42,
            parameters=parameters,
        ),
        evaluations=[_plan_evaluation()],
    )


def _run_plan() -> StudyRunPlan:
    return StudyRunPlan(
        path="plan.jsonl",
        sha256="c" * 64,
        bytes=100,
        trials=1,
        evaluation_jobs=1,
    )


def _study_run(
    status: StudyRunStatus,
    *,
    plan: StudyRunPlan | None,
    finished_at: object | None,
) -> StudyRun:
    return StudyRun(
        schema="mjtensu.mldb/study-run/v1",
        id=StudyRunId("sr-20260908-001"),
        status=status,
        study=StudyId("study-core-v1"),
        execution=StudyRunExecution(started_at="start", finished_at=finished_at),
        plan=plan,
    )


if __name__ == "__main__":
    unittest.main()
