"""Private Study planning preflight for MLDB v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TypeAlias

from mldb_v2.src.catalog.architecture import Architecture, _load_architecture_definition
from mldb_v2.src.catalog.corpus import Corpus, _load_corpus
from mldb_v2.src.catalog.task import Task, _load_task
from mldb_v2.src.common.ids import (
    ArchitectureId,
    CorpusId,
    DefinitionKind,
    EntityKind,
    StudyId,
    TaskId,
    TrainProtocolId,
    EvaluationProtocolId,
)
from mldb_v2.src.evaluation.evaluation_protocol import (
    EvaluationProtocol,
    _load_evaluation_protocol_definition,
)
from mldb_v2.src.repository.resolution import CanonicalRepositoryResolver
from mldb_v2.src.study._study_validation import _load_study_definition
from mldb_v2.src.study.study import (
    EvaluationParameterGrid,
    EvaluationStage,
    Study,
    TrainingParameterGrid,
)
from mldb_v2.src.training.train_protocol import (
    TrainProtocol,
    _load_train_protocol_definition,
)
from mldb_v2.src.verification.definition_lifecycle import DefinitionVerifier


_CanonicalRecord: TypeAlias = Mapping[str, object]


class _PlanningPreflightError(ValueError):
    """Bounded private failure raised before Study expansion begins."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class _EvaluationPlanningInput:
    stage: EvaluationStage
    corpus: Corpus
    protocol: EvaluationProtocol
    @property
    def parameter_axes(self) -> EvaluationParameterGrid:
        return self.stage["parameters"]

    @property
    def parameter_declarations(self):
        return self.protocol["parameters"]


@dataclass(frozen=True)
class _TrainingCasePlanningInput:
    architecture: Architecture
    parameter_overrides: Mapping[str, object]
    seed: int


@dataclass(frozen=True)
class _TrainingPlanningInput:
    corpus: Corpus
    protocol: TrainProtocol
    architectures: tuple[Architecture, ...]
    parameter_axes: TrainingParameterGrid
    seeds: tuple[int, ...]
    cases: tuple[_TrainingCasePlanningInput, ...] | None = None

    @property
    def parameter_declarations(self):
        return self.protocol["parameters"]


@dataclass(frozen=True)
class _ExistingModelPlanningEntry:
    model: _CanonicalRecord
    training_result: _CanonicalRecord
    task_id: str
    architecture_id: str


@dataclass(frozen=True)
class _ExistingModelPlanningInput:
    models: tuple[_ExistingModelPlanningEntry, ...]
_ModelPlanningInput: TypeAlias = _TrainingPlanningInput | _ExistingModelPlanningInput


@dataclass(frozen=True)
class _StudyPlanningInput:
    study: Study
    task: Task
    model: _ModelPlanningInput
    evaluations: tuple[_EvaluationPlanningInput, ...]


class _StudyPlanningPreflight:
    """Resolve one sealed, W002-verifiable Study into planning-only inputs."""

    def __init__(
        self,
        *,
        mldb_data_root: str | Path,
        verifier: DefinitionVerifier,
    ) -> None:
        self._root = Path(mldb_data_root)
        self._resolver = CanonicalRepositoryResolver(self._root)
        self._verifier = verifier

    def prepare(self, study_id: StudyId | str) -> _StudyPlanningInput:
        try:
            study = _load_study_definition(self._root, study_id)
        except (OSError, UnicodeError, ValueError, TypeError) as error:
            raise _PlanningPreflightError("study_unavailable") from error

        if study["status"] != "sealed":
            raise _PlanningPreflightError("study_not_sealed")
        try:
            verification = self._verifier.verify(
                request={"kind": DefinitionKind.STUDY, "id": study["id"]}
            )
        except Exception as error:
            raise _PlanningPreflightError("study_verification_failed") from error

        if (
            type(verification) is not dict
            or verification.get("valid") is not True
            or verification.get("diagnostics") != []
        ):
            raise _PlanningPreflightError("study_verification_failed")

        try:
            model = study["model"]
            if "train" in model:
                planning_model, task = self._prepare_training(model["train"])
            else:
                planning_model, task = self._prepare_existing(model["existing"])
            evaluations = self._prepare_evaluations(study["evaluations"])
        except (OSError, UnicodeError, ValueError, TypeError, KeyError) as error:
            raise _PlanningPreflightError("planning_input_resolution_failed") from error

        return _StudyPlanningInput(
            study=study,
            task=task,
            model=planning_model,
            evaluations=evaluations,
        )

    def _prepare_training(self, source) -> tuple[_TrainingPlanningInput, Task]:
        corpus = _load_corpus(self._resolver, CorpusId(str(source["corpus"])))
        protocol = _load_train_protocol_definition(
            self._root, TrainProtocolId(str(source["protocol"]))
        )
        task = _load_task(self._resolver, TaskId(str(corpus["task"])))

        if "cases" in source:
            architecture_by_id: dict[str, Architecture] = {}
            case_inputs: list[_TrainingCasePlanningInput] = []
            for raw_case in source["cases"]:
                architecture_id = str(raw_case["architecture"])
                architecture = architecture_by_id.get(architecture_id)
                if architecture is None:
                    architecture = _load_architecture_definition(
                        self._root, ArchitectureId(architecture_id)
                    )
                    architecture_by_id[architecture_id] = architecture
                case_inputs.append(
                    _TrainingCasePlanningInput(
                        architecture=architecture,
                        parameter_overrides=raw_case["parameters"],
                        seed=int(raw_case["seed"]),
                    )
                )
            return (
                _TrainingPlanningInput(
                    corpus=corpus,
                    protocol=protocol,
                    architectures=tuple(architecture_by_id.values()),
                    parameter_axes={},
                    seeds=(),
                    cases=tuple(case_inputs),
                ),
                task,
            )

        architectures = tuple(
            _load_architecture_definition(self._root, ArchitectureId(str(architecture_id)))
            for architecture_id in source["architectures"]
        )
        return (
            _TrainingPlanningInput(
                corpus=corpus,
                protocol=protocol,
                architectures=architectures,
                parameter_axes=source["parameters"],
                seeds=tuple(source["seeds"]),
            ),
            task,
        )

    def _prepare_existing(self, model_ids) -> tuple[_ExistingModelPlanningInput, Task]:
        entries: list[_ExistingModelPlanningEntry] = []
        for model_id in model_ids:
            model = self._resolver.resolve(kind=EntityKind.MODEL, entity_id=str(model_id))
            training_result_id = model["training_result"]
            training_result = self._resolver.resolve(
                kind=EntityKind.TRAINING_RESULT,
                entity_id=str(training_result_id),
            )
            entries.append(
                _ExistingModelPlanningEntry(
                    model=model,
                    training_result=training_result,
                    task_id=str(training_result["task"]),
                    architecture_id=str(training_result["architecture"]),
                )
            )
        task = _load_task(self._resolver, TaskId(entries[0].task_id))
        return _ExistingModelPlanningInput(models=tuple(entries)), task

    def _prepare_evaluations(
        self, stages: list[EvaluationStage]
    ) -> tuple[_EvaluationPlanningInput, ...]:
        result: list[_EvaluationPlanningInput] = []
        for stage in stages:
            corpus = _load_corpus(self._resolver, CorpusId(str(stage["corpus"])))
            protocol = _load_evaluation_protocol_definition(
                self._root, EvaluationProtocolId(str(stage["protocol"]))
            )
            result.append(
                _EvaluationPlanningInput(
                    stage=stage,
                    corpus=corpus,
                    protocol=protocol,
                )
            )
        by_name = {item.stage["stage"]: item for item in result}
        for target in result:
            refs = target.stage.get("inputs", {})
            required = target.protocol.get("artifact_inputs", {})
            if set(refs) != set(required):
                raise ValueError(
                    f"Study stage {target.stage['stage']}: inputs {set(refs)} "
                    f"do not match protocol artifact_inputs {set(required)}"
                )
            for alias, reference in refs.items():
                source = by_name[reference["from_stage"]]
                declaration = source.protocol["artifacts"].get(reference["artifact"])
                if declaration is None:
                    raise ValueError(
                        f"Study stage {target.stage['stage']}: from_stage "
                        f"{reference['from_stage']} has no artifact {reference['artifact']}"
                    )
                if not declaration["required"]:
                    raise ValueError(
                        f"Study stage {target.stage['stage']}: optional producer artifact "
                        f"{reference['artifact']} is not supported"
                    )
                if declaration["format"] != required[alias]:
                    raise ValueError(
                        f"Study stage {target.stage['stage']}: input {alias} format mismatch "
                        f"({declaration['format']} != {required[alias]})"
                    )
                if target.protocol.get("artifact_only") and not refs:
                    raise ValueError("artifact_only stage requires inputs")
        return tuple(result)


def _preflight_study(
    *,
    mldb_data_root: str | Path,
    study_id: StudyId | str,
    verifier: DefinitionVerifier,
) -> _StudyPlanningInput:
    return _StudyPlanningPreflight(
        mldb_data_root=mldb_data_root,
        verifier=verifier,
    ).prepare(study_id)
