"""Typed canonical asset resolution for MLDB runtime definitions."""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from ..catalog.architecture import (
    Architecture, ArchitectureImplementation, ArchitectureInterface, ArchitectureStatus,
    ArchitectureStructure, validate_architecture_metadata,
)
from ..catalog.corpus import Corpus, CorpusArtifact, CorpusDataSpec, validate_corpus
from ..catalog.task import CategoricalTarget, Task, TaskInput, TaskScope, validate_task
from ..common.errors import NotFoundError, ValidationFailedError, ValidationIssue, ValidationReport
from ..common.ids import (
    ArchitectureId, CorpusId, EvaluationProtocolId, ModelId, StudyId, TaskId,
    TrainingRunId, TrainProtocolId,
)
from ..common.parameters import PublicParameterDeclaration
from ..evaluation.protocol import (
    EvaluationArtifactDeclaration, EvaluationMetricDeclaration, EvaluationOutputs,
    EvaluationProtocol, EvaluationProtocolImplementation, EvaluationProtocolStatus,
    validate_evaluation_protocol_metadata,
)
from ..repository.layout import RepositoryLayout
from ..repository.ports import FilesystemPort
from ..study.definition import (
    Study, StudyEvaluationStage, StudyExistingModelSource, StudyStatus,
    StudyTrainingModelSource, StudyTrainingParameterAxis, validate_study_metadata,
)
from ..training.protocol import (
    TrainProtocol, TrainProtocolImplementation, TrainProtocolStatus,
    validate_train_protocol_metadata,
)
from ..training.run import (
    TrainingRun, TrainingRunExecution, TrainingRunFailure, TrainingRunResult,
    TrainingRunStatus, TrainingRunStudyLineage, validate_training_run,
)
from ..training.weights import CanonicalWeightsArtifact
from ._yaml import _YamlError, _load_yaml
from .catalog_handles import ArchitectureHandle, CorpusHandle, TaskHandle
from .definition_handles import EvaluationProtocolHandle, StudyHandle, TrainProtocolHandle

if TYPE_CHECKING:
    from ..model.loading import ModelHandle


def resolve_task(task_id: TaskId, layout: RepositoryLayout, filesystem: FilesystemPort) -> TaskHandle:
    path = layout.task_metadata_path(task_id)
    raw = _read_mapping(path, filesystem)
    metadata = _normalize_task(raw)
    _accept(validate_task(metadata), "Task")
    _require_identity(task_id, metadata.id, "Task")
    return TaskHandle(metadata=metadata, metadata_path=path)


def resolve_corpus(corpus_id: CorpusId, layout: RepositoryLayout, filesystem: FilesystemPort) -> CorpusHandle:
    metadata_path = layout.corpus_metadata_path(corpus_id)
    artifact_path = layout.corpus_artifact_path(corpus_id)
    builder_path = layout.corpus_builder_path(corpus_id)
    _require_file(metadata_path, filesystem)
    _require_file(artifact_path, filesystem)
    _require_file(builder_path, filesystem)
    metadata = _normalize_corpus(_read_mapping(metadata_path, filesystem, required=False))
    _accept(validate_corpus(metadata), "Corpus")
    _require_identity(corpus_id, metadata.id, "Corpus")
    task = resolve_task(metadata.task, layout, filesystem)
    artifact_bytes = filesystem.read_bytes(artifact_path)
    digest = hashlib.sha256(artifact_bytes).hexdigest()
    if digest.lower() != metadata.artifact.sha256.lower():
        _fail("corpus.artifact.sha256.mismatch", "Corpus SQLite SHA-256 does not match metadata.", "artifact.sha256")
    if metadata.artifact.bytes is not None and len(artifact_bytes) != metadata.artifact.bytes:
        _fail("corpus.artifact.bytes.mismatch", "Corpus SQLite byte size does not match metadata.", "artifact.bytes")
    _validate_corpus_sqlite(metadata, task.metadata, artifact_path)
    return CorpusHandle(metadata=metadata, metadata_path=metadata_path, artifact_path=artifact_path, builder_path=builder_path)


def resolve_architecture(architecture_id: ArchitectureId, layout: RepositoryLayout, filesystem: FilesystemPort) -> ArchitectureHandle:
    metadata_path = layout.architecture_metadata_path(architecture_id)
    implementation_path = layout.architecture_implementation_path(architecture_id)
    _require_file(metadata_path, filesystem)
    _require_file(implementation_path, filesystem)
    metadata = _normalize_architecture(_read_mapping(metadata_path, filesystem, required=False))
    _accept(validate_architecture_metadata(metadata), "Architecture")
    _require_identity(architecture_id, metadata.id, "Architecture")
    resolve_task(metadata.task, layout, filesystem)
    return ArchitectureHandle(metadata=metadata, metadata_path=metadata_path, implementation_path=implementation_path)


def resolve_train_protocol(train_protocol_id: TrainProtocolId, layout: RepositoryLayout, filesystem: FilesystemPort) -> TrainProtocolHandle:
    metadata_path = layout.train_protocol_metadata_path(train_protocol_id)
    implementation_path = layout.train_protocol_implementation_path(train_protocol_id)
    _require_file(metadata_path, filesystem)
    _require_file(implementation_path, filesystem)
    metadata = _normalize_train_protocol(_read_mapping(metadata_path, filesystem, required=False))
    _accept(validate_train_protocol_metadata(metadata), "Train Protocol")
    _require_identity(train_protocol_id, metadata.id, "Train Protocol")
    resolve_task(metadata.task, layout, filesystem)
    return TrainProtocolHandle(metadata=metadata, metadata_path=metadata_path, implementation_path=implementation_path)


def resolve_model(model_id: ModelId, layout: RepositoryLayout, filesystem: FilesystemPort) -> ModelHandle:
    from ..model.identity import validate_model_metadata
    from ..model.loading import ModelHandle
    metadata_path = layout.model_metadata_path(model_id)
    raw = _read_mapping(metadata_path, filesystem)
    metadata = _normalize_model(raw)
    _accept(validate_model_metadata(metadata), "Model")
    _require_identity(model_id, metadata.id, "Model")
    run_paths = layout.training_run_paths(metadata.training_run)
    run = _normalize_training_run(_read_mapping(run_paths.metadata_path, filesystem))
    _accept(validate_training_run(run), "Training Run")
    _require_identity(metadata.training_run, run.id, "Training Run")
    if run.status is not TrainingRunStatus.COMPLETED or run.result is None:
        _fail("model.training_run.not_completed", "Model lineage Training Run must be completed with canonical weights.", "training_run")
    if run.result.weights.path != "artifacts/weights.pt":
        _fail("model.weights.path.invalid", "Training Run weights path is not canonical.", "result.weights.path")
    _require_file(run_paths.weights_path, filesystem)
    architecture = resolve_architecture(run.architecture, layout, filesystem)
    return ModelHandle(metadata=metadata, metadata_path=metadata_path, training_run=run,
                       training_run_metadata_path=run_paths.metadata_path,
                       weights_path=run_paths.weights_path, architecture=architecture)


def resolve_evaluation_protocol(evaluation_protocol_id: EvaluationProtocolId, layout: RepositoryLayout, filesystem: FilesystemPort) -> EvaluationProtocolHandle:
    metadata_path = layout.evaluation_protocol_metadata_path(evaluation_protocol_id)
    implementation_path = layout.evaluation_protocol_implementation_path(evaluation_protocol_id)
    _require_file(metadata_path, filesystem)
    _require_file(implementation_path, filesystem)
    metadata = _normalize_evaluation_protocol(_read_mapping(metadata_path, filesystem, required=False))
    _accept(validate_evaluation_protocol_metadata(metadata), "Evaluation Protocol")
    _require_identity(evaluation_protocol_id, metadata.id, "Evaluation Protocol")
    resolve_task(metadata.task, layout, filesystem)
    return EvaluationProtocolHandle(metadata=metadata, metadata_path=metadata_path, implementation_path=implementation_path)


def resolve_study(study_id: StudyId, layout: RepositoryLayout, filesystem: FilesystemPort) -> StudyHandle:
    path = layout.study_metadata_path(study_id)
    metadata = _normalize_study(_read_mapping(path, filesystem))
    _accept(validate_study_metadata(metadata), "Study")
    _require_identity(study_id, metadata.id, "Study")
    if isinstance(metadata.model, StudyTrainingModelSource):
        resolve_corpus(metadata.model.corpus, layout, filesystem)
        resolve_train_protocol(metadata.model.protocol, layout, filesystem)
        for architecture_id in metadata.model.architectures:
            resolve_architecture(architecture_id, layout, filesystem)
    else:
        for model_id in metadata.model.models:
            resolve_model(model_id, layout, filesystem)
    for evaluation in metadata.evaluations:
        resolve_corpus(evaluation.corpus, layout, filesystem)
        resolve_evaluation_protocol(evaluation.protocol, layout, filesystem)
    return StudyHandle(metadata=metadata, metadata_path=path)


def _read_mapping(path: Path, filesystem: FilesystemPort, *, required: bool = True) -> dict[str, object]:
    if required:
        _require_file(path, filesystem)
    try:
        value = _load_yaml(filesystem.read_text(path, encoding="utf-8"))
    except (OSError, UnicodeError, _YamlError, ValueError) as error:
        _fail("metadata.parse.invalid", f"Invalid MLDB YAML at {path}: {error}")
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        _fail("metadata.shape.invalid", "MLDB metadata root must be a string-keyed mapping.")
    return value


def _require_file(path: Path, filesystem: FilesystemPort) -> None:
    if not filesystem.file_exists(path):
        raise NotFoundError(f"required canonical file does not exist: {path}")


def _require_identity(requested: object, recorded: object, kind: str) -> None:
    if requested != recorded:
        _fail("resolution.identity.mismatch", f"{kind} requested ID and recorded ID disagree.", "id")


def _accept(report: ValidationReport, kind: str) -> None:
    if not report.valid:
        raise ValidationFailedError(report)


def _fail(code: str, message: str, path: str | None = None) -> None:
    raise ValidationFailedError(ValidationReport((ValidationIssue(code=code, message=message, path=path),)))


def _mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        _fail("metadata.field.mapping_required", f"{path} must be a mapping.", path)
    return value


def _sequence(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        _fail("metadata.field.sequence_required", f"{path} must be a sequence.", path)
    return value


def _required(raw: Mapping[str, object], key: str) -> object:
    if key not in raw:
        _fail("metadata.field.required", f"Required field {key!r} is missing.", key)
    return raw[key]


def _enum(enum_type: type, value: object, path: str):
    try:
        return enum_type(value)
    except (ValueError, TypeError):
        _fail("metadata.enum.invalid", f"Invalid value for {path}.", path)


def _normalize_task(raw: Mapping[str, object]) -> Task:
    inp = _mapping(_required(raw, "input"), "input")
    target = _mapping(_required(raw, "target"), "target")
    scope = _mapping(_required(raw, "scope"), "scope")
    return Task(
        schema=_required(raw, "schema"), id=TaskId(_required(raw, "id")),
        name=_required(raw, "name"), problem_type=_required(raw, "problem_type"),
        description=_required(raw, "description"),
        input=TaskInput(semantic_unit=_required(inp, "semantic_unit")),
        target=CategoricalTarget(type=_required(target, "type"), labels=tuple(_sequence(_required(target, "labels"), "target.labels"))),
        semantics=_mapping(_required(raw, "semantics"), "semantics"),
        scope=TaskScope(includes=tuple(_sequence(_required(scope, "includes"), "scope.includes")),
                        excludes=tuple(_sequence(_required(scope, "excludes"), "scope.excludes"))),
    )


def _normalize_corpus(raw: Mapping[str, object]) -> Corpus:
    artifact = _mapping(_required(raw, "artifact"), "artifact")
    data = _mapping(_required(raw, "data"), "data")
    builder = _mapping(_required(raw, "builder"), "builder")
    return Corpus(
        schema=_required(raw, "schema"), id=CorpusId(_required(raw, "id")), task=TaskId(_required(raw, "task")),
        artifact=CorpusArtifact(format=_required(artifact, "format"), sha256=_required(artifact, "sha256"), bytes=artifact.get("bytes")),
        data=CorpusDataSpec(schema=_required(data, "schema"), table=_required(data, "table")),
        representation=_mapping(_required(raw, "representation"), "representation"),
        builder_parameters=_mapping(_required(builder, "parameters"), "builder.parameters"),
        splits=_mapping(_required(raw, "splits"), "splits"), description=raw.get("description"),
        statistics=raw.get("statistics"), origin=raw.get("origin"),
    )


def _normalize_architecture(raw: Mapping[str, object]) -> Architecture:
    implementation = _mapping(_required(raw, "implementation"), "implementation")
    interface = _mapping(_required(raw, "interface"), "interface")
    structure = _mapping(_required(raw, "structure"), "structure")
    traits = structure.get("traits")
    return Architecture(
        schema=_required(raw, "schema"), id=ArchitectureId(_required(raw, "id")),
        status=_enum(ArchitectureStatus, _required(raw, "status"), "status"), task=TaskId(_required(raw, "task")),
        name=_required(raw, "name"), family=_required(raw, "family"), description=_required(raw, "description"),
        implementation=ArchitectureImplementation(framework=_required(implementation, "framework"),
            entrypoint=_required(implementation, "entrypoint"), sha256=implementation.get("sha256")),
        interface=ArchitectureInterface(input=_mapping(_required(interface, "input"), "interface.input"),
                                        output=_mapping(_required(interface, "output"), "interface.output")),
        structure=ArchitectureStructure(summary=_required(structure, "summary"),
            traits=None if traits is None else tuple(_sequence(traits, "structure.traits"))),
        parameters=raw.get("parameters"),
    )


def _parameter_declarations(value: object, path: str) -> dict[str, PublicParameterDeclaration]:
    mapping = _mapping(value, path)
    result: dict[str, PublicParameterDeclaration] = {}
    for key, raw_declaration in mapping.items():
        declaration = _mapping(raw_declaration, f"{path}.{key}")
        result[key] = PublicParameterDeclaration(default=_required(declaration, "default"))
    return result


def _normalize_train_protocol(raw: Mapping[str, object]) -> TrainProtocol:
    implementation = _mapping(_required(raw, "implementation"), "implementation")
    return TrainProtocol(
        schema=_required(raw, "schema"), id=TrainProtocolId(_required(raw, "id")),
        status=_enum(TrainProtocolStatus, _required(raw, "status"), "status"), task=TaskId(_required(raw, "task")),
        name=_required(raw, "name"), description=_required(raw, "description"),
        implementation=TrainProtocolImplementation(entrypoint=_required(implementation, "entrypoint"), sha256=implementation.get("sha256")),
        parameters=_parameter_declarations(_required(raw, "parameters"), "parameters"), notes=raw.get("notes"),
    )


def _normalize_evaluation_protocol(raw: Mapping[str, object]) -> EvaluationProtocol:
    implementation = _mapping(_required(raw, "implementation"), "implementation")
    outputs = _mapping(_required(raw, "outputs"), "outputs")
    metric_raw = _mapping(_required(outputs, "metrics"), "outputs.metrics")
    artifact_raw = _mapping(_required(outputs, "artifacts"), "outputs.artifacts")
    metrics = {key: EvaluationMetricDeclaration(type=_required(_mapping(value, f"outputs.metrics.{key}"), "type"),
               description=_mapping(value, f"outputs.metrics.{key}").get("description")) for key, value in metric_raw.items()}
    artifacts = {key: EvaluationArtifactDeclaration(
        format=_required(_mapping(value, f"outputs.artifacts.{key}"), "format"),
        schema=_required(_mapping(value, f"outputs.artifacts.{key}"), "schema"),
        required=_required(_mapping(value, f"outputs.artifacts.{key}"), "required")) for key, value in artifact_raw.items()}
    return EvaluationProtocol(
        schema=_required(raw, "schema"), id=EvaluationProtocolId(_required(raw, "id")),
        status=_enum(EvaluationProtocolStatus, _required(raw, "status"), "status"), task=TaskId(_required(raw, "task")),
        name=_required(raw, "name"), description=_required(raw, "description"),
        implementation=EvaluationProtocolImplementation(entrypoint=_required(implementation, "entrypoint"), sha256=implementation.get("sha256")),
        parameters=_parameter_declarations(_required(raw, "parameters"), "parameters"),
        outputs=EvaluationOutputs(metrics=metrics, artifacts=artifacts),
    )


def _normalize_model(raw: Mapping[str, object]):
    allowed = {"schema", "id", "training_run"}
    extras = set(raw) - allowed
    if extras:
        _fail("model.field.unexpected", f"Unexpected Model v1 fields: {sorted(extras)!r}.")
    from ..model.identity import Model
    return Model(schema=_required(raw, "schema"), id=ModelId(_required(raw, "id")),
                 training_run=TrainingRunId(_required(raw, "training_run")))


def _normalize_training_run(raw: Mapping[str, object]) -> TrainingRun:
    execution = _mapping(_required(raw, "execution"), "execution")
    result_raw = raw.get("result")
    result = None
    if result_raw is not None:
        result_mapping = _mapping(result_raw, "result")
        weights = _mapping(_required(result_mapping, "weights"), "result.weights")
        result = TrainingRunResult(weights=CanonicalWeightsArtifact(
            format=_required(weights, "format"), path=_required(weights, "path"),
            sha256=_required(weights, "sha256"), bytes=_required(weights, "bytes")))
    failure_raw = raw.get("failure")
    failure = None if failure_raw is None else TrainingRunFailure(
        type=_required(_mapping(failure_raw, "failure"), "type"),
        message=_required(_mapping(failure_raw, "failure"), "message"))
    study_raw = raw.get("study")
    study = None if study_raw is None else TrainingRunStudyLineage(
        run=_required(_mapping(study_raw, "study"), "run"),
        trial=_required(_mapping(study_raw, "study"), "trial"))
    return TrainingRun(
        schema=_required(raw, "schema"), id=TrainingRunId(_required(raw, "id")),
        status=_enum(TrainingRunStatus, _required(raw, "status"), "status"),
        corpus=CorpusId(_required(raw, "corpus")), architecture=ArchitectureId(_required(raw, "architecture")),
        train_protocol=TrainProtocolId(_required(raw, "train_protocol")),
        parameters=_mapping(_required(raw, "parameters"), "parameters"),
        execution=TrainingRunExecution(seed=_required(execution, "seed"), started_at=_required(execution, "started_at"),
                                       finished_at=execution.get("finished_at")),
        result=result, failure=failure, study=study,
        environment=None if raw.get("environment") is None else _mapping(raw.get("environment"), "environment"),
        work=None if raw.get("work") is None else _mapping(raw.get("work"), "work"),
    )


def _normalize_study(raw: Mapping[str, object]) -> Study:
    model_raw = _mapping(_required(raw, "model"), "model")
    if set(model_raw) == {"train"}:
        train = _mapping(model_raw["train"], "model.train")
        axes_raw = _mapping(_required(train, "parameters"), "model.train.parameters")
        axes = {key: StudyTrainingParameterAxis(values=_sequence(_required(_mapping(value, f"model.train.parameters.{key}"), "values"),
                                                            f"model.train.parameters.{key}.values"))
                for key, value in axes_raw.items()}
        model = StudyTrainingModelSource(
            corpus=CorpusId(_required(train, "corpus")), protocol=TrainProtocolId(_required(train, "protocol")),
            architectures=[ArchitectureId(value) for value in _sequence(_required(train, "architectures"), "model.train.architectures")],
            parameters=axes, seeds=_sequence(_required(train, "seeds"), "model.train.seeds"))
    elif set(model_raw) == {"existing"}:
        model = StudyExistingModelSource(models=[ModelId(value) for value in _sequence(model_raw["existing"], "model.existing")])
    else:
        _fail("study.model_source.invalid", "Study model must contain exactly one of train or existing.", "model")
    evaluations = []
    for index, value in enumerate(_sequence(_required(raw, "evaluations"), "evaluations")):
        item = _mapping(value, f"evaluations[{index}]")
        evaluations.append(StudyEvaluationStage(
            stage=_required(item, "stage"), corpus=CorpusId(_required(item, "corpus")),
            protocol=EvaluationProtocolId(_required(item, "protocol")),
            parameters=_mapping(_required(item, "parameters"), f"evaluations[{index}].parameters")))
    return Study(schema=_required(raw, "schema"), id=StudyId(_required(raw, "id")),
                 status=_enum(StudyStatus, _required(raw, "status"), "status"),
                 name=_required(raw, "name"), description=_required(raw, "description"),
                 model=model, evaluations=evaluations)


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _validate_corpus_sqlite(corpus: Corpus, task: Task, artifact_path: Path) -> None:
    if corpus.data.schema != "mjtensu.mldb/image-classification-corpus/v1":
        _fail("corpus.data.schema.unsupported", "Unsupported concrete Corpus data schema.", "data.schema")
    representation = corpus.representation
    if representation.get("kind") != "image" or representation.get("dtype") != "uint8":
        _fail("corpus.representation.unsupported", "Image-classification v1 requires image/uint8 representation.", "representation")
    payload_column = representation.get("payload_column")
    shape = representation.get("shape")
    if not isinstance(payload_column, str) or not payload_column:
        _fail("corpus.representation.payload_column.invalid", "payload_column must be a non-empty string.", "representation.payload_column")
    if not isinstance(shape, list) or not shape or any(type(dimension) is not int or dimension <= 0 for dimension in shape):
        _fail("corpus.representation.shape.invalid", "shape must contain positive integer dimensions.", "representation.shape")
    element_count = 1
    for dimension in shape:
        element_count *= dimension
    try:
        connection = sqlite3.connect(f"{artifact_path.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as error:
        _fail("corpus.sqlite.open_failed", f"Could not open canonical SQLite artifact read-only: {error}")
    try:
        table = _quote_identifier(corpus.data.table)
        info = connection.execute(f"PRAGMA table_info({table})").fetchall()
        columns = {row[1] for row in info}
        required = {"sample_id", "split", "target", "class_index", payload_column}
        missing = required - columns
        if missing:
            _fail("corpus.sqlite.columns.missing", f"Required sample columns are missing: {sorted(missing)!r}.")
        duplicate = connection.execute(f"SELECT sample_id FROM {table} GROUP BY sample_id HAVING COUNT(*) > 1 LIMIT 1").fetchone()
        if duplicate is not None:
            _fail("corpus.sample_id.duplicate", "sample_id must be unique within the Corpus.", "sample_id")
        empty_split = connection.execute(f"SELECT 1 FROM {table} WHERE split IS NULL OR split = '' LIMIT 1").fetchone()
        if empty_split is not None:
            _fail("corpus.split.invalid", "split must be non-empty for every sample.", "split")
        labels = task.target.labels
        label_to_index = {label: index for index, label in enumerate(labels)}
        payload_sql = _quote_identifier(payload_column)
        for target, class_index, storage_type, payload_size in connection.execute(
            f"SELECT target, class_index, typeof({payload_sql}), length({payload_sql}) FROM {table}"
        ):
            if target not in label_to_index or type(class_index) is not int or class_index != label_to_index[target]:
                _fail("corpus.class_index.mismatch", "target/class_index disagrees with Task ordered labels.", "class_index")
            if storage_type != "blob" or payload_size != element_count:
                _fail("corpus.payload.mismatch", "Image payload is not the declared uint8 BLOB shape.", f"representation.{payload_column}")
        actual_splits = dict(connection.execute(f"SELECT split, COUNT(*) FROM {table} GROUP BY split").fetchall())
        if actual_splits != dict(corpus.splits):
            _fail("corpus.splits.mismatch", "Corpus split summary disagrees with canonical sample table.", "splits")
    except sqlite3.Error as error:
        _fail("corpus.sqlite.invalid", f"Invalid canonical SQLite artifact: {error}")
    finally:
        connection.close()
