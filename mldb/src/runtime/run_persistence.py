from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ..common.errors import LifecycleConflictError, NotFoundError, ValidationFailedError
from ..common.ids import EntityKind, EvaluationRunId, StudyRunId, TrainingRunId
from ..evaluation.interface import UnavailableOutput
from ..evaluation.result_validation import AcceptedEvaluationArtifact, EvaluationValidationIssue
from ..evaluation.run import (
    EvaluationRun, EvaluationRunExecution, EvaluationRunFailure, EvaluationRunResult,
    EvaluationRunStatus, EvaluationRunStudyLineage, validate_evaluation_run,
    validate_evaluation_run_transition,
)
from ..repository.layout import RepositoryLayout
from ..repository.ports import FilesystemPort
from ..study.plan import StudyPlan, StudyPlanEvaluation, StudyPlanRow, StudyPlanTraining, validate_study_plan
from ..study.run import (
    StudyRun, StudyRunEvaluationSummary, StudyRunExecution, StudyRunPlan, StudyRunStatus,
    StudyRunSummary, StudyRunTrainingSummary, validate_study_run, validate_study_run_transition,
)
from ..training.run import (
    TrainingRun, TrainingRunExecution, TrainingRunFailure, TrainingRunResult, TrainingRunStatus,
    TrainingRunStudyLineage, validate_training_run, validate_training_run_transition,
)
from ..training.weights import CanonicalWeightsArtifact
from ._run_serialization import decode_timestamp, dump_yaml_document, load_yaml_document

if TYPE_CHECKING:
    from ..evaluation.preflight import EvaluationPreflight
    from ..study.preflight import PreparedStudyExecution
    from ..training.preflight import TrainingPreflight

_LOCK_GUARD = threading.Lock()
_ROOT_LOCKS: dict[str, threading.Lock] = {}
_TRAINING_ID = re.compile(r"tr-([0-9]{8})-([0-9]{3})\Z")
_EVALUATION_ID = re.compile(r"ev-([0-9]{8})-([0-9]{3})\Z")
_STUDY_ID = re.compile(r"sr-([0-9]{8})-([0-9]{3})\Z")


def allocate_training_run(
    preflight: TrainingPreflight,
    allocation_date: date,
    started_at: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    *,
    study: TrainingRunStudyLineage | None = None,
) -> TrainingRun:
    with _allocation_lock(layout):
        run_id = TrainingRunId(_next_id(layout, filesystem, EntityKind.TRAINING_RUN, "tr", allocation_date, _TRAINING_ID))
        run = TrainingRun(
            schema="mjtensu.mldb/training-run/v1", id=run_id, status=TrainingRunStatus.RUNNING,
            corpus=preflight.corpus.metadata.id, architecture=preflight.architecture.metadata.id,
            train_protocol=preflight.protocol.metadata.id, parameters=dict(preflight.parameters),
            execution=TrainingRunExecution(seed=preflight.seed, started_at=started_at), study=study,
        )
        _require_valid(validate_training_run(run))
        paths = layout.training_run_paths(run_id)
        filesystem.ensure_directory(paths.directory)
        filesystem.ensure_directory(paths.work_dir)
        filesystem.ensure_directory(paths.artifacts_dir)
        if filesystem.file_exists(paths.metadata_path):
            raise LifecycleConflictError(f"Training Run already exists: {run_id}")
        filesystem.replace_text(paths.metadata_path, _training_yaml(run), encoding="utf-8")
        return run


def allocate_evaluation_run(
    preflight: EvaluationPreflight,
    allocation_date: date,
    started_at: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    *,
    study: EvaluationRunStudyLineage | None = None,
) -> EvaluationRun:
    with _allocation_lock(layout):
        run_id = EvaluationRunId(_next_id(layout, filesystem, EntityKind.EVALUATION_RUN, "ev", allocation_date, _EVALUATION_ID))
        run = EvaluationRun(
            schema="mjtensu.mldb/evaluation-run/v1", id=run_id, status=EvaluationRunStatus.RUNNING,
            model=preflight.model.metadata.id, corpus=preflight.corpus.metadata.id,
            evaluation_protocol=preflight.protocol.metadata.id, parameters=dict(preflight.parameters),
            execution=EvaluationRunExecution(started_at=started_at), study=study,
        )
        _require_valid(validate_evaluation_run(run))
        paths = layout.evaluation_run_paths(run_id)
        filesystem.ensure_directory(paths.directory)
        filesystem.ensure_directory(paths.work_dir)
        filesystem.ensure_directory(paths.artifacts_dir)
        if filesystem.file_exists(paths.metadata_path):
            raise LifecycleConflictError(f"Evaluation Run already exists: {run_id}")
        filesystem.replace_text(paths.metadata_path, _evaluation_yaml(run), encoding="utf-8")
        return run


def allocate_study_run(
    prepared: PreparedStudyExecution,
    allocation_date: date,
    started_at: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> StudyRun:
    with _allocation_lock(layout):
        run_id = StudyRunId(_next_id(layout, filesystem, EntityKind.STUDY_RUN, "sr", allocation_date, _STUDY_ID))
        run = StudyRun(
            schema="mjtensu.mldb/study-run/v1", id=run_id, status=StudyRunStatus.RUNNING,
            study=prepared.study.metadata.id, execution=StudyRunExecution(started_at=started_at),
        )
        _require_valid(validate_study_run(run))
        paths = layout.study_run_paths(run_id)
        filesystem.ensure_directory(paths.directory)
        if filesystem.file_exists(paths.metadata_path):
            raise LifecycleConflictError(f"Study Run already exists: {run_id}")
        filesystem.replace_text(paths.metadata_path, _study_yaml(run), encoding="utf-8")
        return run


def finalize_study_plan(
    study_run_id: StudyRunId,
    plan: StudyPlan,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> StudyRun:
    _require_valid(validate_study_plan(plan))
    with _allocation_lock(layout):
        current = read_study_run(study_run_id, layout, filesystem)
        if current.status is not StudyRunStatus.RUNNING or current.plan is not None:
            raise LifecycleConflictError("Study plan is immutable after finalization or terminalization")
        data = _encode_plan(plan)
        metadata = StudyRunPlan(
            path="plan.jsonl", sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
            trials=len(plan), evaluation_jobs=sum(len(row.evaluations) for row in plan),
        )
        updated = StudyRun(
            schema=current.schema, id=current.id, status=current.status, study=current.study,
            execution=current.execution, plan=metadata, summary=current.summary,
        )
        _require_valid(validate_study_run(updated))
        paths = layout.study_run_paths(study_run_id)
        filesystem.replace_bytes(paths.plan_path, data)
        filesystem.replace_text(paths.metadata_path, _study_yaml(updated), encoding="utf-8")
        return updated


def persist_training_run_transition(
    next_run: TrainingRun,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> None:
    with _allocation_lock(layout):
        current = read_training_run(next_run.id, layout, filesystem)
        _require_valid(validate_training_run(next_run))
        _require_valid(validate_training_run_transition(current.status, next_run.status))
        if _training_identity(current) != _training_identity(next_run):
            raise LifecycleConflictError("Training Run immutable execution identity changed")
        filesystem.replace_text(layout.training_run_paths(next_run.id).metadata_path, _training_yaml(next_run), encoding="utf-8")


def persist_evaluation_run_transition(
    next_run: EvaluationRun,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> None:
    with _allocation_lock(layout):
        current = read_evaluation_run(next_run.id, layout, filesystem)
        _require_valid(validate_evaluation_run(next_run))
        _require_valid(validate_evaluation_run_transition(current.status, next_run.status))
        if _evaluation_identity(current) != _evaluation_identity(next_run):
            raise LifecycleConflictError("Evaluation Run immutable execution identity changed")
        filesystem.replace_text(layout.evaluation_run_paths(next_run.id).metadata_path, _evaluation_yaml(next_run), encoding="utf-8")


def persist_study_run_transition(
    next_run: StudyRun,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> None:
    with _allocation_lock(layout):
        current = read_study_run(next_run.id, layout, filesystem)
        _require_valid(validate_study_run(next_run))
        _require_valid(validate_study_run_transition(current.status, next_run.status))
        if (current.schema, current.id, current.study, current.execution.started_at, current.plan) != (
            next_run.schema, next_run.id, next_run.study, next_run.execution.started_at, next_run.plan
        ):
            raise LifecycleConflictError("Study Run immutable execution identity or plan changed")
        filesystem.replace_text(layout.study_run_paths(next_run.id).metadata_path, _study_yaml(next_run), encoding="utf-8")


def read_training_run(
    run_id: TrainingRunId,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> TrainingRun:
    path = layout.training_run_paths(run_id).metadata_path
    if not filesystem.file_exists(path):
        raise NotFoundError(f"Training Run not found: {run_id}")
    run = _training_from(load_yaml_document(filesystem.read_text(path, encoding="utf-8")))
    if run.id != run_id:
        raise LifecycleConflictError("Training Run directory identity disagrees with run.yaml")
    _require_valid(validate_training_run(run))
    return run


def read_evaluation_run(
    run_id: EvaluationRunId,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> EvaluationRun:
    path = layout.evaluation_run_paths(run_id).metadata_path
    if not filesystem.file_exists(path):
        raise NotFoundError(f"Evaluation Run not found: {run_id}")
    run = _evaluation_from(load_yaml_document(filesystem.read_text(path, encoding="utf-8")))
    if run.id != run_id:
        raise LifecycleConflictError("Evaluation Run directory identity disagrees with run.yaml")
    _require_valid(validate_evaluation_run(run))
    return run


def read_study_run(
    run_id: StudyRunId,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> StudyRun:
    path = layout.study_run_paths(run_id).metadata_path
    if not filesystem.file_exists(path):
        raise NotFoundError(f"Study Run not found: {run_id}")
    run = _study_from(load_yaml_document(filesystem.read_text(path, encoding="utf-8")))
    if run.id != run_id:
        raise LifecycleConflictError("Study Run directory identity disagrees with run.yaml")
    _require_valid(validate_study_run(run))
    return run


def read_study_plan(
    study_run_id: StudyRunId,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> StudyPlan:
    run = read_study_run(study_run_id, layout, filesystem)
    if run.plan is None:
        raise LifecycleConflictError("Study Run has no finalized plan")
    path = layout.study_run_paths(study_run_id).plan_path
    if not filesystem.file_exists(path):
        raise NotFoundError(f"Study plan not found: {study_run_id}")
    data = filesystem.read_bytes(path)
    if len(data) != run.plan.bytes or hashlib.sha256(data).hexdigest() != run.plan.sha256:
        raise LifecycleConflictError("Study plan bytes disagree with finalized metadata")
    plan = _decode_plan(data)
    _require_valid(validate_study_plan(plan))
    if len(plan) != run.plan.trials or sum(len(row.evaluations) for row in plan) != run.plan.evaluation_jobs:
        raise LifecycleConflictError("Study plan counts disagree with finalized metadata")
    return plan


def list_training_runs(
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> tuple[TrainingRun, ...]:
    return tuple(read_training_run(TrainingRunId(name), layout, filesystem) for name in _listed_ids(layout, filesystem, EntityKind.TRAINING_RUN, _TRAINING_ID))


def list_evaluation_runs(
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> tuple[EvaluationRun, ...]:
    return tuple(read_evaluation_run(EvaluationRunId(name), layout, filesystem) for name in _listed_ids(layout, filesystem, EntityKind.EVALUATION_RUN, _EVALUATION_ID))


def list_study_runs(
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> tuple[StudyRun, ...]:
    return tuple(read_study_run(StudyRunId(name), layout, filesystem) for name in _listed_ids(layout, filesystem, EntityKind.STUDY_RUN, _STUDY_ID))


def _allocation_lock(layout: RepositoryLayout) -> threading.Lock:
    key = str(layout.root.absolute())
    with _LOCK_GUARD:
        return _ROOT_LOCKS.setdefault(key, threading.Lock())


def _next_id(layout: RepositoryLayout, filesystem: FilesystemPort, kind: EntityKind, prefix: str, day: date, pattern: re.Pattern[str]) -> str:
    container = layout.entity_directory(kind)
    filesystem.ensure_directory(container)
    day_text = day.strftime("%Y%m%d")
    highest = 0
    for entry in filesystem.list_directory(container):
        match = pattern.fullmatch(entry.name)
        if match is not None and match.group(1) == day_text:
            highest = max(highest, int(match.group(2)))
    if highest >= 999:
        raise LifecycleConflictError(f"{prefix} Run ID sequence exhausted for {day_text}")
    return f"{prefix}-{day_text}-{highest + 1:03d}"


def _listed_ids(layout: RepositoryLayout, filesystem: FilesystemPort, kind: EntityKind, pattern: re.Pattern[str]) -> tuple[str, ...]:
    container = layout.entity_directory(kind)
    if not filesystem.directory_exists(container):
        return ()
    return tuple(entry.name for entry in filesystem.list_directory(container) if pattern.fullmatch(entry.name) is not None)


def _require_valid(report: Any) -> None:
    if not report.valid:
        raise ValidationFailedError(report)


def _training_identity(run: TrainingRun) -> tuple[Any, ...]:
    return (run.schema, run.id, run.corpus, run.architecture, run.train_protocol, dict(run.parameters), run.execution.seed, run.execution.started_at, run.study)


def _evaluation_identity(run: EvaluationRun) -> tuple[Any, ...]:
    return (run.schema, run.id, run.model, run.corpus, run.evaluation_protocol, dict(run.parameters), run.execution.started_at, run.study)


def _training_yaml(run: TrainingRun) -> str:
    d: dict[str, Any] = {"schema": run.schema, "id": run.id, "status": run.status.value, "corpus": run.corpus, "architecture": run.architecture, "train_protocol": run.train_protocol, "parameters": dict(run.parameters), "execution": {"seed": run.execution.seed, "started_at": run.execution.started_at}}
    if run.execution.finished_at is not None: d["execution"]["finished_at"] = run.execution.finished_at
    if run.result is not None: d["result"] = {"weights": {"format": run.result.weights.format, "path": run.result.weights.path, "sha256": run.result.weights.sha256, "bytes": run.result.weights.bytes}}
    if run.failure is not None: d["failure"] = {"type": run.failure.type, "message": run.failure.message}
    if run.study is not None: d["study"] = {"run": run.study.run, "trial": run.study.trial}
    if run.environment is not None: d["environment"] = dict(run.environment)
    if run.work is not None: d["work"] = dict(run.work)
    return dump_yaml_document(d)


def _evaluation_yaml(run: EvaluationRun) -> str:
    d: dict[str, Any] = {"schema": run.schema, "id": run.id, "status": run.status.value, "model": run.model, "corpus": run.corpus, "evaluation_protocol": run.evaluation_protocol, "parameters": dict(run.parameters), "execution": {"started_at": run.execution.started_at}}
    if run.execution.finished_at is not None: d["execution"]["finished_at"] = run.execution.finished_at
    if run.study is not None: d["study"] = {"run": run.study.run, "trial": run.study.trial, "stage": run.study.stage}
    if run.result is not None:
        d["result"] = {"metrics": dict(run.result.metrics), "artifacts": {k: {"path": v.path, "format": v.format, "schema": v.schema, "sha256": v.sha256, "bytes": v.bytes} for k, v in run.result.artifacts.items()}}
    if run.unavailable_outputs: d["unavailable_outputs"] = [{"output": v.output, "type": v.type, "message": v.message} for v in run.unavailable_outputs]
    if run.validation_issues: d["validation_issues"] = [{"output": v.output, "type": v.type, "message": v.message} for v in run.validation_issues]
    if run.failure is not None: d["failure"] = {"type": run.failure.type, "message": run.failure.message}
    if run.environment is not None: d["environment"] = dict(run.environment)
    return dump_yaml_document(d)


def _study_yaml(run: StudyRun) -> str:
    d: dict[str, Any] = {"schema": run.schema, "id": run.id, "status": run.status.value, "study": run.study, "execution": {"started_at": run.execution.started_at}}
    if run.execution.finished_at is not None: d["execution"]["finished_at"] = run.execution.finished_at
    if run.plan is not None: d["plan"] = {"path": run.plan.path, "sha256": run.plan.sha256, "bytes": run.plan.bytes, "trials": run.plan.trials, "evaluation_jobs": run.plan.evaluation_jobs}
    if run.summary is not None:
        summary: dict[str, Any] = {}
        if run.summary.training is not None: summary["training"] = {"completed": run.summary.training.completed, "failed": run.summary.training.failed, "cancelled": run.summary.training.cancelled}
        if run.summary.evaluation is not None: summary["evaluation"] = {"completed": run.summary.evaluation.completed, "completed_partial": run.summary.evaluation.completed_partial, "failed": run.summary.evaluation.failed, "cancelled": run.summary.evaluation.cancelled, "blocked": run.summary.evaluation.blocked}
        d["summary"] = summary
    return dump_yaml_document(d)


def _training_from(d: dict[str, Any]) -> TrainingRun:
    ex=d["execution"]; result=d.get("result"); failure=d.get("failure"); study=d.get("study")
    weights = result.get("weights") if isinstance(result, dict) else None
    return TrainingRun(schema=d["schema"], id=TrainingRunId(d["id"]), status=TrainingRunStatus(d["status"]), corpus=d["corpus"], architecture=d["architecture"], train_protocol=d["train_protocol"], parameters=d["parameters"], execution=TrainingRunExecution(seed=ex["seed"], started_at=decode_timestamp(ex["started_at"]), finished_at=decode_timestamp(ex.get("finished_at"))), result=TrainingRunResult(CanonicalWeightsArtifact(format=weights["format"], path=weights["path"], sha256=weights["sha256"], bytes=weights["bytes"])) if weights else None, failure=TrainingRunFailure(type=failure["type"], message=failure["message"]) if failure else None, study=TrainingRunStudyLineage(run=study["run"], trial=study["trial"]) if study else None, environment=d.get("environment"), work=d.get("work"))


def _evaluation_from(d: dict[str, Any]) -> EvaluationRun:
    ex=d["execution"]; result=d.get("result"); failure=d.get("failure"); study=d.get("study")
    parsed_result = None
    if result is not None:
        parsed_result = EvaluationRunResult(metrics=result.get("metrics", {}), artifacts={k: AcceptedEvaluationArtifact(path=v["path"], format=v["format"], schema=v["schema"], sha256=v["sha256"], bytes=v["bytes"]) for k,v in result.get("artifacts", {}).items()})
    return EvaluationRun(schema=d["schema"], id=EvaluationRunId(d["id"]), status=EvaluationRunStatus(d["status"]), model=d["model"], corpus=d["corpus"], evaluation_protocol=d["evaluation_protocol"], parameters=d["parameters"], execution=EvaluationRunExecution(started_at=decode_timestamp(ex["started_at"]), finished_at=decode_timestamp(ex.get("finished_at"))), result=parsed_result, unavailable_outputs=tuple(UnavailableOutput(**v) for v in d.get("unavailable_outputs", [])), validation_issues=tuple(EvaluationValidationIssue(**v) for v in d.get("validation_issues", [])), failure=EvaluationRunFailure(type=failure["type"], message=failure["message"]) if failure else None, study=EvaluationRunStudyLineage(run=study["run"], trial=study["trial"], stage=study["stage"]) if study else None, environment=d.get("environment"))


def _study_from(d: dict[str, Any]) -> StudyRun:
    ex=d["execution"]; plan=d.get("plan"); summary=d.get("summary")
    parsed_summary=None
    if summary is not None:
        t=summary.get("training"); e=summary.get("evaluation")
        parsed_summary=StudyRunSummary(training=StudyRunTrainingSummary(**t) if t else None, evaluation=StudyRunEvaluationSummary(**e) if e else None)
    return StudyRun(schema=d["schema"], id=StudyRunId(d["id"]), status=StudyRunStatus(d["status"]), study=d["study"], execution=StudyRunExecution(started_at=decode_timestamp(ex["started_at"]), finished_at=decode_timestamp(ex.get("finished_at"))), plan=StudyRunPlan(**plan) if plan else None, summary=parsed_summary)


def _encode_plan(plan: StudyPlan) -> bytes:
    lines=[]
    for row in plan:
        item: dict[str, Any] = {"trial": row.trial}
        if row.training is not None:
            item["training"] = {"architecture": row.training.architecture, "corpus": row.training.corpus, "protocol": row.training.protocol, "seed": row.training.seed, "parameters": dict(row.training.parameters)}
        else:
            item["model"] = row.model
        item["evaluations"] = [{"stage": e.stage, "corpus": e.corpus, "protocol": e.protocol, "parameters": dict(e.parameters)} for e in row.evaluations]
        lines.append(json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _decode_plan(data: bytes) -> StudyPlan:
    text=data.decode("utf-8")
    rows=[]
    for line in text.splitlines():
        if not line.strip():
            raise ValueError("blank Study plan line")
        d=json.loads(line)
        training=d.get("training")
        rows.append(StudyPlanRow(trial=d["trial"], training=StudyPlanTraining(architecture=training["architecture"], corpus=training["corpus"], protocol=training["protocol"], seed=training["seed"], parameters=training["parameters"]) if training is not None else None, model=d.get("model"), evaluations=tuple(StudyPlanEvaluation(stage=e["stage"], corpus=e["corpus"], protocol=e["protocol"], parameters=e["parameters"]) for e in d["evaluations"])))
    return tuple(rows)
