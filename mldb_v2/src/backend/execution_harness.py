"""Backend-neutral process entrypoint for one immutable planned stage."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, cast

from mldb_v2.src.backend.candidate_outcome import (
    CompletedEvaluationCandidate,
    CompletedTrainingCandidate,
    EvaluationStageKey,
    FailedEvaluationCandidate,
    FailedTrainingCandidate,
    StageKey,
    TerminalCandidate,
    TrainingStageKey,
)
from mldb_v2.src.backend.stage_input import EvaluationStageInput, StageInput, TrainingStageInput
from mldb_v2.src.catalog._executable_definition_loading import _validate_source_path
from mldb_v2.src.catalog.architecture import _load_architecture_definition
from mldb_v2.src.catalog.corpus import _load_corpus
from mldb_v2.src.common.ids import (
    EntityKind,
    _validate_evaluation_coordinate_id,
    _validate_trial_id,
    _validate_typed_reference,
)
from mldb_v2.src.common.diagnostic import Diagnostic, _validate_diagnostic
from mldb_v2.src.common.telemetry import _AcceptedScalarEvent
from mldb_v2.src.evaluation import runtime as evaluation_runtime
from mldb_v2.src.evaluation.evaluation_protocol import _load_evaluation_protocol_definition
from mldb_v2.src.repository.resolution import CanonicalRepositoryResolver
from mldb_v2.src.source.git_snapshot import (
    _read_committed_source_bytes,
    _repo_root,
    _require_commit,
    _run_git,
    _validate_commit_id,
    _working_tree_source_matches_commit,
)
from mldb_v2.src.storage._paths import _validate_safe_relative_path
from mldb_v2.src.storage.corpus_manifest import _verify_corpus_manifest
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
from mldb_v2.src.study._plan_build import _validate_study_plan
from mldb_v2.src.study.plan import PlanPin, StudyPlan
from mldb_v2.src.training import runtime as training_runtime
from mldb_v2.src.training.train_protocol import _load_train_protocol_definition


class ExecutionHarness(Protocol):
    """Execute one admitted StageInput through the common MLDB harness."""

    def __call__(self, stage_input: StageInput) -> TerminalCandidate:
        ...


_STAGE_INPUT_FIELDS = {
    "schema",
    "study_result",
    "plan",
    "plan_sha256",
    "trial",
    "kind",
    "coordinate",
    "source_commit",
    "pins",
    "stage",
    "runtime_model",
}
_PIN_DOMAIN = {
    "task": "tasks",
    "corpus": "corpora",
    "architecture": "architectures",
    "train_protocol": "train_protocols",
    "evaluation_protocol": "evaluation_protocols",
    "study": "studies",
    "model": "models",
    "training_result": "training_results",
}
_EXECUTABLE_LOADERS = {
    "architecture": _load_architecture_definition,
    "train_protocol": _load_train_protocol_definition,
    "evaluation_protocol": _load_evaluation_protocol_definition,
}
_FAILURE_DIAGNOSTIC: Diagnostic = {
    "code": "stage_execution_failed",
    "message": "Stage execution failed.",
}


def _nonempty_string(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _validate_started_at(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value.endswith("Z"):
        raise ValueError("started_at must be RFC3339 UTC using Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("started_at must be RFC3339 UTC using Z") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("started_at must be RFC3339 UTC using Z")
    return value


def _ended_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _establish_stage_key(value: object) -> StageKey:
    if type(value) is not dict or set(value) != _STAGE_INPUT_FIELDS:
        raise ValueError("StageInput fields do not match schema")
    if value["schema"] != "mjtensu.mldb-v2/stage-input/v1":
        raise ValueError("unsupported StageInput schema")
    study_result = _validate_typed_reference(value["study_result"])
    plan = _validate_typed_reference(value["plan"])
    trial = str(_validate_trial_id(value["trial"]))
    source_commit = _validate_commit_id(value["source_commit"])
    kind = value["kind"]
    if kind == "training":
        if value["coordinate"] is not None:
            raise ValueError("training StageInput coordinate must be null")
        return TrainingStageKey(
            study_result=study_result,
            plan=plan,
            trial=trial,
            kind="training",
            coordinate=None,
            source_commit=source_commit,
        )
    if kind == "evaluation":
        coordinate = str(_validate_evaluation_coordinate_id(value["coordinate"]))
        return EvaluationStageKey(
            study_result=study_result,
            plan=plan,
            trial=trial,
            kind="evaluation",
            coordinate=coordinate,
            source_commit=source_commit,
        )
    raise ValueError("StageInput kind must be training or evaluation")


def _resolved_mldb_prefix(repository_root: Path, mldb_data_root: Path) -> str:
    root = mldb_data_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("pinned mldb_data root must be a directory")
    try:
        relative = root.relative_to(repository_root)
    except ValueError as error:
        raise ValueError("pinned mldb_data root must be inside repository root") from error
    if relative == Path("."):
        raise ValueError("pinned mldb_data root must not be repository root")
    return relative.as_posix()


def _verify_core_snapshot(repository_root: Path, commit: str) -> None:
    _require_commit(repository_root, commit)
    tree = _run_git(
        repository_root,
        "ls-tree",
        "-r",
        "--name-only",
        commit,
        "--",
        "mldb_v2/src",
    )
    if tree.returncode != 0 or not tree.stdout.strip():
        raise ValueError("pinned commit does not contain mldb_v2/src")
    diff = _run_git(repository_root, "diff", "--quiet", commit, "--", "mldb_v2/src")
    if diff.returncode != 0:
        raise ValueError("mldb_v2/src does not match pinned source commit")
    untracked = _run_git(
        repository_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        "mldb_v2/src",
    )
    if untracked.returncode != 0 or untracked.stdout.strip():
        raise ValueError("mldb_v2/src does not match pinned source commit")


def _verified_committed_bytes(repository_root: Path, *, commit: str, path: str) -> bytes:
    relative = _validate_safe_relative_path(path, label="pinned source path")
    data = _read_committed_source_bytes(repository_root, commit=commit, path=relative)
    if not _working_tree_source_matches_commit(repository_root, commit=commit, path=relative):
        raise ValueError(f"working tree source does not match pinned commit: {relative}")
    return data


def _pin_yaml_relative(prefix: str, pin: PlanPin) -> str:
    kind = pin["kind"]
    entity_id = str(pin["id"])
    if kind == "namespace":
        return f"{prefix}/{entity_id}/namespace.yaml"
    domain = _PIN_DOMAIN[kind]
    namespace, local_id = entity_id.split("/", 1)
    return f"{prefix}/{namespace}/{domain}/{local_id}.yaml"


def _pin_companion_relative(prefix: str, *, kind: str, entity_id: str) -> str:
    namespace, local_id = entity_id.split("/", 1)
    return f"{prefix}/{namespace}/{_PIN_DOMAIN[kind]}/{local_id}.py"


def _verify_executable_pin(
    *, repository_root: Path, pinned_root: Path, prefix: str, commit: str, pin: PlanPin
) -> None:
    kind = pin["kind"]
    entity_id = str(pin["id"])
    loader = _EXECUTABLE_LOADERS[kind]
    definition = loader(pinned_root, entity_id)
    implementation = definition["implementation"]
    if type(implementation) is not dict:
        raise ValueError("pinned executable implementation is invalid")
    if implementation.get("sha256") != pin["companion_sha256"]:
        raise ValueError("pinned executable companion digest does not match definition")
    declared_sources = implementation.get("sources", [])
    if declared_sources != pin["sources"]:
        raise ValueError("pinned executable sources do not match definition")
    companion = _verified_committed_bytes(
        repository_root,
        commit=commit,
        path=_pin_companion_relative(prefix, kind=kind, entity_id=entity_id),
    )
    if hashlib.sha256(companion).hexdigest() != pin["companion_sha256"]:
        raise ValueError("pinned executable companion sha256 mismatch")
    for source in pin["sources"]:
        path = _validate_source_path(source["path"])
        data = _verified_committed_bytes(
            repository_root,
            commit=commit,
            path=path,
        )
        if hashlib.sha256(data).hexdigest() != source["sha256"]:
            raise ValueError("pinned executable source sha256 mismatch")


def _verify_corpus_pin(
    *, repository_root: Path, pinned_root: Path, prefix: str, commit: str, pin: PlanPin
) -> None:
    entity_id = str(pin["id"])
    resolver = CanonicalRepositoryResolver(pinned_root)
    corpus = _load_corpus(resolver, entity_id)
    manifest = corpus["manifest"]
    if manifest["sha256"] != pin["manifest_sha256"]:
        raise ValueError("pinned Corpus manifest sha256 does not match definition")
    if manifest["entries"] != pin["manifest_entries"]:
        raise ValueError("pinned Corpus manifest count does not match definition")
    namespace, _local_id = entity_id.split("/", 1)
    manifest_path = f"{prefix}/{namespace}/corpora/{manifest['file']}"
    manifest_bytes = _verified_committed_bytes(
        repository_root,
        commit=commit,
        path=manifest_path,
    )
    _verify_corpus_manifest(
        manifest_bytes,
        expected_sha256=cast(str, pin["manifest_sha256"]),
        expected_count=cast(int, pin["manifest_entries"]),
    )
    builder = corpus.get("builder")
    expected_builder = pin["companion_sha256"]
    if builder is None:
        if expected_builder is not None:
            raise ValueError("pinned Corpus builder does not match definition")
        return
    if type(builder) is not dict or builder.get("sha256") != expected_builder:
        raise ValueError("pinned Corpus builder digest does not match definition")
    builder_bytes = _verified_committed_bytes(
        repository_root,
        commit=commit,
        path=_pin_companion_relative(prefix, kind="corpus", entity_id=entity_id),
    )
    if hashlib.sha256(builder_bytes).hexdigest() != expected_builder:
        raise ValueError("pinned Corpus builder sha256 mismatch")


def _verify_plan_pins(
    *, repository_root: Path, pinned_root: Path, commit: str, pins: list[PlanPin]
) -> None:
    prefix = _resolved_mldb_prefix(repository_root, pinned_root)
    for pin in pins:
        yaml_bytes = _verified_committed_bytes(
            repository_root,
            commit=commit,
            path=_pin_yaml_relative(prefix, pin),
        )
        if hashlib.sha256(yaml_bytes).hexdigest() != pin["yaml_sha256"]:
            raise ValueError("pinned canonical YAML sha256 mismatch")
        if pin["kind"] in _EXECUTABLE_LOADERS:
            _verify_executable_pin(
                repository_root=repository_root,
                pinned_root=pinned_root,
                prefix=prefix,
                commit=commit,
                pin=pin,
            )
        elif pin["kind"] == "corpus":
            _verify_corpus_pin(
                repository_root=repository_root,
                pinned_root=pinned_root,
                prefix=prefix,
                commit=commit,
                pin=pin,
            )


def _plan_trial(plan: StudyPlan, trial_id: str) -> dict[str, object]:
    matches = [trial for trial in plan["trials"] if trial["trial"] == trial_id]
    if len(matches) != 1:
        raise ValueError("StageInput trial does not name exactly one Plan trial")
    return cast(dict[str, object], matches[0])


def _verify_stage_against_plan(stage_input: StageInput, plan: StudyPlan, stage_key: StageKey) -> None:
    trial = _plan_trial(plan, stage_key["trial"])
    source = trial["source"]
    if stage_key["kind"] == "training":
        if type(source) is not dict or source.get("kind") != "training":
            raise ValueError("training StageInput does not match Plan trial source")
        expected = {key: value for key, value in source.items() if key != "kind"}
        if stage_input["stage"] != expected:
            raise ValueError("training StageInput stage does not match Plan source")
        return
    evaluations = trial["evaluations"]
    if type(evaluations) is not list:
        raise ValueError("Plan trial evaluations are invalid")
    matches = [
        item
        for item in evaluations
        if type(item) is dict and item.get("coordinate") == stage_key["coordinate"]
    ]
    if len(matches) != 1:
        raise ValueError("Evaluation coordinate does not match Plan trial")
    coordinate = matches[0]
    expected = {
        "name": coordinate["stage"],
        "task": coordinate["task"],
        "corpus": coordinate["corpus"],
        "evaluation_protocol": coordinate["evaluation_protocol"],
        "parameters": coordinate["parameters"],
    }
    if stage_input["stage"] != expected:
        raise ValueError("evaluation StageInput stage does not match Plan coordinate")


def _verify_common_preflight(
    stage_input: StageInput,
    *,
    stage_key: StageKey,
    repository_root: Path,
    pinned_mldb_data_root: Path,
    runtime_mldb_data_root: Path,
) -> None:
    resolver = CanonicalRepositoryResolver(runtime_mldb_data_root)
    raw_plan = resolver.resolve(kind=EntityKind.STUDY_PLAN, entity_id=stage_key["plan"])
    plan = _validate_study_plan(raw_plan)
    if stage_input["plan"] != plan["id"]:
        raise ValueError("StageInput plan does not match StudyPlan id")
    if stage_input["plan_sha256"] != plan["content_sha256"]:
        raise ValueError("StageInput plan_sha256 does not match StudyPlan content_sha256")
    if stage_input["source_commit"] != plan["source_commit"]:
        raise ValueError("StageInput source_commit does not match StudyPlan")
    if stage_input["pins"] != plan["pins"]:
        raise ValueError("StageInput pins do not exactly match StudyPlan pins")
    _verify_stage_against_plan(stage_input, plan, stage_key)
    _verify_core_snapshot(repository_root, stage_key["source_commit"])
    _verify_plan_pins(
        repository_root=repository_root,
        pinned_root=pinned_mldb_data_root,
        commit=stage_key["source_commit"],
        pins=plan["pins"],
    )


def _attempt(
    *, backend: str, execution_id: str, started_at: str | None, status: str, diagnostic: Diagnostic | None
) -> dict[str, object]:
    ended_at = _ended_at()
    if started_at is not None:
        start = datetime.fromisoformat(started_at[:-1] + "+00:00")
        end = datetime.fromisoformat(ended_at[:-1] + "+00:00")
        if end < start:
            raise ValueError("started_at must not be later than ended_at")
    return {
        "backend": backend,
        "execution_id": execution_id,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "diagnostic": diagnostic,
    }


class CommonExecutionHarness:
    """Compose common immutable preflight and one domain execution attempt."""

    def __init__(
        self,
        *,
        repository_root: str | Path,
        pinned_mldb_data_root: str | Path,
        runtime_mldb_data_root: str | Path,
        object_bytes: _ObjectByteAccess,
        corpus_destination_root: str | Path,
        work_dir: str | Path,
        training_weights_uri: str | None = None,
        evaluation_artifact_uris: Mapping[str, str] | None = None,
        backend: str,
        execution_id: str,
        started_at: str | None = None,
        telemetry_sink: Callable[[_AcceptedScalarEvent], None] | None = None,
    ) -> None:
        self._repository_root = _repo_root(repository_root)
        self._pinned_mldb_data_root = Path(pinned_mldb_data_root)
        self._runtime_mldb_data_root = Path(runtime_mldb_data_root)
        self._object_bytes = object_bytes
        self._corpus_destination_root = Path(corpus_destination_root)
        self._work_dir = Path(work_dir)
        self._training_weights_uri = training_weights_uri
        self._evaluation_artifact_uris = (
            evaluation_artifact_uris if evaluation_artifact_uris is not None else {}
        )
        self._backend = _nonempty_string(backend, label="backend")
        self._execution_id = _nonempty_string(execution_id, label="execution_id")
        self._started_at = _validate_started_at(started_at)
        self._telemetry_sink = telemetry_sink

    def __call__(self, stage_input: StageInput) -> TerminalCandidate:
        stage_key = _establish_stage_key(stage_input)
        try:
            _verify_common_preflight(
                stage_input,
                stage_key=stage_key,
                repository_root=self._repository_root,
                pinned_mldb_data_root=self._pinned_mldb_data_root,
                runtime_mldb_data_root=self._runtime_mldb_data_root,
            )
            self._work_dir.mkdir(parents=True, exist_ok=True)
            if not self._work_dir.is_dir():
                raise ValueError("work_dir must be a directory")
            if stage_key["kind"] == "training":
                if type(self._training_weights_uri) is not str or not self._training_weights_uri:
                    raise ValueError("training weights URI is required")
                result = training_runtime._execute_training_stage(
                    cast(TrainingStageInput, stage_input),
                    mldb_data_root=self._pinned_mldb_data_root,
                    object_bytes=self._object_bytes,
                    corpus_destination_root=self._corpus_destination_root,
                    work_dir=self._work_dir,
                    weights_uri=self._training_weights_uri,
                    telemetry_sink=self._telemetry_sink,
                )
                attempt = _attempt(
                    backend=self._backend,
                    execution_id=self._execution_id,
                    started_at=self._started_at,
                    status="completed",
                    diagnostic=None,
                )
                return CompletedTrainingCandidate(
                    state="terminal",
                    stage_key=cast(TrainingStageKey, stage_key),
                    attempts=[cast(object, attempt)],
                    status="completed",
                    diagnostic=None,
                    result=result,
                )
            result = evaluation_runtime._execute_evaluation_stage(
                cast(EvaluationStageInput, stage_input),
                pinned_mldb_data_root=self._pinned_mldb_data_root,
                runtime_mldb_data_root=self._runtime_mldb_data_root,
                object_bytes=self._object_bytes,
                corpus_destination_root=self._corpus_destination_root,
                work_dir=self._work_dir,
                artifact_uris=self._evaluation_artifact_uris,
                telemetry_sink=self._telemetry_sink,
            )
            attempt = _attempt(
                backend=self._backend,
                execution_id=self._execution_id,
                started_at=self._started_at,
                status="completed",
                diagnostic=None,
            )
            return CompletedEvaluationCandidate(
                state="terminal",
                stage_key=cast(EvaluationStageKey, stage_key),
                attempts=[cast(object, attempt)],
                status="completed",
                diagnostic=None,
                result=result,
            )
        except Exception:
            diagnostic = cast(Diagnostic, dict(_validate_diagnostic(dict(_FAILURE_DIAGNOSTIC))))
            attempt = _attempt(
                backend=self._backend,
                execution_id=self._execution_id,
                started_at=self._started_at,
                status="failed",
                diagnostic=diagnostic,
            )
            if stage_key["kind"] == "training":
                return FailedTrainingCandidate(
                    state="terminal",
                    stage_key=cast(TrainingStageKey, stage_key),
                    attempts=[cast(object, attempt)],
                    status="failed",
                    diagnostic=diagnostic,
                    result=None,
                )
            return FailedEvaluationCandidate(
                state="terminal",
                stage_key=cast(EvaluationStageKey, stage_key),
                attempts=[cast(object, attempt)],
                status="failed",
                diagnostic=diagnostic,
                result=None,
            )
