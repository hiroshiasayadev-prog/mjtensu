"""Private committed-source pin collection for MLDB v2 Study planning."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TypeAlias

from mldb_v2.src.catalog._executable_definition_loading import _validate_source_path
from mldb_v2.src.common.ids import EntityKind, _canonical_json_bytes
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
from mldb_v2.src.study._planning_preflight import (
    _ExistingModelPlanningInput,
    _StudyPlanningInput,
    _TrainingPlanningInput,
)

_PinRecord: TypeAlias = Mapping[str, object]

_KIND_ORDER = (
    "namespace",
    "task",
    "corpus",
    "architecture",
    "train_protocol",
    "evaluation_protocol",
    "study",
    "model",
    "training_result",
)
_DOMAIN_BY_KIND = {
    "task": "tasks",
    "corpus": "corpora",
    "architecture": "architectures",
    "train_protocol": "train_protocols",
    "evaluation_protocol": "evaluation_protocols",
    "study": "studies",
    "model": "models",
    "training_result": "training_results",
}
_EXECUTABLE_KINDS = {"architecture", "train_protocol", "evaluation_protocol"}
_ENTITY_KIND_BY_PIN_KIND = {
    "task": EntityKind.TASK,
    "corpus": EntityKind.CORPUS,
    "architecture": EntityKind.ARCHITECTURE,
    "train_protocol": EntityKind.TRAIN_PROTOCOL,
    "evaluation_protocol": EntityKind.EVALUATION_PROTOCOL,
    "study": EntityKind.STUDY,
    "model": EntityKind.MODEL,
    "training_result": EntityKind.TRAINING_RESULT,
}

class _SourcePinningError(ValueError):
    """Bounded private source-pinning failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class _PinnedExecutableSource:
    path: str
    sha256: str


@dataclass(frozen=True)
class _CommittedSourcePin:
    kind: str
    id: str
    yaml_sha256: str
    companion_sha256: str | None
    sources: tuple[_PinnedExecutableSource, ...]
    manifest_sha256: str | None
    manifest_entries: int | None


@dataclass(frozen=True)
class _CommittedSourcePinCollection:
    source_commit: str
    pins: tuple[_CommittedSourcePin, ...]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolved_mldb_root(
    repository_root: Path, mldb_data_root: str | Path
) -> tuple[Path, str]:
    try:
        root = Path(mldb_data_root).resolve(strict=True)
    except OSError as error:
        raise _SourcePinningError("pin_graph_invalid") from error
    if not root.is_dir():
        raise _SourcePinningError("pin_graph_invalid")
    try:
        relative = root.relative_to(repository_root)
    except ValueError as error:
        raise _SourcePinningError("pin_graph_invalid") from error
    if relative == Path("."):
        raise _SourcePinningError("pin_graph_invalid")
    return root, relative.as_posix()


def _selected_commit(repository_root: Path, value: object) -> str:
    try:
        commit = _validate_commit_id(value)
        _require_commit(repository_root, commit)
    except (ValueError, FileNotFoundError, OSError, RuntimeError) as error:
        raise _SourcePinningError("invalid_commit") from error
    return commit


def _check_mldb_core_source_clean(repository_root: Path, commit: str) -> None:
    tree = _run_git(
        repository_root,
        "ls-tree",
        "-r",
        "--name-only",
        commit,
        "--",
        "mldb_v2/src",
    )
    if tree.returncode != 0:
        raise _SourcePinningError("required_source_invalid")
    try:
        committed_paths = [line.decode("utf-8") for line in tree.stdout.splitlines() if line]
    except UnicodeDecodeError as error:
        raise _SourcePinningError("required_source_invalid") from error
    if not committed_paths:
        raise _SourcePinningError("required_source_missing")

    diff = _run_git(repository_root, "diff", "--quiet", commit, "--", "mldb_v2/src")
    if diff.returncode == 1:
        raise _SourcePinningError("mldb_core_source_dirty")
    if diff.returncode != 0:
        raise _SourcePinningError("required_source_invalid")

    untracked = _run_git(
        repository_root, "ls-files", "--others", "--exclude-standard", "--", "mldb_v2/src"
    )
    if untracked.returncode != 0:
        raise _SourcePinningError("required_source_invalid")
    if untracked.stdout.strip():
        raise _SourcePinningError("mldb_core_source_dirty")


def _required_committed_bytes(
    repository_root: Path,
    *,
    commit: str,
    path: str,
) -> bytes:
    try:
        relative = _validate_safe_relative_path(path, label="required source path")
        committed = _read_committed_source_bytes(
            repository_root, commit=commit, path=relative
        )
    except FileNotFoundError as error:
        raise _SourcePinningError("required_source_missing") from error
    except (ValueError, OSError, RuntimeError) as error:
        raise _SourcePinningError("required_source_invalid") from error
    try:
        matches = _working_tree_source_matches_commit(
            repository_root, commit=commit, path=relative
        )
    except FileNotFoundError as error:
        raise _SourcePinningError("required_source_missing") from error
    except (ValueError, OSError, RuntimeError) as error:
        raise _SourcePinningError("required_source_invalid") from error
    if not matches:
        raise _SourcePinningError("required_source_dirty")
    return committed


def _typed_id(record: _PinRecord) -> str:
    value = record.get("id")
    if type(value) is not str or value.count("/") != 1:
        raise _SourcePinningError("pin_graph_invalid")
    namespace, local_id = value.split("/", 1)
    if not namespace or not local_id:
        raise _SourcePinningError("pin_graph_invalid")
    return value


def _canonical_relative(mldb_prefix: str, kind: str, entity_id: str) -> str:
    if kind == "namespace":
        if not entity_id or "/" in entity_id:
            raise _SourcePinningError("pin_graph_invalid")
        return f"{mldb_prefix}/{entity_id}/namespace.yaml"
    domain = _DOMAIN_BY_KIND.get(kind)
    if domain is None or entity_id.count("/") != 1:
        raise _SourcePinningError("pin_graph_invalid")
    namespace, local_id = entity_id.split("/", 1)
    return f"{mldb_prefix}/{namespace}/{domain}/{local_id}.yaml"


def _companion_relative(mldb_prefix: str, kind: str, entity_id: str) -> str:
    namespace, local_id = entity_id.split("/", 1)
    return f"{mldb_prefix}/{namespace}/{_DOMAIN_BY_KIND[kind]}/{local_id}.py"


def _manifest_relative(mldb_prefix: str, entity_id: str) -> str:
    namespace, local_id = entity_id.split("/", 1)
    return f"{mldb_prefix}/{namespace}/corpora/{local_id}.manifest.jsonl"


def _same_record(left: _PinRecord, right: _PinRecord) -> bool:
    try:
        return _canonical_json_bytes(dict(left)) == _canonical_json_bytes(dict(right))
    except (TypeError, ValueError):
        return False


class _StudySourcePinCollector:
    """Collect deterministic committed pins from one validated planning graph."""

    def __init__(self, *, repository_root: str | Path, mldb_data_root: str | Path) -> None:
        try:
            self._repository_root = _repo_root(repository_root)
        except (OSError, ValueError) as error:
            raise _SourcePinningError("pin_graph_invalid") from error
        self._mldb_data_root, self._mldb_prefix = _resolved_mldb_root(
            self._repository_root, mldb_data_root
        )
        self._resolver = CanonicalRepositoryResolver(self._mldb_data_root)

    def collect(
        self,
        *,
        selected_commit: str,
        planning: _StudyPlanningInput,
    ) -> _CommittedSourcePinCollection:
        if not isinstance(planning, _StudyPlanningInput):
            raise _SourcePinningError("pin_graph_invalid")
        commit = _selected_commit(self._repository_root, selected_commit)
        _check_mldb_core_source_clean(self._repository_root, commit)

        graph = self._pin_graph(planning)
        pins: list[_CommittedSourcePin] = []
        for kind in _KIND_ORDER:
            entity_ids = sorted(
                entity_id for pin_kind, entity_id in graph if pin_kind == kind
            )
            for entity_id in entity_ids:
                pins.append(
                    self._pin_one(
                        kind=kind,
                        entity_id=entity_id,
                        record=graph[(kind, entity_id)],
                        commit=commit,
                    )
                )
        return _CommittedSourcePinCollection(source_commit=commit, pins=tuple(pins))

    def _pin_graph(
        self, planning: _StudyPlanningInput
    ) -> dict[tuple[str, str], _PinRecord]:
        graph: dict[tuple[str, str], _PinRecord] = {}

        def add(kind: str, record: _PinRecord) -> None:
            entity_id = _typed_id(record)
            key = (kind, entity_id)
            previous = graph.get(key)
            if previous is not None and not _same_record(previous, record):
                raise _SourcePinningError("pin_graph_invalid")
            graph[key] = record

        add("study", planning.study)
        add("task", planning.task)
        if isinstance(planning.model, _TrainingPlanningInput):
            add("corpus", planning.model.corpus)
            add("train_protocol", planning.model.protocol)
            for architecture in planning.model.architectures:
                add("architecture", architecture)
        elif isinstance(planning.model, _ExistingModelPlanningInput):
            self._add_existing_model_graph(graph, planning.model)
        else:
            raise _SourcePinningError("pin_graph_invalid")

        for evaluation in planning.evaluations:
            add("corpus", evaluation.corpus)
            add("evaluation_protocol", evaluation.protocol)

        namespaces = sorted({entity_id.split("/", 1)[0] for _, entity_id in graph})
        for namespace in namespaces:
            graph[("namespace", namespace)] = {"id": namespace}
        return graph

    def _add_existing_model_graph(
        self,
        graph: dict[tuple[str, str], _PinRecord],
        model_input: _ExistingModelPlanningInput,
    ) -> None:
        def add(kind: str, record: _PinRecord) -> None:
            entity_id = _typed_id(record)
            key = (kind, entity_id)
            previous = graph.get(key)
            if previous is not None and not _same_record(previous, record):
                raise _SourcePinningError("pin_graph_invalid")
            graph[key] = record

        for entry in model_input.models:
            add("model", entry.model)
            add("training_result", entry.training_result)
            try:
                architecture = self._resolver.resolve(
                    kind=EntityKind.ARCHITECTURE, entity_id=entry.architecture_id
                )
            except (OSError, UnicodeError, ValueError, TypeError, FileNotFoundError) as error:
                raise _SourcePinningError("pin_graph_invalid") from error
            add("architecture", architecture)

    def _pin_one(
        self,
        *,
        kind: str,
        entity_id: str,
        record: _PinRecord,
        commit: str,
    ) -> _CommittedSourcePin:
        yaml_bytes = _required_committed_bytes(
            self._repository_root,
            commit=commit,
            path=_canonical_relative(self._mldb_prefix, kind, entity_id),
        )
        if kind != "namespace":
            self._require_planning_binding(kind=kind, entity_id=entity_id, record=record)
        companion_sha256: str | None = None
        sources: tuple[_PinnedExecutableSource, ...] = ()
        manifest_sha256: str | None = None
        manifest_entries: int | None = None

        if kind in _EXECUTABLE_KINDS:
            companion_sha256, sources = self._pin_executable(
                kind=kind,
                entity_id=entity_id,
                record=record,
                commit=commit,
            )
        elif kind == "corpus":
            companion_sha256, manifest_sha256, manifest_entries = self._pin_corpus(
                entity_id=entity_id,
                record=record,
                commit=commit,
            )

        return _CommittedSourcePin(
            kind=kind,
            id=entity_id,
            yaml_sha256=_sha256(yaml_bytes),
            companion_sha256=companion_sha256,
            sources=sources,
            manifest_sha256=manifest_sha256,
            manifest_entries=manifest_entries,
        )

    def _require_planning_binding(
        self, *, kind: str, entity_id: str, record: _PinRecord
    ) -> None:
        entity_kind = _ENTITY_KIND_BY_PIN_KIND.get(kind)
        if entity_kind is None:
            raise _SourcePinningError("pin_graph_invalid")
        try:
            current = self._resolver.resolve(kind=entity_kind, entity_id=entity_id)
        except (FileNotFoundError, OSError, TypeError, ValueError) as error:
            raise _SourcePinningError("planning_input_stale") from error
        if not _same_record(record, current):
            raise _SourcePinningError("planning_input_stale")

    def _pin_executable(
        self,
        *,
        kind: str,
        entity_id: str,
        record: _PinRecord,
        commit: str,
    ) -> tuple[str, tuple[_PinnedExecutableSource, ...]]:
        implementation = record.get("implementation")
        if type(implementation) is not dict:
            raise _SourcePinningError("pin_graph_invalid")
        recorded_companion = implementation.get("sha256")
        if type(recorded_companion) is not str:
            raise _SourcePinningError("pin_graph_invalid")
        companion = _required_committed_bytes(
            self._repository_root,
            commit=commit,
            path=_companion_relative(self._mldb_prefix, kind, entity_id),
        )
        companion_sha256 = _sha256(companion)
        if companion_sha256 != recorded_companion:
            raise _SourcePinningError("committed_hash_mismatch")

        raw_sources = implementation.get("sources", [])
        if type(raw_sources) is not list:
            raise _SourcePinningError("pin_graph_invalid")
        pinned_sources: list[_PinnedExecutableSource] = []
        for source in raw_sources:
            if type(source) is not dict:
                raise _SourcePinningError("pin_graph_invalid")
            path = source.get("path")
            recorded_sha256 = source.get("sha256")
            if type(path) is not str or type(recorded_sha256) is not str:
                raise _SourcePinningError("pin_graph_invalid")
            try:
                relative = _validate_source_path(
                    path, namespace=entity_id.split("/", 1)[0], mldb_prefix=self._mldb_prefix
                )
            except ValueError as error:
                raise _SourcePinningError("required_source_invalid") from error
            committed = _required_committed_bytes(
                self._repository_root,
                commit=commit,
                path=relative,
            )
            actual_sha256 = _sha256(committed)
            if actual_sha256 != recorded_sha256:
                raise _SourcePinningError("committed_hash_mismatch")
            pinned_sources.append(
                _PinnedExecutableSource(path=relative, sha256=actual_sha256)
            )
        pinned_sources.sort(key=lambda source: source.path)
        if len({source.path for source in pinned_sources}) != len(pinned_sources):
            raise _SourcePinningError("pin_graph_invalid")
        return companion_sha256, tuple(pinned_sources)

    def _pin_corpus(
        self,
        *,
        entity_id: str,
        record: _PinRecord,
        commit: str,
    ) -> tuple[str | None, str, int]:
        manifest = record.get("manifest")
        if type(manifest) is not dict:
            raise _SourcePinningError("pin_graph_invalid")
        expected_sha256 = manifest.get("sha256")
        expected_entries = manifest.get("entries")
        if type(expected_sha256) is not str or type(expected_entries) is not int:
            raise _SourcePinningError("pin_graph_invalid")
        manifest_bytes = _required_committed_bytes(
            self._repository_root,
            commit=commit,
            path=_manifest_relative(self._mldb_prefix, entity_id),
        )
        try:
            entries = _verify_corpus_manifest(
                manifest_bytes,
                expected_sha256=expected_sha256,
                expected_count=expected_entries,
            )
        except ValueError as error:
            raise _SourcePinningError("corpus_manifest_mismatch") from error
        manifest_sha256 = _sha256(manifest_bytes)

        builder = record.get("builder")
        if builder is None:
            return None, manifest_sha256, len(entries)
        if type(builder) is not dict or type(builder.get("sha256")) is not str:
            raise _SourcePinningError("pin_graph_invalid")
        builder_bytes = _required_committed_bytes(
            self._repository_root,
            commit=commit,
            path=_companion_relative(self._mldb_prefix, "corpus", entity_id),
        )
        builder_sha256 = _sha256(builder_bytes)
        if builder_sha256 != builder["sha256"]:
            raise _SourcePinningError("committed_hash_mismatch")
        return builder_sha256, manifest_sha256, len(entries)


def _collect_study_source_pins(
    *,
    repository_root: str | Path,
    mldb_data_root: str | Path,
    selected_commit: str,
    planning: _StudyPlanningInput,
) -> _CommittedSourcePinCollection:
    """Collect committed source identity for one validated Study planning input."""

    return _StudySourcePinCollector(
        repository_root=repository_root,
        mldb_data_root=mldb_data_root,
    ).collect(
        selected_commit=selected_commit,
        planning=planning,
    )
