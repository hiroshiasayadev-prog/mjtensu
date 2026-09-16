"""Deterministic namespace-first canonical repository inventory and Study Result listing."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Collection, Literal, Sequence, TypeAlias, TypedDict

from mldb_v2.src.common.diagnostic import Diagnostic
from mldb_v2.src.common.ids import (
    EntityKind,
    NamespaceId,
    StudyId,
    _validate_namespace_id,
    _validate_typed_reference,
)

from .resolution import (
    CanonicalDocument,
    CanonicalRepositoryResolver,
    _CANONICAL_KIND_ORDER,
    _DOMAIN_BY_KIND,
    _RepositoryDocumentError,
    _read_namespace_document,
)

StudyResultStatus: TypeAlias = Literal[
    "submitted",
    "cancelling",
    "completed",
    "completed_with_failures",
    "failed",
    "cancelled",
]


class CanonicalListing(TypedDict):
    items: Sequence[CanonicalDocument]
    issues: Sequence[Diagnostic]


_STUDY_RESULT_STATUSES = frozenset(
    {"submitted", "cancelling", "completed", "completed_with_failures", "failed", "cancelled"}
)
_EXECUTABLE_KINDS = frozenset(
    {EntityKind.ARCHITECTURE, EntityKind.TRAIN_PROTOCOL, EntityKind.EVALUATION_PROTOCOL}
)
_PYTHON_PERMITTED_KINDS = _EXECUTABLE_KINDS | {EntityKind.CORPUS}
_SOURCE_DOMAIN = "lib"
_RECOGNIZED_DOMAINS = frozenset(_DOMAIN_BY_KIND.values()) | {_SOURCE_DOMAIN}


def _issue(code: str, message: str) -> Diagnostic:
    return {"code": code, "message": message}


def _display(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _valid_local_id(value: str) -> bool:
    try:
        _validate_namespace_id(value)
    except ValueError:
        return False
    return True


def _namespace_source_layout_issues(namespace_path: Path, root: Path) -> list[Diagnostic]:
    source_root = namespace_path / _SOURCE_DOMAIN
    if not source_root.is_dir():
        return []
    issues: list[Diagnostic] = []
    for entry in sorted(source_root.rglob("*"), key=lambda path: path.as_posix()):
        if entry.is_dir():
            continue
        if entry.suffix != ".py":
            issues.append(_issue("repository_unexpected_source_file", f"{_display(entry, root)}: namespace lib may contain Python source only"))
    return issues


def _parse_created_at(value: object) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise ValueError("created_at must be RFC3339 UTC using Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("created_at must be RFC3339 UTC using Z") from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("created_at must be UTC")
    return parsed


class CanonicalRepositoryListing:
    """Broad deterministic inventory beneath one canonical ``mldb_data`` root."""

    def __init__(self, mldb_data_root: str | Path) -> None:
        self._root = Path(mldb_data_root)
        self._resolver = CanonicalRepositoryResolver(self._root)

    def list_entities(
        self,
        *,
        kind: EntityKind | None = None,
        namespace: NamespaceId | None = None,
    ) -> CanonicalListing:
        if kind is not None and not isinstance(kind, EntityKind):
            raise ValueError(f"unsupported entity kind: {kind!r}")
        namespace_filter = None
        if namespace is not None:
            namespace_filter = str(_validate_namespace_id(namespace))

        namespaces, issues = self._discover_namespaces(namespace_filter)
        items: list[CanonicalDocument] = []
        for namespace_id, namespace_document in namespaces:
            issues.extend(self._namespace_layout_issues(namespace_id))
            if kind is None or kind is EntityKind.NAMESPACE:
                items.append(namespace_document)
            if kind is EntityKind.NAMESPACE:
                continue
            kinds = _CANONICAL_KIND_ORDER if kind is None else (kind,)
            for current_kind in kinds:
                documents, domain_issues = self._list_kind(namespace_id, current_kind)
                items.extend(documents)
                issues.extend(domain_issues)

        return {"items": tuple(items), "issues": tuple(issues)}

    def list_study_results(
        self,
        *,
        namespace: NamespaceId | None = None,
        study: StudyId | None = None,
        statuses: Collection[StudyResultStatus] | None = None,
        created_at_from: datetime | None = None,
        created_at_to: datetime | None = None,
        limit: int | None = None,
    ) -> CanonicalListing:
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("limit must be a positive integer")
        study_filter = None
        if study is not None:
            study_filter = _validate_typed_reference(study)
        status_filter: frozenset[str] | None = None
        if statuses is not None:
            raw_statuses = tuple(statuses)
            if any(type(value) is not str or value not in _STUDY_RESULT_STATUSES for value in raw_statuses):
                raise ValueError("statuses contains an invalid Study Result status")
            status_filter = frozenset(raw_statuses)

        listing = self.list_entities(kind=EntityKind.STUDY_RESULT, namespace=namespace)
        items: list[CanonicalDocument] = []
        issues = list(listing["issues"])
        for document in listing["items"]:
            if study_filter is not None:
                recorded_study = document.get("study")
                if type(recorded_study) is not str:
                    issues.append(_issue("repository_study_result_study_invalid", f"{document.get('id')!r}: invalid study field"))
                    continue
                if recorded_study != study_filter:
                    continue
            if status_filter is not None:
                recorded_status = document.get("status")
                if type(recorded_status) is not str or recorded_status not in _STUDY_RESULT_STATUSES:
                    issues.append(_issue("repository_study_result_status_invalid", f"{document.get('id')!r}: invalid status field"))
                    continue
                if recorded_status not in status_filter:
                    continue
            if created_at_from is not None or created_at_to is not None:
                try:
                    created_at = _parse_created_at(document.get("created_at"))
                except ValueError:
                    issues.append(_issue("repository_study_result_created_at_invalid", f"{document.get('id')!r}: invalid created_at field"))
                    continue
                if created_at_from is not None and created_at < created_at_from:
                    continue
                if created_at_to is not None and created_at > created_at_to:
                    continue
            items.append(document)

        if limit is not None:
            items = items[:limit]
        return {"items": tuple(items), "issues": tuple(issues)}

    def _discover_namespaces(
        self, namespace_filter: str | None
    ) -> tuple[list[tuple[str, CanonicalDocument]], list[Diagnostic]]:
        if not self._root.exists():
            raise FileNotFoundError(f"repository root not found: {self._root}")
        if not self._root.is_dir():
            raise NotADirectoryError(str(self._root))

        if namespace_filter is not None:
            candidate = self._root / namespace_filter
            candidates = [candidate] if candidate.is_dir() and (candidate / "namespace.yaml").is_file() else []
        else:
            candidates = [
                child
                for child in self._root.iterdir()
                if child.is_dir() and (child / "namespace.yaml").is_file()
            ]

        namespaces: list[tuple[str, CanonicalDocument]] = []
        issues: list[Diagnostic] = []
        for candidate in sorted(candidates, key=lambda path: path.name):
            if not _valid_local_id(candidate.name):
                issues.append(_issue("repository_namespace_path_invalid", f"{candidate.name}: invalid Namespace directory name"))
                continue
            try:
                document = _read_namespace_document(self._root, candidate.name)
            except _RepositoryDocumentError as error:
                issues.append(_issue(error.code, error.message))
                continue
            except FileNotFoundError:
                issues.append(_issue("repository_namespace_missing", f"{candidate.name}/namespace.yaml: Namespace metadata disappeared during listing"))
                continue
            namespaces.append((candidate.name, document))

        namespaces.sort(key=lambda item: str(item[1]["id"]))
        return namespaces, issues

    def _namespace_layout_issues(self, namespace_id: str) -> list[Diagnostic]:
        namespace_path = self._root / namespace_id
        issues: list[Diagnostic] = []
        for entry in sorted(namespace_path.iterdir(), key=lambda path: path.name):
            if entry.name == "namespace.yaml":
                continue
            if entry.name in _RECOGNIZED_DOMAINS:
                if not entry.is_dir():
                    issues.append(_issue("repository_domain_not_directory", f"{_display(entry, self._root)}: canonical domain must be a directory"))
                continue
            if entry.is_dir():
                issues.append(_issue("repository_unknown_domain", f"{_display(entry, self._root)}: unknown direct Namespace domain"))
                continue
            issues.append(_issue("repository_unexpected_namespace_entry", f"{_display(entry, self._root)}: unexpected file in canonical Namespace root"))
        issues.extend(_namespace_source_layout_issues(namespace_path, self._root))
        return issues

    def _list_kind(
        self, namespace_id: str, kind: EntityKind
    ) -> tuple[list[CanonicalDocument], list[Diagnostic]]:
        domain = self._root / namespace_id / _DOMAIN_BY_KIND[kind]
        if not domain.exists() or not domain.is_dir():
            return [], []
        entries = sorted(domain.iterdir(), key=lambda path: path.name)
        issues: list[Diagnostic] = []
        documents: list[CanonicalDocument] = []
        valid_documents: dict[str, CanonicalDocument] = {}
        yaml_files: dict[str, Path] = {}

        for entry in entries:
            if entry.is_dir():
                issues.append(_issue("repository_disallowed_subdirectory", f"{_display(entry, self._root)}: nested domain directories are not canonical inventory"))
                continue
            if not entry.name.endswith(".yaml"):
                continue
            local_id = entry.name[:-5]
            yaml_files[local_id] = entry
            if not _valid_local_id(local_id):
                issues.append(_issue("repository_invalid_yaml_basename", f"{_display(entry, self._root)}: invalid canonical YAML basename"))
                continue
            typed_id = f"{namespace_id}/{local_id}"
            try:
                document = self._resolver.resolve(kind=kind, entity_id=typed_id)
            except _RepositoryDocumentError as error:
                issues.append(_issue(error.code, error.message))
                continue
            except FileNotFoundError:
                issues.append(_issue("repository_candidate_missing", f"{_display(entry, self._root)}: canonical candidate disappeared during listing"))
                continue
            documents.append(document)
            valid_documents[local_id] = document

        for entry in entries:
            if entry.is_dir() or not entry.name.endswith(".py"):
                continue
            local_id = entry.name[:-3]
            matching_yaml = domain / f"{local_id}.yaml"
            if kind not in _PYTHON_PERMITTED_KINDS:
                issues.append(_issue("repository_disallowed_python", f"{_display(entry, self._root)}: Python companion is not permitted for {kind.value}"))
                continue
            if not _valid_local_id(local_id):
                issues.append(_issue("repository_disallowed_python", f"{_display(entry, self._root)}: invalid/shared Python module is not a canonical companion"))
                continue
            if not matching_yaml.is_file():
                issues.append(_issue("repository_orphan_executable", f"{_display(entry, self._root)}: executable companion has no same-basename YAML"))
                continue
            if kind is EntityKind.CORPUS and local_id in valid_documents:
                if "builder" not in valid_documents[local_id]:
                    issues.append(_issue("repository_disallowed_python", f"{_display(entry, self._root)}: Corpus Python companion requires a builder declaration"))

        if kind in _EXECUTABLE_KINDS:
            for local_id in sorted(valid_documents):
                implementation = domain / f"{local_id}.py"
                if not implementation.is_file():
                    issues.append(_issue("repository_missing_executable", f"{_display(implementation, self._root)}: required same-basename executable companion is missing"))

        for entry in entries:
            if entry.is_dir() or not entry.name.endswith(".manifest.jsonl"):
                continue
            local_id = entry.name[: -len(".manifest.jsonl")]
            if kind is not EntityKind.CORPUS:
                issues.append(_issue("repository_disallowed_manifest", f"{_display(entry, self._root)}: manifest companion is only permitted for Corpus"))
                continue
            if not _valid_local_id(local_id):
                issues.append(_issue("repository_manifest_placement_invalid", f"{_display(entry, self._root)}: invalid manifest companion basename"))
                continue
            if not (domain / f"{local_id}.yaml").is_file():
                issues.append(_issue("repository_manifest_orphan", f"{_display(entry, self._root)}: manifest companion has no same-basename Corpus YAML"))

        for entry in entries:
            if entry.is_dir():
                continue
            if entry.name.endswith(".yaml") or entry.name.endswith(".py") or entry.name.endswith(".manifest.jsonl"):
                continue
            issues.append(_issue("repository_unexpected_domain_file", f"{_display(entry, self._root)}: unexpected file in canonical domain"))

        if kind is EntityKind.CORPUS:
            for local_id, document in sorted(valid_documents.items()):
                expected = f"{local_id}.manifest.jsonl"
                manifest = document.get("manifest")
                if type(manifest) is not dict or manifest.get("file") != expected:
                    issues.append(_issue("repository_manifest_placement_invalid", f"{namespace_id}/corpora/{local_id}.yaml: manifest.file must name the same-basename manifest"))
                    continue
                manifest_path = domain / expected
                if not manifest_path.is_file():
                    issues.append(_issue("repository_manifest_missing", f"{_display(manifest_path, self._root)}: declared Corpus manifest is missing"))

        return documents, issues
