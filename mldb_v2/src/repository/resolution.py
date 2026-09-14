"""Exact namespace-first canonical repository resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, TypeAlias

from mldb_v2.src.common.ids import (
    EntityKind,
    NamespaceId,
    TypedEntityId,
    _validate_namespace_id,
    _validate_typed_reference,
)

from ._yaml import _YamlError, _load_yaml

CanonicalEntityId: TypeAlias = NamespaceId | TypedEntityId
CanonicalDocument: TypeAlias = Mapping[str, object]

_DOMAIN_BY_KIND: dict[EntityKind, str] = {
    EntityKind.TASK: "tasks",
    EntityKind.CORPUS: "corpora",
    EntityKind.ARCHITECTURE: "architectures",
    EntityKind.TRAIN_PROTOCOL: "train_protocols",
    EntityKind.EVALUATION_PROTOCOL: "evaluation_protocols",
    EntityKind.STUDY: "studies",
    EntityKind.STUDY_PLAN: "study_plans",
    EntityKind.TRAINING_RESULT: "training_results",
    EntityKind.MODEL: "models",
    EntityKind.EVALUATION_RESULT: "evaluation_results",
    EntityKind.STUDY_RESULT: "study_results",
}

_SCHEMA_BY_KIND: dict[EntityKind, str] = {
    EntityKind.NAMESPACE: "mjtensu.mldb-v2/namespace/v1",
    EntityKind.TASK: "mjtensu.mldb-v2/task/v1",
    EntityKind.CORPUS: "mjtensu.mldb-v2/corpus/v1",
    EntityKind.ARCHITECTURE: "mjtensu.mldb-v2/architecture/v1",
    EntityKind.TRAIN_PROTOCOL: "mjtensu.mldb-v2/train-protocol/v1",
    EntityKind.EVALUATION_PROTOCOL: "mjtensu.mldb-v2/evaluation-protocol/v1",
    EntityKind.STUDY: "mjtensu.mldb-v2/study/v1",
    EntityKind.STUDY_PLAN: "mjtensu.mldb-v2/study-plan/v1",
    EntityKind.TRAINING_RESULT: "mjtensu.mldb-v2/training-result/v1",
    EntityKind.MODEL: "mjtensu.mldb-v2/model/v1",
    EntityKind.EVALUATION_RESULT: "mjtensu.mldb-v2/evaluation-result/v1",
    EntityKind.STUDY_RESULT: "mjtensu.mldb-v2/study-result/v1",
}

_CANONICAL_KIND_ORDER: tuple[EntityKind, ...] = tuple(_DOMAIN_BY_KIND)


class _RepositoryDocumentError(ValueError):
    """Private structural-document failure with a diagnostic-compatible code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def _relative_display(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _load_document(path: Path, root: Path) -> dict[str, object]:
    display = _relative_display(path, root)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise _RepositoryDocumentError(
            "repository_yaml_malformed", f"{display}: cannot read canonical YAML as UTF-8"
        ) from error
    try:
        value = _load_yaml(text)
    except _YamlError as error:
        raise _RepositoryDocumentError(
            "repository_yaml_malformed", f"{display}: malformed canonical YAML ({error})"
        ) from error
    if type(value) is not dict:
        raise _RepositoryDocumentError(
            "repository_yaml_malformed", f"{display}: canonical YAML root must be a mapping"
        )
    return value


def _require_schema(
    document: Mapping[str, object], *, kind: EntityKind, path: Path, root: Path
) -> None:
    expected = _SCHEMA_BY_KIND[kind]
    if document.get("schema") == expected:
        return
    code = (
        "repository_namespace_schema_mismatch"
        if kind is EntityKind.NAMESPACE
        else "repository_schema_kind_mismatch"
    )
    raise _RepositoryDocumentError(
        code,
        f"{_relative_display(path, root)}: expected schema {expected!r}",
    )


def _read_namespace_document(root: Path, namespace: str) -> dict[str, object]:
    path = root / namespace / "namespace.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"namespace not found: {namespace}")
    document = _load_document(path, root)
    _require_schema(document, kind=EntityKind.NAMESPACE, path=path, root=root)
    recorded = document.get("id")
    try:
        recorded_id = _validate_namespace_id(recorded)
    except ValueError as error:
        raise _RepositoryDocumentError(
            "repository_namespace_id_invalid",
            f"{_relative_display(path, root)}: invalid Namespace id",
        ) from error
    if recorded_id != namespace:
        raise _RepositoryDocumentError(
            "repository_namespace_id_mismatch",
            f"{_relative_display(path, root)}: Namespace id {recorded_id!r} does not match directory {namespace!r}",
        )
    return document


class CanonicalRepositoryResolver:
    """Resolve exactly one typed canonical document beneath an ``mldb_data`` root."""

    def __init__(self, mldb_data_root: str | Path) -> None:
        self._root = Path(mldb_data_root)

    def resolve(
        self,
        *,
        kind: EntityKind,
        entity_id: CanonicalEntityId,
    ) -> CanonicalDocument:
        if not isinstance(kind, EntityKind):
            raise ValueError(f"unsupported entity kind: {kind!r}")
        if kind is EntityKind.NAMESPACE:
            namespace = str(_validate_namespace_id(entity_id))
            return _read_namespace_document(self._root, namespace)

        if kind not in _DOMAIN_BY_KIND:
            raise ValueError(f"unsupported entity kind: {kind!r}")
        typed_id = _validate_typed_reference(entity_id)
        namespace, local_id = typed_id.split("/", 1)
        _read_namespace_document(self._root, namespace)

        path = self._root / namespace / _DOMAIN_BY_KIND[kind] / f"{local_id}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"entity not found: {kind.value} {typed_id}")
        document = _load_document(path, self._root)
        _require_schema(document, kind=kind, path=path, root=self._root)
        if document.get("id") != typed_id:
            raise _RepositoryDocumentError(
                "repository_document_id_mismatch",
                f"{_relative_display(path, self._root)}: document id does not match canonical path identity {typed_id!r}",
            )
        return document
