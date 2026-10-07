"""Canonical MLDB namespace -> ClearML logical Project routing."""

from __future__ import annotations

from mldb_v2.src.common.ids import _validate_typed_reference


_PROJECT_NAMESPACE_ALIASES: dict[str, str] = {
    "nanodet": "tile-detector",
}


def _clearml_project_for_namespace(namespace: str) -> str:
    if type(namespace) is not str or not namespace:
        raise ValueError("namespace must be a non-empty string")
    project_local = _PROJECT_NAMESPACE_ALIASES.get(namespace, namespace)
    return f"mldb/{project_local}"


def _clearml_project_for_reference(reference: object) -> str:
    namespace = _validate_typed_reference(reference).split("/", 1)[0]
    return _clearml_project_for_namespace(namespace)
