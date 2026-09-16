"""Private helpers for executable definition parsing and companion loading."""

from __future__ import annotations

import hashlib
import importlib.util
from importlib.machinery import ModuleSpec
import inspect
import re
import sys
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Mapping

from mldb_v2.src.common.ids import EntityKind, _validate_typed_reference
from mldb_v2.src.common.parameters import (
    PublicParameterDeclaration,
    _validate_public_parameter_declaration,
    _validate_public_parameter_value,
)
from mldb_v2.src.repository.resolution import CanonicalRepositoryResolver
from mldb_v2.src.verification.executable_integrity import ExecutableSource


_SHA256_RE = re.compile(r"[0-9a-f]{64}", re.ASCII)
_VERSIONED_LOCAL_ID_RE = re.compile(r".+-v([1-9][0-9]*)", re.ASCII)
_VERSIONED_SCHEMA_RE = re.compile(r".+/v([1-9][0-9]*)", re.ASCII)
_DOMAIN_BY_KIND = {
    EntityKind.ARCHITECTURE: "architectures",
    EntityKind.TRAIN_PROTOCOL: "train_protocols",
    EntityKind.EVALUATION_PROTOCOL: "evaluation_protocols",
}


def _require_exact_keys(
    value: object,
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    label: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be a mapping")
    keys = set(value)
    if keys - required - optional:
        raise ValueError(f"{label} contains unknown field")
    if required - keys:
        raise ValueError(f"{label} is missing required field")
    return value


def _require_string(value: object, *, label: str, nonempty: bool = False) -> str:
    if type(value) is not str or (nonempty and value == ""):
        requirement = "a non-empty string" if nonempty else "a string"
        raise ValueError(f"{label} must be {requirement}")
    return value


def _validate_definition_id(value: object) -> str:
    typed_id = _validate_typed_reference(value)
    _namespace, local_id = typed_id.split("/", 1)
    if _VERSIONED_LOCAL_ID_RE.fullmatch(local_id) is None:
        raise ValueError("definition local id must end in -v<positive-integer>")
    return typed_id


def _validate_task_reference(value: object) -> str:
    return _validate_definition_id(value)


def _validate_status(value: object) -> str:
    if value not in {"draft", "sealed"} or type(value) is not str:
        raise ValueError("status must be exactly draft or sealed")
    return value


def _validate_sha256(value: object, *, label: str = "sha256") -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase 64-hex SHA-256")
    return value


def _validate_source_path(
    value: object,
    *,
    namespace: str | None = None,
    mldb_prefix: str = "mldb_data",
) -> str:
    path = _require_string(value, label="source path", nonempty=True)
    if "\\" in path or any(char in path for char in "*?["):
        raise ValueError("source path must be a repository-relative file path")
    pure = PurePosixPath(path)
    if pure.is_absolute() or path != pure.as_posix() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("source path must be a canonical repository-relative file path")
    if ":" in pure.parts[0]:
        raise ValueError("source path must be a repository-relative file path")
    if namespace is not None:
        prefix = PurePosixPath(mldb_prefix) / namespace / "lib"
        if pure == prefix or not pure.is_relative_to(prefix) or pure.suffix != ".py":
            raise ValueError("source path must be a Python helper under the executable namespace lib directory")
    return path


def _validate_sources(
    value: object,
    *,
    namespace: str | None = None,
    mldb_prefix: str = "mldb_data",
) -> list[ExecutableSource]:
    if type(value) is not list:
        raise ValueError("implementation.sources must be a list")
    validated: list[ExecutableSource] = []
    paths: list[str] = []
    for entry in value:
        mapping = _require_exact_keys(entry, required={"path", "sha256"}, label="implementation.sources entry")
        path = _validate_source_path(mapping["path"], namespace=namespace, mldb_prefix=mldb_prefix)
        sha256 = _validate_sha256(mapping["sha256"], label="source sha256")
        paths.append(path)
        validated.append({"path": path, "sha256": sha256})
    if len(set(paths)) != len(paths):
        raise ValueError("implementation.sources paths must be unique")
    if paths != sorted(paths):
        raise ValueError("implementation.sources must be sorted lexicographically by path")
    return validated


def _validate_implementation(
    value: object,
    *,
    entrypoint: str,
    framework: str | None = None,
    sealed: bool,
    namespace: str | None = None,
) -> dict[str, object]:
    required = {"entrypoint"}
    optional = {"sha256", "sources"}
    if framework is not None:
        required.add("framework")
    mapping = _require_exact_keys(
        value,
        required=required,
        optional=optional,
        label="implementation",
    )
    if mapping["entrypoint"] != entrypoint:
        raise ValueError(f"implementation.entrypoint must be exactly {entrypoint}")
    if framework is not None and mapping["framework"] != framework:
        raise ValueError(f"implementation.framework must be exactly {framework}")
    if sealed and "sha256" not in mapping:
        raise ValueError("sealed executable definition requires implementation.sha256")
    if "sha256" in mapping:
        _validate_sha256(mapping["sha256"], label="implementation.sha256")
    if "sources" in mapping:
        _validate_sources(mapping["sources"], namespace=namespace)
    return mapping


def _validate_json_mapping(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be a mapping")
    _validate_public_parameter_value(value)
    return value


def _validate_parameter_declarations(
    value: object,
) -> dict[str, PublicParameterDeclaration]:
    if type(value) is not dict:
        raise ValueError("parameters must be a mapping")
    result: dict[str, PublicParameterDeclaration] = {}
    for key, declaration in value.items():
        if type(key) is not str:
            raise ValueError("public parameter keys must be strings")
        result[key] = _validate_public_parameter_declaration(declaration)
    return result


def _validate_versioned_schema(value: object) -> str:
    schema = _require_string(value, label="artifact schema", nonempty=True)
    if _VERSIONED_SCHEMA_RE.fullmatch(schema) is None:
        raise ValueError("artifact schema must end in /v<positive-integer>")
    return schema


def _resolve_document(
    mldb_data_root: str | Path,
    *,
    kind: EntityKind,
    entity_id: str,
) -> tuple[dict[str, object], Path]:
    if kind not in _DOMAIN_BY_KIND:
        raise ValueError("unsupported executable definition kind")
    typed_id = _validate_definition_id(entity_id)
    resolver = CanonicalRepositoryResolver(mldb_data_root)
    document = resolver.resolve(kind=kind, entity_id=typed_id)
    namespace, local_id = typed_id.split("/", 1)
    yaml_path = Path(mldb_data_root) / namespace / _DOMAIN_BY_KIND[kind] / f"{local_id}.yaml"
    return dict(document), yaml_path


def _package_module(name: str, path: Path) -> ModuleType:
    module = ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    spec = ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(path)]
    module.__spec__ = spec
    return module


def _load_companion_module(
    mldb_data_root: str | Path,
    *,
    kind: EntityKind,
    entity_id: str,
) -> ModuleType:
    if kind not in _DOMAIN_BY_KIND:
        raise ValueError("unsupported executable definition kind")
    typed_id = _validate_definition_id(entity_id)
    namespace, local_id = typed_id.split("/", 1)
    namespace_root = (Path(mldb_data_root) / namespace).resolve()
    companion = namespace_root / _DOMAIN_BY_KIND[kind] / f"{local_id}.py"
    if not companion.is_file():
        raise FileNotFoundError(f"same-basename executable companion not found: {companion}")
    absolute = companion.resolve()
    digest = hashlib.sha256(str(absolute).encode("utf-8")).hexdigest()[:24]
    package_name = f"_mldb_v2_execpkg_{digest}"
    domain_package = f"{package_name}.{_DOMAIN_BY_KIND[kind]}"
    module_name = f"{domain_package}.entry"
    previous = {name: module for name, module in tuple(sys.modules.items()) if name == package_name or name.startswith(package_name + ".")}
    for name in previous:
        sys.modules.pop(name, None)
    sys.modules[package_name] = _package_module(package_name, namespace_root)
    sys.modules[domain_package] = _package_module(domain_package, absolute.parent)
    spec = importlib.util.spec_from_file_location(module_name, absolute)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot create module spec for companion: {companion}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)
        sys.modules.update(previous)
        raise ValueError(f"failed to load executable companion: {companion}") from error
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
    return module


def _validate_callable_signature(callable_value: object, *, entrypoint: str, arity: int) -> None:
    if not callable(callable_value):
        raise ValueError(f"{entrypoint} must be callable")
    try:
        signature = inspect.signature(callable_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"cannot inspect {entrypoint} signature") from error
    parameters = tuple(signature.parameters.values())
    if len(parameters) != arity:
        raise ValueError(f"{entrypoint} must accept exactly {arity} caller argument(s)")
    for parameter in parameters:
        if parameter.kind not in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            raise ValueError(f"{entrypoint} has incompatible callable signature")


def _load_executable_callable(
    mldb_data_root: str | Path,
    *,
    kind: EntityKind,
    entity_id: str,
    entrypoint: str,
    arity: int,
):
    # Establish the exact canonical definition before deriving/importing its sibling.
    _resolve_document(mldb_data_root, kind=kind, entity_id=entity_id)
    module = _load_companion_module(
        mldb_data_root,
        kind=kind,
        entity_id=entity_id,
    )
    if not hasattr(module, entrypoint):
        raise ValueError(f"executable companion is missing {entrypoint}")
    callable_value = getattr(module, entrypoint)
    _validate_callable_signature(callable_value, entrypoint=entrypoint, arity=arity)
    return callable_value
