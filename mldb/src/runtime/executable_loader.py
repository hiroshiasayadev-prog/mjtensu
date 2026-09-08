"""Executable loading for resolved MLDB Python assets."""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
from types import ModuleType
from typing import Callable

from ..catalog.architecture import ArchitectureBuild, ArchitectureStatus
from ..common.errors import MldbError
from ..evaluation.interface import EvaluationEntrypoint
from ..evaluation.protocol import EvaluationProtocolStatus
from ..training.protocol import TrainEntrypoint, TrainProtocolStatus
from .catalog_handles import ArchitectureHandle
from .definition_handles import EvaluationProtocolHandle, TrainProtocolHandle


def load_architecture_build(
    architecture: ArchitectureHandle,
) -> ArchitectureBuild:
    _verify_sealed_hash(
        architecture.implementation_path,
        architecture.metadata.status is ArchitectureStatus.SEALED,
        architecture.metadata.implementation.sha256,
        "Architecture",
    )
    module = _load_module(architecture.implementation_path)
    entrypoint = _required_entrypoint(module, architecture.metadata.implementation.entrypoint)
    _require_arity(entrypoint, 0, "Architecture build")
    return entrypoint


def load_train_entrypoint(
    protocol: TrainProtocolHandle,
) -> TrainEntrypoint:
    _verify_sealed_hash(
        protocol.implementation_path,
        protocol.metadata.status is TrainProtocolStatus.SEALED,
        protocol.metadata.implementation.sha256,
        "Train Protocol",
    )
    module = _load_module(protocol.implementation_path)
    entrypoint = _required_entrypoint(module, protocol.metadata.implementation.entrypoint)
    _require_arity(entrypoint, 1, "Train Protocol train")
    return entrypoint


def load_evaluation_entrypoint(
    protocol: EvaluationProtocolHandle,
) -> EvaluationEntrypoint:
    _verify_sealed_hash(
        protocol.implementation_path,
        protocol.metadata.status is EvaluationProtocolStatus.SEALED,
        protocol.metadata.implementation.sha256,
        "Evaluation Protocol",
    )
    module = _load_module(protocol.implementation_path)
    entrypoint = _required_entrypoint(module, protocol.metadata.implementation.entrypoint)
    _require_arity(entrypoint, 1, "Evaluation Protocol evaluate")
    return entrypoint


def _verify_sealed_hash(path: Path, sealed: bool, expected: str | None, kind: str) -> None:
    if not path.is_file():
        raise MldbError(f"{kind} implementation file does not exist: {path}")
    if not sealed:
        return
    if expected is None:
        raise MldbError(f"sealed {kind} has no implementation SHA-256")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual.lower() != expected.lower():
        raise MldbError(f"{kind} implementation SHA-256 mismatch")


def _load_module(path: Path) -> ModuleType:
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    module_name = f"_mldb_asset_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise MldbError(f"could not create Python module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException as error:
        sys.modules.pop(module_name, None)
        raise MldbError(f"failed to import MLDB executable {path}: {error}") from error
    return module


def _required_entrypoint(module: ModuleType, name: str) -> Callable[..., object]:
    value = getattr(module, name, None)
    if not callable(value):
        raise MldbError(f"required callable entrypoint {name!r} is missing")
    return value


def _require_arity(callable_value: Callable[..., object], positional_count: int, label: str) -> None:
    try:
        signature = inspect.signature(callable_value)
    except (TypeError, ValueError) as error:
        raise MldbError(f"could not inspect {label} entrypoint") from error
    parameters = tuple(signature.parameters.values())
    if len(parameters) != positional_count:
        raise MldbError(f"{label} entrypoint has incompatible invocation shape")
    if any(parameter.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
           for parameter in parameters):
        raise MldbError(f"{label} entrypoint has incompatible invocation shape")
