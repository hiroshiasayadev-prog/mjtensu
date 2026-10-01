"""Runtime-registry application service."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .core import RegistryStore, RuntimeSnapshot, SnapshotMetadata


class CandidateValidator(Protocol):
    def validate(self, pyproject_toml: bytes, uv_lock: bytes) -> None: ...


@dataclass(frozen=True)
class SetResult:
    metadata: SnapshotMetadata
    created: bool


class RuntimeRegistryService:
    def __init__(self, store: RegistryStore, validator: CandidateValidator) -> None:
        self._store = store
        self._validator = validator

    def get(self, version: int | None = None) -> RuntimeSnapshot:
        return self._store.get(version)

    def set(self, pyproject_toml: bytes, uv_lock: bytes) -> SetResult:
        if not pyproject_toml or not uv_lock:
            raise ValueError("pyproject_toml and uv_lock must both be non-empty")
        self._validator.validate(pyproject_toml, uv_lock)
        metadata, created = self._store.publish(pyproject_toml, uv_lock)
        return SetResult(metadata=metadata, created=created)
