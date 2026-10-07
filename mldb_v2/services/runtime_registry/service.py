"""Runtime-registry application service."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .core import (
    RegistryStore,
    RuntimeImageMetadata,
    RuntimeSnapshot,
    SnapshotMetadata,
)


class CandidateValidator(Protocol):
    def validate(self, pyproject_toml: bytes, uv_lock: bytes) -> None: ...


@dataclass(frozen=True)
class RuntimeImagePolicy:
    profile: str
    recipe_version: str
    repository: str
    base_image: str

    def tag_for(self, version: int) -> str:
        return f"rr-v{version}"


@dataclass(frozen=True)
class SetResult:
    metadata: SnapshotMetadata
    created: bool
    images: tuple[RuntimeImageMetadata, ...] = ()


class RuntimeRegistryService:
    def __init__(
        self,
        store: RegistryStore,
        validator: CandidateValidator,
        image_policies: tuple[RuntimeImagePolicy, ...] = (),
    ) -> None:
        self._store = store
        self._validator = validator
        self._image_policies = {policy.profile: policy for policy in image_policies}

    def get(self, version: int | None = None) -> RuntimeSnapshot:
        return self._store.get(version)

    def ensure_image(self, version: int, profile: str) -> RuntimeImageMetadata:
        existing = self._store.get_image(version, profile)
        if existing is not None and existing.state == "READY":
            return self._store.touch_image(version, profile)
        if existing is not None and existing.state != "MISSING":
            return existing
        policy = self._image_policies.get(profile)
        if policy is None:
            raise ValueError(f"runtime image profile {profile!r} is not configured")
        return self._store.ensure_image(
            version,
            profile=profile,
            recipe_version=policy.recipe_version,
            repository=policy.repository,
            tag=policy.tag_for(version),
            base_image=policy.base_image,
        )

    def set(self, pyproject_toml: bytes, uv_lock: bytes) -> SetResult:
        if not pyproject_toml or not uv_lock:
            raise ValueError("pyproject_toml and uv_lock must both be non-empty")
        self._validator.validate(pyproject_toml, uv_lock)
        metadata, created = self._store.publish(pyproject_toml, uv_lock)
        images = tuple(
            self.ensure_image(metadata.version, profile)
            for profile in sorted(self._image_policies)
        )
        return SetResult(metadata=metadata, created=created, images=images)
