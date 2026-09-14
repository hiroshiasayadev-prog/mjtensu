"""Operational backend configuration values for MLDB v2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


class BackendConfigurationError(ValueError):
    """Raised when operational backend configuration is malformed."""


def _validate_backend_type(value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise BackendConfigurationError("backend type must be a non-empty trimmed string")
    return value


def _freeze_options(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BackendConfigurationError("backend options must be a mapping")
    copied: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or key.strip() != key:
            raise BackendConfigurationError(
                "backend option keys must be non-empty trimmed strings"
            )
        copied[key] = item
    return MappingProxyType(copied)


@dataclass(frozen=True)
class BackendConfig:
    """Backend activation data that never becomes canonical MLDB state."""

    backend_type: str
    options: Mapping[str, object] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend_type", _validate_backend_type(self.backend_type))
        object.__setattr__(self, "options", _freeze_options(self.options))
