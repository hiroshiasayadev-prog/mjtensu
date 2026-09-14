"""Generic execution-backend registration and configured resolution."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias, cast

from mldb_v2.src.backend._config import BackendConfig, _validate_backend_type
from mldb_v2.src.backend.backend_port import BackendPort

BackendFactory: TypeAlias = Callable[[BackendConfig], BackendPort]


class BackendRegistrationError(ValueError):
    """Raised when a backend registration is invalid or duplicated."""


class UnknownBackendError(LookupError):
    """Raised when no backend factory is registered for a configured type."""


def _require_backend_port(value: object) -> BackendPort:
    missing = [
        name
        for name in ("admit", "observe", "collect", "cancel_study")
        if not callable(getattr(value, name, None))
    ]
    if missing:
        raise TypeError(
            "backend factory returned an object missing BackendPort methods: "
            + ", ".join(missing)
        )
    return cast(BackendPort, value)


class BackendRegistry:
    """Resolve operational backend configuration to one BackendPort instance."""

    def __init__(self) -> None:
        self._factories: dict[str, BackendFactory] = {}

    def register(self, backend_type: str, factory: BackendFactory) -> None:
        try:
            validated_type = _validate_backend_type(backend_type)
        except ValueError as exc:
            raise BackendRegistrationError(str(exc)) from exc
        if not callable(factory):
            raise BackendRegistrationError("backend factory must be callable")
        if validated_type in self._factories:
            raise BackendRegistrationError(
                f"backend type is already registered: {validated_type}"
            )
        self._factories[validated_type] = factory

    def resolve(self, config: BackendConfig) -> BackendPort:
        if not isinstance(config, BackendConfig):
            raise TypeError("config must be a BackendConfig")
        try:
            factory = self._factories[config.backend_type]
        except KeyError as exc:
            raise UnknownBackendError(
                f"unknown backend type: {config.backend_type}"
            ) from exc
        return _require_backend_port(factory(config))
