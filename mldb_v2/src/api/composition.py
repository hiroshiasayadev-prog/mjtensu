"""Public production composition root for the MLDB v2 Application."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from mldb_v2.src.backend._config import BackendConfig
from mldb_v2.src.backend._registry import BackendRegistry
from mldb_v2.src.backend.clearml_backend import register_clearml_backend
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
from mldb_v2.src.storage.s3_transport import (
    S3ObjectByteTransport,
    _S3TransportConfig,
    _create_s3_transport,
)

from .application import Application
from .application_interface import ApplicationInterface

_DEFAULT_BACKEND_ENV = "MLDB_V2_DEFAULT_BACKEND"


@dataclass(frozen=True)
class ApplicationComposition:
    """Public runtime application plus adapter-owned default-backend selection."""

    application: ApplicationInterface
    default_backend: str | None


def _optional_environment_value(
    environment: Mapping[str, str], name: str
) -> str | None:
    value = environment.get(name)
    if value is None or value == "":
        return None
    if type(value) is not str:
        raise TypeError(f"runtime environment value {name} must be a string")
    return value


def _configured_default_backend(environment: Mapping[str, str]) -> str | None:
    value = _optional_environment_value(environment, _DEFAULT_BACKEND_ENV)
    if value is None:
        return None
    if value.strip() != value:
        raise ValueError(f"{_DEFAULT_BACKEND_ENV} must be a non-empty trimmed backend name")
    return value


def _clearml_backend_config(environment: Mapping[str, str]) -> BackendConfig:
    options: dict[str, object] = {}
    for environment_name, option_name in (
        ("CLEARML_API_HOST", "api_host"),
        ("CLEARML_WEB_HOST", "web_host"),
        ("CLEARML_FILES_HOST", "files_host"),
        ("CLEARML_API_ACCESS_KEY", "access_key"),
        ("CLEARML_API_SECRET_KEY", "secret_key"),
    ):
        value = _optional_environment_value(environment, environment_name)
        if value is not None:
            options[option_name] = value

    for environment_name, option_name in (
        ("MLDB_V2_CLEARML_QUEUE", "queue"),
        ("MLDB_V2_CLEARML_REPOSITORY", "repository"),
    ):
        value = _optional_environment_value(environment, environment_name)
        if value is not None:
            options[option_name] = value

    runtime_data_root = _optional_environment_value(
        environment, "MLDB_V2_RUNTIME_DATA_ROOT"
    )
    if runtime_data_root is not None:
        options["runtime_data_root"] = runtime_data_root

    artifact_prefix = _optional_environment_value(
        environment, "MLDB_V2_ARTIFACT_URI_PREFIX"
    )
    if artifact_prefix is None:
        bucket = _optional_environment_value(environment, "MLDB_S3_BUCKET")
        if bucket is not None:
            artifact_prefix = f"s3://{bucket}/mldb-v2"
    if artifact_prefix is not None:
        options["artifact_uri_prefix"] = artifact_prefix.rstrip("/")

    return BackendConfig(backend_type="clearml", options=options)


def _s3_transport_config(environment: Mapping[str, str]) -> _S3TransportConfig:
    return _S3TransportConfig(
        endpoint_url=_optional_environment_value(environment, "MLDB_S3_ENDPOINT_URL"),
        region_name=_optional_environment_value(environment, "MLDB_S3_REGION"),
        access_key_id=(
            _optional_environment_value(environment, "AWS_ACCESS_KEY_ID")
            or _optional_environment_value(environment, "MINIO_ROOT_USER")
        ),
        secret_access_key=(
            _optional_environment_value(environment, "AWS_SECRET_ACCESS_KEY")
            or _optional_environment_value(environment, "MINIO_ROOT_PASSWORD")
        ),
        session_token=_optional_environment_value(environment, "AWS_SESSION_TOKEN"),
    )


class _LazyS3ObjectByteTransport:
    """Delay boto3/client activation until object bytes are actually required."""

    def __init__(self, config: _S3TransportConfig) -> None:
        self._config = config
        self._resolved: S3ObjectByteTransport | None = None

    def _transport(self) -> S3ObjectByteTransport:
        if self._resolved is None:
            self._resolved = _create_s3_transport(self._config)
        return self._resolved

    def read_bytes(self, uri: str) -> bytes:
        return self._transport().read_bytes(uri)

    def publish_bytes_immutable(self, uri: str, data: bytes) -> None:
        self._transport().publish_bytes_immutable(uri, data)


def compose_application(
    *,
    repository_root: str | Path,
    environment: Mapping[str, str] | None = None,
) -> ApplicationComposition:
    """Compose the production Application without exposing lower-layer construction."""

    runtime_environment = os.environ if environment is None else environment
    registry = BackendRegistry()
    register_clearml_backend(registry)
    application = Application(
        repository_root=repository_root,
        backend_registry=registry,
        object_bytes=_ObjectByteAccess(
            _LazyS3ObjectByteTransport(_s3_transport_config(runtime_environment))
        ),
        backend_configs={"clearml": _clearml_backend_config(runtime_environment)},
        mldb_tests_root=Path(repository_root) / "mldb_tests",
    )
    return ApplicationComposition(
        application=application,
        default_backend=_configured_default_backend(runtime_environment),
    )
