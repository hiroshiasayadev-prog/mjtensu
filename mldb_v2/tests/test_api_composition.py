from __future__ import annotations

import inspect
from pathlib import Path
from typing import get_type_hints

import pytest

import mldb_v2.src.api as public_api
import mldb_v2.src.api.composition as composition
from mldb_v2.src.api import (
    ApplicationComposition,
    ApplicationInterface,
    compose_application,
)
from mldb_v2.src.api.application import Application
from mldb_v2.src.backend._clearml_sdk import ClearMLSDKAdapter
from mldb_v2.src.backend._registry import BackendRegistry
from mldb_v2.src.common.ids import EntityKind


def _empty_repository(tmp_path: Path) -> Path:
    (tmp_path / "mldb_data").mkdir()
    return tmp_path


def test_public_composition_constructs_application_without_default(tmp_path: Path) -> None:
    repository_root = _empty_repository(tmp_path)
    namespace: dict[str, object] = {}
    exec("from mldb_v2.src.api import compose_application", namespace)
    public_compose = namespace["compose_application"]

    result = public_compose(repository_root=repository_root, environment={})

    assert isinstance(result, ApplicationComposition)
    assert isinstance(result.application, Application)
    assert result.default_backend is None
    hints = get_type_hints(ApplicationComposition)
    assert hints == {
        "application": ApplicationInterface,
        "default_backend": str | None,
    }


def test_default_backend_is_explicit_exact_and_bounded(tmp_path: Path) -> None:
    repository_root = _empty_repository(tmp_path)
    configured = compose_application(
        repository_root=repository_root,
        environment={"MLDB_V2_DEFAULT_BACKEND": "clearml"},
    )
    assert configured.default_backend == "clearml"
    assert compose_application(
        repository_root=repository_root,
        environment={"MLDB_V2_DEFAULT_BACKEND": ""},
    ).default_backend is None

    for malformed in (" clearml", "clearml ", " ", "\tclearml"):
        with pytest.raises(ValueError, match="MLDB_V2_DEFAULT_BACKEND"):
            compose_application(
                repository_root=repository_root,
                environment={"MLDB_V2_DEFAULT_BACKEND": malformed},
            )

def test_composition_does_not_activate_backend_or_s3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root = _empty_repository(tmp_path)

    def fail_backend(*args, **kwargs):
        raise AssertionError("composition must not resolve a backend")

    def fail_s3(*args, **kwargs):
        raise AssertionError("composition must not create an S3 client")

    monkeypatch.setattr(BackendRegistry, "resolve", fail_backend)
    monkeypatch.setattr(composition, "_create_s3_transport", fail_s3)

    result = compose_application(repository_root=repository_root, environment={})
    assert result.default_backend is None

    listing = result.application.list_entities(resource=EntityKind.TASK)
    assert listing == {"items": (), "issues": ()}


def test_public_surface_does_not_expose_lower_layer_construction(tmp_path: Path) -> None:
    repository_root = _empty_repository(tmp_path)
    result = public_api.compose_application(repository_root=repository_root, environment={})
    signature = str(inspect.signature(public_api.compose_application))

    assert isinstance(result, public_api.ApplicationComposition)
    assert {"ApplicationComposition", "compose_application"} <= set(public_api.__all__)
    for forbidden in (
        "BackendRegistry",
        "BackendConfig",
        "_ObjectByteAccess",
        "ClearML",
        "S3",
        "CanonicalRepositoryResolver",
    ):
        assert forbidden not in signature

def test_existing_runtime_configuration_mapping_matches_consumers() -> None:
    environment = {
        "CLEARML_API_HOST": "https://api.example.invalid",
        "CLEARML_WEB_HOST": "https://web.example.invalid",
        "CLEARML_FILES_HOST": "https://files.example.invalid",
        "CLEARML_API_ACCESS_KEY": "test-access",
        "CLEARML_API_SECRET_KEY": "test-secret",
        "MLDB_V2_CLEARML_QUEUE": "default",
        "MLDB_V2_CLEARML_REPOSITORY": "https://github.com/example/repo.git",
        "MLDB_V2_CLEARML_DOCKER_IMAGE": "python:3.10-slim-bookworm",
        "MLDB_V2_CLEARML_DOCKER_ENV_FILE": "/srv/bugrat/clearml/.env",
        "MLDB_V2_CLEARML_DOCKER_GPU": "all",
        "MLDB_V2_CLEARML_DOCKER_SHM_SIZE": "2g",
        "MLDB_V2_RUNTIME_DATA_ROOT": "runtime-data",
        "MLDB_V2_ARTIFACT_URI_PREFIX": "s3://bucket/prefix/",
        "MLDB_S3_ENDPOINT_URL": "https://s3.example.invalid",
        "MLDB_S3_REGION": "test-region",
        "AWS_ACCESS_KEY_ID": "s3-access",
        "AWS_SECRET_ACCESS_KEY": "s3-secret",
        "AWS_SESSION_TOKEN": "s3-token",
    }
    backend_config = composition._clearml_backend_config(environment)
    adapter = ClearMLSDKAdapter.from_backend_config(backend_config)
    settings = adapter._settings

    assert settings.api_host == environment["CLEARML_API_HOST"]
    assert settings.web_host == environment["CLEARML_WEB_HOST"]
    assert settings.files_host == environment["CLEARML_FILES_HOST"]
    assert settings.access_key == environment["CLEARML_API_ACCESS_KEY"]
    assert settings.secret_key == environment["CLEARML_API_SECRET_KEY"]
    assert backend_config.options["queue"] == "default"
    assert settings.repository == "https://github.com/example/repo.git"
    assert settings.docker_image == "python:3.10-slim-bookworm"
    assert settings.docker_env_file == "/srv/bugrat/clearml/.env"
    assert settings.docker_gpu == "all"
    assert settings.docker_shm_size == "2g"
    assert settings.s3_endpoint_url == environment["MLDB_S3_ENDPOINT_URL"]
    assert settings.s3_region == environment["MLDB_S3_REGION"]
    assert settings.runtime_data_root == "runtime-data"
    assert settings.artifact_uri_prefix == "s3://bucket/prefix"

    s3_config = composition._s3_transport_config(environment)
    assert s3_config.endpoint_url == environment["MLDB_S3_ENDPOINT_URL"]
    assert s3_config.region_name == environment["MLDB_S3_REGION"]
    assert s3_config.access_key_id == environment["AWS_ACCESS_KEY_ID"]
    assert s3_config.secret_access_key == environment["AWS_SECRET_ACCESS_KEY"]
    assert s3_config.session_token == environment["AWS_SESSION_TOKEN"]

    sdk_source = Path(inspect.getfile(ClearMLSDKAdapter)).read_text(encoding="utf-8")
    for existing_name in (
        "MLDB_V2_RUNTIME_DATA_ROOT",
        "MLDB_V2_ARTIFACT_URI_PREFIX",
        "MLDB_S3_BUCKET",
        "MLDB_S3_ENDPOINT_URL",
        "MLDB_S3_REGION",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "MINIO_ROOT_USER",
        "MINIO_ROOT_PASSWORD",
    ):
        assert existing_name in sdk_source


def test_composition_does_not_serialize_operational_config(tmp_path: Path) -> None:
    repository_root = _empty_repository(tmp_path)
    result = compose_application(
        repository_root=repository_root,
        environment={
            "CLEARML_API_SECRET_KEY": "canonical-leak-sentinel-clearml",
            "AWS_SECRET_ACCESS_KEY": "canonical-leak-sentinel-s3",
            "MLDB_S3_ENDPOINT_URL": "https://endpoint.example.invalid",
        },
    )
    assert result.application.list_entities(resource=EntityKind.TASK)["items"] == ()
    canonical_files = [path for path in (repository_root / "mldb_data").rglob("*") if path.is_file()]
    assert canonical_files == []
    assert "canonical-leak-sentinel-clearml" not in repr(result)
    assert "canonical-leak-sentinel-s3" not in repr(result)


def test_composition_runtime_dependency_hygiene() -> None:
    source = Path(composition.__file__).read_text(encoding="utf-8")
    assert "mldb_v2.skeleton" not in source
    assert "mldb_v2.src.cli" not in source
    assert "default" not in {composition._configured_default_backend({})}

    signature = inspect.signature(compose_application)
    assert tuple(signature.parameters) == ("repository_root", "environment")
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
