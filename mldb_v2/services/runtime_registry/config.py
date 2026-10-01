"""Environment configuration for the standalone runtime-registry service."""
from __future__ import annotations

from dataclasses import dataclass
import os

from .core import RegistryStore
from .object_store import S3ObjectStore
from .service import RuntimeRegistryService
from .validator import UvSyncValidator


@dataclass(frozen=True)
class RegistryConfig:
    db_path: str
    bucket: str
    prefix: str
    endpoint_url: str | None
    region: str | None
    access_key: str | None
    secret_key: str | None
    session_token: str | None
    uv_binary: str
    uv_cache_dir: str | None
    uv_timeout_seconds: int

    @classmethod
    def from_env(cls) -> "RegistryConfig":
        env = os.environ
        bucket = env.get("MLDB_RUNTIME_REGISTRY_S3_BUCKET") or env.get("MLDB_S3_BUCKET")
        if not bucket:
            raise RuntimeError("MLDB_RUNTIME_REGISTRY_S3_BUCKET or MLDB_S3_BUCKET is required")
        return cls(
            db_path=env.get("MLDB_RUNTIME_REGISTRY_DB_PATH", "/data/registry.sqlite3"),
            bucket=bucket,
            prefix=env.get("MLDB_RUNTIME_REGISTRY_S3_PREFIX", "mldb-runtime-registry"),
            endpoint_url=env.get("MLDB_RUNTIME_REGISTRY_S3_ENDPOINT_URL") or env.get("MLDB_S3_ENDPOINT_URL"),
            region=env.get("MLDB_RUNTIME_REGISTRY_S3_REGION") or env.get("MLDB_S3_REGION"),
            access_key=env.get("AWS_ACCESS_KEY_ID") or env.get("MINIO_ROOT_USER"),
            secret_key=env.get("AWS_SECRET_ACCESS_KEY") or env.get("MINIO_ROOT_PASSWORD"),
            session_token=env.get("AWS_SESSION_TOKEN"),
            uv_binary=env.get("MLDB_RUNTIME_REGISTRY_UV_BINARY", "uv"),
            uv_cache_dir=env.get("MLDB_RUNTIME_REGISTRY_UV_CACHE_DIR", "/data/uv-cache"),
            uv_timeout_seconds=int(env.get("MLDB_RUNTIME_REGISTRY_UV_TIMEOUT_SECONDS", "1800")),
        )


def compose_service(config: RegistryConfig | None = None) -> RuntimeRegistryService:
    cfg = config or RegistryConfig.from_env()
    try:
        import boto3
    except ModuleNotFoundError as exc:
        raise RuntimeError("boto3 is required by the runtime-registry service") from exc
    client = boto3.client(
        "s3",
        endpoint_url=cfg.endpoint_url,
        region_name=cfg.region,
        aws_access_key_id=cfg.access_key,
        aws_secret_access_key=cfg.secret_key,
        aws_session_token=cfg.session_token,
    )
    store = RegistryStore(cfg.db_path, S3ObjectStore(client), cfg.bucket, cfg.prefix)
    validator = UvSyncValidator(cfg.uv_binary, cfg.uv_timeout_seconds, cfg.uv_cache_dir)
    return RuntimeRegistryService(store, validator)
