"""Asynchronous immutable GPU runtime image materializer."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .config import RegistryConfig, compose_store
from .core import RegistryStore, RuntimeImageMetadata


_ACCEPT_MANIFEST = (
    "application/vnd.oci.image.manifest.v1+json,"
    "application/vnd.docker.distribution.manifest.v2+json"
)


class RuntimeImageBuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class BuilderConfig:
    poll_seconds: float = 2.0
    retention: int = 3
    registry_scheme: str = "https"
    gc_container: str | None = None

    @classmethod
    def from_env(cls) -> "BuilderConfig":
        env = os.environ
        retention = int(env.get("MLDB_RUNTIME_IMAGE_RETENTION", "3"))
        if retention < 1:
            raise RuntimeError("MLDB_RUNTIME_IMAGE_RETENTION must be positive")
        return cls(
            poll_seconds=float(env.get("MLDB_RUNTIME_IMAGE_POLL_SECONDS", "2")),
            retention=retention,
            registry_scheme=env.get("MLDB_RUNTIME_IMAGE_REGISTRY_SCHEME", "https"),
            gc_container=env.get("MLDB_RUNTIME_IMAGE_GC_CONTAINER"),
        )


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        # The builder image carries the standalone Docker CLI but not the
        # buildx plugin. Use the daemon's legacy builder for this small recipe;
        # the immutable inputs/digest, not the builder frontend, define identity.
        env={**os.environ, "DOCKER_BUILDKIT": "0"},
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "command failed").strip()
        if len(detail) > 12000:
            detail = detail[-12000:]
        raise RuntimeImageBuildError(
            f"{' '.join(command[:3])} failed with exit {completed.returncode}: {detail}"
        )
    return completed.stdout


def _registry_parts(repository: str) -> tuple[str, str]:
    host, separator, name = repository.partition("/")
    if not separator or not host or not name:
        raise RuntimeImageBuildError(
            "runtime image repository must include registry host and repository path"
        )
    return host, name


def _manifest_digest(repository: str, tag: str, scheme: str) -> str:
    host, name = _registry_parts(repository)
    url = f"{scheme}://{host}/v2/{quote(name, safe='/')}/manifests/{quote(tag, safe='')}"
    request = Request(url, method="HEAD", headers={"Accept": _ACCEPT_MANIFEST})
    with urlopen(request, timeout=30) as response:
        digest = response.headers.get("Docker-Content-Digest")
    if digest is None or not digest.startswith("sha256:"):
        raise RuntimeImageBuildError("registry did not return Docker-Content-Digest")
    return digest


def _delete_manifest(image: RuntimeImageMetadata, scheme: str) -> None:
    if image.digest is None:
        return
    host, name = _registry_parts(image.repository)
    url = (
        f"{scheme}://{host}/v2/{quote(name, safe='/')}/manifests/"
        f"{quote(image.digest, safe=':')}"
    )
    request = Request(url, method="DELETE", headers={"Accept": _ACCEPT_MANIFEST})
    try:
        with urlopen(request, timeout=30) as response:
            if response.status not in (202, 200):
                raise RuntimeImageBuildError(
                    f"registry manifest delete returned HTTP {response.status}"
                )
    except HTTPError as exc:
        if exc.code != 404:
            raise


def _recipe_path(recipe_version: str) -> Path:
    if (
        not recipe_version
        or "/" in recipe_version
        or "\\" in recipe_version
        or recipe_version.startswith(".")
    ):
        raise RuntimeImageBuildError("invalid runtime image recipe version")
    path = Path(__file__).with_name("image_recipes") / f"{recipe_version}.Dockerfile"
    if not path.is_file():
        raise RuntimeImageBuildError(f"runtime image recipe {recipe_version!r} is missing")
    return path


def _build_one(
    store: RegistryStore,
    image: RuntimeImageMetadata,
    *,
    uv_binary: str,
    scheme: str,
) -> str:
    snapshot = store.get(image.version)
    recipe = _recipe_path(image.recipe_version)
    with tempfile.TemporaryDirectory(prefix=f"mldb-runtime-image-v{image.version}-") as tmp:
        root = Path(tmp)
        shutil.copy2(recipe, root / "Dockerfile")
        shutil.copy2(uv_binary, root / "uv")
        (root / "pyproject.toml").write_bytes(snapshot.pyproject_toml)
        (root / "uv.lock").write_bytes(snapshot.uv_lock)
        tagged = f"{image.repository}:{image.tag}"
        _run(
            [
                "docker",
                "build",
                "--build-arg",
                f"BASE_IMAGE={image.base_image}",
                "--build-arg",
                f"RUNTIME_REGISTRY_VERSION={image.version}",
                "--build-arg",
                f"RUNTIME_REGISTRY_SNAPSHOT_SHA256={snapshot.metadata.snapshot_sha256}",
                "-t",
                tagged,
                ".",
            ],
            cwd=root,
        )
        _run(["docker", "push", tagged])
        digest = _manifest_digest(image.repository, image.tag, scheme)
        expected = _run(
            [
                "docker",
                "image",
                "inspect",
                tagged,
                "--format",
                "{{ index .Config.Labels \"org.mjtensu.runtime-registry.snapshot-sha256\" }}",
            ]
        ).strip()
        if expected != snapshot.metadata.snapshot_sha256:
            raise RuntimeImageBuildError("built image snapshot label mismatch")
        return digest


def _prune(
    store: RegistryStore,
    *,
    profile: str,
    retention: int,
    scheme: str,
) -> int:
    ready = store.ready_images(profile)
    removed = 0
    for image in ready[retention:]:
        _delete_manifest(image, scheme)
        store.mark_image_missing(image.version, image.profile)
        removed += 1
    return removed


def _garbage_collect(container: str | None) -> None:
    if not container:
        return
    try:
        _run(
            [
                "docker",
                "exec",
                container,
                "registry",
                "garbage-collect",
                "--delete-untagged",
                "/etc/distribution/config.yml",
            ]
        )
    except Exception as exc:
        print(f"runtime image registry GC deferred: {exc}", flush=True)


def run_forever() -> None:
    registry_config = RegistryConfig.from_env()
    builder_config = BuilderConfig.from_env()
    store = compose_store(registry_config)
    print(
        "runtime image builder started "
        f"retention={builder_config.retention} poll={builder_config.poll_seconds}s",
        flush=True,
    )
    while True:
        image = store.next_image_build()
        if image is None:
            time.sleep(builder_config.poll_seconds)
            continue
        print(
            f"building runtime image version={image.version} profile={image.profile} "
            f"tag={image.repository}:{image.tag}",
            flush=True,
        )
        try:
            digest = _build_one(
                store,
                image,
                uv_binary=registry_config.uv_binary,
                scheme=builder_config.registry_scheme,
            )
            store.mark_image_ready(image.version, image.profile, digest)
            removed = _prune(
                store,
                profile=image.profile,
                retention=builder_config.retention,
                scheme=builder_config.registry_scheme,
            )
            if removed:
                _garbage_collect(builder_config.gc_container)
            print(
                f"runtime image ready version={image.version} digest={digest} "
                f"pruned={removed}",
                flush=True,
            )
        except Exception as exc:
            detail = str(exc)
            if len(detail) > 4000:
                detail = detail[-4000:]
            store.mark_image_failed(image.version, image.profile, detail)
            print(
                f"runtime image build failed version={image.version}: {detail}",
                flush=True,
            )


def main() -> None:
    run_forever()


if __name__ == "__main__":
    main()
