from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from mldb_v2.services.runtime_registry.core import RegistryCorruption, RegistryStore
from mldb_v2.services.runtime_registry.service import RuntimeImagePolicy, RuntimeRegistryService
from mldb_v2.services.runtime_registry.validator import UvSyncValidationError, UvSyncValidator


class MemoryObjectStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def publish_bytes_immutable(self, uri: str, data: bytes) -> None:
        existing = self.objects.get(uri)
        if existing is not None and existing != data:
            raise FileExistsError(uri)
        self.objects[uri] = data

    def read_bytes(self, uri: str) -> bytes:
        return self.objects[uri]


class RecordingValidator:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, bytes]] = []

    def validate(self, pyproject_toml: bytes, uv_lock: bytes) -> None:
        self.calls.append((pyproject_toml, uv_lock))


class RuntimeRegistryTests(unittest.TestCase):
    def make_service(self, root: Path):
        objects = MemoryObjectStore()
        validator = RecordingValidator()
        store = RegistryStore(root / "registry.sqlite3", objects, "registry-bucket")
        return RuntimeRegistryService(store, validator), objects, validator

    def test_set_get_version_and_idempotent_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service, _objects, validator = self.make_service(Path(tmp))
            p1 = b"[project]\nname='r1'\nversion='0'\n"
            l1 = b"version = 1\nrevision = 1\n"
            first = service.set(p1, l1)
            duplicate = service.set(p1, l1)
            second = service.set(p1.replace(b"r1", b"r2"), l1)
            self.assertTrue(first.created)
            self.assertEqual(first.metadata.version, 1)
            self.assertFalse(duplicate.created)
            self.assertEqual(duplicate.metadata.version, 1)
            self.assertTrue(second.created)
            self.assertEqual(second.metadata.version, 2)
            self.assertEqual(service.get().metadata.version, 2)
            self.assertEqual(service.get(1).pyproject_toml, p1)
            self.assertEqual(len(validator.calls), 3)

    def test_corrupt_garage_object_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service, objects, _validator = self.make_service(Path(tmp))
            result = service.set(b"p", b"l")
            objects.objects[result.metadata.uv_lock_uri] = b"corrupt"
            with self.assertRaises(RegistryCorruption):
                service.get(result.metadata.version)

    def test_validation_failure_does_not_publish(self) -> None:
        class Reject:
            def validate(self, pyproject_toml: bytes, uv_lock: bytes) -> None:
                raise UvSyncValidationError("nope")

        with tempfile.TemporaryDirectory() as tmp:
            objects = MemoryObjectStore()
            store = RegistryStore(Path(tmp) / "registry.sqlite3", objects, "registry-bucket")
            service = RuntimeRegistryService(store, Reject())
            with self.assertRaises(UvSyncValidationError):
                service.set(b"p", b"l")
            self.assertEqual(objects.objects, {})

    def test_set_queues_runtime_image_without_waiting_for_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            objects = MemoryObjectStore()
            validator = RecordingValidator()
            store = RegistryStore(Path(tmp) / "registry.sqlite3", objects, "registry-bucket")
            policy = RuntimeImagePolicy(
                profile="gpu-cu124",
                recipe_version="gpu-cu124-v1",
                repository="registry.example/mjtensu/gpu-runtime",
                base_image="registry.example/mjtensu/gpu-base@sha256:" + "a" * 64,
            )
            service = RuntimeRegistryService(store, validator, (policy,))
            result = service.set(b"p", b"l")

            self.assertTrue(result.created)
            self.assertEqual(result.images[0].state, "BUILDING")
            self.assertEqual(result.images[0].tag, "rr-v1")
            self.assertIsNone(result.images[0].image_ref)

            store.mark_image_ready(1, "gpu-cu124", "sha256:" + "b" * 64)
            ready = service.ensure_image(1, "gpu-cu124")
            self.assertEqual(ready.state, "READY")
            self.assertEqual(
                ready.image_ref,
                "registry.example/mjtensu/gpu-runtime@sha256:" + "b" * 64,
            )

            store.mark_image_missing(1, "gpu-cu124")
            rebuilt = service.ensure_image(1, "gpu-cu124")
            self.assertEqual(rebuilt.state, "BUILDING")
            self.assertEqual(rebuilt.recipe_version, "gpu-cu124-v1")

    def test_ready_image_cache_orders_by_recent_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            objects = MemoryObjectStore()
            validator = RecordingValidator()
            store = RegistryStore(Path(tmp) / "registry.sqlite3", objects, "registry-bucket")
            policy = RuntimeImagePolicy(
                profile="gpu-cu124",
                recipe_version="gpu-cu124-v1",
                repository="registry.example/mjtensu/gpu-runtime",
                base_image="registry.example/mjtensu/gpu-base@sha256:" + "a" * 64,
            )
            service = RuntimeRegistryService(store, validator, (policy,))
            for version in range(1, 4):
                result = service.set(f"p{version}".encode(), f"l{version}".encode())
                self.assertEqual(result.metadata.version, version)
                store.mark_image_ready(version, "gpu-cu124", "sha256:" + str(version) * 64)

            service.ensure_image(1, "gpu-cu124")
            order = [image.version for image in store.ready_images("gpu-cu124")]
            self.assertEqual(order[0], 1)
            self.assertEqual(set(order), {1, 2, 3})

    def test_uv_validator_uses_clean_locked_sync(self) -> None:
        seen: dict[str, object] = {}

        def fake_run(command, **kwargs):
            seen["command"] = command
            root = Path(kwargs["cwd"])
            self.assertEqual((root / "pyproject.toml").read_bytes(), b"project")
            self.assertEqual((root / "uv.lock").read_bytes(), b"lock")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with mock.patch("mldb_v2.services.runtime_registry.validator.subprocess.run", side_effect=fake_run):
            UvSyncValidator(uv_binary="/opt/uv").validate(b"project", b"lock")
        command = seen["command"]
        self.assertEqual(command[:4], ["/opt/uv", "sync", "--locked", "--no-install-project"])
        self.assertIn("--project", command)


if __name__ == "__main__":
    unittest.main()
