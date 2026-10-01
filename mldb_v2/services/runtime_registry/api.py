"""Tiny HTTP API: GET registry snapshot and PUT (SET) a new snapshot."""
from __future__ import annotations

from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from urllib.parse import parse_qs, urlsplit

from .config import compose_service
from .core import RegistryCorruption, RegistryNotFound
from .service import RuntimeRegistryService
from .validator import UvSyncValidationError

_MAX_BODY_BYTES = 32 * 1024 * 1024


def _metadata_dict(metadata) -> dict[str, object]:
    return asdict(metadata)


def make_handler(service: RuntimeRegistryService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "mldb-runtime-registry/1"

        def _json(self, status: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _path_and_query(self):
            parsed = urlsplit(self.path)
            return parsed.path.rstrip("/") or "/", parse_qs(parsed.query)

        def do_GET(self) -> None:  # noqa: N802
            path, query = self._path_and_query()
            if path != "/":
                self._json(404, {"error": "not_found"})
                return
            raw_version = query.get("version", [None])[0]
            try:
                version = None if raw_version is None else int(raw_version)
                if version is not None and version <= 0:
                    raise ValueError
                snapshot = service.get(version)
            except ValueError:
                self._json(400, {"error": "invalid_version"})
                return
            except RegistryNotFound as exc:
                self._json(404, {"error": "registry_not_found", "detail": str(exc)})
                return
            except RegistryCorruption as exc:
                self._json(500, {"error": "registry_corruption", "detail": str(exc)})
                return
            payload = _metadata_dict(snapshot.metadata)
            payload["pyproject_toml"] = snapshot.pyproject_toml.decode("utf-8")
            payload["uv_lock"] = snapshot.uv_lock.decode("utf-8")
            self._json(200, payload)

        def do_PUT(self) -> None:  # noqa: N802
            path, _query = self._path_and_query()
            if path != "/":
                self._json(404, {"error": "not_found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                size = 0
            if size <= 0 or size > _MAX_BODY_BYTES:
                self._json(400, {"error": "invalid_body_size"})
                return
            try:
                payload = json.loads(self.rfile.read(size))
                pyproject = payload["pyproject_toml"].encode("utf-8")
                uv_lock = payload["uv_lock"].encode("utf-8")
                result = service.set(pyproject, uv_lock)
            except UvSyncValidationError as exc:
                self._json(422, {"error": "uv_sync_failed", "detail": str(exc)})
                return
            except (json.JSONDecodeError, KeyError, AttributeError, TypeError, ValueError) as exc:
                self._json(400, {"error": "invalid_request", "detail": str(exc)})
                return
            response = _metadata_dict(result.metadata)
            response["created"] = result.created
            self._json(201 if result.created else 200, response)

    return Handler


def main() -> None:
    service = compose_service()
    host = os.environ.get("MLDB_RUNTIME_REGISTRY_HOST", "0.0.0.0")
    port = int(os.environ.get("MLDB_RUNTIME_REGISTRY_PORT", "8080"))
    server = ThreadingHTTPServer((host, port), make_handler(service))
    server.serve_forever()


if __name__ == "__main__":
    main()
