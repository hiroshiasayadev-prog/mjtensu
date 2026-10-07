# MLDB Global Runtime Registry

This service owns the versioned Python runtime specification for MLDB.

A published version consists of immutable `pyproject.toml` + `uv.lock`. OCI Task images are derived artifacts built asynchronously from that snapshot and a versioned image recipe.

Snapshot API:

- `GET /`: latest published snapshot.
- `GET /?version=N`: immutable historical snapshot.
- `PUT /`: validate and publish a candidate snapshot.

`PUT` performs `uv sync --locked --no-install-project` before publication. It returns after snapshot publication and image-request enqueue; it does not wait for Docker build/push.
Image API:

- `GET /image?version=N&profile=gpu-cu124`
- `200`: READY and returns `image_ref=repository@sha256:...`
- `202`: BUILDING
- `424`: FAILED
- MISSING is re-queued as BUILDING when requested.

The asynchronous builder is:

    python -m runtime_registry.image_builder

It builds from `image_recipes/<recipe-version>.Dockerfile`, pushes to the configured OCI registry, verifies the manifest digest, and records READY only after push succeeds.

Runtime images are retained as a bounded cache; immutable snapshots and versioned recipes are the reconstruction authority.
