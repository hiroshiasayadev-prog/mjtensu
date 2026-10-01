# MLDB Global Runtime Registry

This service owns one global, versioned Python runtime definition for MLDB workers.
Architecture, Train Protocol, Evaluation Protocol, and Study definitions do not declare package dependencies.

API surface:

- `GET /` returns the latest published snapshot.
- `GET /?version=N` returns an immutable historical snapshot.
- `PUT /` sets a candidate `{pyproject_toml, uv_lock}` snapshot.

`PUT` first creates a clean temporary environment and runs `uv sync --locked --no-install-project`.
Only a successful candidate is published. Published blobs are content-addressed in Garage; SQLite stores only version metadata.
Repeatedly setting the current snapshot is idempotent and does not allocate another registry version.
