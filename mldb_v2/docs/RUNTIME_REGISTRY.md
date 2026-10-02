# MLDB v2 Global Runtime Registry

This document defines the operational contract for the global Python runtime registry used by MLDB v2.

The purpose is to keep worker Python environments consistent without adding dependency declarations to Study, Architecture, Train Protocol, or Evaluation Protocol definitions. Experiment source and runtime package state are versioned separately: Git/Study Plan pins source, while the Study Result pins one immutable runtime-registry version.

## 1. Authority and ownership

The runtime registry is the package-version authority for formal MLDB execution. A registry snapshot contains an immutable `pyproject.toml` plus `uv.lock`; the registry version identifies that complete package environment.

Do not copy package inventories into canonical Training/Evaluation results. `runtime_registry_version` on the Study Result is the environment-provenance handle. Given that immutable version, the registry can reconstruct the exact package snapshot.

The registry is deployment/runtime state, not experiment-schema state. Do not add package lists, repository selectors, environment names, virtualenv identifiers, or registry versions to Study, Architecture, Train Protocol, or Evaluation Protocol YAML.

## 2. Service and storage

Production endpoint:

    https://mjtensu-dev.home.arpa/mldb-runtime-registry/

Source:

    mldb_v2/services/runtime_registry/

Persistent service state:

    /srv/bugrat/mldb-runtime-registry/

Immutable snapshot objects are stored in the dedicated Garage bucket `mldb-runtime-registry`; SQLite stores registry version metadata and object hashes. Snapshot objects are content-addressed and verified on read.

The API surface is intentionally small:

- `GET /` returns the latest published immutable snapshot.
- `GET /?version=N` returns one historical immutable snapshot.
- `PUT /` publishes a candidate snapshot.

`PUT` validates the candidate in a clean temporary environment with `uv sync --locked --no-install-project`. A failed validation does not allocate a new registry version. Re-publishing bytes identical to the current latest snapshot is idempotent and does not create another version.

## 3. Study Run pinning

A fresh Study Run resolves registry `latest` exactly once, when the new canonical Study Result is created. The returned positive integer is stored as:

    runtime_registry_version: N

Every Training/Evaluation StageInput in that Study carries the same value. Child stages must never resolve `latest` independently.

Lifecycle behavior is deliberately reproducible:

- `resume` keeps the existing Study Result and therefore keeps its registry version.
- stage retry keeps the same Study Result and registry version.
- `rerun` creates a fresh Study Result from the source run's immutable Plan but reuses the source run's `runtime_registry_version` rather than resolving current latest.
- pre-registry historical Study Results remain readable, but MLDB does not invent a missing runtime version for reproducible re-execution.

The runtime-registry version is immutable Study Result identity. Canonical update/retry paths reject a version change.

## 4. Current registry baseline

Registry v1 was published on 2026-10-02 from the package environments observed on the active ClearML workers. It contained 66 direct pins / 85 resolved packages. The only observed worker drift before v1 publication was on `dev-wsl-gpu3060`: `botocore` and `PyJWT` were one patch newer than the other workers, so v1 normalized to those newer observed versions.

Registry v2 is the current snapshot. It adds the Python dependency required by the recognition functional-video render path:

    direct pins:             67
    resolved packages:       86
    Python:                  3.11
    torch:                   2.5.1+cu124
    torchvision:             0.20.1+cu124
    numpy:                   1.26.4
    onnx:                    1.22.0
    onnxruntime:             1.28.0
    opencv-python-headless:  4.11.0.86

Chrome and ffmpeg remain image/OS-level dependencies for the functional render worker; they are not Python registry entries.

## 5. Managed worker materialization

Runtime-registry enforcement is currently implemented for:

- `precision5820-gpu3060` on the new server;
- `old-gpu3090` on the old server.

The new-server reference worker has a dedicated validation queue also named `precision5820-gpu3060`. The old RTX 3090 worker retains both `default` and `recognition-functional`; registry enablement does not replace or merge those queues.

Each worker reuses one managed virtual environment rooted below a worker-specific host directory, for example:

    /srv/bugrat/data-lv/clearml/runtime-registry/precision5820-gpu3060/
    /srv/bugrat/data-lv/clearml/runtime-registry/old-gpu3090/

The Task container sees that root as `/mldb-runtime-registry` and receives the pinned target version through `MLDB_RUNTIME_REGISTRY_VERSION`.

The managed venv is created with `--system-site-packages`. This is intentional: the prebuilt worker image already contains the large CUDA/PyTorch stack, and an ordinary `uv sync` into a venv would reinstall those packages even when the base-visible versions already match. The reference worker therefore performs differential convergence.

For a target registry version it:

1. loads and validates the immutable registry snapshot;
2. compares the snapshot's resolved package versions with distributions visible from the managed interpreter;
3. installs/upgrades/downgrades only mismatched packages with `uv pip`;
4. removes packages only when they were previously registry-managed and are locally removable from the managed venv;
5. verifies the complete resolved target state;
6. atomically replaces `current.json` only after verification succeeds;
7. re-executes the MLDB harness under `/mldb-runtime-registry/venv/bin/python`.

A real ClearML smoke verified the mechanism first on `precision5820-gpu3060` without reinstalling PyTorch. A second real smoke on `old-gpu3090` used registry v2 and the `recognition-functional-v1` render image together: the worker kept base-visible `torch==2.5.1+cu124`, `cv2==4.11.0`, Chrome, and ffmpeg, re-executed MLDB through the managed venv, and produced the accepted functional trace and overlay MP4.

## 6. Concurrency and locking

The reference worker uses one reusable managed venv, not one venv per registry version.

A runtime lock protects that mutable environment:

- a version transition takes the exclusive lock;
- a Task running under a verified managed environment retains a shared lock;
- another Task cannot switch the shared venv to a different registry version while that Task is executing.

The version marker is fail-closed. A failed download, install, downgrade, removal, verification, or re-exec preparation must not advance `current.json`.

## 7. ClearML boundary

MLDB passes `runtime_registry_version` through StageInput and injects the same value into the Task container as `MLDB_RUNTIME_REGISTRY_VERSION`.

The reference worker deployment additionally supplies the worker identity, managed-runtime root, registry service endpoint, and read-only `uv` binary mount. These are deployment concerns and must not leak into experiment YAML.

`MLDB_V2_CLEARML_PREBUILT_RUNTIME=true` still prevents ClearML Agent from building its own per-Task environment. On the registry-aware reference worker, the global registry is the package-version authority; ClearML's smaller Task requirement list is compatibility/visibility metadata, not the authoritative runtime definition.

## 8. Worker rollout status

The pinned registry snapshot is currently enforced by:

- `precision5820-gpu3060`
- `old-gpu3090`

These active workers are not yet migrated to the managed-runtime bootstrap:

- `old-cpu`
- `old-iphone`
- `dev-wsl-gpu3060`

Every new Study Result carries a pinned `runtime_registry_version`, but package enforcement is guaranteed only on migrated workers until the bootstrap is copied and smoke-tested on each remaining worker.

Use queue `precision5820-gpu3060` when the purpose is to isolate the new-server registry path. Use `recognition-functional` for the RTX 3090 functional trace/render stage. Normal stage-routing rules remain independent of registry pinning.

## 9. Publishing a new registry version

A package-set change is an operational runtime change. Build a candidate `pyproject.toml` with exact direct pins, resolve it to `uv.lock`, and publish both through `PUT /`.

Modified external OSS must first be published as a uniquely named Python package with a non-conflicting import namespace. The default public-fork path is independent Git repository under `external/` -> Git tag -> wheel build -> GitHub Release asset -> runtime-registry dependency/lock. See `EXTERNAL_FORK_PACKAGES.md`; do not install a modified fork under its upstream import name or make experiment YAML select a fork/repository.

The server performs a clean `uv sync --locked` before publication. Only after that validation succeeds are the immutable objects stored and the next SQLite registry version committed.

Publishing a registry version does not rewrite an already-started Study. New Study Runs may resolve the new latest version; existing runs, retries, resumes, and reruns keep their previously pinned version.

## 10. Known limitation

Because the reference managed venv uses `--system-site-packages`, it can overlay a base-image package with another version but cannot make a package that exists only in the base image disappear completely. If a future registry snapshot requires total removal of such a base package, materialization intentionally fails closed.

Resolve that case by changing the base-runtime/worker design or otherwise making removal enforceable. Do not mark the registry version as active while the visible environment still violates the snapshot.

## 11. Operational checks

For registry service health, verify latest and historical GETs return hash-valid snapshots and that the requested version exists.

For a migrated worker, verify:

    /srv/bugrat/data-lv/clearml/runtime-registry/<worker-id>/current.json
    /srv/bugrat/data-lv/clearml/runtime-registry/<worker-id>/venv/

A running registry-managed MLDB Task should execute with:

    /mldb-runtime-registry/venv/bin/python

Do not manually edit the version marker, snapshot cache, canonical Study Result version, or package list to repair a failed run. Diagnose the first failed boundary and preserve the previous verified environment until convergence succeeds.
