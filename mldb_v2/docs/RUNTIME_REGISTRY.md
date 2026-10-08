# MLDB v2 Global Runtime Registry

This document defines the operational contract for the global Python runtime registry used by MLDB v2.

The registry is the source of truth for Python package state. Docker/OCI images are derived execution artifacts materialized from one immutable registry snapshot plus one versioned base-image recipe. Study, Architecture, Train Protocol, and Evaluation Protocol definitions do not declare package dependencies.

## 1. Authority and identity

One registry snapshot contains an immutable `pyproject.toml` plus `uv.lock`. Its positive integer registry version identifies that complete Python package environment.

A canonical Study Result stores only:

    runtime_registry_version: N

Source identity is pinned separately by the Study Plan Git commit. Runtime-image repository names, tags, worker paths, package inventories, and virtualenv identifiers are deployment state and must not be copied into canonical experiment YAML.

The OCI image digest is derived runtime state. MLDB resolves it from `runtime_registry_version` when creating the ClearML Task.

## 2. Registry service and snapshot storage

Production endpoint:

    https://mjtensu-dev.home.arpa/mldb-runtime-registry/

Source:

    mldb_v2/services/runtime_registry/

Persistent service state:

    /srv/bugrat/mldb-runtime-registry/
Immutable Python snapshots are stored in the Garage bucket `mldb-runtime-registry`. SQLite stores registry-version metadata, object hashes, and runtime-image materialization state.

Snapshot API:

- `GET /` returns the latest published immutable snapshot.
- `GET /?version=N` returns one historical immutable snapshot.
- `PUT /` validates and publishes a candidate `{pyproject_toml, uv_lock}`.

`PUT` runs `uv sync --locked --no-install-project` in a clean temporary environment. A failed validation allocates no new registry version. Re-publishing bytes identical to the current latest snapshot is idempotent.

A successful `PUT` does **not** wait for Docker build/push. It commits the small immutable Python snapshot, creates the image-materialization request, and returns immediately.

## 3. OCI registry and Garage storage

Task images are served by the local OCI Distribution registry:

    https://mldb-registry.thebugrat.dev/

Its blob/manifest backend is the dedicated Garage bucket:

    mldb-container-registry

Workers use the OCI API only; they do not access the Garage bucket directly.

Current GPU base artifact:

    mldb-registry.thebugrat.dev/mjtensu/gpu-base@sha256:ec5ca38bc108e8d4379d26237d6bd4dca40307a682fd4d46bf9eeae6a3fff370

The corresponding rebuild recipe is versioned under:

    mldb_v2/services/runtime_registry/image_recipes/base/

Runtime-image recipes are versioned beside it. Do not modify a recipe after it has been used as a published materialization identity; create a new recipe version instead.
## 4. Asynchronous runtime-image materialization

The separate `runtime_registry.image_builder` process polls image requests. A materialization record has one of four states:

- `BUILDING`: queued or actively building.
- `READY`: an immutable image digest is available.
- `FAILED`: the last build/push attempt failed.
- `MISSING`: the previously materialized image was intentionally removed and may be rebuilt.

Image API:

    GET /image?version=N&profile=gpu-cu124

Response behavior:

- `200` for READY;
- `202` for BUILDING;
- `424` for FAILED.

A request for MISSING re-queues materialization from the retained registry snapshot and recipe.

For `gpu-cu124`, the builder copies the immutable `pyproject.toml` and `uv.lock` into the build context and runs:

    UV_PROJECT_ENVIRONMENT=/opt/conda       uv sync --locked --no-install-project --project /opt/mldb-runtime-spec

The completed image is pushed once, its registry manifest digest is read back, and only then is the materialization marked READY.

## 5. Image retention

The OCI registry is a materialized-artifact cache, not the long-term environment authority.

The current runtime-image retention target is three READY images per profile, managed as a small LRU cache. READY image resolution refreshes its access timestamp; after a build, the least-recently-used excess manifest is deleted and marked MISSING. The immutable Python snapshot and versioned build recipes remain available for reconstruction, so requesting a pruned historical version rebuilds it and keeps that rebuilt image in the cache.

Each GPU host also runs `mldb-runtime-image-pruner` once per hour. It queries the OCI registry's currently retained runtime manifests and removes only stale local `mjtensu/gpu-runtime` images; active containers are protected. The pruner fails closed if the registry cannot provide any retained manifests. It does not pre-pull all three central images, so a worker may locally hold fewer than three until those versions are actually used.

Base images change much less frequently. Base recipes are versioned separately so an old runtime can be reconstructed even after routine runtime-image pruning. Keep only the small number of base artifacts needed operationally; do not accumulate every multi-gigabyte runtime image indefinitely.
## 6. Study lifecycle pinning

A fresh Study Run resolves registry `latest` exactly once when its canonical Study Result is created. Every Training/Evaluation StageInput in that Study carries the same `runtime_registry_version`.

Lifecycle behavior:

- `resume` retains the existing registry version.
- stage retry retains the existing registry version.
- `rerun` reuses the source Study Result registry version.
- historical pre-registry results remain readable, but MLDB does not invent a missing version.

The registry version is immutable Study Result identity. Child stages never resolve `latest` independently.

## 7. ClearML execution boundary

For a GPU stage, the controller resolves:

    runtime_registry_version
        -> image profile
        -> READY OCI digest
        -> repository@sha256:...

ClearML receives the digest-qualified image reference, never an execution tag such as `latest`.

The Task container also receives `MLDB_RUNTIME_REGISTRY_VERSION` for provenance. `MLDB_V2_CLEARML_PREBUILT_RUNTIME=true` tells ClearML to use the baked interpreter instead of creating a per-Task venv.

All three GPU parent workers use the same digest-pinned ClearML worker image with `clearml-agent-bootstrap==1.0.6`. The bootstrap payload is unpacked into a host-visible cache so sibling Task containers can mount it through the host Docker socket. GPU worker configuration sets `agent.package_manager.pip_version=[]`, and the baked Task interpreter is selected with `CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=/opt/conda/bin/python`. A live Task smoke verified that ClearML kept the existing pip, found all requested packages preinstalled, installed zero additional packages, and started the MLDB entrypoint without modifying the baked Python environment.

Stage routing is independent of registry version pinning. A route may set `runtime_image_profile: null` to opt out of baked GPU materialization.

Current routing uses `gpu-cu124` for GPU stages, `recognition-functional-video`, and the `onnx-cpu-latency` stage. CPU latency inherits the pinned Registry image with `docker_gpu: null` so only CPU providers run; iPhone-specialized stages retain `runtime_image_profile: null` and the existing static-image/device-runner path.

## 8. GPU worker rollout

The old per-worker managed-venv bootstrap is retired for GPU workers. There is no worker-local `current.json`, shared runtime lock, differential `uv pip`, or runtime re-exec on:

- `precision5820-gpu3060`
- `old-gpu3090`
- `dev-wsl-gpu3060`

The worker only starts the image selected by the Task.

A live cross-host smoke verified one identical RR v6 digest on all three GPUs:

    mldb-registry.thebugrat.dev/mjtensu/gpu-runtime@sha256:934ac28b0c110ca08278e4a20cb39ff25b1c7f9f07b53f3a29117cfa8c4cd123

The same image exposed `torch 2.5.1+cu124`, `opencv 4.11.0`, `onnxruntime 1.28.0`, and `matplotlib 3.10.9` on the new RTX 3060, old RTX 3090, and WSL RTX 3060; Chrome/ffmpeg remain supplied by the shared GPU base.

A real ClearML Task on queue `precision5820-gpu3060` launched with that exact v6 digest through bootstrap v1.0.6. Its environment setup kept pip 24.2, reported zero additional packages to install, and reached the MLDB entrypoint. The intentionally cloned smoke then failed only because it did not carry a real StageInput/runtime snapshot; the runtime-image/bootstrap boundary itself had already passed.

## 9. Current Python baseline

Registry v2 contains the original baked GPU package baseline, including:

    Python                   3.11
    torch                    2.5.1+cu124
    torchvision              0.20.1+cu124
    numpy                    1.26.4
    onnx                     1.22.0
    onnxruntime              1.28.0
    opencv-python-headless   4.11.0.86

Registry v3 added `clearml-agent==3.0.3` to the authoritative lock. v4 added the NanoDet fork plus its Lightning/metrics dependencies, v5 added `matplotlib==3.10.9`, v6 added `pycocotools==2.0.11`, and v7 moved the NanoDet fork to fork.2 and added `pytest==9.1.1`. GPU Task startup uses the parent worker's separately pinned compiled bootstrap v1.0.6, so Task startup does not install or upgrade packages in the baked image.

Chrome, ffmpeg, CUDA userspace, and other OS-level dependencies belong to the base-image recipe, not the Python registry snapshot.

## 10. Publishing and rebuilding

To change Python packages:

1. edit exact direct pins in a candidate `pyproject.toml`;
2. resolve `uv.lock`;
3. compare the candidate against the current snapshot for unintended changes;
4. `PUT` both files to the registry;
5. let the asynchronous builder materialize the configured image profiles.

A successful SET is Python-spec publication, not proof that every image profile is READY. Task submission waits for the requested profile to become READY and fails closed if materialization fails or exceeds its configured wait.

If an old runtime image was pruned, requesting that version/profile changes MISSING back to BUILDING and recreates the artifact from the retained snapshot and recipes.

Modified external OSS must first be published under a unique package/import namespace. See `EXTERNAL_FORK_PACKAGES.md`; experiment YAML must not select package repositories or forks.

## 11. Operational checks

For snapshot health, verify current and historical GETs validate object hashes.

For image health, verify the image endpoint returns READY plus a `repository@sha256:...` reference, then pull that exact digest.

For a ClearML smoke, the agent log must show the same digest in `docker_cmd`, and the actual Task container image must match it.

Do not repair failures by editing canonical Study Results, SQLite rows, image manifests, or package lists by hand. Fix the failed boundary, then requeue/rebuild the derived image artifact.
