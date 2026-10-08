# MLDB v2 Operations Manual

This is the day-to-day runbook for executing existing MLDB v2 Studies from the Windows development repository through ClearML to the GPU worker.

> Current execution model (2026-10-02): one MLDB Study Result maps to one ClearML Pipeline Run. MLDB owns semantic release/canonical acceptance, ClearML owns physical child-Task execution, and the Study Result pins one immutable global runtime-registry version for the full run.

## 1. Proven deployment

Repository root:

    C:\Users\imved\projects\mjtensu

Supported Windows entrypoint:

    .\mldb.cmd <command> ...

`mldb.cmd` changes to the repository root, sets `MLDB_REPO_ROOT`, prefers `.venv\Scripts\python.exe`, and invokes `python -m mldb_v2.src.cli`.

The currently validated external services are:

| Service | Current endpoint / identity |
|---|---|
| ClearML Web | `https://clearml.thebugrat.dev/` |
| ClearML API | `https://clearml-api.thebugrat.dev/` |
| ClearML Files | `https://clearml-files.thebugrat.dev/` |
| ClearML default queue | `default` |
| Registry-reference queue | `precision5820-gpu3060` |
| Canonical CPU-latency queue | `latency-cpu` |
| Registry-aware reference worker | `precision5820-gpu3060` (new server, RTX 3060) |
| Registry-aware GPU/render worker | `old-gpu3090` (RTX 3090; `default` + `recognition-functional`) |
| Canonical latency worker | `old-cpu` (dedicated CPU-only ClearML Agent on the old host; not yet registry-managed) |
| Runtime registry | `https://mjtensu-dev.home.arpa/mldb-runtime-registry/` |
| S3 endpoint | `https://mldb-s3.thebugrat.dev` |
| S3 region | `us-east-1` |

Treat the local `.env` and actual infrastructure as the live configuration source; the values above are the last validated deployment, not a reason to ignore current state.

## 2. Environment setup

`mldb.cmd` automatically loads the repository-root `.env` before starting the CLI. The file is Git-ignored and is the normal place for local ClearML/S3 credentials and stable runtime defaults.

Explicit process environment wins over `.env`. This makes normal use zero-bootstrap while still allowing one-off overrides such as:

    $env:MLDB_V2_CLEARML_QUEUE = 'another-queue'
    .\mldb.cmd run <study-id>

`mldb.cmd` also sets `MLDB_REPO_ROOT` itself. Do not put another repository root in `.env`.

The loader intentionally expects simple dotenv entries of the form `KEY=value`, with blank lines and `#` comments allowed. Keep secrets unquoted unless there is a concrete need to extend the loader. Never commit `.env`.

Environment variables consumed by the production composition include:

| Variable | Consumer / purpose | Secret? |
|---|---|---|
| `CLEARML_API_HOST` | ClearML API endpoint | no |
| `CLEARML_WEB_HOST` | ClearML UI endpoint | no |
| `CLEARML_FILES_HOST` | ClearML file endpoint | no |
| `CLEARML_API_ACCESS_KEY` | ClearML authentication | yes |
| `CLEARML_API_SECRET_KEY` | ClearML authentication | yes |
| `MLDB_V2_DEFAULT_BACKEND` | default backend used when `run` omits `--backend` | no |
| `MLDB_V2_CLEARML_QUEUE` | default ClearML queue name | no |
| `MLDB_V2_CLEARML_STAGE_ROUTES_JSON` | backend-only JSON map from logical stage name to queue / Docker GPU overrides | no |
| `MLDB_V2_CLEARML_PREBUILT_RUNTIME` | reuse the prebuilt image's system Python instead of creating a per-Task venv | no |
| `MLDB_V2_CLEARML_REPOSITORY` | Git repository URL used by ClearML source execution | no |
| `MLDB_V2_CLEARML_DOCKER_IMAGE` | worker container image | no |
| `MLDB_V2_CLEARML_DOCKER_ENV_FILE` | optional Docker env-file path | may contain secrets |
| `MLDB_V2_CLEARML_DOCKER_GPU` | Docker GPU selector | no |
| `MLDB_V2_CLEARML_DOCKER_SHM_SIZE` | container shared-memory size | no |
| `MLDB_V2_RUNTIME_DATA_ROOT` | optional runtime canonical snapshot root | no |
| `MLDB_V2_RUNTIME_REGISTRY_URL` | global runtime-registry endpoint used to resolve immutable runtime snapshots | no |
| `MLDB_V2_ARTIFACT_URI_PREFIX` | explicit artifact URI prefix | no |
| `MLDB_S3_BUCKET` | fallback artifact bucket | no |
| `MLDB_S3_ENDPOINT_URL` | S3/MinIO endpoint | no |
| `MLDB_S3_REGION` | S3 region | no |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` | S3 credentials | yes |
| `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` | S3 credential fallback | yes |

The validated local `.env` now also carries these non-secret runtime defaults so ordinary execution does not require a PowerShell bootstrap:

    MLDB_V2_DEFAULT_BACKEND=clearml
    MLDB_V2_CLEARML_QUEUE=default
    MLDB_V2_CLEARML_STAGE_ROUTES_JSON={"onnx-cpu-latency":{"queue":"latency-cpu","docker_gpu":null}}
    MLDB_V2_CLEARML_REPOSITORY=https://github.com/hiroshiasayadev-prog/mjtensu.git
    MLDB_V2_CLEARML_DOCKER_IMAGE=mldb-clearml-runner:torch2.5.1-cu124-v1
    MLDB_V2_CLEARML_PREBUILT_RUNTIME=true
    MLDB_V2_CLEARML_DOCKER_GPU=all
    MLDB_V2_CLEARML_DOCKER_SHM_SIZE=2g

The validated path does not use `MLDB_V2_CLEARML_DOCKER_ENV_FILE`; required object-store credentials are supplied through the configured runtime environment. Introduce a Docker env-file only for a concrete deployment need.

Normal startup is therefore just:

    Set-Location C:\Users\imved\projects\mjtensu
    .\mldb.cmd doctor

If you bypass `mldb.cmd` and invoke `python -m mldb_v2.src.cli` directly, `.env` is not loaded by the Python module and `MLDB_REPO_ROOT` is not supplied automatically. Prefer the wrapper for normal operation.

### Global runtime registry

Every new Study Result resolves the current runtime-registry `latest` exactly once when the run is created and persists that positive integer as `runtime_registry_version`. Every Training/Evaluation StageInput for that Study carries the same version. `resume` and stage retry retain it; `rerun` creates a fresh Study Result from the old immutable Plan but reuses the source Study Result's runtime-registry version rather than resolving a newer `latest`.

This is deployment/runtime state, not experiment authoring state. Do not add package, repository, environment, virtualenv, or registry-version fields to Study, Architecture, Train Protocol, or Evaluation Protocol YAML. Publish package-set changes through the global registry instead.

GPU registry enforcement is image-based. The controller resolves the Study's pinned registry version to a READY `gpu-cu124` image digest and gives that exact `repository@sha256:...` reference to ClearML. The GPU workers do not maintain or mutate a registry-specific venv. A per-host hourly pruner removes local runtime-image digests that are no longer retained by the central three-entry LRU cache, while leaving active containers untouched.

`old-gpu3090` additionally serves queue `recognition-functional`. Chrome, ffmpeg, CUDA userspace, and other OS dependencies live in the versioned GPU base recipe; the Python package set comes from the registry snapshot. The same RR v6 image digest was executed successfully on the new RTX 3060, old RTX 3090, and WSL RTX 3060. CPU ONNX latency Tasks inherit the normal `gpu-cu124` Registry image profile with `docker_gpu: null`, while iPhone-specialized routes still opt out with `runtime_image_profile: null` and use the static fallback image. See `RUNTIME_REGISTRY.md` for the full publication/materialization rules.

A pre-registry historical Study Result remains readable, but MLDB does not invent a runtime version for reproducible re-execution. A rerun/recovery path that requires an unknown version fails closed rather than silently substituting current `latest`.

### ClearML run hygiene and archiving

Keep the default ClearML experiment view focused on currently relevant results. Once a diagnostic/intermediate run has served its purpose and its replacement has been verified, archive the superseded ClearML controller Task **and all child Tasks** together. Archiving is a UI/lifecycle operation only: do not delete the canonical MLDB Study Result, Evaluation Results, source commit, or persisted artifacts merely to reduce ClearML clutter.

Recommended rule:

- keep the latest accepted run for an active Study visible;
- archive failed/cancelled troubleshooting runs after the cause is recorded and a replacement run is verified;
- archive successful validation runs once a newer accepted revision supersedes them;
- never archive a still-running run or the only accepted result for a Study;
- archive the controller and child Tasks as one unit so Pipeline and task lists stay consistent.

ClearML SDK exposes this as `Task.set_archived(True)`. Archived Tasks remain recoverable with archived-task views / API queries.

### CPU latency queue and comparability

`onnx-cpu-latency` is a benchmark stage, not ordinary throughput work. Route it only to `latency-cpu`. The canonical Linux worker is the dedicated CPU-only Agent `old-cpu`; its ONNX CPU benchmark Tasks now inherit the globally pinned `gpu-cu124` runtime-image profile while setting `docker_gpu: null`. The Registry supplies the same digest-qualified image as GPU Tasks, but Docker exposes no GPU to the CPU Task. The CPU parent Agent remains a separate ClearML queue consumer; it does not maintain its own Python package set. The GPU Agent `old-gpu3090` serves `default` and `recognition-functional`, but not `latency-cpu`. The two Agents may execute concurrently on the same physical old host, so CPU latency work no longer occupies the GPU worker slot. Do **not** subscribe a Windows development worker, another CPU model, or any heterogeneous machine to `latency-cpu`; doing so changes benchmark hardware and invalidates direct historical/model comparison. Additional workers may subscribe to `default` for ordinary Training/Evaluation once they have the required runtime image and data access.

Queue isolation fixes worker identity, but same-host GPU work and unrelated host processes can still perturb timing. For comparable latency numbers, keep other CPU-heavy host workloads away from the benchmark window. If host contention cannot be controlled, treat the latency values as non-comparable and rerun under controlled load. Keep the Protocol's batch size, ORT provider, intra/inter-op thread counts, execution mode, and runtime versions fixed as well.

For GPU stages, `MLDB_V2_CLEARML_PREBUILT_RUNTIME=true` uses the materialized registry image directly. ClearML Task requirements remain projected for compatibility/visibility, but the authoritative package set is the registry lock baked into the image. GPU parent workers use the digest-pinned worker image with compiled `clearml-agent-bootstrap==1.0.6`, a host-visible bootstrap cache, and `agent.package_manager.pip_version=[]`; Task startup therefore keeps the baked pip/interpreter and installs no runtime packages when the image matches the pinned registry snapshot.

## 3. Preflight before a run

Start with repository and runtime health:

    git status --short
    .\mldb.cmd doctor

Then validate the exact definition scope you intend to use, for example:

    .\mldb.cmd validate study tile-classifier/<study-id> --json
    .\mldb.cmd verify train-protocol tile-classifier/<protocol-id> --json

## 4. Source pinning before ClearML execution

Formal planning selects the current Git `HEAD` as the Study source commit. Every required v2 source input must match that commit exactly.

The clean set is scoped, not repository-wide. It includes `mldb_v2/src/` plus the referenced Namespace/Study/Task/Corpus/Architecture/Protocol definitions, same-basename executable siblings, declared same-namespace `lib/` helpers, manifests/builders, and any referenced canonical Model/result inputs.

Unrelated files may remain dirty.

Before planning a changed experiment:

1. Inspect `git status --short` and the referenced definition closure.
2. Stage only the required experiment/runtime source; never use `git add .` in a dirty tree.
3. Review `git diff --cached --name-status`, `--stat`, and `--check`.
4. Commit the required source.
5. Push the commit to a branch/ref the ClearML worker can fetch.
6. Stay on that source commit while planning the Study.

If planning reports `source_not_pinned`, do not attach the working-tree patch to the backend. See `TROUBLESHOOTING.md`.

## 5. Plan and execute

Plan one sealed Study explicitly:

    .\mldb.cmd plan <namespace>/<study-id>

Normal foreground execution:

    .\mldb.cmd run <namespace>/<study-id>

or, without a configured default backend:

    .\mldb.cmd run <namespace>/<study-id> --backend clearml

`run` plans/compiles the Study, resolves registry `latest` once, and creates a fresh durable Study Result containing that `runtime_registry_version`. It then creates/recovers one ClearML Pipeline Run for that Study Result; ClearML owns child-Task scheduling while the foreground MLDB process reconciles canonical results until terminal. Interrupting the local process does not mean cancellation; the backend Pipeline may continue. Use `resume` to reconnect/reconcile or `cancel` to request cancellation.

## 6. Monitor and recover

Useful read/control commands:

    .\mldb.cmd ps --all
    .\mldb.cmd status <study-result-id>
    .\mldb.cmd watch <study-result-id>
    .\mldb.cmd logs <study-result-id> --failed
    .\mldb.cmd resume <study-result-id>
    .\mldb.cmd advance <study-result-id>
    .\mldb.cmd retry-stage <study-result-id> --trial <trial-id> --coordinate <evaluation-coordinate>
    .\mldb.cmd cancel <study-result-id>

`watch` is read-only. `advance` performs one explicit canonical reconciliation pass and is mainly for recovery/debugging. `rerun` creates a fresh Study Result from the immutable Plan of an earlier execution rather than recompiling the current mutable Study definition, and it preserves the source Study Result's `runtime_registry_version`.

`retry-stage` is different from `rerun`. It synchronously retries exactly one **failed Evaluation** inside an existing `completed_with_failures` Study Result. The Study Result ID, execution key, Plan/source commit, logical evaluation coordinate, Evaluation Result ID, training result, and completed sibling evaluations are retained. The backend creates one new physical attempt; the failed physical attempt and its logs remain preserved. Canonical Evaluation Result `attempts` is extended by exactly one ordered attempt. If the retry succeeds, that same stage slot changes `failed -> completed` and the Study Result becomes `completed` only when every planned stage is completed. If the retry fails again, the Study Result remains `completed_with_failures`.

Version 1 deliberately does not reopen terminal canonical state to a generic pending state and does not support Training-stage retry. In-progress retry visibility is therefore backend/ClearML visibility only; the CLI returns the updated canonical Study Result after the physical retry reaches a terminal outcome. A completed or otherwise non-failed Evaluation cannot be retried.

After W011 implementation, the primary ClearML UI entrypoint for a running/completed Study is its Pipeline Run. Use child Pipeline Tasks for detailed logs/Charts and the controller summary for selected Study-level projections. Until W011 closes, existing executions still appear as flat Tasks.

## 7. Read the result from canonical MLDB state

After a terminal execution, prefer canonical objects over ClearML UI state:

    .\mldb.cmd get runs <study-result-id> --json
    .\mldb.cmd get training-results <training-result-id> --json
    .\mldb.cmd get models <model-id> --json
    .\mldb.cmd get evaluation-results <evaluation-result-id> --json

Namespace-first files are stored under paths such as:

    mldb_data/<namespace>/study_results/
    mldb_data/<namespace>/training_results/
    mldb_data/<namespace>/models/
    mldb_data/<namespace>/evaluation_results/

Use Evaluation Result metrics for scientific comparison. The Study Result's `runtime_registry_version` is the canonical environment-provenance handle for that run; the immutable registry snapshot reconstructs the package set, so canonical results do not duplicate a full package dump. ClearML telemetry is for understanding execution behavior, not for replacing canonical result metrics.

## 8. Known-good telemetry evidence

W010 actual closure proved:

- detector Training: `validation/{f1, recall, mean_iou, loss}` at steps 1, 2, 3;
- detector Evaluation: validated metrics under the exact Evaluation stage name at step 0;
- classifier Training: `optimization/cross_entropy_loss` at steps 1, 2, 3;
- classifier Evaluation: validated metrics under `angle-robustness` at step 0.

Those names are Protocol-specific examples, not generic MLDB requirements. See `TELEMETRY.md` before authoring a new Protocol.

## 9. Operational invariants

- Do not manually edit StudyResult, TrainingResult, EvaluationResult, Model, or artifact identities to repair a run.
- Do not treat ClearML Task status as canonical experiment truth.
- Do not commit `.env` or secrets.
- Do not merge v1 SSH-worker procedures into the v2 ClearML path.
- Do not resolve runtime-registry `latest` separately per stage; one Study Result owns one pinned version.
- Do not add package/environment declarations to experiment YAML to bypass the global registry.
- Do not advance a worker runtime marker after partial/failed convergence.
- Do not modify MLDB core for an experiment unless the existing surface demonstrably blocks that experiment.
