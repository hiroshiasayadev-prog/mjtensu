# MLDB v2 Operations Manual

This is the day-to-day runbook for executing existing MLDB v2 Studies from the Windows development repository through ClearML to the GPU worker.

> W011 transition note (2026-09-17): the approved target maps one MLDB Study Result to one ClearML Pipeline Run and delegates physical child-Task scheduling/retry/liveness to ClearML. The current runtime still uses the earlier flat stage-Task mapping until W011 implementation closes. See `CLEARML_PIPELINES.md` before changing backend execution code.

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
| Canonical CPU-latency queue | `latency-cpu` |
| Validated general/GPU worker | `bugrat-gpu0` |
| Canonical latency worker | `bugrat-gpu0` (same single worker process; `latency-cpu` has higher queue priority) |
| Validated GPU | NVIDIA GeForce RTX 3090, 24 GB |
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

`onnx-cpu-latency` is a benchmark stage, not ordinary throughput work. Route it only to `latency-cpu`. The canonical Linux worker `bugrat-gpu0` subscribes to `latency-cpu` first and `default` second. Because it is one ClearML Agent process, it executes at most one MLDB Task at a time on that host, so a default Evaluation cannot overlap a latency benchmark there. Do **not** subscribe a Windows development worker, another CPU model, or any heterogeneous machine to `latency-cpu`; doing so changes benchmark hardware and invalidates direct historical/model comparison. Additional workers may subscribe to `default` for ordinary Training/Evaluation once they have the required runtime image and data access.

Queue isolation fixes worker identity and prevents same-agent overlap, but unrelated host processes can still perturb timing. For comparable latency numbers, keep other CPU-heavy host workloads away from the benchmark window. If host contention cannot be controlled, treat the latency values as non-comparable and rerun under controlled load. Keep the Protocol's batch size, ORT provider, intra/inter-op thread counts, execution mode, and runtime versions fixed as well.

The prebuilt task image `mldb-clearml-runner:torch2.5.1-cu124-v1` contains the pinned MLDB remote runtime dependencies. `MLDB_V2_CLEARML_PREBUILT_RUNTIME=true` tells the ClearML task launcher to reuse `/opt/conda/bin/python` instead of building a fresh virtualenv; Task requirements remain declared and are reconciled against that interpreter. Rebuild the image when `_REMOTE_PACKAGES` changes.

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

`run` plans/compiles the Study and creates a fresh durable Study Result. Under the W011 target mapping it then creates/recover one ClearML Pipeline Run for that Study Result; ClearML owns child-Task scheduling while the foreground MLDB process reconciles canonical results until terminal. Interrupting the local process does not mean cancellation; the backend Pipeline may continue. Use `resume` to reconnect/reconcile or `cancel` to request cancellation.

## 6. Monitor and recover

Useful read/control commands:

    .\mldb.cmd ps --all
    .\mldb.cmd status <study-result-id>
    .\mldb.cmd watch <study-result-id>
    .\mldb.cmd logs <study-result-id> --failed
    .\mldb.cmd resume <study-result-id>
    .\mldb.cmd advance <study-result-id>
    .\mldb.cmd cancel <study-result-id>

`watch` is read-only. `advance` performs one explicit canonical reconciliation pass and is mainly for recovery/debugging. `rerun` creates a fresh Study Result from the immutable Plan of an earlier execution rather than recompiling the current mutable Study definition.

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

Use Evaluation Result metrics for scientific comparison. ClearML telemetry is for understanding execution behavior, not for replacing canonical result metrics.

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
- Do not modify MLDB core for an experiment unless the existing surface demonstrably blocks that experiment.
