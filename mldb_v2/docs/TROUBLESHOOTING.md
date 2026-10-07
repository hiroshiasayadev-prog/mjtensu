# MLDB v2 Troubleshooting Runbook

Use this document symptom-first. Do not repair canonical records by hand to make an execution look successful.

Start every investigation with:

    git status --short
    .\mldb.cmd doctor
    .\mldb.cmd status <study-result-id>
    .\mldb.cmd logs <study-result-id> --failed

Then identify the first boundary that failed: planning/source pinning, admission, queue/worker, source checkout, executable integrity, S3/corpus, CUDA/Protocol execution, telemetry, artifact publication, candidate collection, or canonical acceptance.

## `source_not_pinned`

Meaning: one or more formal v2 inputs consumed by the Study do not match the selected Git commit. This is a required safety gate, not a ClearML outage.

Recovery:

1. Confirm current `HEAD` and branch.
2. Determine the exact referenced canonical inputs and same-basename executable companions for the Study.
3. Compare required working-tree bytes against `HEAD`; ignore unrelated dirty files.
4. Stage only required source paths. Never use `git add .` merely to clear the error.
5. Review the staged file list and diff.
6. Commit and push to a branch/ref ClearML can fetch.
7. Stay on that commit and run `mldb plan` again.
8. Confirm the persisted Plan `source_commit` equals the new commit.

Do not attach the uncommitted patch to the ClearML Task as a workaround.

## Task remains queued / worker does not pick it up

Check in this order:

1. ClearML queue configured by `MLDB_V2_CLEARML_QUEUE` (validated deployment: `default`).
2. Worker registration/activity. For registry-enforcement validation, use `precision5820-gpu3060`; legacy/default workers may still be active on `default`.
3. That the agent is actually attached to the intended queue.
4. Whether an earlier Task is still occupying the worker/GPU.
5. Admission metadata/ownership rather than resubmitting blindly.

A retry must not create a second logical stage just because admission status was ambiguous.

## ClearML Task fails before the MLDB harness runs

Typical causes are source checkout, package/container setup, Docker arguments, runtime-registry bootstrap, or credentials. Verify the pinned Git commit is remote-reachable, repository URL is correct, the configured Docker image exists, and ClearML credentials/endpoints are present in the launching process.

For a GPU Task, verify that the controller resolved the pinned `MLDB_RUNTIME_REGISTRY_VERSION` to a READY `gpu-cu124` image and that ClearML's `docker_cmd` uses the returned `repository@sha256:...` reference. The worker no longer mounts or mutates a managed runtime directory.

Use the Task logs. Do not classify this as a Training Protocol failure until the harness actually enters domain execution.

## Runtime registry / materialized image failure

First distinguish run-level pinning from image materialization. The Study Result should contain one positive `runtime_registry_version`, and every StageInput for that Study should carry the same value. If a stage shows a different value, treat it as a lifecycle/provenance defect; do not resolve `latest` again to repair it.

For a GPU stage, check:

1. the pinned snapshot exists through `GET /?version=N`;
2. `GET /image?version=N&profile=gpu-cu124` is READY rather than BUILDING/FAILED;
3. the returned reference is digest-qualified (`repository@sha256:...`);
4. the ClearML Task `docker_cmd` uses exactly that reference;
5. the GPU parent worker reports compiled bootstrap v1.0.6 with a host-visible bootstrap directory;
6. Task setup reports that no pip version is configured, keeps the baked pip, and installs zero additional packages.

If an old image is MISSING, request the image endpoint and let the builder reconstruct it. Do not edit SQLite/materialization rows or canonical Study Results by hand.

Pre-registry historical Study Results may lack `runtime_registry_version`. They remain readable, but reproducible rerun/recovery must not silently substitute current `latest`.

## S3 / artifact / corpus failure

Check `MLDB_S3_ENDPOINT_URL`, region, bucket or artifact prefix, and one valid credential path (`AWS_*` or MinIO fallback). Confirm the launching process loaded `.env`; the supported `mldb.cmd` wrapper loads the repository-root `.env` automatically, while direct `python -m mldb_v2.src.cli` invocation does not.

Do not change artifact URIs or hashes in canonical records to match whatever bytes happen to exist. Object bytes must satisfy the formal URI/size/SHA contract.

## CUDA / GPU failure

Separate scheduling from domain execution. A Task reaching a worker does not prove CUDA is available inside the Task container.

Check the digest-qualified runtime image, GPU selector, worker Docker/NVIDIA runtime, and Protocol-specific requirements. GPU availability still depends on the selected worker and Docker GPU route; materializing a registry image does not provision a GPU device by itself.

Classifier verification can require `cache_device=cuda` to prevent silent CPU fallback. The rotated detector Protocol already requires CUDA.

## Training fails with zero accepted telemetry

A successful Train Protocol is not allowed to return a trained module while emitting no accepted scalar observations.

Check whether the current Protocol version calls `context.telemetry.report_scalar(...)`, whether the call is on every successful path, and whether its values satisfy strict scalar validation. Do not weaken the runtime enforcement to preserve an old silent fixture/Protocol; update the Protocol semantics/version instead.

## Evaluation fails with zero accepted telemetry

Validated numeric Evaluation candidate metrics are automatically projected. Therefore first check whether the EvaluationCandidate actually declares/returns valid numeric metrics.

Artifact-only Evaluation must explicitly report at least one useful scalar. Invalid metrics must not be accepted as telemetry merely to satisfy the event count.

## ClearML shows `No chart data`

Trace the path in order:

1. Protocol emits useful scalar events (or Evaluation returns validated numeric metrics).
2. Runtime accepts the event and successful-stage telemetry count is non-zero.
3. `CommonExecutionHarness` receives/threads the telemetry sink.
4. ClearML remote harness binds the sink to the current Task.
5. Logger mapping preserves group/title, series, value, and step/iteration.
6. Final Task flush completes.

Do not start by changing metric names. W010 proved the generic plumbing with classifier `optimization/cross_entropy_loss`, detector `validation/{f1,recall,mean_iou,loss}`, and automatic Evaluation metrics.

A ClearML reporting outage after acceptance is operational loss; it must not invalidate an otherwise valid canonical result.

## Backend Pipeline/Task completed but canonical Study is not terminal

Read canonical `status` plus backend Pipeline/child observations. A green backend node is not canonical acceptance. Use `mldb resume <study-result-id>` for normal continuation or `mldb advance <study-result-id>` for one explicit idempotent reconciliation pass.

Do not edit StudyResult/TrainingResult/EvaluationResult YAML to terminal status manually. If candidate acceptance fails, investigate the exact result/artifact/source validation failure. Under the W011 mapping, also verify that the controller did not release a dependent Evaluation before the accepted Training Result/Model semantic gate.

## Local process was interrupted

Under the W011 target mapping, interrupting `run`/`resume` stops local canonical reconciliation but does not cancel the ClearML Pipeline; backend work may continue. Inspect `mldb status`, then `resume` the existing Study Result to reconnect/reconcile. Use `cancel` only when cancellation is actually intended.

The W011 Pipeline mapping is implemented. Interruption recovery should reconnect to the existing Study Result/Pipeline and therefore preserve its pinned runtime-registry version.

## Historical result looks wrong

Do not mutate historical sealed definitions or completed canonical results. Determine whether the issue is a definition bug, Protocol bug, data issue, or interpretation issue. Fix it in a new version/Study and produce a new execution.

## When to change MLDB core

Only after reproducing a blocker where a requested experiment cannot be represented or executed through the existing public surface. Environment mistakes, missing commits, bad Protocol code, queue state, or broken source definitions are not automatically core defects.