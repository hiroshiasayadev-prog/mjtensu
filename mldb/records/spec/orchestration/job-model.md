# Concept: MLDB orchestration job model

- **id**: `spec:mldb.orchestration.job_model`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.orchestration`

## What this is

Defines the logical compute-work units derived from one complete Study Run plan and the boundary between Queue jobs and immutable MLDB Run attempts.

This concept fixes job kinds and dependencies without fixing the Queue persistence schema or detailed job state machine.

## Concept model

V1 has two logical compute-work kinds:

```text
training
evaluation
```

All queued compute work is derived from a complete valid Study Run `plan.jsonl`.

Training-derived trial:

```text
Study Run / trial
    ↓
training job
    ↓ completed Training Run
Model
    ↓
evaluation job / stage
```

Existing-Model trial:

```text
Study Run / trial / existing Model
    ↓
evaluation job / stage
```

Worker processes one Training or Evaluation execution attempt at a time. Worker does not process an entire Study or entire trial as one compute unit.

## Rules

### Queue admission

Study execution must pass complete Study validation before Study Run allocation.

Queue jobs may be derived only after complete valid Study Run plan materialization succeeds.

| condition | Queue consequence |
|---|---|
| Study validation fails before Study Run allocation | Create no Queue jobs. |
| Study Run plan materialization fails before a complete plan exists | Create no Queue jobs from the partial plan. |
| Complete valid plan exists | Derive the logical jobs represented by that plan. |

Queue does not independently author Architecture, Corpus, Protocol, seed, parameter, Model, or evaluation-stage intent that disagrees with the immutable plan.

### Training job

A training job exists for each plan row containing `training`.

Its durable execution intent is identified by:

```text
Study Run
+ trial
```

The exact resolved training inputs are read from the immutable plan.

A training job has no corresponding Training Run while it is merely waiting for capacity or dependency-free scheduling.

Before executable invocation, Controller confirms the required concrete preflight, allocates a new Training Run ID, and creates the schema-valid `running` Run record.

Retry creates another Training Run attempt for the same logical job and Study lineage.

### Evaluation job

An evaluation job exists for every plan row and every evaluation stage in that row.

Its durable execution intent is identified by:

```text
Study Run
+ trial
+ stage
```

For a training-derived row, the job is not runnable until the trial has a completed Training Run and resulting Model.

For an existing-Model row, the immutable plan already identifies the Model, so no training dependency exists.

Before executable invocation, Controller confirms the required concrete preflight, allocates a new Evaluation Run ID, and creates the schema-valid `running` Run record.

Retry creates another Evaluation Run attempt for the same logical job and Study lineage.

### Job and Run identity

Queue job identity and MLDB Run identity are different.

One logical job may have zero, one, or multiple historical Run attempts.

| situation | relationship |
|---|---|
| Waiting Queue job | No Run is required. |
| First execution attempt | New Training Run or Evaluation Run. |
| Retry after terminal unsuccessful attempt | New Run ID for the same logical job coordinate. |
| Controller-accepted successful attempt | Planned coordinate becomes satisfied. |

Worker-local completion is not sufficient to satisfy a job. Controller must accept and commit the corresponding canonical Run result.

A previous failed, cancelled, or partial Run remains historical evidence after a later retry succeeds.

### Recovery boundary

Queue progress is operational, but recovery must preserve already accepted MLDB work.

Controller restart or Queue reconciliation must use:

- the immutable Study Run plan;
- completed or terminal Training Runs with matching Study lineage;
- Models produced by completed training-derived trials;
- existing Models named directly by existing-Model plan rows;
- Evaluation Runs with matching Study lineage.

Completed training must not be repeated merely because downstream evaluation remains unfinished.

Durable Queue state names, lease semantics, stale-attempt handling, and reconciliation authority are defined by `spec:mldb.orchestration.queue_lifecycle`. Exact SQLite DDL and transaction layout remain a later storage-format contract.

## Boundary

| concern | owner |
|---|---|
| Study source and immutable plan rows | `spec:mldb.study`. |
| Controller / Queue / Worker authority | `spec:mldb.orchestration.responsibility_model`. |
| Training Run lifecycle | `spec:mldb.training.training_run_lifecycle`. |
| Evaluation Run lifecycle | `spec:mldb.evaluation.evaluation_run_lifecycle`. |
| Durable Queue state machine and reconciliation | `spec:mldb.orchestration.queue_lifecycle`. |
| Exact SQLite persistence schema | `spec:mldb.orchestration.queue_storage_format`. |
| Worker wire payload | Future Worker API contract. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.orchestration` | Parent orchestration overview. |
| `spec:mldb.orchestration.responsibility_model` | Defines ownership of Queue and Worker behavior. |
| `spec:mldb.orchestration.queue_lifecycle` | Defines durable job states, leases, and restart reconciliation. |
| `spec:mldb.orchestration.queue_storage_format` | Defines the concrete v1 SQLite representation of jobs and attempts. |
| `spec:mldb.study.plan_format` | Provides authoritative job intent. |
| `spec:mldb.study.study_run_lifecycle` | Defines plan retry and reconciliation semantics. |
