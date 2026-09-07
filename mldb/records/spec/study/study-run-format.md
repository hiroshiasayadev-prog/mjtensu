# Contract: Study Run format

- **id**: `spec:mldb.study.study_run_format`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.study`
- **contract_class**: `format`

## What this is

Defines `run.yaml` for one concrete execution of one sealed Study.

Study Run records Study-level lifecycle and immutable plan integrity. Child Training Runs and Evaluation Runs remain authoritative for their own execution facts and results.

## Current contract

Study Run metadata uses:

```text
mjtensu.mldb/study-run/v1
```

Canonical placement is:

```text
mldb_data/study_runs/<study-run-id>/run.yaml
```

Example:

```yaml
schema: mjtensu.mldb/study-run/v1
id: sr-20260904-001
status: completed_with_failures
study: tile-classifier-lr-search-v1

execution:
  started_at: 2026-09-04T10:00:00+09:00
  finished_at: 2026-09-04T15:42:11+09:00

plan:
  path: plan.jsonl
  sha256: 0123456789abcdef...
  bytes: 18492
  trials: 24
  evaluation_jobs: 48

summary:
  training:
    completed: 23
    failed: 1
    cancelled: 0
  evaluation:
    completed: 42
    completed_partial: 2
    failed: 2
    cancelled: 0
    blocked: 2
```

## Rules

Every Study Run requires:

| field | contract |
|---|---|
| `schema` | Exactly `mjtensu.mldb/study-run/v1`. |
| `id` | Study Run event ID `sr-YYYYMMDD-NNN`. |
| `status` | Study Run lifecycle state. |
| `study` | Exactly one sealed Study ID. |
| `execution.started_at` | Study Run execution start timestamp. |

Once complete plan materialization succeeds, the Study Run additionally requires:

| field | contract |
|---|---|
| `plan.path` | Exactly `plan.jsonl` in v1. |
| `plan.sha256` | SHA-256 of the complete immutable plan. |
| `plan.bytes` | Plan byte size. |
| `plan.trials` | Number of Study-local trial rows in the plan, regardless of whether Models come from training or `model.existing`. |
| `plan.evaluation_jobs` | Total number of planned evaluation stage instances across all trials. |

`completed` and `completed_with_failures` always require a complete valid plan and therefore require all plan fields.

A `failed` or `cancelled` Study Run may omit plan fields only when no complete valid plan was established before termination. A partial plan file does not satisfy the plan contract.

If a complete valid plan was established before a later failure or cancellation, the plan and all plan metadata remain required and immutable.

A terminal Study Run additionally requires `execution.finished_at`.

Study Run event IDs do not use terminal `-vN` revision syntax.

### Summary

`summary` is optional derived information.

A summary may include evaluation completed/completed_partial/failed/cancelled/blocked counts. A training-derived Study may additionally include training completed/failed/cancelled counts; an existing-Model Study has no planned training coordinates.

These counts describe final outcomes of planned coordinates, not the raw number of historical child Run attempts. A failed attempt followed by a successful retry contributes to the successful final coordinate count while the failed child Run remains historical evidence.

The summary is not more authoritative than the immutable plan and child Run records. A summary disagreement must not redefine child history.

### Child lineage

Study Run does not maintain an authoritative mutable list of child Run IDs.

Training-derived trials create Training Runs that reference Study Run and trial through their own lineage fields. Evaluation Runs reference Study Run, trial, and evaluation stage for both Study Model-source modes.

A Model created by Study training requires no additional Study Run field because it inherits lineage through its originating Training Run. An existing Model selected by `model.existing` remains immutable input and is not modified to acquire Study lineage.

### Metrics

Study Run does not store authoritative Evaluation Run metrics, rankings, best-model selections, or aggregate model scores.

Those are derived views over child execution records.

## Validation rules

- Reject unsupported `schema`.
- Reject directory name and `id` disagreement.
- Reject an ID outside `sr-YYYYMMDD-NNN`.
- Reject a status outside the Study Run lifecycle contract.
- Reject a Study Run that references a non-sealed Study for execution.
- Reject a `plan.path` other than `plan.jsonl` in v1.
- Reject missing plan path, hash, byte size, trial count, or evaluation-job count after valid plan materialization.
- Reject `completed` or `completed_with_failures` without a complete valid plan.
- Permit `failed` or `cancelled` to omit plan metadata only when plan materialization never completed.
- Reject omission of plan metadata from a `failed` or `cancelled` Study Run when a complete valid plan had already been established.
- Reject plan integrity metadata that disagrees with the persisted complete plan.
- Reject `plan.trials` that disagrees with the number of plan rows.
- Reject `plan.evaluation_jobs` that disagrees with the sum of evaluation entries in the plan.
- Reject a terminal Study Run without `execution.finished_at`.
- Reject mutation of terminal Study Run historical facts.

## Boundary

| concern | owner |
|---|---|
| Study definition fields | `spec:mldb.study.study_format`. |
| `plan.jsonl` row schema | `spec:mldb.study.plan_format`. |
| Grid expansion and parameter resolution | `spec:mldb.study.grid_expansion`. |
| Status transitions and child failure interpretation | `spec:mldb.study.study_run_lifecycle`. |
| Child Training Run details | `spec:mldb.training`. |
| Child Evaluation Run details | `spec:mldb.evaluation`. |
| Queue job state | `spec:mldb.orchestration`. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.study` | Parent Study overview. |
| `spec:mldb.study.plan_format` | Defines the plan whose integrity metadata is recorded here. |
| `spec:mldb.study.study_run_lifecycle` | Defines Study Run status semantics. |
