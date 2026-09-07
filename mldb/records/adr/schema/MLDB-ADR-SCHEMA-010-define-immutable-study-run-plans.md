# MLDB-ADR-SCHEMA-010: Define immutable Study Run plans

- **status**: accepted
- **date**: 2026-09-04
- **depends_on**: MLDB-ADR-SCHEMA-005, MLDB-ADR-SCHEMA-006, MLDB-ADR-SCHEMA-008, MLDB-ADR-SCHEMA-009
- **supersedes**:
- **migrated_to_spec**:

## Context

Study defines a reusable grid experiment but does not record one concrete execution of that Study.

The same sealed Study may be executed multiple times. Each execution needs a stable identity, a frozen record of the exact expanded trial plan, lifecycle status, and a lineage key that child Training Runs and Evaluation Runs can use to identify which Study execution and trial they belong to.

The concrete expanded plan must be persisted rather than reconstructed later from queue state. Queue implementations may change, crash, retry jobs, or be deleted entirely, but MLDB must still be able to answer what the Study intended to execute and how each produced Run relates to that plan.

Failure handling must also be explicit. One failed training or evaluation job must not stop independent trials. A Study Run that reaches the end of its plan with some failed or blocked jobs is meaningfully different from an orchestration failure that prevented the plan from being executed at all.

## Decision

Introduce `StudyRun` as the immutable execution-history entity for one materialization and orchestration of one sealed Study.

A Study Run owns:

- one Study reference;
- one materialized immutable trial plan;
- Study-level execution timing and lifecycle;
- derived progress/summary information where useful.

The individual Training Runs and Evaluation Runs remain the authoritative execution records for their own work and results.

### Physical placement

Study Runs live under:

```text
mldb_data/
  study_runs/
```

Each Study Run occupies one directory:

```text
mldb_data/study_runs/<study-run-id>/
  run.yaml
  plan.jsonl
```

Example:

```text
mldb_data/study_runs/sr-20260904-001/
  run.yaml
  plan.jsonl
```

`run.yaml` is the Study Run lifecycle record.

`plan.jsonl` is the immutable materialized plan for the execution.

### Study Run ID

Study Run is an execution event and does not use `-vN` revision syntax.

The initial ID form is:

```text
sr-YYYYMMDD-NNN
```

where the date is the local date on which the Study Run is allocated and `NNN` is a zero-padded per-date sequence number.

Examples:

```text
sr-20260904-001
sr-20260904-002
```

Re-running the same Study receives another Study Run ID.

### Lifecycle

Study Run status is one of:

```text
running
completed
completed_with_failures
failed
cancelled
```

A Study Run is created as `running` after its Study has been validated and its plan materialization has begun.

Terminal states are:

```text
completed
completed_with_failures
failed
cancelled
```

A terminal Study Run is immutable.

The statuses mean:

- `completed`: all planned training and evaluation work that was runnable completed successfully;
- `completed_with_failures`: orchestration reached the end of the plan, but one or more child Runs failed, were cancelled, or were blocked by failed dependencies;
- `failed`: Study-level planning or orchestration failed in a way that prevented the materialized plan from being carried through normally;
- `cancelled`: the Study Run was intentionally stopped before completion.

A single child Run failure must not automatically set Study Run to `failed` and must not stop independent work.

### Metadata format

Study Run metadata uses:

```text
mjtensu.mldb/study-run/v1
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
    completed: 44
    failed: 2
    cancelled: 0
    blocked: 2
```

Required fields are:

- `schema`;
- `id`;
- `status`;
- `study`;
- `execution.started_at`;
- `plan.path`;
- `plan.sha256`;
- `plan.bytes`;
- `plan.trials`;
- `plan.evaluation_jobs`.

Terminal Study Runs additionally require `execution.finished_at`.

`summary` is optional and is derived information. It must never be treated as more authoritative than the child Training Run and Evaluation Run records.

### Plan materialization

The Study Run plan is created by deterministically expanding the referenced sealed Study.

For each training grid coordinate, Study Run assigns one Study-local trial ID:

```text
trial-0001
trial-0002
...
```

Trial IDs are stable only within one Study Run.

`plan.jsonl` contains exactly one JSON object per training trial.

Example:

```json
{"trial":"trial-0001","training":{"architecture":"plain-cnn-v1","corpus":"gray35-train-v1","protocol":"tile-classifier-standard-v1","seed":42,"parameters":{"epochs":150,"batch_size":512,"learning_rate":0.001,"weight_decay":0.0001}},"evaluations":[{"stage":"final-holdout","corpus":"gray35-final-holdout-v1","protocol":"tile-classifier-standard-eval-v1","parameters":{"batch_size":4096}},{"stage":"real-holdout","corpus":"gray35-real-holdout-v1","protocol":"tile-classifier-robustness-eval-v1","parameters":{"batch_size":2048}}]}
```

Every parameter mapping stored in the plan is fully resolved. Omitted protocol parameters have already been replaced by their sealed protocol defaults.

The plan therefore records exactly what the Study Run intends to execute without requiring later lookup of historical default resolution behavior.

### Plan integrity and immutability

Once materialization completes successfully, `plan.jsonl` is immutable and its SHA-256 and byte size are recorded in `run.yaml`.

Queue workers do not mutate the plan to record progress.

Execution progress, retries, claims, and scheduling state belong to queue infrastructure and child Run records.

If plan generation itself fails before a valid complete plan can be materialized, the Study Run may terminate as `failed`. Any partial plan file must not be treated as a valid immutable plan.

### Child Run lineage

Training Runs created from a Study Run include Study lineage:

```yaml
study:
  run: sr-20260904-001
  trial: trial-0001
```

Evaluation Runs created for one planned evaluation stage include:

```yaml
study:
  run: sr-20260904-001
  trial: trial-0001
  stage: final-holdout
```

These references are downstream-to-upstream lineage.

Study Run does not maintain an authoritative mutable array of all child Run IDs. Child Runs can be discovered by their Study lineage.

Model requires no additional Study field because it resolves its originating Training Run and therefore inherits Study lineage transitively.

### Trial execution semantics

For each planned trial:

1. orchestration creates the Training Run using the exact materialized Architecture, Corpus, Train Protocol, seed, and resolved parameters;
2. if training completes, MLDB automatically creates the Model according to MLDB-ADR-SCHEMA-006;
3. each planned evaluation stage may then create an Evaluation Run for that Model;
4. evaluation stages for the same Model are independent siblings.

The child Run records remain responsible for their own lifecycle and result validity.

### Failure isolation

Study Run orchestration is not fail-fast across independent work.

If one Training Run fails:

- that trial produces no Model;
- evaluation jobs that require that Model become blocked;
- unrelated trials continue.

If one Evaluation Run fails:

- sibling evaluation stages for that Model continue if independently runnable;
- evaluations for other trials continue;
- no successful Model or Training Run is invalidated.

A blocked evaluation is an orchestration job state rather than a synthetic failed Evaluation Run. If an Evaluation Run was never started because its required Model does not exist, no Evaluation Run record needs to be created solely to represent the block.

At Study Run completion, the presence of failed, cancelled, or blocked child work results in `completed_with_failures` if orchestration itself successfully processed the plan to terminality.

### Derived summary

Study Run tooling may maintain or finalize a `summary` section for convenience.

Useful counts include:

- training completed/failed/cancelled;
- evaluation completed/failed/cancelled/blocked.

These counts are derived from plan coordinates, queue outcome, and child Run records.

A summary mismatch does not redefine child history. Tooling may regenerate a derived summary if needed as long as terminal historical facts are not changed.

### Queue boundary

Study Run defines the durable plan and lineage but does not standardize the queue implementation.

Operational queue data may include:

- job IDs;
- dependency edges;
- queued/claimed/running/blocked state;
- worker identity;
- attempt counts;
- leases and heartbeats;
- retry timing;
- priority.

This state is non-authoritative operational data and may live in SQLite, Redis, another database, or another scheduler.

Deleting or rebuilding the queue must not erase completed MLDB Training Run, Model, Evaluation Run, Study, or Study Run history.

### Reconciliation and idempotency

Orchestration should be restartable.

Given one Study Run plan, tooling may reconcile which planned coordinates already have child Runs and which jobs still need to be created or retried.

Creation of child work should therefore use the Study Run/trial/stage lineage to avoid accidentally creating duplicate jobs during scheduler restart when the intended behavior is resume.

Whether a user explicitly requests a new repeated execution of an already completed coordinate is a higher-level orchestration decision and should create new child Run IDs rather than mutate historical Runs.

### No metrics in Study Run

Study Run does not duplicate Evaluation Run metrics into its authoritative schema.

Rankings, best-model selection, Pareto analysis, aggregate tables, and visualization are derived views over child Evaluation Runs and Models.

External tools such as MLflow may mirror those values, but MLDB child Run records remain the source of truth.

## Rationale

Persisting a materialized plan makes a Study execution auditable independently of queue internals. The exact resolved parameter combinations and evaluation stages remain visible even if scheduler state disappears.

A separate Study Run allows the same sealed Study to be executed repeatedly without making the Study definition mutable.

Child-to-parent lineage keeps the relationship scalable without placing growing mutable Run-ID arrays into Study Run metadata.

Failure isolation matches the purpose of automated parameter sweeps: one bad Architecture/parameter combination should not throw away dozens of independent useful experiments.

`completed_with_failures` makes partial success explicit without conflating expected child failures with a Study-level orchestration failure.

Keeping aggregate metrics out of Study Run prevents another duplicate source of truth and allows MLflow or project-owned dashboards to remain replaceable views.

## Rejected alternatives

### Store only the Study reference and reconstruct the grid later

This would make historical execution dependent on future expansion code and protocol-default lookup. Persisting a resolved plan provides direct evidence of what the orchestration intended to execute.

### Store all child Run IDs in Study Run

The list would grow and mutate throughout execution and would reverse the established downstream-to-upstream lineage direction.

Child Runs instead identify their Study Run/trial/stage.

### Fail the entire Study Run when one child fails

Parameter sweeps are specifically intended to explore combinations, some of which may fail. Fail-fast semantics would discard independent useful work and make large sweeps fragile.

Independent jobs continue; the final Study status can be `completed_with_failures`.

### Create failed Evaluation Runs for blocked evaluations

No evaluation execution occurred when the required training Model was never produced. Creating a synthetic failed Evaluation Run would confuse dependency blocking with evaluator failure.

Blocked state belongs to orchestration, while Evaluation Run exists only when evaluation execution actually begins.

### Make queue storage authoritative MLDB data

Queue state is volatile operational machinery. Tying durable experiment identity to one scheduler database would make MLDB history fragile and harder to migrate.

The materialized Study Run plan and child Run records are authoritative instead.

## Consequences

Future Study Run tooling should be able to:

- allocate `sr-YYYYMMDD-NNN` IDs;
- validate the referenced Study is sealed;
- deterministically expand the Study grid;
- resolve all Train Protocol and Evaluation Protocol defaults;
- assign stable Study-local `trial-NNNN` identifiers;
- write and hash immutable `plan.jsonl`;
- create queue jobs from the plan;
- create Training Runs with `study.run` and `study.trial` lineage;
- wait for automatically generated Models before releasing dependent evaluation jobs;
- create Evaluation Runs with `study.run`, `study.trial`, and `study.stage` lineage;
- block only dependency-invalid jobs while continuing independent jobs;
- terminate normally as `completed_with_failures` when child work fails but plan orchestration itself succeeds;
- reconcile a Study Run after queue/worker restart from the immutable plan and existing child records;
- derive progress summaries without making them the source of truth.

With Study and Study Run defined, queue and worker implementation can be designed independently as execution infrastructure rather than additional MLDB schema entities.

## Evidence

The project's intended workflow is to vary Architecture, public training parameters, and seeds, automatically train every combination, create a Model for each successful Training Run, and evaluate those Models under common protocols. This naturally forms a repeated train -> Model -> evaluation DAG per trial.

MLDB-ADR-SCHEMA-005 and MLDB-ADR-SCHEMA-008 already isolate Training Run and Evaluation Run failures. Study Run extends the same principle across a larger experiment so one failed validation or training combination does not terminate unrelated queued work.
