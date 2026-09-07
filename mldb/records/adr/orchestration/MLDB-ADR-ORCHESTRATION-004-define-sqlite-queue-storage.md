# MLDB-ADR-ORCHESTRATION-004: Define SQLite Queue storage

- **status**: accepted
- **date**: 2026-09-04
- **depends_on**: MLDB-ADR-ORCHESTRATION-002, MLDB-ADR-ORCHESTRATION-003
- **supersedes**:
- **migrated_to_spec**: 2026-09-04

## Context

MLDB-ADR-ORCHESTRATION-003 selects Controller-local SQLite as the durable v1 Queue backend and defines logical job states, attempts, leases, and reconciliation authority.

Implementation now needs a concrete persistence contract that preserves those semantics without duplicating immutable Study Run plan inputs into mutable Queue storage.

The current dependency graph is intentionally small. A training-derived Evaluation job depends on exactly one Training job for the same Study Run trial, while an existing-Model Evaluation job has no Queue dependency. The project does not need a generic dependency graph table, persistent Worker registry, or distributed Controller coordination in v1.

Queue state must survive ordinary Controller restart, support one active attempt per logical job, and permit efficient lookup of ready work and expired leases. Queue storage remains operational and rebuildable rather than canonical experiment history.

## Decision

Persist v1 Queue state at:

```text
.local/mldb/queue.sqlite
```

relative to the repository root.

The database is non-canonical operational state and remains outside `mldb_data/`.

Use exactly two application tables in v1:

```text
jobs
attempts
```

Use SQLite `PRAGMA user_version = 1` for the Queue schema version rather than adding a metadata table.

### SQLite runtime settings

Every Controller connection that mutates or reconciles Queue state must enable:

```sql
PRAGMA foreign_keys = ON;
```

The v1 Queue database uses WAL journaling and full synchronous durability:

```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = FULL;
```

Connection busy timeout and other performance tuning are implementation choices.

### Timestamp representation

Persist Queue timestamps as UTC RFC 3339 text with exactly six fractional-second digits and a `Z` suffix:

```text
YYYY-MM-DDTHH:MM:SS.ffffffZ
```

Example:

```text
2026-09-04T10:00:14.123456Z
```

This fixed UTC representation is lexicographically sortable for Queue scheduling comparisons.

### Jobs table

`jobs` stores one logical Study Run work coordinate and its current operational progress.

The required logical columns are:

```text
job_id
study_run_id
trial_id
kind
stage
status
dependency_job_id
retry_not_before
created_at
updated_at
```

`job_id` is an SQLite integer primary key and is only an operational Queue identity.

`kind` is exactly:

```text
training
evaluation
```

`status` is exactly the lifecycle defined by ORCHESTRATION-003:

```text
blocked
ready
active
retry_wait
satisfied
failed
cancelled
```

Training jobs require `stage IS NULL`.

Evaluation jobs require a non-null `stage`.

A training job has no dependency job.

A training-derived Evaluation job references the Training job for the same Study Run and trial through `dependency_job_id`.

An existing-Model Evaluation job has `dependency_job_id IS NULL`.

`retry_not_before` is non-null only while the job is `retry_wait`. An immediately retryable job may use the current timestamp and be promoted to `ready` without an artificial delay.

The database enforces one logical Training job per Study Run and trial, and one logical Evaluation job per Study Run, trial, and stage.

### Attempts table

`attempts` stores concrete Worker execution attempts that reached Run allocation.

The required logical columns are:

```text
attempt_id
job_id
attempt_no
run_id
worker_id
lease_token
lease_until
started_at
finished_at
close_reason
```

`attempt_id` is an SQLite integer primary key and is only an operational Queue identity.

`attempt_no` is a positive per-job sequence beginning at 1.

`run_id` is the unique Training Run or Evaluation Run allocated for the attempt. Its Run kind must match the parent job kind.

`worker_id` identifies the Worker selected for the attempt. Worker registration is not stored in a separate v1 table.

`lease_token` is an opaque unique identity for the active authorization of that attempt.

`lease_until` is the current expiry timestamp and may be extended by heartbeat while the attempt remains active.

`finished_at IS NULL` means the attempt remains open. At most one open attempt may exist for a job.

`close_reason` is optional concise operational metadata. V1 does not standardize a complete close-reason vocabulary because canonical child Run status and failure information remain authoritative.

A concrete preflight failure before Run allocation creates no `attempts` row.

### No duplicated execution payload

Queue tables do not persist independent copies of:

- Architecture ID;
- Corpus ID;
- Train Protocol ID;
- Evaluation Protocol ID;
- training seed;
- resolved public parameters;
- existing Model ID;
- Evaluation stage inputs beyond the Study-local `stage` coordinate.

Those execution inputs remain authoritative in the immutable Study Run plan and referenced MLDB records.

### No Worker table

V1 does not persist a Worker registry table.

Worker identity and lease ownership needed to recover an active attempt live on the attempt row. Capability advertisement, Worker session behavior, and availability discovery belong to the Worker API contract.

A persistent Worker registry may be introduced later if concrete scheduling requirements need it.

### No generic dependency table

V1 does not create a `job_dependencies` table.

`dependency_job_id` is sufficient because the only Queue-level dependency is one Evaluation job waiting for its trial Training job.

A future workflow requiring multiple upstream dependencies may replace or extend this representation through a Queue schema revision.

### Queue admission transaction

After a complete valid Study Run plan exists, Controller inserts the complete logical job set for that plan in one SQLite transaction.

The transaction must commit either the complete derived job set or none of it.

Coordinate uniqueness makes repeated admission or reconciliation idempotent when the existing rows match the immutable plan.

An existing coordinate whose stored job kind, stage, or dependency relationship disagrees with the immutable plan is an inconsistency to reconcile or reject rather than silently overwrite as a different job.

### Attempt-start ordering

V1 supports one Controller process as the Queue writer and dispatcher.

Controller must serialize the dispatch-start critical section so concurrent Worker pull requests cannot start two attempts for one `ready` job.

For one selected job, the ordering is:

```text
select eligible job
  -> concrete preflight
  -> allocate child Run ID
  -> persist schema-valid running child Run
  -> insert attempt row
  -> move job to active
  -> commit Queue transaction
  -> return assignment to Worker
```

The attempt insert and `job -> active` mutation occur in one SQLite transaction.

Controller does not return the assignment before that transaction commits.

A crash after child Run creation but before the Queue transaction commits leaves a `running` Run without recoverable active lease ownership; reconciliation resolves it as an interrupted attempt before another Run is scheduled.

A crash after Queue commit but before the Worker receives the assignment leaves an `active` leased attempt that can expire and be reconciled normally.

### Result-close ordering

Canonical MLDB acceptance precedes Queue closure.

After Controller accepts a successful child result, it first finalizes canonical MLDB state and only then closes the attempt and moves the job to `satisfied` in one Queue transaction.

For Training, canonical acceptance includes the completed Training Run and required Model creation.

For Evaluation, canonical acceptance requires `completed`; `completed_partial` does not satisfy the job.

If Controller crashes after canonical completion but before Queue closure, startup reconciliation repairs the Queue from canonical history without rerunning the accepted work.

## Rationale

Two tables are sufficient because immutable Study Run plans already own execution intent while Queue only needs current logical progress and concrete attempt ownership.

A single nullable dependency foreign key matches the v1 dependency graph without introducing a generic DAG persistence model before it is needed.

Avoiding a Worker registry keeps Worker identity operational and localized to active attempts while the Worker API remains undecided.

Partial unique indexes can enforce coordinate identity and one-open-attempt invariants directly in SQLite.

WAL plus full synchronous durability provides a strong local durability default for a Queue whose primary purpose is surviving Controller restart, while canonical MLDB history remains the final reconciliation authority.

Using `PRAGMA user_version` preserves the two-table application model while still giving future schema migrations an explicit version boundary.

## Rejected alternatives

### Store resolved execution payload in `jobs`

The immutable Study Run plan already contains the exact execution intent. Copying those values into Queue would create two mutable representations that could disagree.

### Add a persistent Worker table immediately

V1 only needs Worker identity and lease ownership for concrete attempts. Capability discovery and session management are better designed with the Worker API rather than guessed into the storage schema.

### Add a generic many-to-many dependency table

Current Evaluation work has at most one Training dependency. A generic dependency graph would add complexity with no present workflow requiring it.

### Use timestamps as local time

Queue scheduling and lease comparison must not depend on host timezone or daylight-saving behavior. Fixed UTC text avoids that ambiguity.

### Put Queue SQLite under `mldb_data/`

`mldb_data/` contains canonical MLDB entity and artifact state. Queue is operational and rebuildable, so it belongs in local runtime storage instead.

### Add a third metadata table for schema version

SQLite already supplies `PRAGMA user_version`, so another application table is unnecessary.

## Consequences

The Queue storage format contract can define exact v1 DDL and indexes for `jobs` and `attempts`.

Controller implementation needs a repository-local `.local/mldb/` runtime directory but must not treat it as canonical MLDB data.

Queue admission, attempt start, attempt closure, retry promotion, lease update, and reconciliation must preserve the defined transaction and authority boundaries.

A future multi-Controller deployment would require another coordination decision because v1 assumes one Controller process owns Queue dispatch mutation.

A future workflow with multiple Queue dependencies or persistent Worker scheduling state may require a schema revision while leaving immutable Study Run plans and child Run history unchanged.

## Evidence

Study Run plans provide deterministic Training and Evaluation coordinates and already exclude operational queue fields.

ORCHESTRATION-003 requires durable job state, one active leased attempt, and reconciliation from canonical MLDB history.

The repository already ignores `.local/`, making it appropriate for non-canonical machine-local runtime state without coupling Queue bytes to versioned MLDB definitions.
