# Contract: Queue SQLite storage format

- **id**: `spec:mldb.orchestration.queue_storage_format`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.orchestration`
- **contract_class**: `format`

## What this is

Defines the v1 SQLite persistence format for durable operational Queue state.

The database persists logical Study-derived jobs and concrete Worker attempts across Controller restart. It does not duplicate immutable Study Run execution payloads and is not authoritative experiment history.

## Current contract

The Queue database lives at:

```text
.local/mldb/queue.sqlite
```

relative to the repository root.

The file is machine-local operational state outside canonical `mldb_data/`.

V1 uses:

```sql
PRAGMA user_version = 1;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = FULL;
```

Every Controller connection that mutates or reconciles Queue state must also enable:

```sql
PRAGMA foreign_keys = ON;
```

V1 has exactly two application tables:

```text
jobs
attempts
```

### Timestamp format

Every Queue timestamp is UTC RFC 3339 text with exactly six fractional-second digits and a `Z` suffix:

```text
YYYY-MM-DDTHH:MM:SS.ffffffZ
```

Example:

```text
2026-09-04T10:00:14.123456Z
```

### V1 DDL

```sql
CREATE TABLE jobs (
    job_id INTEGER PRIMARY KEY,
    study_run_id TEXT NOT NULL,
    trial_id TEXT NOT NULL,
    kind TEXT NOT NULL
        CHECK (kind IN ('training', 'evaluation')),
    stage TEXT,
    status TEXT NOT NULL
        CHECK (status IN (
            'blocked',
            'ready',
            'active',
            'retry_wait',
            'satisfied',
            'failed',
            'cancelled'
        )),
    dependency_job_id INTEGER,
    retry_not_before TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,

    FOREIGN KEY (dependency_job_id)
        REFERENCES jobs(job_id)
        ON DELETE RESTRICT,

    CHECK (
        (kind = 'training' AND stage IS NULL AND dependency_job_id IS NULL)
        OR
        (kind = 'evaluation' AND stage IS NOT NULL)
    ),

    CHECK (
        (status = 'retry_wait' AND retry_not_before IS NOT NULL)
        OR
        (status <> 'retry_wait' AND retry_not_before IS NULL)
    )
);

CREATE UNIQUE INDEX uq_jobs_training_coordinate
    ON jobs(study_run_id, trial_id)
    WHERE kind = 'training';

CREATE UNIQUE INDEX uq_jobs_evaluation_coordinate
    ON jobs(study_run_id, trial_id, stage)
    WHERE kind = 'evaluation';

CREATE INDEX idx_jobs_study_status
    ON jobs(study_run_id, status);

CREATE INDEX idx_jobs_schedulable
    ON jobs(status, retry_not_before, job_id);

CREATE INDEX idx_jobs_dependency
    ON jobs(dependency_job_id, status)
    WHERE dependency_job_id IS NOT NULL;

CREATE TABLE attempts (
    attempt_id INTEGER PRIMARY KEY,
    job_id INTEGER NOT NULL,
    attempt_no INTEGER NOT NULL
        CHECK (attempt_no > 0),
    run_id TEXT NOT NULL UNIQUE,
    worker_id TEXT NOT NULL,
    acquire_token TEXT NOT NULL UNIQUE,
    lease_token TEXT NOT NULL UNIQUE,
    lease_until TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    close_reason TEXT,

    FOREIGN KEY (job_id)
        REFERENCES jobs(job_id)
        ON DELETE RESTRICT,

    UNIQUE (job_id, attempt_no)
);

CREATE UNIQUE INDEX uq_attempts_one_open_per_job
    ON attempts(job_id)
    WHERE finished_at IS NULL;

CREATE UNIQUE INDEX uq_attempts_one_open_per_worker
    ON attempts(worker_id)
    WHERE finished_at IS NULL;

CREATE INDEX idx_attempts_open_lease
    ON attempts(lease_until)
    WHERE finished_at IS NULL;

PRAGMA user_version = 1;
```

## Rules

### Database identity and initialization

A new Queue database is initialized only when the target database does not yet contain a Queue schema.

A v1 initialized database must report:

```text
PRAGMA user_version = 1
```

An unsupported nonzero `user_version` must not be silently treated as v1.

Future migration behavior belongs to a later schema-migration contract. V1 implementation may refuse mutation when it encounters an unsupported Queue schema version.

### Jobs

One `jobs` row represents exactly one logical work coordinate from one complete immutable Study Run plan.

| kind | coordinate identity | stage | dependency |
|---|---|---|---|
| `training` | `study_run_id + trial_id` | `NULL` | `NULL` |
| `evaluation`, training-derived | `study_run_id + trial_id + stage` | non-null | Training job for the same Study Run and trial. |
| `evaluation`, existing-Model | `study_run_id + trial_id + stage` | non-null | `NULL`. |

`job_id` is an SQLite-local operational identity and must not be persisted into Study Run plans or child Run records.

`study_run_id`, `trial_id`, and `stage` must match the immutable Study Run plan.

`kind`, `stage`, and dependency relationship must not redefine or contradict the plan.

`created_at` is immutable after row creation.

`updated_at` must be replaced with the current Queue timestamp whenever the job row changes operational state or retry timing.

`retry_not_before` is present only for `retry_wait`. When retry is immediately allowed, the value may be the current timestamp.

A scheduler may promote an eligible `retry_wait` job to `ready` once the current UTC time is greater than or equal to `retry_not_before`.

### Attempts

One `attempts` row represents one concrete Worker execution that reached child Run allocation.

A concrete preflight failure before Run allocation creates no attempt row.

`attempt_no` begins at 1 for each logical job and increases monotonically by one for each later concrete attempt represented in the current Queue database. Because attempt numbering is operational rather than canonical history, a full Queue rebuild may renumber or restart operational attempt sequences without changing child Run identity.

`run_id` must identify the concrete child Run allocated for that attempt.

| parent job kind | required Run kind |
|---|---|
| `training` | Training Run ID. |
| `evaluation` | Evaluation Run ID. |

`worker_id` must be a non-empty Worker-process identity supplied by the Worker/API boundary.

`acquire_token` is the opaque Worker-generated identity of the logical acquire operation that created the attempt. It is persisted so retry of an acquire whose response was lost can recover the same still-authorized assignment without allocating another Run.

`lease_token` is an opaque unique token for that attempt's current authorization.

`lease_until` is the current Worker-heartbeat inactivity deadline. Assignment initializes it to one hour after assignment establishment, and every successfully accepted heartbeat replaces it with one hour after the accepted heartbeat time.

`finished_at IS NULL` means the attempt is open.

`finished_at IS NOT NULL` means the attempt is closed and must never be reopened.

At most one open attempt may exist for one logical job.

At most one open attempt may exist for one `worker_id` because one Worker process executes at most one assignment at a time in v1.

A job in `active` must have exactly one open attempt. A job in any other state must have no open attempt after reconciliation completes.

`close_reason` may contain a concise machine-readable operational reason. Its vocabulary is not normative in v1 and must not replace canonical child Run status or failure information.

### Dependency validation beyond DDL

SQLite foreign keys alone do not prove semantic dependency correctness.

Controller validation must reject or reconcile an Evaluation job whose non-null `dependency_job_id`:

- does not reference a `training` job;
- references another Study Run;
- references another trial;
- contradicts an existing-Model plan row;
- is absent for a training-derived plan row.

### No execution payload duplication

The Queue database must not independently persist the execution payload already owned by the immutable Study Run plan.

In particular, `jobs` and `attempts` do not add authoritative columns for:

- Architecture;
- Corpus;
- Train Protocol;
- Evaluation Protocol;
- seed;
- resolved public parameters;
- existing Model;
- formal result declarations.

Worker assignments resolve those inputs through Study Run coordinates and canonical MLDB records.

### Queue admission transaction

Controller derives the complete logical job set only from a complete valid Study Run plan.

Admission for one Study Run is all-or-nothing in one SQLite transaction.

For a training-derived row, Controller creates:

```text
one training job: ready
one evaluation job per stage: blocked, dependent on the training job
```

For an existing-Model row, Controller creates:

```text
one evaluation job per stage: ready, no dependency job
```

Coordinate uniqueness must make repeated admission safe when existing rows agree with the immutable plan.

An existing coordinate with incompatible `kind`, `stage`, or dependency relationship is an inconsistency and must not be silently rewritten into different execution intent.

### Dispatch-start critical section

V1 has one Controller process responsible for Queue dispatch mutation.

Controller must serialize the dispatch-start critical section so two concurrent Worker pull requests cannot both begin the same `ready` job.

For one selected job, the required cross-store ordering is:

```text
select eligible ready job
  -> concrete preflight
  -> allocate child Run ID
  -> persist canonical running child Run
  -> BEGIN Queue transaction
       insert attempt
       update job to active
     COMMIT
  -> return assignment to Worker
```

The attempt row stores the request's `acquire_token`, and the attempt row plus `active` job state are committed together.

If the assignment response is lost, a repeated acquire carrying the same `acquire_token` must resolve the existing open attempt and return its same still-authorized assignment rather than allocate another child Run.

The assignment must not be returned before the Queue transaction commits.

A crash after the canonical `running` Run is written but before Queue commit is recovered as a `running` child Run without recoverable active lease ownership according to `spec:mldb.orchestration.queue_lifecycle`.

A crash after Queue commit but before assignment delivery leaves a leased `active` attempt that may expire normally.

### Heartbeat update

A valid heartbeat for the currently open attempt may extend `lease_until`.

Heartbeat handling must match both:

- the open attempt identity;
- the current `lease_token`.

Heartbeat must not reopen a closed attempt or revive a superseded lease.

The healthy heartbeat cadence is outside this format contract. The lease extension duration is fixed at one hour from each successfully accepted heartbeat by the Queue lifecycle and Worker API contracts.

### Attempt closure

Closing an attempt sets `finished_at` exactly once.

For a satisfying result, Controller first commits canonical MLDB result state and then, in one Queue transaction:

```text
attempt.finished_at = now
job.status = satisfied
job.updated_at = now
```

For a training job, satisfaction additionally requires the canonical Model created from the completed Training Run.

For an Evaluation job, only an accepted `completed` Evaluation Run satisfies the job.

`completed_partial` does not satisfy an Evaluation job.

For an unsuccessful attempt that remains retryable, Controller closes the attempt and moves the job to `retry_wait` with a non-null `retry_not_before` in one Queue transaction.

For an unsuccessful attempt with no further retry, Controller closes the attempt and moves the job to `failed`.

Intentional Study cancellation may close active attempts under the cancellation contract and move affected logical jobs to `cancelled`.

### Reconciliation

Queue reconciliation uses immutable Study Run plans and canonical child history as authority for facts already accepted by MLDB.

Reconciliation may insert missing jobs, repair stale operational states, close or resolve unrecoverable attempts, and release dependencies.

It must not:

- mutate immutable Study Run plan intent;
- reopen terminal child Runs;
- repeat accepted completed training merely because downstream Evaluation work remains unfinished;
- treat Queue-only state as stronger evidence than a completed canonical Run or Model.

A missing Queue database may be rebuilt for running Study Runs from immutable plans and canonical child lineage, with loss of purely operational attempt history tolerated where it does not alter canonical experiment history.

## Validation rules

- Reject a Queue database whose nonzero `PRAGMA user_version` is unsupported.
- Reject mutation when foreign-key enforcement cannot be enabled.
- Reject `kind` outside `training` or `evaluation`.
- Reject `status` outside the Queue lifecycle contract.
- Reject a training job with non-null `stage` or dependency.
- Reject an evaluation job with null `stage`.
- Reject a `retry_wait` job without `retry_not_before`.
- Reject a non-`retry_wait` job with `retry_not_before`.
- Reject duplicate Training coordinates.
- Reject duplicate Evaluation coordinates.
- Reject a dependency relationship that disagrees with the immutable Study Run plan.
- Reject `attempt_no <= 0`.
- Reject duplicate `run_id`, `acquire_token`, or `lease_token`.
- Reject duplicate `(job_id, attempt_no)`.
- Reject more than one open attempt for one job.
- Reject more than one open attempt for one `worker_id`.
- Reject an `active` job without exactly one open attempt after reconciliation.
- Reject a non-`active` job that retains an open attempt after reconciliation.
- Reject an attempt whose Run kind does not match its parent job kind.
- Reject malformed Queue timestamps or timestamps not encoded in the required UTC format.
- Reject reopening an attempt whose `finished_at` is already set.
- Reject satisfying a Training job without a completed Training Run and its required Model.
- Reject satisfying an Evaluation job from `completed_partial`, `failed`, or `cancelled` Evaluation Run history.
- Reject Queue admission from a partial or invalid Study Run plan.
- Reject an attempt row created for a concrete preflight failure that never allocated a child Run.

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.orchestration` | Parent orchestration overview. |
| `spec:mldb.orchestration.job_model` | Defines Study-derived job identity and dependencies represented here. |
| `spec:mldb.orchestration.queue_lifecycle` | Defines the states, leases, retry, and reconciliation semantics persisted by this format. |
| `spec:mldb.orchestration.responsibility_model` | Defines Controller ownership of Queue mutation and canonical MLDB authority. |
| `spec:mldb.orchestration.worker_api` | Supplies Worker-process identity, acquisition replay token, and lease operations persisted by this format. |
| `spec:mldb.study.plan_format` | Supplies immutable execution intent referenced by Queue coordinates. |
| `spec:mldb.repository.layout` | Keeps this operational database outside canonical `mldb_data/`. |
