# Concept: MLDB Queue lifecycle

- **id**: `spec:mldb.orchestration.queue_lifecycle`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.orchestration`

## What this is

Defines the durable operational lifecycle for Study-derived Training and Evaluation jobs in the v1 MLDB Queue.

The Queue uses Controller-local SQLite so scheduling progress survives ordinary Controller restart while canonical MLDB Runs, Models, formal artifacts, and Study Run plans remain authoritative experiment history.

## Concept model

A logical Queue job represents one planned Study coordinate that orchestration wants to satisfy.

A job may create multiple concrete execution attempts over time:

```text
logical job
  -> attempt 1 -> child Run failed
  -> attempt 2 -> child Run completed
  -> satisfied
```

V1 job states are:

```text
blocked
ready
active
retry_wait
satisfied
failed
cancelled
```

Conceptually:

```text
blocked --dependency satisfied--> ready
                                  |
                                  v
                                active
                               /   |   \
                              /    |    \
                    success /  retry    \ stop
                            v      v       v
                      satisfied retry_wait failed
                                    |
                                    v
                                  ready

blocked / ready / active / retry_wait --intentional stop--> cancelled
```

The exact SQLite schema is defined by `spec:mldb.orchestration.queue_storage_format`. Worker operation and communication-retry semantics are defined by `spec:mldb.orchestration.worker_api`; exact HTTP wire details remain separate.

## Rules

### Durable Queue backend

V1 Queue state is persisted in a Controller-local SQLite database.

SQLite may store logical jobs, execution attempts, dependencies, Worker association, leases, retry timing, and other operational scheduler state needed to resume safely.

The Queue database is outside canonical `mldb_data` entity and result storage.

Queue persistence must not redefine immutable Study Run plans or terminal child Run history.

### Job state meanings

| state | contract |
|---|---|
| `blocked` | The job exists in the immutable plan but an upstream dependency required for execution is not satisfied. |
| `ready` | The job has no unsatisfied dependency and may be selected for execution. |
| `active` | One currently authorized execution attempt exists for the job under an active lease. |
| `retry_wait` | The latest attempt did not satisfy the job, another attempt remains permitted, and the job is waiting for retry eligibility. |
| `satisfied` | Canonical MLDB history contains an accepted child result that fully satisfies this planned coordinate. |
| `failed` | The coordinate remains unsatisfied and orchestration will not schedule another attempt in this Study Run. |
| `cancelled` | The coordinate was intentionally stopped and will not be scheduled again in this Study Run. |

`satisfied`, `failed`, and `cancelled` are terminal operational states for one logical job.

A Queue job state is not a Training Run, Evaluation Run, or Study Run lifecycle state.

`cancelled` represents intentional cancellation of the logical planned work, such as stopping the Study Run. A child attempt ending `cancelled` for a transient execution reason does not by itself require the logical job to become `cancelled`; it may still be retried.

### Initial readiness

For one training-derived Study trial:

```text
training job = ready
evaluation jobs = blocked
```

The training job becomes `satisfied` only after:

1. Controller accepts the canonical Training Run result;
2. the Training Run is `completed`;
3. the deterministic Model exists and is valid.

That satisfaction releases the trial's dependent evaluation jobs from `blocked` to `ready` when no other required preflight condition prevents execution.

For one existing-Model Study trial, no training job exists and evaluation jobs may begin as `ready` because the immutable plan already identifies the Model.

### Evaluation satisfaction

An evaluation job is `satisfied` only by an accepted Evaluation Run with status `completed` for its exact Study Run, trial, and stage coordinate.

| Evaluation Run outcome | Queue consequence |
|---|---|
| `completed` | The job becomes `satisfied`. |
| `completed_partial` | The job remains unsatisfied; move to `retry_wait` if retry is allowed, otherwise `failed`. |
| `failed` | Move to `retry_wait` if retry is allowed, otherwise `failed`. |
| `cancelled` | Move to `retry_wait` if retry remains allowed, otherwise `cancelled` or `failed` according to the orchestration stop reason. |

A partial evaluation may remain useful historical evidence even though the logical job is not satisfied.

### Attempts

One logical job may have zero, one, or multiple attempts.

At most one attempt for a logical job may be active at a time.

A concrete attempt is associated with exactly one Training Run or Evaluation Run after successful concrete preflight and Controller Run allocation.

If concrete preflight fails before Run allocation, no child Run and no concrete execution attempt is created. The logical job may enter `retry_wait` when retry is appropriate, or `failed` when orchestration will not retry.

A retry after any terminal unsuccessful child attempt allocates a new child Run ID.

Attempt sequence, Worker identity, lease identity, lease expiry, timestamps, and retry metadata are operational Queue facts rather than fields of the immutable Study Run plan.

### Lease semantics

An `active` job has one currently authorized Worker attempt protected by a lease.

The Worker may extend the lease through heartbeat under `spec:mldb.orchestration.worker_api`.

Retryable Controller communication failure during heartbeat does not itself terminalize the child Run from the Worker side. Worker retries heartbeat communication every 10 seconds without a finite retry count, while Controller retains authority to invalidate an actually expired or superseded lease.

When Controller determines that a lease is stale or expired:

| condition | action |
|---|---|
| Corresponding child Run is still `running` | Terminalize that Run as an unsuccessful interrupted attempt according to the child lifecycle. |
| Retry remains permitted | Close the attempt and move the job to `retry_wait`. |
| Retry is no longer permitted | Close the attempt and move the job to `failed`. |

A result reported under an invalidated or superseded lease must not be accepted as the canonical result for a newer attempt.

Controller restart alone does not invalidate a recoverable active attempt. When SQLite still contains the active attempt and lease identity, its child Run is still `running`, and the lease remains valid under the configured recovery rules, Controller may continue the same attempt and Run after restart.

An active attempt expires after one continuous hour without a successfully accepted heartbeat for its current lease. Each accepted heartbeat resets the inactivity deadline to one hour from that heartbeat. The initial assignment establishes the first one-hour deadline until the first heartbeat is accepted.

The normal healthy heartbeat cadence, retry count for domain jobs, and retry backoff remain outside this contract. Retryable communication failure still follows the Worker API's 10-second indefinite communication retry rule.

### Canonical commit ordering

Queue satisfaction occurs after canonical MLDB acceptance, never before it.

Training success ordering is:

```text
Worker candidate
  -> validate
  -> commit canonical learned artifact
  -> Training Run completed
  -> ensure Model
  -> Queue job satisfied
  -> release evaluation dependencies
```

Evaluation success ordering is:

```text
Worker candidate
  -> validate formal outputs
  -> commit accepted canonical artifacts
  -> Evaluation Run completed
  -> Queue job satisfied
```

Worker-local completion or successful candidate upload alone does not satisfy a job.

### Reconciliation

Controller must reconcile Queue SQLite with canonical MLDB state before normal scheduling after startup.

Reconciliation uses the immutable Study Run plan and child lineage as the authority for already accepted experiment facts.

At minimum it must be able to:

- derive the expected logical jobs for every running Study Run with a complete valid plan;
- restore missing Queue jobs from plan coordinates;
- identify completed training coordinates from Training Runs and Models;
- identify completed evaluation coordinates from Evaluation Runs;
- repair stale Queue jobs to `satisfied` when canonical history already satisfies them;
- recompute `blocked` versus `ready` from plan dependencies and canonical upstream results;
- inspect active attempts and their corresponding child Runs;
- resolve stale or unrecoverable active leases before another attempt is started;
- avoid creating a duplicate Run for a coordinate that already has an authoritative active or satisfying child execution;
- permit later Study Run finalization from reconciled coordinate outcomes.

Examples:

| Queue state before restart | Canonical MLDB fact | Reconciled result |
|---|---|---|
| training `ready` | completed Training Run + valid Model | training `satisfied`; dependent evaluations reconsidered. |
| evaluation `ready` | matching Evaluation Run `completed` | evaluation `satisfied`. |
| evaluation `blocked` | upstream training now completed + Model exists | evaluation becomes `ready`. |
| job absent | valid plan coordinate exists and remains unsatisfied | recreate operational job from plan. |
| job `active` | persisted lease remains valid and child Run remains `running` | keep the same active attempt and Run; do not restart computation solely because Controller restarted. |
| job `active` | lease cannot be recovered and child Run remains `running` | resolve interrupted Run, then retry or fail according to policy. |

Already accepted training must not be rerun merely because some downstream evaluation remains incomplete.

### Queue database loss

The Queue database is durable for ordinary operation but remains replaceable operational state.

If Queue state is lost, expected logical jobs can be reconstructed from complete immutable Study Run plans and canonical child lineage.

Completed Training Runs, Models, and Evaluation Runs remain valid regardless of Queue loss.

Purely operational history such as previous lease details or old attempt timing may be unrecoverable without affecting canonical experiment history.

A `running` child Run whose active lease information cannot be recovered safely must not simply be adopted as a successful attempt. Controller must resolve it as an interrupted attempt before a new Run is scheduled for the coordinate.

Conversely, ordinary Controller restart with intact SQLite lease state should preserve an otherwise valid active Worker attempt rather than forcing avoidable retraining or re-evaluation.

### Cross-store crash recovery

Queue SQLite and filesystem-backed MLDB persistence do not share one atomic transaction.

Controller therefore uses ordered idempotent writes plus reconciliation rather than a cross-store distributed transaction.

The required invariants are:

- canonical result acceptance precedes Queue `satisfied`;
- terminal child Run facts are never rewritten to match stale Queue state;
- Queue may be repaired when it lags canonical history;
- a new child Run is not allocated while reconciliation finds an existing active or satisfying child execution for the same coordinate;
- result handling may be retried after Controller restart without accepting the same candidate as two distinct canonical results;
- artifact commit should use staging and atomic replacement where the applicable artifact contract permits it.

## Boundary

| concern | owner |
|---|---|
| Study-derived job identity and dependencies | `spec:mldb.orchestration.job_model`. |
| Controller / Queue / Worker authority | `spec:mldb.orchestration.responsibility_model`. |
| SQLite table DDL, indexes, v1 persisted fields, and transaction boundaries | `spec:mldb.orchestration.queue_storage_format`. |
| Future Queue schema migrations beyond v1 | Future Queue migration contract. |
| Worker acquire/heartbeat/outcome operation semantics and communication retry | `spec:mldb.orchestration.worker_api`. |
| Exact HTTP routes/authentication/streaming | Future Worker wire contract. |
| One-hour heartbeat-inactivity expiry and 10-second communication retry | `spec:mldb.orchestration.worker_api` together with this lifecycle. |
| Domain-job retry limit and backoff policy | Future orchestration policy contract. |
| Training Run terminal semantics | `spec:mldb.training.training_run_lifecycle`. |
| Evaluation Run terminal semantics | `spec:mldb.evaluation.evaluation_run_lifecycle`. |
| Study Run final outcome | `spec:mldb.study.study_run_lifecycle`. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.orchestration` | Parent orchestration overview. |
| `spec:mldb.orchestration.job_model` | Defines the Training and Evaluation jobs governed by this lifecycle. |
| `spec:mldb.orchestration.responsibility_model` | Defines Controller authority for canonical commit and Worker lease boundary. |
| `spec:mldb.orchestration.queue_storage_format` | Defines the concrete `.local/mldb/queue.sqlite` v1 schema and persistence rules. |
| `spec:mldb.orchestration.worker_api` | Defines Worker lease communication, idempotent replay, and the distinction between communication and domain failure. |
| `spec:mldb.study.plan_format` | Supplies immutable work intent used for reconstruction and reconciliation. |
| `spec:mldb.study.study_run_lifecycle` | Defines Study-level resume and finalization behavior. |
