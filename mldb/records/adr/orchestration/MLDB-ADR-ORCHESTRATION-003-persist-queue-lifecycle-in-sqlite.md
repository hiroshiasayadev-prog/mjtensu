# MLDB-ADR-ORCHESTRATION-003: Persist Queue lifecycle in SQLite

- **status**: accepted
- **date**: 2026-09-04
- **depends_on**: MLDB-ADR-ORCHESTRATION-001, MLDB-ADR-ORCHESTRATION-002, MLDB-ADR-SCHEMA-005, MLDB-ADR-SCHEMA-008, MLDB-ADR-SCHEMA-010, MLDB-ADR-SCHEMA-017
- **supersedes**:
- **migrated_to_spec**: 2026-09-04

## Context

MLDB now derives all v1 queued compute work from immutable Study Run plans and executes one Training or Evaluation attempt at a time on Workers.

The Queue remains operational rather than canonical MLDB history, but purely in-memory scheduling is insufficient. Controller restart must not cause already completed training to be repeated merely because downstream evaluation is unfinished, and Worker loss must be distinguishable from completed accepted work.

Study Run plans and child Run lineage already provide the durable experiment facts needed to reconstruct intended work. Queue storage should therefore preserve current scheduling progress across ordinary process restarts while remaining repairable from canonical MLDB history when operational state and durable history disagree.

Controller and Queue are expected to run together in the initial deployment. The project does not currently need a distributed queue service, multi-controller consensus, or an external broker.

## Decision

Use a Controller-local SQLite database as the v1 durable Queue backend.

SQLite stores operational Queue state across Controller restarts. It does not become authoritative experiment history and remains outside canonical `mldb_data` entity and result storage.

The exact database filename, table DDL, indexes, migration mechanism, and column layout are deferred to a Queue storage contract.

### Logical job lifecycle

A logical Queue job uses the following v1 lifecycle states:

```text
blocked
ready
active
retry_wait
satisfied
failed
cancelled
```

Their meanings are:

| state | meaning |
|---|---|
| `blocked` | The immutable plan contains the job, but an upstream dependency required for execution is not yet satisfied. |
| `ready` | The job has no unsatisfied dependency and may be selected for execution. |
| `active` | One current Worker execution attempt is established for the job and is protected by an active lease. |
| `retry_wait` | The latest attempt did not satisfy the job, and policy allows another attempt after any applicable retry delay. |
| `satisfied` | Controller has accepted canonical child Run results that fully satisfy the planned coordinate. |
| `failed` | The planned coordinate remains unsatisfied and orchestration will not schedule another attempt in this Study Run. |
| `cancelled` | The planned coordinate was intentionally stopped and will not be scheduled again in this Study Run. |

`satisfied`, `failed`, and `cancelled` are terminal Queue states for one logical job within one Study Run.

Queue state is not copied into Study Run, Training Run, or Evaluation Run lifecycle fields.

### Dependency progression

Training-derived Study trials begin conceptually as:

```text
training job = ready
associated evaluation jobs = blocked
```

A training job becomes `satisfied` only after Controller has completed canonical Training Run acceptance and the required Model exists.

That accepted Model releases the trial's evaluation jobs from `blocked` to `ready` when their other preflight conditions remain valid.

Existing-Model Study trials have no training job. Their evaluation jobs may begin as `ready` because the immutable plan already identifies the Model dependency.

Evaluation job `satisfied` requires an accepted Evaluation Run with status `completed` for the planned Study Run, trial, and stage coordinate.

An Evaluation Run with status `completed_partial` retains historical output but does not satisfy the planned evaluation coordinate. It may lead to `retry_wait` when retry remains allowed, or `failed` when orchestration stops retrying.

### Attempts and Run identity

Queue must durably distinguish one logical job from its concrete execution attempts.

One logical job may have zero, one, or multiple attempts over its lifetime.

Each established attempt may reference exactly one concrete Training Run or Evaluation Run allocated by Controller after successful concrete preflight and before executable invocation.

Retry never reuses a terminal child Run ID.

An attempt may record operational information such as Worker identity, attempt sequence, lease identity, lease expiry, and timestamps. Exact persisted fields belong to the Queue storage contract.

At most one attempt for one logical job may be active at a time.

If concrete preflight fails before Run allocation, no child Run and no concrete execution attempt is created. The logical job may move to `retry_wait` when the condition is considered recoverable and retry remains allowed, or to `failed` when orchestration will not retry.

### Lease and Worker-loss semantics

An `active` job is associated with a lease that identifies the currently authorized Worker attempt.

Worker heartbeat may extend that lease according to the future Worker API and Queue storage contracts.

When an active lease is determined to be stale or expired, Controller must ensure that the corresponding still-`running` child Run does not remain indefinitely active.

The stale attempt is terminalized as an unsuccessful child Run according to the applicable Training Run or Evaluation Run lifecycle, with concise failure information when available.

The logical job then becomes `retry_wait` if another attempt is permitted or `failed` otherwise.

Once a lease has been invalidated or superseded, a late result reported only under that stale lease must not be accepted as the canonical result for a newer attempt.

Controller process restart alone does not invalidate a recoverable active attempt. If SQLite still contains the active attempt and lease identity, the corresponding child Run is still `running`, and the lease remains valid under the configured recovery rules, Controller may resume lease handling for that same attempt and Run rather than restarting the computation.

Exact lease duration, heartbeat interval, retry count, backoff policy, and stale-worker grace period are deferred.

### Canonical acceptance precedes Queue satisfaction

Worker-local success is not job completion.

The ordering for a successful Training attempt is conceptually:

```text
Worker result candidate
  -> Controller validation
  -> canonical learned artifact commit
  -> Training Run completed
  -> Model ensured
  -> Queue training job satisfied
  -> dependent evaluation jobs released
```

The ordering for a successful Evaluation attempt is:

```text
Worker result candidate
  -> Controller result validation
  -> canonical formal artifact commit
  -> Evaluation Run completed
  -> Queue evaluation job satisfied
```

Queue must not record `satisfied` before the corresponding canonical MLDB result is accepted.

### Reconciliation authority

Controller reconciles Queue SQLite against canonical MLDB state on startup and when an inconsistency is detected.

Canonical MLDB history wins over stale Queue progress for facts that MLDB already owns.

Reconciliation must be able to use:

- every `running` Study Run with a complete valid immutable plan;
- plan rows and their Study-local trial and stage coordinates;
- Training Runs with matching Study lineage;
- Models produced by completed training-derived trials;
- existing Models named directly by existing-Model plan rows;
- Evaluation Runs with matching Study lineage.

Reconciliation must restore or correct operational progress so already accepted work is not repeated.

For example, if Queue says a training job is `ready` but canonical history already contains a completed Training Run and its Model for that Study coordinate, the job must be repaired to `satisfied` and its dependent evaluations reconsidered.

If an evaluation stage already has an accepted `completed` Evaluation Run, the corresponding evaluation job must be repaired to `satisfied`.

If Queue state is missing for a valid running Study Run, its expected logical jobs may be reconstructed from the immutable plan and canonical child lineage.

A recoverable `active` attempt whose persisted lease identity and corresponding `running` child Run remain valid across Controller restart may continue as the same attempt and Run.

Loss of Queue-only attempt or lease information must not invalidate completed canonical Runs. If a canonical Run is still `running` but its active Queue lease cannot be recovered safely, Controller must resolve that Run as an unsuccessful interrupted attempt before scheduling a new Run for the same logical job.

### Cross-store crash consistency

SQLite Queue state and filesystem-backed canonical MLDB records are not one atomic transaction domain.

V1 does not require a distributed transaction across SQLite and canonical MLDB files.

Controller operations must instead be ordered and idempotent so reconciliation can resolve crashes between durable writes.

In particular:

- accepted canonical result state is written before Queue becomes `satisfied`;
- a new Run must not be allocated for a logical job if reconciliation discovers an already active or already satisfying child Run for that coordinate;
- result artifact acceptance should use staging and atomic replacement where the existing artifact contract permits it;
- restart reconciliation must tolerate Queue being either ahead of or behind the latest canonical MLDB write without redefining terminal Run history.

### Queue durability boundary

SQLite is durable operational state, not a second experiment database.

The Queue may persist enough job, attempt, dependency, Worker, lease, and retry information to resume scheduling efficiently.

Architecture, Corpus, Protocol, seed, parameters, existing Model selection, and evaluation-stage intent remain authoritative in the immutable Study Run plan and referenced MLDB definitions rather than being independently authored by Queue storage.

The Queue database may be repaired or rebuilt from Study Run plans and canonical child history where possible. Rebuilding may lose purely operational history such as old lease details while preserving experiment history.

## Rationale

SQLite provides transactional durable state with very low operational overhead for the intended single Controller deployment.

Separating logical jobs from attempts preserves the existing rule that retries produce new immutable child Runs while one Study coordinate remains the same intended work.

Using canonical MLDB history as reconciliation authority guarantees that completed training and evaluation survive scheduler restart independently of Queue freshness.

Lease-based active attempts allow Worker loss and late stale results to be handled without allowing two attempts to become authoritative for one job.

Avoiding a distributed transaction keeps the first implementation small while explicit ordering and reconciliation provide crash recovery across SQLite and filesystem-backed MLDB state.

## Rejected alternatives

### Keep Queue only in memory

Controller restart would lose scheduling progress and active attempt ownership. Although the plan can reconstruct intent, routine recovery would be unnecessarily expensive and Worker-attempt handling would be ambiguous.

### Make Queue SQLite authoritative for completed experiment history

Queue state is mutable operational data and may be repaired or rebuilt. Completed Runs, Models, plans, and formal artifacts already provide the canonical historical record.

### Represent one Queue job per whole Study or trial

Training and evaluation have independent dependencies, retries, and Worker scheduling needs. ORCHESTRATION-002 already fixes Training and Evaluation as the Worker compute units.

### Reuse the same Run ID when a Worker is lost

A terminal Run is immutable historical evidence. A retry is a new execution attempt and therefore receives a new Training Run or Evaluation Run ID.

### Accept late results from an expired lease

Once another attempt can start, accepting a stale Worker's result could race with the current attempt and make attempt authority ambiguous.

### Introduce Redis or an external broker immediately

The initial deployment has one Controller and does not require distributed queue coordination. SQLite is sufficient while preserving a future replacement boundary.

## Consequences

The Queue storage contract must define SQLite schema and transactional invariants for logical jobs, attempts, leases, and any Worker records needed by the implementation.

The Worker API contract must carry enough lease identity to authenticate heartbeat and result reporting for the active attempt.

Controller startup must perform Queue/MLDB reconciliation before normal scheduling resumes.

Controller result handling must be idempotent across restart and must never mark a Queue job `satisfied` before canonical acceptance.

A completed training coordinate and its Model remain reusable after Controller restart even when one or more downstream evaluations are still pending.

A lost Worker may cause its current child Run to fail and be retried, but it must not force already accepted upstream work or unrelated Study coordinates to rerun.

Exact SQLite tables, indexes, migrations, lease timing, retry limits, backoff, and Worker API payloads remain follow-up contracts.

## Evidence

Study Run plans already persist complete immutable execution intent independently of Queue state.

Training Run and Evaluation Run lineage already identifies Study Run, trial, and evaluation stage coordinates, allowing canonical history to be matched back to logical Queue work.

Study Run retry semantics already require failed child attempts to remain historical while a later Run can satisfy the same planned coordinate.

The current project deployment already favors a small Controller-plus-Queue server with separate Workers, making an embedded transactional Queue database an appropriate first durable scheduler backend.
