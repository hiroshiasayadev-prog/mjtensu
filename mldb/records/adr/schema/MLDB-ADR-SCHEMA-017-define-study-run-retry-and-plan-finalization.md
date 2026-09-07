# MLDB-ADR-SCHEMA-017: Define Study Run retry and plan finalization

- **status**: accepted
- **date**: 2026-09-04
- **depends_on**: MLDB-ADR-SCHEMA-005, MLDB-ADR-SCHEMA-008, MLDB-ADR-SCHEMA-010, MLDB-ADR-SCHEMA-016
- **supersedes**:
- **migrated_to_spec**: 2026-09-04

## Context

MLDB-ADR-SCHEMA-010 requires Study Run plan metadata while also allowing Study Run plan materialization itself to fail before a complete valid plan exists.

Those requirements cannot both be unconditional. Before complete plan materialization succeeds, there is no authoritative `plan.jsonl` hash, byte size, trial count, or evaluation-job count to persist.

Study orchestration must also survive process and worker interruption. A `running` Study Run should be resumable from durable MLDB state when possible, while terminal Study Runs and terminal child Runs must remain immutable under their existing lifecycle contracts.

Retry semantics must preserve history without making a temporary failed child attempt permanently prevent a Study Run from reaching full completion when a later retry satisfies the same planned coordinate.

## Decision

Clarify Study Run plan-finalization and retry semantics.

### Plan metadata finalization

Every Study Run always requires:

- `schema`;
- `id`;
- `status`;
- `study`;
- `execution.started_at`.

The following plan metadata becomes required only after complete valid plan materialization succeeds:

- `plan.path`;
- `plan.sha256`;
- `plan.bytes`;
- `plan.trials`;
- `plan.evaluation_jobs`.

A `completed` or `completed_with_failures` Study Run always has a complete valid plan and therefore always requires all plan metadata.

A `failed` or `cancelled` Study Run may omit plan metadata only when no complete valid plan was established before terminality.

If complete valid plan materialization succeeded before a later failure or cancellation, the plan and all plan metadata remain required and immutable.

A partial plan file produced before successful finalization is not an MLDB plan artifact and must not be treated as authoritative.

### Resume of a running Study Run

A Study Run whose status remains `running` is resumable rather than retried as a new Study Run merely because a worker, scheduler, or process stopped.

If no complete valid plan exists yet, orchestration may restart plan materialization for the same `running` Study Run. Any partial unfinalized plan bytes may be discarded or replaced because they have not entered the immutable plan contract.

If a complete valid plan already exists, orchestration resumes from that immutable plan and existing child lineage. The plan must not be regenerated into a different execution intent or mutated to record progress.

Resume must use Study Run, trial, and stage lineage to avoid accidental duplicate child creation caused only by scheduler restart.

Queue-specific stale-worker detection, leases, attempts, and retry timing remain outside the persistent Study Run contract.

### Retry of a terminal Study Run

A terminal Study Run never transitions back to `running`.

Retrying a terminal `failed`, `cancelled`, or `completed_with_failures` Study Run creates a new Study Run ID and performs a new Study Run execution of the same sealed Study.

Explicitly repeating a `completed` Study Run also creates a new Study Run ID.

The previous terminal Study Run remains immutable historical evidence.

### Retry of terminal child Runs

Retrying terminal child work never mutates or reopens the existing child Run.

A retried Training Run receives a new Training Run ID while retaining the same `study.run` and `study.trial` lineage when it is retrying that planned training coordinate inside the same still-running Study Run.

A retried Evaluation Run receives a new Evaluation Run ID while retaining the same `study.run`, `study.trial`, and `study.stage` lineage when it is retrying that planned evaluation coordinate inside the same still-running Study Run.

This applies to failed or cancelled child Runs, and to `completed_partial` Evaluation Runs when orchestration intentionally retries to obtain a complete result.

The Study Run plan is not changed by child retries.

### Planned-coordinate satisfaction

Study Run terminal status is determined by whether the immutable plan is ultimately satisfied, not by whether every historical child attempt succeeded on its first try.

A planned training coordinate is fully satisfied when at least one matching child Training Run has completed successfully and produced its Model.

A planned evaluation coordinate is fully satisfied when at least one matching child Evaluation Run reaches `completed` for the Model selected by orchestration for that trial.

A `completed_partial` Evaluation Run is usable historical output but does not fully satisfy the planned evaluation coordinate while the Study Run remains open for retry.

A failed or cancelled historical child attempt that is later followed by a successful retry does not by itself force the Study Run to `completed_with_failures`.

At normal orchestration end:

- `completed` means every runnable planned coordinate is fully satisfied;
- `completed_with_failures` means orchestration reached the end but one or more planned coordinates remain unsatisfied because training/evaluation failed, was cancelled, remained partial, or an evaluation was blocked by a missing Model;
- `failed` remains reserved for Study-level planning or orchestration failure that prevented normal processing;
- `cancelled` remains intentional Study-level termination.

### Derived summary semantics

Study Run summary counts describe final planned-coordinate outcomes rather than the raw number of historical child Run attempts.

A failed attempt followed by a successful retry contributes to the final successful coordinate count, while the failed child Run remains discoverable as historical evidence through lineage.

Attempt counts and queue retry counters are operational or derived views and are not part of the Study Run authoritative summary contract.

## Rationale

Conditional plan metadata resolves the contradiction in MLDB-ADR-SCHEMA-010 without inventing placeholder hashes or counts for a plan that never existed.

Keeping `running` Study Runs resumable allows scheduler or worker crashes to recover without creating a second Study execution merely because operational infrastructure restarted.

Keeping terminal Run records immutable preserves the established event-history model. Retry adds another execution record instead of rewriting history.

Judging Study completion by final plan satisfaction makes retry useful. A transient OOM, worker crash, or partial evaluation can be retried and recovered without permanently degrading the Study Run status after all intended coordinates eventually succeed.

Keeping retry attempts out of `plan.jsonl` preserves the distinction between durable execution intent and operational attempt history.

## Rejected alternatives

### Require plan metadata on Study Runs that never produced a valid plan

This would require fabricated or partial path/hash/count fields and would make an invalid partial file look authoritative.

### Retry a terminal Study Run by setting it back to running

This would violate terminal immutability and erase the distinction between the original failed execution and a later retry.

### Rewrite failed child Runs during retry

Training Run and Evaluation Run are immutable execution events. Reusing their IDs would destroy failure history and contradict their lifecycle contracts.

### Make any historical failed attempt permanently force completed_with_failures

This would make successful retry unable to restore full Study completion even after every planned coordinate is ultimately satisfied.

### Store retry attempts in plan.jsonl

The plan defines intended coordinates, not scheduler attempts. Mutating it for retries would destroy its immutable-intent role and couple MLDB persistence to queue behavior.

## Consequences

Study Run tooling must distinguish unfinalized plan bytes from a complete immutable plan.

A running Study Run may resume plan generation when no valid plan exists, or reconcile child work from an existing immutable plan when one does.

Retrying or explicitly repeating any terminal Study Run creates a new Study Run ID.

Child retries inside a still-running Study Run create new child Run IDs with the same Study lineage while leaving the plan unchanged.

Study Run finalization evaluates the final satisfaction state of planned coordinates rather than merely checking whether any failed child Run exists in history.

Derived Study summaries count final coordinate outcomes; historical attempts remain queryable from child Runs and operational retry metadata.

## Evidence

MLDB-ADR-SCHEMA-010 already states that orchestration should be restartable from the Study Run plan and child lineage, and that a partial plan is invalid when plan generation fails.

Training Run and Evaluation Run lifecycle contracts already require new Run IDs for retry of terminal executions.

The consistency review identified the unconditional Study Run plan-field list as incompatible with failure before complete plan materialization, and implementation planning identified retry as necessary for long-running Study execution.
