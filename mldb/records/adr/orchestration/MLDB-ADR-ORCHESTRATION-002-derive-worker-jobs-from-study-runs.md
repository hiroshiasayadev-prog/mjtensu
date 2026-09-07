# MLDB-ADR-ORCHESTRATION-002: Derive Worker jobs from Study Runs

- **status**: accepted
- **date**: 2026-09-04
- **depends_on**: MLDB-ADR-ORCHESTRATION-001, MLDB-ADR-SCHEMA-010, MLDB-ADR-SCHEMA-012, MLDB-ADR-SCHEMA-017, MLDB-ADR-SCHEMA-020
- **supersedes**:
- **migrated_to_spec**: 2026-09-04

## Context

MLDB orchestration already separates Controller, Queue, and Worker responsibilities, but the top-level source of queued work and the Worker work-unit granularity were not fixed.

Allowing independent top-level Training or Evaluation requests would create multiple durable execution-intent paths and complicate queue recovery. Study can now either train new Models or select existing Models before applying evaluation stages, so both primary workflows can be expressed through one durable Study definition and one immutable Study Run plan.

Workers should not execute whole Studies or whole trials because training and evaluation have different dependencies, retry independently, and may be scheduled to different compute capabilities.

Queue insertion should occur only after the authored Study and its dependencies have passed validation. Invalid Studies should not create operational work that can never execute correctly.

## Decision

Use Study execution as the v1 top-level queued execution entry.

Controller accepts execution of a sealed Study, validates the complete Study and referenced immutable inputs, materializes one Study Run plan, and derives Queue jobs from that complete immutable plan.

Training Run and Evaluation Run executors remain internal runtime capabilities invoked as child work. They are not separate v1 user-authored queue-entry definitions.

### Validation before Queue insertion

Study execution proceeds in this order:

```text
Study execution request
  -> validate Study and referenced immutable inputs
  -> reject on validation failure
  -> allocate Study Run
  -> materialize complete plan.jsonl
  -> reject Queue insertion if complete plan materialization fails
  -> derive Queue jobs from the valid immutable plan
```

A Study validation failure before Study Run allocation creates neither a Study Run nor Queue jobs.

A failure after Study Run allocation but before complete plan materialization follows the Study Run failure contract and creates no Queue jobs from a partial plan.

Queue must consume only a complete valid Study Run plan.

### Worker job kinds

Queue has two logical compute-work kinds in v1:

```text
training
evaluation
```

A Worker processes one assigned Training or Evaluation execution attempt at a time.

Worker does not own Study-wide progression, trial completion, dependency release, or Study Run finalization.

### Training jobs

A training job exists only for a Study Run plan row whose Model source is `training`.

The job refers to the Study Run and trial. The exact Architecture, Corpus, Train Protocol, seed, and resolved parameters are authoritative in the immutable plan rather than re-authored as independent Queue intent.

A training job becomes a concrete Training Run only when execution is starting, required concrete preflight succeeds, and Controller allocates the Training Run ID.

Successful Controller acceptance of the Training result produces the Model required by that trial's evaluation jobs.

### Evaluation jobs

Every plan row produces one evaluation job for every materialized evaluation stage.

An evaluation job refers to the Study Run, trial, and stage.

For a training-derived row, evaluation work is dependency-blocked until a completed Training Run has produced the trial Model.

For an existing-Model row, the Model is already identified by the immutable plan, so no training dependency exists.

A runnable evaluation job becomes a concrete Evaluation Run only when execution is starting, required concrete preflight succeeds, and Controller allocates the Evaluation Run ID.

### Job identity versus Run identity

Queue job identity remains operational and separate from MLDB Run identity.

One logical training or evaluation job may produce multiple immutable Run attempts through retry.

Queued or dependency-blocked jobs do not require a `running` Training Run or Evaluation Run.

A Worker claim does not itself allocate a Run. Concrete Run allocation occurs after required preflight and before executable invocation.

Worker success reporting does not by itself complete the logical job. Controller must accept and commit the corresponding canonical Run result before Queue may treat the intended coordinate as satisfied.

### Dependency progression

Controller and Queue advance Study Run work according to immutable plan dependencies and canonical MLDB history.

Training-derived trial:

```text
training job
  -> completed Training Run
  -> Model
  -> evaluation jobs become runnable
```

Existing-Model trial:

```text
existing Model
  -> evaluation jobs may be runnable immediately
```

Completed child work must not be repeated merely because Controller or Worker restarts. Queue recovery and reconciliation must use the immutable Study Run plan plus canonical child Run and Model history.

Exact durable Queue schema, job lifecycle states, leases, heartbeat intervals, retry backoff, and reconciliation algorithm are deferred to focused orchestration contracts.

## Rationale

Making Study the sole top-level queued execution definition gives training experiments and existing-Model re-evaluation the same durable planning, retry, and recovery model.

Deriving Queue jobs from `plan.jsonl` avoids duplicating experiment configuration into mutable scheduler state.

Using Training and Evaluation as Worker work units keeps Workers simple and lets training and evaluation retry, schedule, and scale independently.

Delaying Run allocation until execution start preserves the established preflight rule and prevents queued capacity wait from appearing as a long-running MLDB execution attempt.

Requiring Controller acceptance before job satisfaction prevents Worker-local success from becoming authoritative history before result validation and canonical commit.

## Rejected alternatives

### Queue a whole Study as one Worker job

A Study contains many independent training and evaluation operations with dependencies and separate retry behavior. One Worker job would serialize unrelated work and make partial recovery coarse.

### Queue one whole trial as one Worker job

Evaluation stages are independent siblings and may run on different Workers or retry independently. Trial-sized Worker jobs would unnecessarily couple those operations.

### Permit top-level direct Training and Evaluation queue requests in v1

Study now expresses both train-then-evaluate and existing-Model evaluation. Separate queue-entry types would add another durable request and recovery path without a current requirement.

### Allocate all child Runs when Queue jobs are created

Queued work may wait a long time, be blocked on a Model, or never be claimed. Preallocating Runs would make waiting scheduler state look like active execution history.

## Consequences

Controller's v1 user-facing execution action can be centered on executing a sealed Study.

Study validation must complete before Study Run allocation, and complete plan materialization must complete before Queue insertion.

Queue schema design needs logical training and evaluation jobs referenced by Study Run coordinates rather than duplicated execution configuration.

Training jobs exist only for training-derived plan rows.

Evaluation jobs exist for every plan row and evaluation stage.

Worker API design needs two execution assignment/result families: Training and Evaluation.

Recovery design must avoid redoing completed training when only downstream evaluation remains incomplete.

## Evidence

Study Run already persists immutable resolved execution intent and child lineage suitable for queue reconstruction.

MLDB-ADR-SCHEMA-020 allows a Study to select existing Models directly, eliminating the main need for a separate direct-evaluation queue-entry type.

Training Run and Evaluation Run lifecycle contracts already define independent retries with new Run IDs, which maps naturally to one logical Queue job producing multiple execution attempts.
