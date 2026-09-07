# Concept: Evaluation Run lifecycle

- **id**: `spec:mldb.evaluation.evaluation_run_lifecycle`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.evaluation`

## What this is

Defines the state model for one Evaluation Run and the boundary between local Run failure and broader orchestration.

Evaluation Run is an execution-history entity. Re-execution creates another Run rather than revising a previous event.

## Concept model

Evaluation Run v1 has five states.

| state | meaning | terminal |
|---|---|---|
| `running` | One allocated evaluation attempt whose execution or formal-result handling has not reached a terminal outcome. | No. |
| `completed` | Evaluation execution succeeded and the complete declared formal result surface was satisfied. | Yes. |
| `completed_partial` | Evaluation execution succeeded and retained valid formal results, but one or more declared metrics or optional formal artifacts were unavailable or rejected under the partial-result contract. | Yes. |
| `failed` | Evaluation execution or a completion-critical contract failed. | Yes. |
| `cancelled` | The concrete evaluation attempt was intentionally stopped before successful completion. | Yes. |

Allowed transitions are:

```text
running -> completed
running -> completed_partial
running -> failed
running -> cancelled
```

No terminal state transitions back to `running` or to another terminal state.

## Execution boundary

The evaluation executor conceptually performs:

```text
launch request
  -> resolve Model / Corpus / Evaluation Protocol
  -> verify static compatibility and integrity
  -> resolve complete public parameters
  -> preflight succeeds
  -> allocate ev-YYYYMMDD-NNN
  -> create schema-valid run.yaml as running
  -> create work/ and artifacts/
  -> load executable implementation / construct EvaluationContext
  -> invoke evaluate(context)
  -> validate returned formal results
  -> import and hash accepted artifacts
  -> finalize terminal state
```

A preflight rejection creates no Evaluation Run.

A Run may become `completed` or `completed_partial` only after execution returns successfully and all completion-critical validation and artifact materialization succeed.

`completed_partial` additionally requires at least one accepted formal metric or artifact plus an explicit incompleteness reason under the result-validation contract.

## Failure model

The affected Evaluation Run becomes `failed` for post-allocation conditions including:

- executable loading or EvaluationContext construction failure;
- `evaluate(context)` failure;
- invalid formal scalar metrics;
- undeclared returned formal outputs;
- missing or invalid required structured artifacts;
- failure to import or record required formal result material.

Partial-result conditions are distinct from failure.

| condition | lifecycle consequence |
|---|---|
| Declared metric is explicitly unavailable, with other completion-critical outputs valid | `completed_partial`. |
| Optional artifact is explicitly unavailable, with other completion-critical outputs valid | `completed_partial`. |
| Returned optional artifact fails validation, with other completion-critical outputs valid | Omit it, record `validation_issues`, and use `completed_partial`. |
| Optional artifact is simply omitted as allowed by `required: false` | Does not by itself prevent `completed`. |

Failure takes precedence over partial completion.

## Failure isolation

- Failure, cancellation, or partial completion is local to the affected Evaluation Run.
- A `completed_partial` Evaluation Run must not block or cancel sibling evaluations merely because its result surface is incomplete.
- One failed Evaluation Run must not mutate or cancel sibling evaluations.
- One failed Evaluation Run must not invalidate its Model or originating Training Run.
- One failed Evaluation Run must not implicitly cancel unrelated Study trials or queue jobs.
- Orchestration may block only work that explicitly depends on a result requirement the affected Run did not satisfy.

Evaluation stages for one Model are independent siblings unless a later orchestration contract declares another dependency.

## Retry and repetition

Retrying a failed, cancelled, or partially completed evaluation creates a new Evaluation Run ID.

When a Study-owned Evaluation Run is retried for the same planned coordinate while its Study Run remains `running`, the new Run retains the same `study.run`, `study.trial`, and `study.stage` lineage.

Explicitly repeating a completed evaluation also creates a new Evaluation Run.

Evaluation Run itself does not deduplicate identical Model, Corpus, Evaluation Protocol, and parameter combinations. A later orchestration layer may reuse an existing completed result as an optimization without changing Evaluation Run identity semantics.

## Terminal immutability

After reaching `completed`, `completed_partial`, `failed`, or `cancelled`, the Run's historical facts are immutable, including:

- selected Model, Corpus, and Evaluation Protocol;
- resolved parameters;
- timestamps;
- accepted metrics and artifact identities;
- unavailable outputs;
- validation issues;
- failure metadata;
- accepted artifact bytes and integrity metadata.

A later execution must receive another Run ID rather than repurpose a terminal Run.

## Boundary

| concern | owner |
|---|---|
| Exact `run.yaml` fields | `spec:mldb.evaluation.evaluation_run_format`. |
| Metric and formal artifact acceptance | `spec:mldb.evaluation.result_validation`. |
| Evaluation-specific algorithm | Evaluation Protocol implementation. |
| Queue claims, leases, attempts, Worker identity, and blocked-job state | `spec:mldb.orchestration`. |
| Study-wide terminal summary | `spec:mldb.study.study_run_lifecycle`. |
| Promotion or release decisions | Future lifecycle topics. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.evaluation` | Parent evaluation overview. |
| `spec:mldb.evaluation.evaluation_run_format` | Persists lifecycle state and terminal facts. |
| `spec:mldb.evaluation.result_validation` | Determines whether returned formal results permit completion. |
| `spec:mldb.orchestration` | Defines queued retry and separate Worker execution without transferring terminal-state authority away from Controller. |
