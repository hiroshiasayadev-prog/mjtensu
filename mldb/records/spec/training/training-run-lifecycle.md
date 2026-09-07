# Reference: Training Run lifecycle

- **id**: `spec:mldb.training.training_run_lifecycle`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.training`

## What this is

Defines Training Run state transitions, terminal immutability, retry behavior, and local failure scope.

## State model

| state | meaning | terminal |
|---|---|---:|
| `running` | The Run record exists and the execution may still produce facts or a result. | no |
| `completed` | Training succeeded, canonical learned weights were accepted, and successful finalization completed. | yes |
| `failed` | Training or required result validation failed. | yes |
| `cancelled` | Execution was intentionally stopped before successful completion. | yes |

Allowed transitions are:

```text
running -> completed
running -> failed
running -> cancelled
```

No transition out of a terminal state is allowed.

## Rules

- The runtime must complete launch preflight before allocating a Training Run ID.
- Launch preflight must resolve the selected Corpus, Architecture, and Train Protocol, validate required static compatibility and integrity, validate the concrete training seed as integer and not boolean, and resolve the complete public-parameter mapping.
- A launch request rejected during preflight must not create a Training Run.
- After preflight succeeds, the runtime allocates the Training Run ID and creates a schema-valid Run as `running` before executable loading and Train Protocol invocation.
- While `running`, the runtime may add execution facts that become known during the attempt.
- `completed` is allowed only after the canonical learned-weight contract succeeds.
- Successful finalization must ensure the deterministic Model identity exists for the completed Run.
- `failed` and `cancelled` Runs produce no Model.
- Terminal Run inputs, seed, resolved parameters, timestamps, result identity, failure facts, and Study lineage are immutable.
- A failed or cancelled Run remains a valid historical MLDB record.
- Retrying any terminal Run creates a new Training Run ID.
- When a Study-owned failed or cancelled Training Run is retried for the same planned coordinate while its Study Run remains `running`, the new Run retains the same `study.run` and `study.trial` lineage.
- Repeating an identical successful execution creates a new Training Run ID.
- One Run failure must not mutate or cancel unrelated Runs.
- A Study may block downstream evaluations for a trial whose training produced no Model, but that block is not a mutation of the failed Training Run.

## Execution boundary

The lifecycle surrounds, but does not define, the internal Train Protocol algorithm.

```text
launch request
  -> resolve and validate selected assets
  -> validate integer training seed
  -> resolve complete public parameters
  -> preflight succeeds
  -> allocate ID
  -> create schema-valid run.yaml: running
  -> load executable implementation / construct TrainContext
  -> invoke train(context)
  -> validate returned learned state
  -> serialize canonical weights
  -> completed + Model

preflight rejection
  -> no Training Run

post-allocation execution/required-result failure
  -> failed

intentional cancellation after allocation
  -> cancelled
```

Executable loading and context construction occur after allocation. Failure in those stages finalizes the affected Training Run as `failed` because a validated execution attempt already exists.

## Immutability

Terminal immutability applies to the historical meaning of the Run and its canonical result.

- Terminal `run.yaml` must not be repurposed to describe another execution.
- Canonical learned bytes of a completed Run must not be overwritten in place.
- Protocol working files should not be rewritten after terminality when doing so changes historical evidence.
- A tooling defect that requires historical correction must use an explicit future correction or migration mechanism rather than reuse the Run ID for a different event.

## Boundary

| concern | owner |
|---|---|
| Run YAML fields | `spec:mldb.training.training_run_format`. |
| Train Protocol execution | `spec:mldb.training.train_interface`. |
| Canonical learned-state acceptance | `spec:mldb.training.canonical_weights`. |
| Model ID and YAML content | `spec:mldb.model`. |
| Study-level `completed_with_failures` semantics | `spec:mldb.study`. |
| Queue retries, leases, Worker attempts, and heartbeats | `spec:mldb.orchestration`. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.training` | Parent training overview. |
| `spec:mldb.training.training_run_format` | Records the current lifecycle state and terminal timestamps. |
| `spec:mldb.training.canonical_weights` | Must succeed before transition to `completed`. |
| `spec:mldb.orchestration` | Defines how queued retries and separate Workers create new immutable Run attempts without owning terminal Run state. |
