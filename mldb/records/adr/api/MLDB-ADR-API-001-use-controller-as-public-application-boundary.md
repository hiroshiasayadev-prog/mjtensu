# MLDB-ADR-API-001: Use Controller as the public application boundary

- **status**: accepted
- **date**: 2026-09-04
- **depends_on**: MLDB-ADR-SCHEMA-011, MLDB-ADR-SCHEMA-020, MLDB-ADR-ORCHESTRATION-001, MLDB-ADR-ORCHESTRATION-002, MLDB-ADR-ORCHESTRATION-003, MLDB-ADR-ORCHESTRATION-005, MLDB-ADR-ORCHESTRATION-006
- **supersedes**:
- **migrated_to_spec**: 2026-09-04

## Context

MLDB now has detailed contracts for reusable definitions, sealing, Study materialization, immutable Runs and Models, Queue persistence, and Worker execution.

The immediate project need is not additional distributed-systems infrastructure. ShuffleNet architecture-condition experiments are waiting for MLDB to become usable as an experiment runner.

Without one public application boundary, a CLI, web UI, scripts, and future integrations could each reimplement validation, sealing, Study execution, cancellation, and status derivation differently. They could also reach directly into Queue or Worker APIs and bypass the Controller authority already established by orchestration decisions.

Repository-authored YAML and Python definitions are already a workable authoring mechanism for the first usable version. Building a CRUD service for every definition before experiments can run would add implementation work without improving the immediate training workflow.

## Decision

Define one public MLDB application boundary owned by Controller.

CLI, UI, automation, and future HTTP adapters invoke Controller application operations rather than mutating Queue state, invoking Workers directly, or duplicating runtime validation logic.

The Controller application boundary is transport-independent. V1 may first implement it as Python application services. HTTP routing, authentication, and serialization may be added as thin adapters later without changing operation semantics.

### V1 authoring boundary

Reusable definitions remain repository-authored files in v1.

The public application API does not need generic create/update/delete CRUD for Task, Corpus, Architecture, Train Protocol, Evaluation Protocol, or Study before the first usable experiment workflow.

Callers may create or edit draft YAML/Python/test files through normal repository tooling, then ask Controller to validate or seal them.

### Definition validation

Controller exposes a read-only validation operation for reusable definitions.

Validation resolves the target through the normal MLDB runtime and applies the applicable schema, reference, Task compatibility, parameter-interface, lifecycle, and integrity checks without changing lifecycle state.

Validation must not establish a second validation implementation specific to CLI or HTTP.

### Definition sealing

Controller exposes a sealing operation for definitions whose lifecycle supports `draft -> sealed`.

For Architecture, Train Protocol, and Evaluation Protocol, sealing includes the mandatory asset-specific pytest gate and implementation hash behavior already defined by MLDB verification contracts.

For Study, sealing applies its Study-format and referenced-input validation without inventing an executable-asset pytest requirement.

A sealing request against a definition kind without a seal lifecycle is rejected rather than silently interpreted as another operation.

### Study execution

Controller exposes Study execution as the v1 top-level experiment launch operation.

Execution semantics remain:

```text
sealed Study
  -> complete Study validation
  -> allocate Study Run
  -> materialize complete immutable plan
  -> admit derived Training/Evaluation jobs to Queue
  -> return Study Run identity
```

The call is asynchronous with respect to training and evaluation completion. Returning a Study Run ID does not mean child execution is complete.

No public API operation launches a standalone Training Run or Evaluation Run in v1.

### Study Run observation

Controller exposes a Study Run status operation.

The response presents the canonical Study Run lifecycle plus a derived progress view built from the immutable plan, canonical child Runs and Models, and current Queue operational state where useful.

Derived progress must not become more authoritative than canonical child history.

The operation should make it possible to answer at least:

- whether the Study Run is still running or terminal;
- how many training coordinates are satisfied, active, waiting/retrying, or terminally unsatisfied when training exists;
- how many evaluation coordinates are satisfied, active, blocked, waiting/retrying, or terminally unsatisfied;
- which child coordinates currently failed or remain incomplete.

Exact presentation fields may be refined in the interface contract without changing underlying entity semantics.

### Study Run cancellation

Controller exposes an idempotent Study Run cancellation operation.

Cancellation prevents new work from being scheduled for that Study Run and requests cooperative cancellation of currently active Worker attempts through the orchestration boundary.

Controller retains authority for child Run terminalization and Study Run finalization. Callers do not cancel Queue rows or Workers directly.

Cancellation of an already terminal Study Run must not reopen or rewrite it.

### Entity read access

Controller exposes read operations sufficient to retrieve and list MLDB entities and execution records needed by tooling.

Read operations do not reinterpret canonical files and must use the same runtime resolution contracts as execution.

Rich search, ranking, analytics, and visualization are not required for the first usable experiment path.

### Internal-only orchestration operations

Worker acquisition, heartbeat, immutable-asset transfer, candidate upload, result reporting, Queue reconciliation, and Queue mutation are not public user/application operations.

They remain internal Controller/Worker orchestration interfaces.

## Rationale

One application boundary lets MLDB ship a usable experiment runner without coupling callers to repository internals, Queue SQLite, or Worker transport.

Keeping authoring file-based minimizes work before the first real architecture sweep. The project already needs validators and sealing logic regardless of whether a CRUD API exists.

Transport independence avoids blocking ML work on HTTP route design. A CLI and local automation can call the same Controller service directly while later HTTP or UI adapters reuse it.

Centering Study execution preserves the durable planning and recovery model already established while exposing exactly the workflow needed for architecture and hyperparameter comparisons.

A derived status view provides practical observability without duplicating mutable experiment facts into another canonical store.

## Rejected alternatives

### Build full definition CRUD before execution

This would require request schemas, partial-update semantics, file generation, conflict handling, and authoring UX before the project can run its waiting experiments. Repository files are sufficient for initial authoring.

### Let each CLI command call runtime internals directly

That would encourage duplicate validation and lifecycle behavior and make a future UI or HTTP service another separate implementation.

### Expose Queue SQLite as the application API

Queue rows are operational state, not the MLDB semantic boundary. Direct Queue mutation could bypass Study validation, Run allocation rules, canonical result acceptance, and reconciliation.

### Expose Worker API to normal callers

Worker API is an internal compute boundary for one concrete attempt. It is not an experiment-level control API.

### Require HTTP before implementing Controller operations

The immediate value is in domain execution, not network transport. HTTP can remain an adapter over transport-independent Controller operations.

## Consequences

The first usable MLDB implementation should prioritize core runtime/domain behavior and the Controller application operations needed to validate/seal definitions and execute/observe/cancel Studies.

CLI commands should be thin clients over Controller application services rather than independent domain implementations.

A future web or HTTP API should map onto the same operation semantics.

Remote Worker transport, richer search, MLflow synchronization, visualization, and definition CRUD can follow after the first real Study runs end to end.

The project can therefore stop extending distributed execution design for now and focus implementation on reaching a real Study -> Training -> Model -> Evaluation workflow.