# Overview: MLDB application API

- **id**: `spec:mldb.api`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb`

## What this is

Defines the public MLDB application boundary used by CLI, UI, scripts, and future transport adapters.

The application API is owned by Controller and exposes MLDB-level operations such as validation, sealing, Study execution, status observation, cancellation, and entity reads. Queue and Worker APIs remain internal orchestration boundaries.

## Current contract

Public callers interact with MLDB through Controller application operations.

```text
repository-authored definitions
        |
        v
CLI / UI / automation
        |
        v
Controller application API
        |
        +--> runtime validation / sealing
        |
        +--> Study Run materialization
        |
        +--> Queue admission / cancellation
        |
        +--> canonical entity / Run reads
        |
        v
internal Queue / Worker execution
```

The operation semantics are transport-independent. A Python application service may be implemented before HTTP without changing the public MLDB behavior.

V1 keeps reusable-definition authoring file-based. The API validates and seals authored definitions but does not require generic definition CRUD before experiments can run.

## Responsibility boundaries

| concern | owner |
|---|---|
| Public MLDB application operations | `spec:mldb.api.controller`. |
| Reusable definition file formats | Catalog, training, evaluation, and Study specs. |
| Runtime asset resolution and validation | `spec:mldb.runtime`. |
| Executable-asset pytest sealing gate | `spec:mldb.verification`. |
| Study plan semantics | `spec:mldb.study`. |
| Queue scheduling and Worker execution | `spec:mldb.orchestration`. |
| Worker acquisition/heartbeat/result communication | `spec:mldb.orchestration.worker_api`. |
| HTTP routing, authentication, and transport serialization | Future API adapter contract. |

## Non-goals

- Require HTTP before the first usable MLDB implementation.
- Provide generic create/update/delete CRUD for all reusable definitions in v1.
- Expose Queue SQLite mutation to normal callers.
- Expose Worker attempt operations as user-facing experiment controls.
- Define MLflow synchronization, dashboards, ranking, or rich analytics.
- Define standalone public Training Run or Evaluation Run launch operations in v1.

## Topics

| title | kind | ref | summary |
|---|---|---|---|
| Controller application interface | Contract | `spec:mldb.api.controller` | Public validation, sealing, Study execution/status/cancellation, and entity-read operation semantics. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb` | Parent MLDB overview. |
| `spec:mldb.runtime` | Supplies common resolution, validation, and execution semantics used by Controller operations. |
| `spec:mldb.study` | Supplies Study execution and Study Run lifecycle semantics. |
| `spec:mldb.orchestration` | Supplies internal Queue/Worker execution behind the public application boundary. |
| `spec:mldb.verification` | Supplies the executable-asset sealing gate invoked by Controller. |