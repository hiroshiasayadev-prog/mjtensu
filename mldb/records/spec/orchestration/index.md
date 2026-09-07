# Overview: MLDB orchestration

- **id**: `spec:mldb.orchestration`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb`

## What this is

Defines the control-plane and compute-plane boundary used to execute MLDB work through replaceable queue infrastructure and separate Workers.

Orchestration coordinates existing MLDB runtime, Study, Training Run, and Evaluation Run contracts. V1 queued execution begins from a validated sealed Study and derives Training or Evaluation work from its immutable Study Run plan.

## Current contract

MLDB orchestration separates three logical responsibilities.

| responsibility | role |
|---|---|
| Controller | Owns launch preflight, Study materialization, Run allocation, canonical runtime-generated MLDB persistence, result acceptance, finalization, Model creation, and reconciliation. |
| Queue | Owns replaceable operational scheduling state such as logical jobs, dependencies, claims, leases, retries, and priorities. |
| Worker | Pulls assigned compute work, executes protocol code with local work/cache resources, and returns result candidates without directly mutating canonical MLDB state. |

Controller and Queue may be deployed together. V1 Queue operational state is persisted in Controller-local SQLite and reconciled against canonical MLDB history on startup. Worker is a separately executable compute participant behind the Worker API boundary.

Conceptually:

```text
CLI / UI / automation
    |
    v
public Controller application API
    |
    v
Controller ------------------------+
    |                               |
    | logical work                  | canonical MLDB state
    v                               v
 Queue                         mldb_data/
    |
    | pull / assignment
    v
 Worker
    |
    | result candidates
    +------------------------------> Controller
```

Normal application callers use `spec:mldb.api` and do not invoke Queue or Worker operations directly.

A logical Queue job is not a Training Run or Evaluation Run.

V1 logical compute jobs are `training` or `evaluation` and are derived only from a complete valid Study Run plan. Worker processes one such execution attempt at a time rather than an entire Study or trial.

Queued work does not create a `running` Run merely because it is waiting for capacity. A concrete Run is allocated only when execution is starting and concrete preflight has succeeded.

Worker-produced files remain execution-side candidates until Controller accepts them under the applicable Training Run or Evaluation Run result contract.

Immutable Corpus artifacts may be reused through Worker-local content-addressed caching after integrity verification. Mutable upstream annotation databases are not Worker execution inputs.

## Responsibility boundaries

| concern | owner |
|---|---|
| MLDB entity and artifact semantics | Existing catalog, training, model, evaluation, and Study specs. |
| Public caller operations | `spec:mldb.api`. |
| Controller / Queue / Worker ownership | `spec:mldb.orchestration.responsibility_model`. |
| Typed asset resolution and common preflight rules | `spec:mldb.runtime`. |
| Study plan and retry semantics | `spec:mldb.study`. |
| Training/Evaluation Queue work units and dependency boundary | `spec:mldb.orchestration.job_model`. |
| Durable SQLite Queue lifecycle, leases, and reconciliation | `spec:mldb.orchestration.queue_lifecycle`. |
| Queue SQLite storage format | `spec:mldb.orchestration.queue_storage_format`. |
| Worker API operation and retry semantics | `spec:mldb.orchestration.worker_api`. |
| Exact HTTP routes, authentication, and byte-stream transport | Future Worker wire contract. |
| Corpus transfer protocol and cache eviction | Future orchestration contract. |
| Worker authentication and transport security | Future orchestration contract. |

## Non-goals

- Define future Queue schema migrations beyond v1.
- Define exact HTTP routes, authentication, or byte-stream transport for Worker communication.
- Define Worker authentication or authorization.
- Define exact Worker cache directory layout or eviction policy.
- Make queue state authoritative MLDB history.
- Define top-level direct Training or Evaluation queue-entry requests in v1.
- Allow Workers to allocate Run IDs or directly commit canonical Model, Run, or formal artifact state.
- Synchronize mutable annotation databases as substitutes for registered Corpus artifacts.

## Topics

| title | kind | ref | summary |
|---|---|---|---|
| Orchestration responsibility model | Concept | `spec:mldb.orchestration.responsibility_model` | Controller, Queue, and Worker responsibilities, dependency direction, Run/job separation, and remote execution boundary. |
| Orchestration job model | Concept | `spec:mldb.orchestration.job_model` | Study-derived Training and Evaluation work units, dependencies, Run-attempt separation, and recovery boundary. |
| Queue lifecycle | Concept | `spec:mldb.orchestration.queue_lifecycle` | Durable SQLite job states, attempt/lease semantics, canonical-commit ordering, and startup reconciliation. |
| Queue SQLite storage format | Contract | `spec:mldb.orchestration.queue_storage_format` | `.local/mldb/queue.sqlite`, v1 jobs/attempts DDL, indexes, timestamps, and transaction boundaries. |
| Worker API | Contract | `spec:mldb.orchestration.worker_api` | Pull acquisition, heartbeat, immutable asset retrieval, candidate upload, outcome reporting, and communication retry semantics. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb` | Parent MLDB overview. |
| `spec:mldb.api` | Provides the public application boundary above internal Queue and Worker orchestration. |
| `spec:mldb.runtime` | Supplies reusable execution semantics consumed by Controller and Worker. |
| `spec:mldb.study` | Supplies immutable plan intent and retry lineage consumed by orchestration. |
| `spec:mldb.training` | Defines Training Run and canonical learned-result semantics. |
| `spec:mldb.evaluation` | Defines Evaluation Run and formal result semantics. |
| `spec:mldb.repository` | Defines canonical MLDB locations that Queue and Worker must not replace. |
| `spec:mldb.orchestration.worker_api` | Defines the pull-oriented communication boundary used by separate Workers. |
