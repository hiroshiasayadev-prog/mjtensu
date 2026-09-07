# Concept: MLDB orchestration responsibility model

- **id**: `spec:mldb.orchestration.responsibility_model`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.orchestration`

## What this is

Defines the logical responsibility boundary between Controller, Queue, and Worker for local or distributed MLDB execution.

The model fixes semantic ownership while leaving concrete process layout, queue technology, and Worker transport details open.

## Concept model

| component | owns | does not own |
|---|---|---|
| Controller | Launch preflight, Study plan materialization, Run allocation, runtime-generated canonical MLDB writes, result acceptance, terminalization, automatic Model creation, Study reconciliation, authoritative immutable execution inputs. | GPU scheduling policy internals, queue leases, Worker-local cache policy, model-family-specific protocol behavior. |
| Queue | Logical jobs, dependency readiness, claims, leases, heartbeats, retry timing, attempt counters, priorities, capability matching, and other operational scheduler state. | Run IDs, canonical Run status, Model identity, formal artifact acceptance, durable MLDB execution history. |
| Worker | Work acquisition, local asset/cache preparation, executable invocation, execution-local work files, candidate result preparation, liveness reporting, and candidate result handoff. | Run ID allocation, canonical `run.yaml` mutation, Model creation, Study plan mutation, final result acceptance, or final MLDB terminal-state choice. |

Controller and Queue are logically separate even when one server process implements both.

Worker is separately executable and communicates across a Worker API boundary.

## Dependency direction

```text
                     canonical MLDB repository
                              ^
                              |
                         Controller
                         ^       |
             result      |       | logical jobs
            candidates   |       v
                         Worker <- Queue
                           ^
                           |
                      pull / lease
```

Queue coordinates availability and scheduling. Controller decides whether an assigned execution is valid MLDB work and owns the resulting durable state. Worker performs the compute attempt.

## Controller rules

- Controller must remain the orchestration authority for runtime-generated canonical MLDB state.
- Controller validates the complete sealed Study and referenced immutable inputs before Study Run allocation.
- Controller derives Queue work only after complete valid Study Run plan materialization succeeds.
- Controller performs or confirms all concrete Run preflight required by the selected plan coordinate before allocating a Training Run or Evaluation Run.
- Controller allocates MLDB Run IDs; Queue and Worker do not.
- Controller creates the initial schema-valid `running` Run record only when concrete execution is starting.
- Controller validates Worker result candidates before accepting them as canonical formal results.
- Controller chooses the final Evaluation Run status according to `spec:mldb.evaluation.evaluation_run_lifecycle` and result-validation rules.
- Controller finalizes completed Training Runs only after canonical learned-state acceptance and ensures the deterministic Model record.
- Controller owns Study Run finalization and reconciliation from immutable plan intent plus child Run lineage.
- Controller may expose immutable input artifacts or artifact retrieval information to Workers.
- Controller must not make Queue operational state the source of truth for completed MLDB history.

Repository authoring of reusable draft definitions is separate from Controller's exclusive ownership of runtime-generated execution state.

## Queue rules

- Queue job identity is operational and distinct from Training Run or Evaluation Run identity.
- V1 Queue compute jobs are exactly `training` or `evaluation` and are derived from complete valid Study Run plan coordinates.
- A training job exists only for a training-derived plan row.
- An evaluation job exists for every plan row and evaluation stage.
- A logical job may wait without a corresponding `running` MLDB Run.
- One logical Study coordinate may produce multiple child Run attempts across retries.
- Queue may record attempt count, claims, leases, heartbeat state, retry timing, priority, and dependency state without adding those fields to MLDB Run records.
- Queue must not reopen or mutate terminal MLDB Runs during retry.
- Queue must not directly create Models or canonical formal artifacts.
- Queue failure or replacement must not invalidate existing MLDB Run, Model, Study Run, plan, or formal artifact history.
- V1 Queue operational state is durably persisted in Controller-local SQLite according to `spec:mldb.orchestration.queue_lifecycle`.
- Study-owned queue state must be reconcilable from the immutable Study Run plan and existing child lineage for already accepted canonical work.

The concrete v1 SQLite representation belongs to `spec:mldb.orchestration.queue_storage_format`. Scheduling policy such as retry limits, backoff, and prioritization remains outside this concept.

## Worker rules

- Worker acquires work through the pull-oriented `spec:mldb.orchestration.worker_api` boundary.
- Retryable Controller communication failure does not by itself become a Training or Evaluation domain failure; Worker retries the same logical communication operation every 10 seconds without a finite retry count.
- Domain execution failure is reported to Controller and is never locally rerun under the same attempt/Run identity.
- Worker processes one Training or Evaluation execution attempt at a time rather than owning an entire Study or trial.
- Worker must not require Controller to establish an inbound connection to the Worker for normal work acquisition.
- Worker consumes only the execution inputs selected by Controller for the assigned attempt.
- Worker may keep execution-local working files and reusable immutable-asset caches outside canonical `mldb_data` result locations.
- Worker verifies every required assigned immutable byte object against the descriptor's supplied integrity identity before use or cache reuse.
- After verification, Worker may materialize those exact bytes at Worker-local execution paths and combine them with the Controller-selected validated metadata to construct the same runtime Handle types consumed by existing Training/Evaluation contexts and loaders. Repository-provenance paths unavailable on Worker remain absent rather than being replaced with fake canonical paths.
- Worker-local Handle execution paths and caches are not canonical repository locations and never become independent authority for asset identity or bytes.
- Worker loads executable assets and invokes the applicable Architecture, Train Protocol, or Evaluation Protocol according to existing runtime contracts.
- Worker reports execution outcome and result candidates to Controller rather than committing canonical Run state itself.
- Worker failure must not mutate unrelated MLDB Runs or Study branches.

For Training execution, Worker may perform the generic CPU tensor-state preparation and `torch.save` candidate serialization defined by `spec:mldb.training.canonical_weights` in execution-local storage.

For Evaluation execution, Worker returns the protocol-produced EvaluationResult candidate, including explicit unavailable outputs when applicable, plus candidate structured artifacts.

## Job-to-Run boundary

Logical scheduling and concrete execution history remain separate. Detailed Study-derived work-unit semantics belong to `spec:mldb.orchestration.job_model`.

```text
logical job
    |
    | assigned to Worker
    v
Controller preflight / confirmation
    |
    | succeeds
    v
allocate concrete Run
    |
    v
Worker execution attempt
    |
    +--> failed/cancelled/partial attempt
    |        |
    |        +--> Queue may schedule retry
    |                  |
    |                  v
    |             new concrete Run
    |
    +--> successful attempt
```

A queue claim does not itself imply that a Run must exist. The implementation may complete the final preflight before or as an assignment is activated, but concrete Run allocation must remain after successful preflight and before executable invocation.

Retry of terminal child work always creates a new Run ID according to Training Run and Evaluation Run lifecycle contracts.

## Corpus distribution boundary

Corpus synchronization uses the immutable registered Corpus artifact rather than its mutable upstream annotation source.

| concern | owner |
|---|---|
| Canonical Corpus identity, metadata, and integrity | MLDB repository and Corpus contracts. |
| Selection of the Corpus required by an execution | Controller. |
| Transfer or retrieval of missing immutable Corpus bytes | Orchestration data-distribution boundary. |
| Local reusable Corpus cache | Worker. |
| Cache hit acceptance | Worker must match the selected Corpus content integrity identity before reuse. |
| Cache eviction and directory layout | Future Worker/cache implementation contract. |

A Worker cache is an optimization and never a second Corpus source of truth.

The Corpus builder is not a distributed execution input. It remains a canonical authoring/materialization sibling used to explain or create the registered Corpus artifact; Training/Evaluation Worker execution consumes the assigned immutable Corpus artifact itself and must not require builder transfer merely to construct a runtime Corpus handle.

## Result handoff boundary

Worker output remains a candidate until Controller accepts it.

| result stage | owner |
|---|---|
| Protocol execution | Worker. |
| Execution-local candidate files | Worker. |
| Candidate transfer/reporting | Worker API boundary. |
| Formal contract validation | Controller using existing runtime contracts. |
| Canonical artifact commit | Controller. |
| Run terminal status mutation | Controller. |
| Automatic Model creation | Controller. |

This split permits Worker-side compute and serialization without granting remote compute processes canonical repository write ownership.

## Deployment boundary

A minimal deployment may physically use:

```text
mldb-server
  Controller
  Queue

mldb-worker
  Worker
  local immutable-asset cache
  local work area
```

The physical process names are illustrative rather than normative.

Local single-machine execution may use the same logical contracts without requiring network transport, provided Controller, Queue, and Worker ownership remains observable and testable.

## Boundary

| concern | owner |
|---|---|
| Core runtime resolution and executable semantics | `spec:mldb.runtime`. |
| Training Run and learned-result semantics | `spec:mldb.training`. |
| Evaluation Run and formal-result semantics | `spec:mldb.evaluation`. |
| Study plan, lineage, and retry semantics | `spec:mldb.study`. |
| Canonical repository placement | `spec:mldb.repository`. |
| Worker API operation, idempotency, and communication-retry semantics | `spec:mldb.orchestration.worker_api`. |
| Exact HTTP endpoint/authentication/streaming contract | Future Worker wire contract. |
| Study-derived Queue work units | `spec:mldb.orchestration.job_model`. |
| Durable Queue job/attempt lifecycle and reconciliation | `spec:mldb.orchestration.queue_lifecycle`. |
| Exact v1 SQLite Queue schema | `spec:mldb.orchestration.queue_storage_format`. |
| Concrete lease timing and retry policy | Future orchestration policy contract. |
| Corpus transfer protocol and cache implementation | Future orchestration contract. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.orchestration` | Parent orchestration overview. |
| `spec:mldb.orchestration.job_model` | Defines Study-derived Training and Evaluation work units and their Run-attempt boundary. |
| `spec:mldb.orchestration.queue_lifecycle` | Defines durable SQLite Queue states, leases, and reconciliation authority. |
| `spec:mldb.orchestration.queue_storage_format` | Defines the concrete v1 jobs/attempts schema and `.local/mldb/queue.sqlite` placement. |
| `spec:mldb.orchestration.worker_api` | Defines acquisition, heartbeat, asset/candidate transfer semantics, idempotent outcome reporting, and communication retry. |
| `spec:mldb.runtime` | Supplies common execution semantics split across Controller and Worker. |
| `spec:mldb.study.study_run_lifecycle` | Supplies retry and reconciliation semantics consumed by Queue and Controller. |
| `spec:mldb.training.canonical_weights` | Defines Training candidate serialization semantics used on Worker and accepted by Controller. |
| `spec:mldb.evaluation.result_validation` | Defines Evaluation candidate acceptance performed by Controller. |
