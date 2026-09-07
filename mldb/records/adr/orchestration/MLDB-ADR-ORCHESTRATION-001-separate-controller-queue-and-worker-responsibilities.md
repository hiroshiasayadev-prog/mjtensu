# MLDB-ADR-ORCHESTRATION-001: Separate Controller, Queue, and Worker responsibilities

- **status**: accepted
- **date**: 2026-09-04
- **depends_on**: MLDB-ADR-SCHEMA-005, MLDB-ADR-SCHEMA-008, MLDB-ADR-SCHEMA-010, MLDB-ADR-SCHEMA-012, MLDB-ADR-SCHEMA-017, MLDB-ADR-SCHEMA-018
- **supersedes**:
- **migrated_to_spec**: 2026-09-04

## Context

MLDB already separates durable experiment history from operational queue state.

Training Runs, Models, Evaluation Runs, Study Runs, immutable plans, and formal artifacts are authoritative MLDB state. Queue claims, leases, attempts, retries, worker identity, and scheduling state are operational and replaceable.

The intended deployment also needs compute workers to run independently from the server that owns MLDB state. Large immutable Corpus artifacts must be available to workers without making workers co-owners of canonical MLDB persistence.

Without an explicit responsibility boundary, distributed execution could allow workers to allocate Run IDs, mutate canonical `run.yaml` files, write Models directly, or couple Study retry behavior to one queue implementation.

MLDB therefore needs a control-plane and compute-plane split before the worker API and queue storage details are designed.

## Decision

Define three logical orchestration responsibilities: Controller, Queue, and Worker.

These are logical responsibility units. Controller and Queue may be deployed in the same process or service. Worker is a separately executable compute client connected through a worker API boundary.

### Controller

Controller owns MLDB control-plane semantics and runtime-generated canonical persistence.

Controller responsibilities include:

- accept direct execution and Study execution requests;
- perform or confirm the launch preflight required before concrete Run allocation;
- materialize immutable Study Run plans;
- allocate Training Run, Evaluation Run, and Study Run IDs under their lifecycle contracts;
- create and mutate runtime-generated canonical Run records while those records are mutable;
- accept Worker result candidates and apply the canonical result-validation contract;
- commit accepted canonical formal artifacts into MLDB-owned storage;
- finalize Training Run and Evaluation Run terminal state;
- ensure automatic Model creation after completed Training Runs;
- finalize or reconcile Study Run state from immutable plan intent and child lineage;
- expose the authoritative immutable assets and metadata required for Worker execution.

Among orchestration components, only Controller may mutate runtime-generated canonical MLDB execution records, Model records, Study Run plans, or formal result storage.

This restriction does not make Controller the only editor of reusable draft definitions. Repository-authoring workflows may still create or edit reusable MLDB definitions according to their own lifecycle contracts.

### Queue

Queue owns operational scheduling state only.

Queue responsibilities may include:

- logical job identity;
- dependency readiness;
- queued, claimed, running, blocked, retrying, or equivalent operational status;
- worker claim ownership;
- leases and heartbeats;
- attempt counters;
- retry timing and backoff;
- scheduling priority;
- worker capability matching.

Queue must not allocate MLDB Run IDs, define MLDB terminal status, mutate canonical MLDB execution records, or become the source of truth for completed experiment history.

A queue job is not a Training Run or Evaluation Run.

One logical planned coordinate may produce multiple immutable child Run attempts over time. For example, a failed Training Run retry creates another Training Run ID while the planned Study coordinate remains the same.

Queue storage technology and exact queue schema are not fixed by this decision.

### Worker

Worker owns one assigned compute attempt and its execution-local resources.

Worker responsibilities include:

- acquire work through the Worker API boundary;
- maintain liveness or execution progress required by the orchestration protocol;
- obtain the resolved immutable assets needed for the assigned execution;
- verify locally consumed immutable artifact integrity when required by the assignment contract;
- maintain execution-local working files and reusable local caches;
- load executable Architecture, Train Protocol, or Evaluation Protocol implementations through the MLDB runtime contract;
- invoke `train(context)` or `evaluate(context)` as assigned;
- perform execution-side generic result preparation required by the canonical runtime contract, including canonical Training Run weight candidate serialization when applicable;
- report successful result candidates, partial evaluation output information, failure, or cancellation back to Controller.

Worker must not allocate MLDB Run IDs, create Model records, choose final MLDB terminal state, or write directly into canonical MLDB execution-record or formal-artifact locations.

A Worker may produce local candidate files that become canonical artifacts only after Controller validates and commits them.

### Worker API direction

Worker execution uses a pull-oriented control boundary.

A Worker initiates work acquisition from the Controller/Queue side rather than requiring the Controller to open an inbound connection to each Worker.

The same boundary carries or enables work acquisition, liveness, execution result reporting, and access to required immutable execution inputs.

Exact HTTP paths, request/response schemas, authentication, streaming behavior, and transport technology are deferred to a dedicated Worker API contract.

### Queue job and Run allocation boundary

A logical job may exist in Queue before any concrete Training Run or Evaluation Run exists.

A queued job alone must not create a `running` MLDB Run.

For Worker-dispatched execution, Controller allocates the concrete Training Run or Evaluation Run only when execution is being started for an assigned Worker and the required launch preflight has succeeded.

This preserves the rule that every persisted `running` Run is a real validated execution attempt rather than waiting capacity in Queue.

Retrying a terminal child execution creates a new child Run ID according to the existing Run lifecycle contracts. Queue retry counters and attempt identifiers remain operational state and do not replace Run history.

### Corpus distribution boundary

Corpus remains an immutable registered MLDB asset owned by the canonical repository contract.

Worker may maintain a reusable local cache of immutable Corpus artifact bytes keyed by content integrity identity such as SHA-256.

On assignment, Worker may reuse a cached Corpus only when its content identity matches the resolved Corpus required by the assignment. Otherwise Worker obtains the immutable Corpus artifact through the orchestration data-distribution boundary and verifies integrity before use.

Mutable upstream annotation databases are not synchronized as Worker execution inputs through this mechanism. Worker consumes the frozen MLDB Corpus artifact selected for the Run.

Exact artifact-transfer endpoints, cache directory layout, eviction policy, and transport are deferred.

### Result handoff

Worker result output is a candidate until Controller accepts it.

For Training Run execution, Worker may create the runtime-defined `pytorch-state-dict` candidate from the returned trained module in execution-local storage. Controller verifies required integrity and result contracts before committing canonical `artifacts/weights.pt`, terminalizing the Training Run, and ensuring the Model record.

For Evaluation Run execution, Worker reports the `EvaluationResult` candidate and candidate structured-artifact files. Controller applies formal result validation and determines `completed`, `completed_partial`, or `failed` according to the Evaluation Run contract.

The mechanism by which candidate bytes are uploaded or transferred is deferred to the Worker API contract.

## Rationale

Controller ownership of canonical runtime persistence keeps one authority for Run lifecycle and Model identity even when compute is distributed across multiple machines.

Keeping Queue operational prevents scheduler technology from becoming part of durable MLDB semantics. Queue state can change, be rebuilt, or be replaced without rewriting immutable experiment history.

Separating logical jobs from immutable Run attempts directly supports retry semantics. A transient failed attempt remains historical evidence while the same planned coordinate can later succeed through another Run.

A pull-oriented Worker boundary simplifies worker discovery, firewall configuration, restart behavior, and dynamic worker count while remaining compatible with lease and heartbeat scheduling.

Content-addressed Worker caching fits immutable Corpus semantics and avoids repeatedly copying large Corpora while preserving integrity checks.

Treating Worker output as a candidate preserves the distinction between compute-side execution and Controller-side canonical acceptance.

## Rejected alternatives

### Let Workers write `mldb_data` directly

Multiple Workers writing Run YAML, Model YAML, and canonical artifacts would create concurrent ownership of lifecycle transitions and repository mutation. It would also make remote execution depend on shared filesystem semantics.

### Treat queue jobs as Training Runs or Evaluation Runs

Jobs may wait for capacity, be retried, lose leases, or produce multiple execution attempts. Mapping one job directly to one immutable Run would either leave misleading `running` Runs while queued or force mutable Run identity across retries.

### Make Worker the authority for terminal status

Worker can report execution facts, but only Controller has the authoritative Run record, formal result declarations, Study lineage, and repository acceptance boundary required to select canonical terminal state.

### Synchronize mutable annotation databases to Workers

Training reproducibility begins from registered immutable Corpus artifacts. Synchronizing the mutable upstream annotation source would bypass the frozen Corpus boundary and produce ambiguous execution inputs.

### Require separate network services for Controller and Queue

The responsibility boundary is logical. A small deployment can keep Controller and Queue in one server process while preserving replaceable queue state internally.

## Consequences

A future server implementation may consist physically of one `mldb-server` process containing Controller plus Queue responsibilities and one or more separate Worker processes.

Worker API design must preserve pull-oriented acquisition and must not grant Workers direct canonical MLDB write ownership.

Queue schema design must model logical work independently from immutable Training Run and Evaluation Run IDs.

Run allocation for queued work occurs at execution start after preflight, not when the logical job merely enters Queue.

Corpus distribution design can use immutable content identity and Worker-local cache without introducing a second mutable Corpus source of truth.

Result-transfer design must distinguish execution-local candidate bytes from Controller-accepted canonical formal artifacts.

Existing runtime, training, evaluation, and Study semantic contracts remain authoritative whether their executable work is performed locally or by a remote Worker.

## Evidence

Study Run retry semantics already require multiple immutable child Run attempts to be able to satisfy one planned coordinate without mutating the Study plan.

Training Run and Evaluation Run preflight rules already separate invalid launch requests from allocated execution attempts.

The repository contract already excludes queue and cache storage from canonical MLDB entity placement, while Corpus and formal result artifacts already carry integrity metadata suitable for remote verification.
