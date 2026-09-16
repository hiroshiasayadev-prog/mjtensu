# MLDB-V2-ADR-ARCHITECTURE-001: Separate experiment protocol from execution backends

- **status**: accepted
- **date**: 2026-09-09
- **depends_on**:
- **supersedes**:
- **migrated_to_spec**: `spec:mldb.v2.architecture`

## Context

MLDB v1 grew from an experiment schema into an execution platform. It owns queue persistence,
worker assignment, leases, heartbeat recovery, retries, remote execution, Run lifecycle, and
canonical result persistence. Those capabilities are useful, but mature execution systems such as
ClearML already provide queues, agents, retries, logs, remote execution, task state, and experiment
visualization.

The project still needs stronger domain contracts than a generic experiment tracker provides:
versioned Task, Corpus, Architecture, Train Protocol, Evaluation Protocol, Study, Model lineage,
deterministic experiment planning, formal result validation, and content integrity.

## Decision

MLDB v2 is an experiment protocol and canonical semantic layer, not a scheduler.

MLDB v2 owns:

- ML asset identity and schema;
- immutable reusable definitions;
- typed references and validation;
- deterministic Study compilation;
- formal model lineage;
- formal result acceptance;
- canonical execution history;
- content-integrity metadata;
- backend-independent submission and collection contracts.

Execution backends own:

- queueing and scheduling;
- worker or agent registration;
- GPU/resource assignment;
- retry mechanics and attempt execution;
- heartbeat, leases, and liveness;
- live logs and progress;
- operational task state;
- visualization UI.

Object storage owns large immutable bytes. Git owns version history of canonical MLDB records.

MLDB v2 MUST NOT introduce a second queue, worker lease system, generic scheduler, heartbeat
protocol, or backend-specific operational database while the selected backend already owns that
responsibility.

## Rationale

This retains the high-value part of MLDB v1 while removing infrastructure that is already solved
well by execution platforms. The protocol remains usable if ClearML is later replaced because
canonical identity, planning, lineage, and results do not depend on ClearML database semantics.

## Rejected alternatives

### Keep the v1 queue/worker stack and use ClearML only as a viewer

This duplicates operational state and creates reconciliation problems between two schedulers.

### Make ClearML the source of truth

A ClearML server loss or backend migration would then destroy the authoritative experiment model.
ClearML Task mutability and metadata are also weaker than the domain contracts required by MLDB.

## Consequences

`mldb_v2/src/` will contain protocol, repository, validation, backend-port, and adapter code.
There is no v2 orchestration/queue/worker subsystem unless a future accepted ADR demonstrates a
backend-independent requirement that cannot be met through the backend port.
