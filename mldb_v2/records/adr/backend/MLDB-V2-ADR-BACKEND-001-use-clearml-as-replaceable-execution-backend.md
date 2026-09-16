# MLDB-V2-ADR-BACKEND-001: Use ClearML as a replaceable execution backend

- **status**: accepted
- **date**: 2026-09-09
- **depends_on**:
  - MLDB-V2-ADR-ARCHITECTURE-001
  - MLDB-V2-ADR-STUDY-001
- **supersedes**:
- **migrated_to_spec**: `spec:mldb.v2.backend.clearml_mapping`

## Context

The project needs remote GPU execution, queues, agents, logs, task status, experiment comparison,
and visualization. ClearML supplies these capabilities and can be self-hosted.

A direct ClearML-shaped core would make later backend replacement difficult and would allow
operational metadata to become canonical by accident.

## Decision

ClearML is the initial implementation of the generic MLDB v2 backend port.

The mapping is:

```text
MLDB Namespace       -> ClearML Project
MLDB Study/Plan      -> Task metadata and grouping fields
MLDB trial stage     -> ClearML Task
MLDB scalar telemetry-> ClearML Scalars
MLDB parameters      -> ClearML Hyperparameters/configuration
MLDB artifact        -> ClearML artifact/model metadata plus canonical object-store reference
```

The default ClearML Project name is `mldb/<namespace>`.

Study is NOT mapped one-to-one to a ClearML Project. Multiple Studies in the same namespace belong
to the same project so they remain easy to compare.

ClearML Task IDs are backend provenance only. They may be recorded in formal MLDB results but never
form an MLDB entity identity.

## Rationale

This gives the project the operational platform it wants without turning ClearML into the semantic
source of truth.

## Rejected alternatives

### ClearML Project per Study

This recreates the cross-experiment comparison fragmentation that motivated the redesign.

### Copy ClearML Task schema into canonical MLDB records

Task schema is backend-specific and mixes operational and semantic concerns.

## Consequences

The ClearML adapter may evolve with ClearML APIs while the generic backend port and canonical
records remain stable.
