# MLDB-V2-ADR-SCHEMA-001: Retain versioned domain definitions and first-class Model identity

- **status**: accepted
- **date**: 2026-09-09
- **depends_on**:
  - MLDB-V2-ADR-ARCHITECTURE-001
  - MLDB-V2-ADR-REPOSITORY-001
- **supersedes**:
- **migrated_to_spec**: `spec:mldb.v2.catalog`

## Context

Replacing the v1 execution machinery does not remove the need to distinguish prediction semantics,
materialized datasets, model topology, training procedure, evaluation procedure, and learned model
identity.

Generic experiment trackers normally expose parameters and artifacts but do not enforce these
semantic boundaries.

## Decision

MLDB v2 keeps the reusable definition classes:

- Namespace;
- Task;
- Corpus;
- Architecture;
- Train Protocol;
- Evaluation Protocol;
- Study.

MLDB v2 also keeps Model as a first-class learned identity.

Task owns prediction semantics. Corpus owns one immutable materialized sample collection.
Architecture owns unweighted model structure. Train Protocol owns reusable training procedure.
Evaluation Protocol owns reusable evaluation procedure and formal output declaration. Study owns
one declarative comparison intent.

A completed Training Result produces exactly one canonical Model. A Model can be evaluated by later
Studies without retraining.

Reusable semantic changes require a new versioned local ID. Backend IDs never replace MLDB IDs.

Public parameter values remain JSON-compatible values. Seeds are integers.

## Rationale

These boundaries are the part of MLDB v1 that prevents experiment history from becoming a bag of
loosely named parameters. First-class Model identity is required for later reevaluation on new
corpora or evaluation protocols.

## Rejected alternatives

### Collapse all definitions into Study parameters

This loses reusable identity and makes lineage depend on ad hoc parameter names.

### Treat a model as only a ClearML artifact

That prevents backend-independent reference and reevaluation.

## Consequences

Backend adapters must translate these entities into backend metadata without weakening their
canonical meaning. The backend is allowed to expose additional operational parameters and metrics,
but those values do not automatically become MLDB definitions or formal results.
