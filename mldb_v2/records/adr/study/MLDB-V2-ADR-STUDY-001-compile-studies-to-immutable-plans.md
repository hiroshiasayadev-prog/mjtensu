# MLDB-V2-ADR-STUDY-001: Compile Studies to immutable deterministic plans

- **status**: accepted
- **date**: 2026-09-09
- **depends_on**:
  - MLDB-V2-ADR-SCHEMA-001
  - MLDB-V2-ADR-STORAGE-001
- **supersedes**:
- **migrated_to_spec**: `spec:mldb.v2.study`

## Context

A Study is convenient for humans when it names candidate architectures, parameter values, seeds,
and common evaluation stages. It is not sufficient evidence of what actually ran if the Study can
change or if a backend dynamically chooses trials.

## Decision

A Study is declarative experiment intent. Before formal submission, MLDB compiles it into an
immutable Study Plan.

Compilation MUST be deterministic for the same Study, referenced definitions, source Git commit,
and ordered parameter values.

A Study Plan materializes:

- the exact source Git commit;
- exact referenced definition identities and integrity metadata;
- Corpus manifest digests;
- every trial in deterministic order;
- every effective public parameter value;
- every seed;
- model source for each trial;
- every evaluation stage and effective parameter value.

Trial IDs are assigned deterministically in plan order.

Study v2 supports both training new models and evaluating existing Model references.

The initial Study contract is static. Conditional/dynamic search spaces and backend-generated HPO
trials are not formal MLDB Study trials unless they are first materialized into an immutable plan.

## Rationale

The plan is the stable bridge between human experiment intent and replaceable execution backends.
It makes formal history reviewable even when the backend database is gone.

## Rejected alternatives

### Treat backend Tasks as the plan

Backend task creation order and metadata are operational state, not a durable experiment contract.

### Let ClearML HPO create formal trials dynamically

That would make the canonical trial set depend on backend behavior. Dynamic search may be supported
later through an explicit materialization contract.

## Consequences

Submission accepts a Study Plan, not an unexpanded Study. Changing a Study after planning never
changes an existing plan.
