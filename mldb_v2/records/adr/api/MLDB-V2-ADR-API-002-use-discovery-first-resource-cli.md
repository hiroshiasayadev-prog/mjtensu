# MLDB-V2-ADR-API-002: Use a discovery-first resource-oriented CLI

- **status**: accepted
- **date**: 2026-09-09
- **depends_on**:
  - MLDB-V2-ADR-API-001
  - MLDB-V2-ADR-REPOSITORY-001
- **supersedes**:
- **migrated_to_spec**: `spec:mldb.v2.cli`

## Context

A CLI that requires users or automation to already know every Namespace, Study, Result, and entity
ID makes routine inspection expensive and encourages filesystem scanning. Authoring checks also need
both single-target and broad repository/namespace scopes; requiring one command per definition is not
a usable operating model.

Existing resource-oriented CLIs establish a useful pattern: discovery/list operations default to a
broad view, selectors narrow that view, mutations require explicit targets, and machine-readable
output is a first-class interface rather than an afterthought.

## Decision

MLDB v2 CLI is discovery-first and resource-oriented. Read/check operations accept optional scope;
omitting scope means the broadest safe applicable scope. `validate` and `verify` support repository,
Namespace, kind, and individual-definition scopes with one shared selector model.
Mutation operations do not infer broad scope accidentally. Multi-target mutation requires explicit
bulk intent. CLI list/check/status operations support deterministic structured output so an AI agent
or automation can inspect repository state without custom filesystem discovery.

`watch` is observational and read-only. Study progression belongs to `run`/`resume` and the
application Study driver; watching state does not itself advance execution.

## Rationale

This keeps common use proportional to intent: one command lists or checks many objects, while an
individual mutation remains explicit. It also makes the CLI a stable discovery surface for both
humans and agents instead of forcing callers to know repository layout details.

## Consequences

The initial CLI includes resource discovery (`get`, `ps`), scoped authoring checks, explicit
execution actions (`run`, `resume`, `rerun`, `cancel`), read-only monitoring (`status`, `watch`,
`logs`), and environment diagnosis (`doctor`). Common selectors/output rules are specified once and
reused across commands where applicable.
