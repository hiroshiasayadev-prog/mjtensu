# MLDB-V2-ADR-API-001: Use one resumable application boundary

- **status**: accepted
- **date**: 2026-09-09
- **depends_on**:
  - MLDB-V2-ADR-ARCHITECTURE-001
  - MLDB-V2-ADR-STUDY-001
  - MLDB-V2-ADR-RESULTS-001
- **supersedes**:
- **migrated_to_spec**: `spec:mldb.v2.api`

## Context

MLDB v1 exposed useful Python operations but normal experiment use still required users or ad hoc
scripts to invoke Python entrypoints and manually bridge planning, submission, waiting, collection,
and dependent Evaluation admission.

MLDB v2 also has semantic dependency progression: Evaluation work becomes ready only after the
upstream Training Result and Model are canonically accepted. A usable interface therefore needs a
resumable Study driver rather than a collection of unrelated helper scripts.

## Decision

MLDB v2 defines one transport-independent application API used by CLI, Python callers, and any
future HTTP adapter. Domain behavior MUST NOT be reimplemented separately by those adapters.
A formal Study execution is progressed by one idempotent application operation that reconciles
canonical Plan intent with backend terminal outcomes, accepts results, marks blocked work skipped
when required, admits newly-ready planned coordinates, and closes the Study Result when possible.

A convenience `run` operation creates a fresh formal execution and repeatedly invokes this same
progression until terminal. A separate `resume` operation does the same for an existing non-terminal
Study Result. Process restart therefore does not require hidden in-memory workflow reconstruction.

The normal human interface is a repository-installed `mldb` command, not direct invocation of
individual Python files. `plan` and one-pass `advance` remain available for inspection/automation.
Read-only observation is separate from progression.

## Rationale

A single application boundary prevents CLI, HTTP, and scripts from drifting into distinct MLDB
semantics. Idempotent progression provides crash/restart recovery without reintroducing a daemon,
queue database, or MLDB-owned worker scheduler.

## Consequences

`mldb run <study-ref>` covers validate -> plan -> start -> advance/wait -> terminal result.
`mldb resume <study-result-ref>` resumes that progression for an existing execution.
`mldb status` and `mldb watch` are observational and do not progress a Study; cancellation is explicit.
Backend queueing remains backend-owned; the application driver owns only Plan-defined semantic DAG
progression and canonical acceptance.
