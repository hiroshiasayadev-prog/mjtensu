# MLDB-V2-ADR-STUDY-003: Pin formal plans to committed canonical inputs

- **status**: accepted
- **date**: 2026-09-09
- **depends_on**:
  - MLDB-V2-ADR-STUDY-001
  - MLDB-V2-ADR-STORAGE-001
- **supersedes**:
- **migrated_to_spec**: `spec:mldb.v2.study.source_pinning`

## Context

Formal experiments need a reproducible Git source boundary, but requiring the entire mjtensu
working tree to be clean would block ML work whenever unrelated application development is in
progress.

## Decision

Formal planning requires only the canonical v2 inputs consumed by the Plan to match the selected
Git commit. Unrelated source files and unrelated namespaces may remain dirty.
Backend execution uses the pinned commit or an equivalent verified snapshot and never captures the
caller's uncommitted patch as formal experiment source.

## Consequences

`mldb plan` checks the referenced `mldb_data/<namespace>/...` records/companions/manifests and other
canonical records used by the Plan against Git. Dirty frontend/backend work does not block planning.
Submission revalidates the persisted Plan/source pin before backend work begins.
