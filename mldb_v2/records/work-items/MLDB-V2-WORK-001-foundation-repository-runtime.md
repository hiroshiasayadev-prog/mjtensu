# MLDB-V2-WORK-001: Foundation and repository runtime

- **status**: completed
- **date**: 2026-09-09
- **depends_on**: []
- **source_refs**: `spec:mldb.v2.common`, `spec:mldb.v2.storage`, `spec:mldb.v2.repository`

## Goal
Implement the shared runtime primitives required by every later v2 component: canonical identity/value validation, repository layout/resolution/listing, canonical writes, short mutation coordination, and reusable Git/object-storage access needed by the frozen contracts.

## Boundary
Own `mldb_v2/src/common/`, `src/storage/`, and `src/repository/`. Do not implement Catalog semantics, Study compilation, backend scheduling, or CLI behavior. Legacy v1 flat layout is never a fallback.

## Task candidates
| task | responsibility | dependency |
|---|---|---|
| MLDB-V2-TASK-001-01 | Implement common ID/public-parameter/diagnostic validation and canonical value helpers matching Skeleton shapes. | - |
| MLDB-V2-TASK-001-02 | Implement exact namespace-first resolution and deterministic repository inventory/listing with structural diagnostics. | T01 |
| MLDB-V2-TASK-001-03 | Implement immutable canonical create, nonterminal StudyResult replacement, and process-safe StudyResult mutation coordination. | T01 |
| MLDB-V2-TASK-001-04 | Implement shared ArtifactRef/manifest integrity and internal Git/object-byte access primitives required by later verification/runtime code. | T01 |
| MLDB-V2-TASK-001-05 | Verify repository/common/storage conformance and failure behavior. | T02,T03,T04 |

## Completion condition
- Exact resolution never scans or guesses kind.
- Broad listing reports malformed candidates instead of silently hiding them.
- Canonical writes are atomic/idempotent under the Spec lifecycle rules.
- Coordination does not become a queue/lease/heartbeat subsystem.
- Artifact and manifest integrity helpers expose no credentials in canonical values.
- Focused tests pass and public runtime shapes conform to the frozen Skeleton.
