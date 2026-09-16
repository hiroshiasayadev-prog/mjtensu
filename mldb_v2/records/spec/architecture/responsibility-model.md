# Contract: Responsibility model

- **id**: `spec:mldb.v2.architecture.responsibility_model`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.architecture`
- **contract_class**: `architecture`

## Ownership

| concern | authoritative owner |
|---|---|
| Definition schema and identity | MLDB canonical records |
| Definition verification/sealing | MLDB |
| Study expansion and trial identity | MLDB |
| Formal metric/artifact contract | MLDB Evaluation Protocol |
| Model lineage | MLDB |
| Formal terminal history | MLDB canonical records |
| Large artifact bytes | content-verified object storage |
| Queue/resource scheduling | execution backend |
| Worker/agent registration | execution backend |
| Attempt retry/liveness | execution backend |
| Live logs/progress | execution backend |
| Visualization and ad hoc plots | execution backend |
| Backend DB | operational projection only |

## Rules

- Backend state MUST NOT overwrite canonical MLDB semantics.
- MLDB MUST NOT infer formal success from a backend task name or UI presentation.
- A backend scalar is formal only after Evaluation Protocol result acceptance.
- ClearML IDs MAY appear only as backend provenance.
- Object-store URIs are meaningful only together with canonical integrity metadata.
- Git commit identity pins transitive repository code used by executable definition siblings.
- MLDB v2 MUST NOT depend on v1 queue or worker state.

## Implementation placement

- MLDB v2 implementation: `mldb_v2/src/`.
- MLDB v2 skeleton: `mldb_v2/skeleton/`.
- MLDB v2 implementation tests: `mldb_v2/tests/`.
- Brewprint records: `mldb_v2/records/`.
- Canonical ML definitions/history: `mldb_data/`.
