# Contract: Responsibility model

- **id**: `spec:mldb.v2.architecture.responsibility_model`
- **status**: draft
- **date**: 2026-09-17
- **parent**: `spec:mldb.v2.architecture`
- **contract_class**: `architecture`

## Ownership

| concern | authoritative owner |
|---|---|
| Definition schema and identity | MLDB canonical records |
| Definition verification/sealing | MLDB |
| Study expansion and trial/stage identity | MLDB |
| Semantic dependency/gating rules | MLDB Plan + canonical accepted results |
| Formal metric/artifact contract | MLDB Evaluation Protocol |
| Model lineage | MLDB |
| Formal terminal history | MLDB canonical records |
| Large artifact bytes | content-verified object storage |
| Study execution container / operational DAG | execution backend |
| Queue/resource scheduling | execution backend |
| Worker/agent registration | execution backend |
| Attempt retry/liveness | execution backend |
| Physical child-step launch/dependency scheduling | execution backend |
| Live logs/progress | execution backend |
| Study/pipeline UI and visualization | execution backend |
| Backend DB | operational projection only |

## Rules

- Backend state MUST NOT overwrite canonical MLDB semantics.
- MLDB defines semantic readiness; the backend may implement the physical DAG and scheduling that realizes those already-defined dependencies.
- A downstream stage that requires newly-produced canonical lineage is not releasable merely because an upstream backend Task is green.
- MLDB MUST NOT implement a duplicate queue, worker registry, retry scheduler, heartbeat service, or resource scheduler when those are backend concerns.
- MLDB MUST NOT infer formal success from a backend Task/Pipeline name or UI presentation.
- A backend scalar is formal only when it is also accepted under the relevant MLDB result contract.
- ClearML Pipeline/Task IDs MAY appear only as backend provenance/navigation metadata.
- Object-store URIs are meaningful only together with canonical integrity metadata.
- Git commit identity pins transitive repository code used by executable definition siblings.
- MLDB v2 MUST NOT depend on v1 queue or worker state.

## ClearML specialization

For the ClearML backend, one canonical Study Result maps to one ClearML Pipeline Run. The Pipeline is the human/operational execution container; child Tasks perform the actual training/evaluation harness calls. ClearML owns physical scheduling and UI grouping, while MLDB retains semantic gates and canonical acceptance.

## Implementation placement

- MLDB v2 implementation: `mldb_v2/src/`.
- MLDB v2 skeleton: `mldb_v2/skeleton/`.
- MLDB v2 implementation tests: `mldb_v2/tests/`.
- Brewprint records: `mldb_v2/records/`.
- Canonical ML definitions/history: `mldb_data/`.
