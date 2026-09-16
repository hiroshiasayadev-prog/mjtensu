# Contract: ClearML backend mapping

- **id**: `spec:mldb.v2.backend.clearml_mapping`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.backend`
- **contract_class**: `adapter`

## Mapping

| MLDB | ClearML |
|---|---|
| Namespace | Project `mldb/<namespace>` |
| Study/Plan identity | Task metadata/tags/configuration |
| one training stage attempt | Task |
| one Evaluation coordinate attempt | Task |
| public parameters | Hyperparameters/configuration |
| accepted execution telemetry | Scalars/plots |
| formal artifact bytes | ClearML artifact/model projection + canonical S3 reference |
| backend attempt identity | ClearML Task ID |

Study is not a ClearML Project. Multiple Studies in one Namespace share the same Project.

## Required Task metadata

Every MLDB-created ClearML Task must expose enough searchable metadata to recover:

- full MLDB namespace;
- Study ID;
- Study Plan ID;
- Study Result/execution ID;
- trial ID;
- stage kind (`training` or `evaluation`);
- evaluation stage name and Evaluation coordinate ID when applicable;
- canonical Architecture/Model/Protocol/Corpus references;
- source Git commit.

Task naming is human-readable presentation and MUST NOT be parsed as identity.

## Source execution

ClearML execution must use the Study Plan's pinned Git commit or an equivalent verified repository
snapshot. Definition companion hashes and Corpus manifest digests must still match the plan before
domain execution starts.

## Authority

ClearML status, parameters, and telemetry are operational projections. Valid scalar events from
`spec:mldb.v2.common.telemetry` are projected to ClearML Scalars/plots. Numeric Evaluation candidate
metrics are projected automatically by the common execution harness. A ClearML reporting outage
MUST NOT invalidate an otherwise valid canonical stage outcome. Collection validates formal results
against the pinned MLDB plan and protocol before canonical persistence.

## Admission ownership

Each MLDB-created ClearML Task carries a deterministic logical ownership key derived from exact
Study Result ID + trial ID + stage kind + Evaluation coordinate when applicable. The adapter uses
that key to recover an already-created Task after ambiguous client failure and MUST NOT create a
second logical stage merely because admission is retried.

ClearML Task creation occurs only for coordinates admitted by `spec:mldb.v2.api.study_driver`.
