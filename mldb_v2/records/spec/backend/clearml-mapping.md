# Contract: ClearML backend mapping

- **id**: `spec:mldb.v2.backend.clearml_mapping`
- **status**: draft
- **date**: 2026-09-17
- **parent**: `spec:mldb.v2.backend`
- **contract_class**: `adapter`

## Mapping

| MLDB | ClearML |
|---|---|
| Namespace | Project `mldb/<namespace>` |
| one Study Result execution | one Pipeline Run / controller Task |
| immutable Study Plan | Pipeline DAG/configuration source |
| trial | logical Pipeline branch plus searchable metadata |
| one training/evaluation attempt | child Pipeline step Task |
| public parameters | Task hyperparameters/configuration |
| accepted execution telemetry | child Task Scalars/plots |
| selected Study-summary projection | Pipeline/controller metrics/artifacts/models |
| formal artifact bytes | ClearML artifact/model projection + canonical S3 reference |
| backend attempt identity | child ClearML Task ID |

A Study is not a ClearML Project. Multiple Study executions in one Namespace share the same Project, while each Study Result has its own Pipeline Run identity.

## Pipeline identity

Pipeline creation/recovery is keyed by exact MLDB Study Result identity plus its immutable Plan/source commit. Retrying client creation after an ambiguous response MUST recover the same logical Pipeline Run rather than create a duplicate execution.

## Pipeline construction

The adapter projects the immutable Study Plan into a ClearML Pipeline DAG. Training and Evaluation nodes remain ordinary ClearML Tasks and continue to invoke the common MLDB execution harness.

MLDB releases only semantically eligible logical stages to the backend. After release, the ClearML adapter owns physical Task creation/enqueue, queue placement, worker selection, retry/liveness, and cancellation. MLDB does not run a second queue or resource scheduler beside ClearML.

Plan dependencies are necessary but not sufficient for downstream release. A newly-trained trial's Evaluation nodes remain semantically gated until MLDB accepts the Training candidate and canonical Model lineage. The Pipeline projection may predeclare those nodes for UI, but the adapter must not create/enqueue the child Task until that acceptance boundary opens.

Existing-Model trials may expose their Evaluation nodes immediately after MLDB validates the existing Model lineage required by the Plan.

## UI projection

The Pipeline Run is the primary ClearML UI entrypoint for one MLDB Study Result. Child Tasks SHOULD be grouped into stable Pipeline stages suitable for human navigation, for example `training`, named Evaluation groups, and deployment/diagnostic groups when the Study contains them.

Selected child metrics/artifacts/models MAY be mirrored to the Pipeline/controller Task for Study-level comparison. Such mirroring is presentation only and MUST NOT create a second formal metric authority.

Task and Pipeline names are human-readable presentation and MUST NOT be parsed as identity.

## Required metadata

The controller Task and every child Task must expose enough searchable metadata to recover the full MLDB namespace, Study ID, Plan ID, Study Result ID, source commit, and backend ownership identity. Child Tasks additionally expose trial, stage kind/name/coordinate, and canonical Architecture/Model/Protocol/Corpus references as applicable.

## Source execution

Every child Task executes from the Study Plan's pinned Git commit or an equivalent verified repository snapshot. Definition companion hashes, source closure, Model lineage, and Corpus manifest digests remain subject to the existing MLDB preflight/runtime integrity checks.

## Retry, cache, and cancellation

ClearML retry/liveness is operational policy for a Pipeline step. Multiple retry Tasks remain attempts of one MLDB logical stage and are collected in deterministic attempt order.

Pipeline step caching/reuse is disabled by default for formal MLDB execution. A fresh Study Result must not silently inherit completion from an older Pipeline Run.

Study cancellation maps to Pipeline/controller cancellation plus active child cancellation. MLDB remains responsible for the canonical `cancelling`/terminal Study Result transition and formal child result acceptance.

## Authority

ClearML Pipeline/Task status, parameters, telemetry, summary projections, and logs are operational views. They never replace canonical MLDB Plan/Result/Model records. A ClearML reporting/UI outage must not invalidate an otherwise valid stage candidate, and a green Pipeline/Task is not sufficient for canonical MLDB success.
