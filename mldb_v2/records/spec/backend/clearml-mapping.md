# Contract: ClearML backend mapping

- **id**: `spec:mldb.v2.backend.clearml_mapping`
- **status**: draft
- **date**: 2026-09-18
- **parent**: `spec:mldb.v2.backend`
- **contract_class**: `adapter`

## Mapping

| MLDB | ClearML |
|---|---|
| Namespace | logical Project root `mldb/<namespace>` |
| Task-level Pipeline UI container | native hidden Pipeline subproject `mldb/<namespace>/.pipelines/<task-local-id>` when required by the ClearML server/UI |
| one Study Result execution | one Pipeline Run / controller Task inside its MLDB Task Pipeline subproject |
| immutable Study Plan | Pipeline DAG/configuration source |
| trial | logical Pipeline branch plus searchable metadata |
| one training/evaluation attempt | child Pipeline step Task |
| public parameters | Task hyperparameters/configuration |
| accepted execution telemetry | child Task Scalars/plots |
| selected Study-summary projection | Pipeline/controller metrics/artifacts/models |
| formal artifact bytes | ClearML artifact/model projection + canonical S3 reference |
| backend attempt identity | child ClearML Task ID |

The MLDB Namespace remains the logical ClearML Project root. A ClearML server/UI that represents Pipelines as hidden subprojects may require one native `.pipelines/<task-local-id>` UI container under that root; this is an adapter/UI detail, not a new MLDB semantic Project. All Studies compatible with the same MLDB Task share that Pipeline subproject, while every Study Result keeps its own Pipeline Run/controller identity and Study/Plan metadata. Task grouping is presentation-only and does not merge Study semantics or execution identity.

## Pipeline identity

Pipeline creation/recovery is keyed by exact MLDB Study Result identity plus its immutable Plan/source commit. Retrying client creation after an ambiguous response MUST recover the same logical Pipeline Run rather than create a duplicate execution.

## Pipeline construction

The adapter projects the immutable Study Plan into a ClearML Pipeline DAG. Training and Evaluation nodes remain ordinary ClearML Tasks and continue to invoke the common MLDB execution harness.

MLDB releases only semantically eligible logical stages to the backend. After release, the ClearML adapter owns physical Task creation/enqueue, queue placement, worker selection, retry/liveness, and cancellation. MLDB does not run a second queue or resource scheduler beside ClearML.

ClearML queue/resource routing is backend operational configuration, not Study/Evaluation semantics. The adapter MAY define a default queue plus exact logical-stage overrides. A stage override MAY select a different ClearML queue and MAY override the Docker GPU selector, including explicit `null` to omit `--gpus` for CPU-only work. Routing metadata MUST NOT be written into canonical `StageInput`, Study, Protocol, Plan, or Result records. The same resolved route MUST be represented in both the native Pipeline node projection and the child Task enqueue operation.

A prebuilt ClearML task image MAY provide the pinned runtime dependencies. When prebuilt-runtime mode is enabled, the adapter MAY instruct ClearML Agent to reuse the image's system Python instead of creating a fresh per-Task virtualenv, while retaining the Task requirements declaration for compatibility checking/provenance. The prebuilt image therefore becomes deployment infrastructure and MUST be rebuilt whenever the pinned remote runtime package contract changes.

Plan dependencies are necessary but not sufficient for downstream release. A newly-trained trial's Evaluation nodes remain semantically gated until MLDB accepts the Training candidate and canonical Model lineage. The Pipeline projection may predeclare those nodes for UI, but the adapter must not create/enqueue the child Task until that acceptance boundary opens.

Existing-Model trials may expose their Evaluation nodes immediately after MLDB validates the existing Model lineage required by the Plan.

## UI projection

The Pipeline Run is the primary ClearML UI entrypoint for one MLDB Study Result. Child Tasks SHOULD be grouped into stable Pipeline stages suitable for human navigation, for example `training`, named Evaluation groups, and deployment/diagnostic groups when the Study contains them.

Selected child metrics/artifacts/models MAY be mirrored to the Pipeline/controller Task for Study-level comparison. When the same Evaluation stage is applied to multiple trial Models, MLDB SHOULD aggregate the accepted canonical Evaluation metrics by stage into a bounded comparison projection. The ClearML adapter SHOULD render that projection as one compact comparison table plus one fixed-height categorical bar plot per formal Evaluation metric, grouped under the Evaluation stage. All metric plots SHOULD be directly visible without a metric-selector dropdown; the native ClearML graph-group collapse control MAY be used to hide/show the Evaluation group as a whole. Detailed per-trial telemetry remains on the child Evaluation Tasks, and controller Scalars MUST NOT be repurposed merely to simulate cross-model categorical comparison. Controller comparison tables/plots MUST be emitted only from a terminal Study summary; in-progress summaries continue to update canonical controller configuration/comment/native node status but MUST NOT append comparison Plot events, because ClearML does not provide reliable overwrite semantics for repeated Plot events at the same metric/variant/iteration and an early partial projection can otherwise remain visible after completion. Terminal comparison events MUST be synchronously flushed before the controller Task transitions to its terminal ClearML status; ClearML status transitions do not themselves flush queued Logger events.

The comparison projection SHOULD carry each formal metric's Evaluation Protocol `preference`. For `higher` or `lower`, the ClearML categorical bars SHOULD use rank-oriented color semantics from preferable to less preferable while preserving trial order; equal values SHOULD receive the same rank color. For `neutral` or omitted preference, the adapter SHOULD leave Plotly's normal bar color unchanged. Preference metadata is visualization semantics only and MUST NOT become an implicit winner-selection or optimization policy.

The comparison projection SHOULD also carry the referenced Evaluation Protocol identity plus its reusable human-facing `name` and `description`, and each formal metric's `description`. The adapter SHOULD surface Protocol-level prose through the controller Task description/comment. It SHOULD additionally render one consolidated controller-level `Evaluation Metrics` table before the per-stage comparisons, with one row per Evaluation metric and columns for logical Evaluation stage, metric name, metric `description`, and `preference`. The adapter SHOULD NOT duplicate the same explanatory prose beneath every per-stage comparison. Evaluation and metric prose belong to the Protocol definition rather than being duplicated in every Study.

Pipeline trial/step labels SHOULD use human-readable Model/Architecture lineage when available while retaining the immutable MLDB trial/coordinate identity in metadata. The native ClearML Pipeline DAG is the execution-flow view; the adapter SHOULD NOT publish a redundant custom execution-flow chart on the controller. Task and Pipeline names are presentation only and MUST NOT be parsed as identity.

Evaluation Protocol artifact `study_view` controls optional Study/controller mirroring. `hidden` or omission leaves the artifact on its child Evaluation Task only. `select` SHOULD produce one controller-level artifact view with trial/model selection when the format is renderable. `all` SHOULD mirror each available trial/model artifact directly. ClearML may implement `select` with native Plotly controls or an equivalent selector, but the selected trial/model is UI state only and MUST NOT be interpreted as a formal winner. Study-level mirroring reads already-accepted artifact references and remains observational; failure to render or retrieve the UI copy MUST NOT change canonical Evaluation success.

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
