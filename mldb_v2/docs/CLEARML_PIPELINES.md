# ClearML Pipeline Mapping

This document defines the intended MLDB v2 -> ClearML operational mapping after W011.

The key principle is simple: **MLDB owns experiment semantics and canonical history; ClearML owns operational execution mechanics and UI.**

## 1. Concept mapping

The natural user-facing mapping is:

```text
MLDB Study Result        <->  ClearML Pipeline Run
MLDB Study Plan          <->  Pipeline DAG/configuration
MLDB trial               <->  logical Pipeline branch
MLDB training/eval work  <->  child Pipeline Task
MLDB telemetry           ->   child Task metrics/plots
selected Study summary   ->   Pipeline/controller metrics/artifacts
```

One Namespace remains the logical ClearML Project root. On ClearML servers that expose Pipelines through native hidden subprojects, the adapter additionally places controller Tasks under `mldb/<namespace>/.pipelines/<task-local-id>` and marks that subproject `pipeline` + `hidden`; this is a ClearML UI implementation detail, not a new MLDB semantic Project. Different Studies that target the same MLDB Task therefore appear as separate Runs inside one Task-level Pipeline card instead of creating one card per Study.

One fresh MLDB Study Result still gets one fresh, recoverable Pipeline Run. The Run name carries the Study identity; Task-level UI grouping does not merge Study Results or change canonical ownership.

## 2. What ClearML owns

ClearML owns the operational mechanics that are not the scientific meaning of the experiment:

- Pipeline/controller lifecycle and Study-level UI projection;
- physical child-Task creation/enqueue after MLDB semantic release;
- queue and worker placement;
- GPU/resource scheduling;
- operational retry/liveness;
- cancellation of Pipeline/active Tasks;
- logs, live status, Charts, and Pipeline UI;
- optional mirroring of selected child metrics/artifacts/models to the Pipeline summary.

MLDB should not grow a second queue, retry scheduler, heartbeat service, worker registry, or resource scheduler beside ClearML.

Stage-specific placement is still allowed at the adapter boundary. Production composition accepts an operational stage-route map; for example `onnx-cpu-latency` can be routed to a dedicated `latency-cpu` queue with `docker_gpu: null`, while every unlisted stage falls back to the normal `default` queue and global Docker GPU setting. This routing does not enter the Study or Evaluation Protocol YAML because queue names and worker topology are deployment concerns. The registry-reference worker additionally subscribes to queue `precision5820-gpu3060` so runtime-registry behavior can be verified without another `default` worker taking the Task.
For the validated deployment, the old Linux compute host runs two ClearML Agent processes: `old-gpu3090` subscribes only to `default`, while the CPU-only `old-cpu` subscribes only to `latency-cpu`. This preserves a single canonical CPU identity without consuming the GPU worker slot during CPU latency evaluation. Because both Agents share one physical host, concurrent GPU work can still perturb CPU timing; runs affected by host contention are not directly comparable. Heterogeneous workers may join `default`, but they must not join `latency-cpu`.

## 3. What MLDB keeps

MLDB also owns the runtime-version identity of a formal run. A fresh Study Result resolves the global runtime registry once and persists `runtime_registry_version`; all child StageInputs inherit that exact integer. Retry/resume retain it, and rerun reuses the source Study Result's version. A child Task must not resolve `latest` independently.

The runtime registry is global deployment state, not experiment-schema state. Architecture, Train Protocol, Evaluation Protocol, and Study YAML do not declare packages, repositories, virtualenvs, or registry versions. The immutable registry snapshot is sufficient package provenance; canonical Training/Evaluation results do not duplicate a package inventory.

Worker enforcement is currently active on `precision5820-gpu3060` and `old-gpu3090`. Each converges one reusable managed venv to the pinned version and re-execs the MLDB harness through that interpreter. The remaining workers have not yet copied this bootstrap. Therefore a pinned Study version is already canonical across MLDB, while package enforcement is guaranteed only on migrated workers until rollout completes.

MLDB remains authoritative for Study/Plan identity, trial/stage semantics, source pinning, Protocol contracts, semantic dependency gates, accepted Model lineage, formal result acceptance, and canonical Training/Evaluation/Study Results.

A ClearML Task or Pipeline becoming green never bypasses MLDB result acceptance.

## 4. Semantic gate between train and eval

The important boundary is newly-trained Model lineage.

```text
training child Task completes
        ↓
terminal candidate collected
        ↓
MLDB validates/persists Training Result + Model
        ↓
semantic gate opens
        ↓
backend releases dependent Evaluation child Tasks
```

A Pipeline may predeclare downstream Evaluation nodes so the DAG is visible, but those nodes must not execute merely because the training Task itself finished. The canonical acceptance boundary is the gate.

Controller callbacks/hooks are acceptable implementation seams when they call generic MLDB lifecycle/acceptance code. They must not copy model-family logic into the ClearML adapter.

## 5. Expected ClearML UI

Users should normally enter through **Pipelines**, not reconstruct a Study from a flat experiment list. One Pipeline Run should show the Study DAG, child status, timing, and navigable Task details.

Child Tasks should be grouped into stable stages so a multi-trial Study is collapsible instead of appearing as an unstructured wall of Tasks. A typical classifier Study may look like:

```text
Pipeline Run: shufflenet-spatial-screen-v1 / run-...
  training        4/4
  angle-eval      4/4
  manzu-diagnostic 4/4
  deployment      4/4
```

Opening a child node shows detailed Task telemetry/artifacts/logs. Training curves and dense Evaluation diagnostics stay on those child Tasks.

For multi-Model Evaluation, MLDB builds a Study-level comparison projection from accepted canonical Evaluation Results. The ClearML controller renders one compact comparison table plus one fixed-height Model Comparison plot per formal Evaluation metric, grouped by Evaluation stage, with readable trial labels derived from Model/Architecture lineage. All metric plots are visible directly; there is no metric-selector dropdown. Long Evaluation groups can be collapsed with ClearML's native graph-group chevron. Evaluation Protocol metric `preference` controls comparison coloring: `higher` and `lower` rank the models within that metric from blue through light-blue/light-red to red, while `neutral` or omitted preference keeps the normal Plotly color. Equal values within the same metric receive the same rank color. Each directional chart labels whether higher/lower is preferable and states that blue means more preferable and red less preferable. Trial ordering is unchanged and the colors do not select a winning model across metrics. Controller Scalars are not used for this cross-sectional comparison; time-series Training/Evaluation telemetry remains on the child Tasks. Comparison tables and plots are emitted only when the Study reaches a terminal status. While a Study is running, MLDB still updates the controller summary/configuration, comment, and native Pipeline node statuses, but deliberately does not publish partial comparison Plot events because repeated ClearML Plot events at the same metric/variant/iteration are not a reliable overwrite surface. The terminal comparison events are flushed synchronously before the controller Task is marked completed/failed/stopped because ClearML status transitions do not flush pending Logger events.

Evaluation explanations come from the reusable Evaluation Protocol definition itself. MLDB projects the Protocol identity, `name`, and `description` into the controller Task description/comment instead of consuming another fixed-height Plot card. Immutable `trial-XXXX` / evaluation-coordinate identities remain in metadata and canonical records even when the UI label is human-readable.

Formal Evaluation artifacts remain visible on their child Tasks. A Protocol may additionally set `artifacts.<name>.study_view` to control Study-level presentation: omitted/`hidden` keeps it child-only; `select` renders one controller surface with a Model/trial selector; `all` mirrors every Model/trial copy. This is intended for diagnostics such as confusion matrices, robustness plots, contact sheets, and layer-separation plots that are meaningful to inspect per Model but should not be silently discarded from the Study view. For `select` PNG artifacts, the controller keeps the same dropdown surface but references each child artifact through the ClearML Web `/files` proxy rather than embedding base64 image bytes in the Plot event; this keeps the controller event bounded and lets the browser fetch only the displayed image. The policy is backend-neutral presentation metadata and never changes canonical acceptance.

ClearML groups controller Plot events by metric and, in Compare/Plots, requests only the latest iteration for that metric. Therefore all mirrored Study Artifact variants under one controller metric such as `Study Artifact - <stage>` must be published as one projection generation at the same iteration. A refresh or repair must not advance only one artifact variant: if any sibling variant needs a new iteration, republish the complete visible sibling set at that same new iteration. Otherwise ClearML can legitimately return only the newly advanced variant and make the older sibling artifacts appear to disappear even though their canonical artifacts still exist. Normal terminal projection uses one common iteration for the full Study Artifact group; maintenance reprojection must preserve the same generation rule.

The native ClearML Pipeline DAG is the execution-flow view. MLDB does not publish a second custom Sankey/"Execution Flow" plot on the controller; for Existing-Model Studies such a duplicate plot degenerates into disconnected circles and adds no execution information.

## 6. Retry, cache, and rerun policy

Operational retry belongs to ClearML. `mldb retry-stage <study-result> --trial <trial> --coordinate <evaluation-coordinate>` is the bounded MLDB entrypoint for requesting one new physical attempt of a failed Evaluation while retaining the same logical Study Result and Evaluation Result.

The original child Task remains the unique logical ownership Task identified by `study_result + trial + kind + coordinate`. Retry Tasks do **not** claim a second logical ownership record; they carry explicit retry-owner/index identity and preserve the same immutable StageInput. Observation reconstructs an ordered attempt sequence from the owner followed by its retry Tasks and fails closed on gaps, duplicates, ownership disagreement, or ambiguous creation recovery. The native Pipeline node remains bound to the original owner Task so ordinary `admit` remains idempotent. A retry Task is associated with the same Pipeline controller as a child, preserving its own physical Task/log history without resetting or erasing the failed Task.

Canonical state is not changed back to `pending` while the retry is running. After terminal collection and normal MLDB result acceptance, the same Evaluation Result ID receives exactly one appended attempt and may remain `failed` or become `completed`. This first implementation is synchronous and supports failed Evaluations only; Training-stage retry is intentionally outside the v1 scope.

ClearML step caching/reuse is disabled by default for formal execution. `mldb rerun` creates a fresh Study Result/Pipeline Run from the exact immutable source Plan and the source Study Result's exact `runtime_registry_version`; it does not silently reuse an old green Task or move to a newer registry snapshot.

ClearML UI actions that create an execution without first allocating a canonical MLDB Study Result are not a supported formal MLDB entrypoint.

## 7. Interruption and cancellation

After the Pipeline exists, closing the local `mldb run`/`resume` process does not cancel backend work. `resume` reconnects to the same Study Result/Pipeline and reconciles completed children.

`mldb cancel` first records canonical cancellation intent, then requests Pipeline/active-child cancellation. Final canonical status still comes from normal result/cancellation reconciliation.

## 8. Authority reminder

ClearML is the operational execution engine and observability UI. It is deliberately not the source of truth for formal MLDB results.

If ClearML and canonical MLDB disagree, investigate and reconcile through the formal lifecycle; do not edit canonical YAML to match the UI.

## 9. Implementation status

As of 2026-10-02, the W011 Pipeline mapping is implemented and the global runtime-registry pin is part of canonical Study execution. An actual two-trial RTX 3090 ClearML Study has completed through training acceptance, dependent evaluation release, recovery/resume, and terminal canonical Study closure. One Study Result creates/recovers one ClearML controller Task with native Pipeline DAG configuration; released child Tasks are bound to exact Pipeline nodes before enqueue, detailed telemetry remains on child Tasks, and bounded Study-summary values are projected to the controller without requiring the ClearML Fileserver.

The controller Task is an operational/UI ownership container, not a second remote MLDB scheduler loop. MLDB reconciliation opens semantic gates; the ClearML backend performs Task creation/enqueue and ClearML queue/agent infrastructure owns worker/resource execution.

Actual verification found that API-server-2.17+ ClearML UIs discover Pipelines through native hidden `.pipelines/<pipeline-name>` subprojects. Controllers created directly in the Namespace Project remained valid Tasks but were invisible in the Pipelines page. The adapter now uses the native subproject layout while retaining recovery compatibility with the earlier flat placement. Final T011-05 closure still requires human-visible Pipeline-page confirmation and the remaining actual cancellation check.

Runtime-registry verification completed on both migrated GPU workers. `precision5820-gpu3060` proved the reusable managed-venv path without reinstalling unchanged base-visible PyTorch packages. `old-gpu3090` then completed `recognition-functional-video` at registry v2 through queue `recognition-functional` using `mldb-clearml-runner:recognition-functional-v1`; the MLDB harness executed as `/mldb-runtime-registry/venv/bin/python`, and the accepted Evaluation Result contained both a prediction trace and overlay MP4. This does not imply that the remaining workers are registry-managed.
