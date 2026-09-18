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

One Namespace remains the logical ClearML Project root. On ClearML servers that expose Pipelines through native hidden subprojects, the adapter additionally places controller Tasks under `mldb/<namespace>/.pipelines/<study-local-id>` and marks that subproject `pipeline` + `hidden`; this is a ClearML UI implementation detail, not a new MLDB semantic Project.

One fresh MLDB Study Result gets one fresh, recoverable Pipeline Run inside the Study's native Pipeline subproject.

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

## 3. What MLDB keeps

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

For multi-Model Evaluation, MLDB builds a Study-level comparison projection from accepted canonical Evaluation Results. The ClearML controller renders one compact comparison table and one bounded multi-metric bar-panel per Evaluation stage, with readable trial labels derived from Model/Architecture lineage. Each metric subplot compares the trial Models directly instead of producing one single-value chart per trial-metric pair. Controller Scalars are not used for this cross-sectional comparison; time-series Training/Evaluation telemetry remains on the child Tasks.

Evaluation explanations come from the reusable Evaluation Protocol definition itself. MLDB projects the Protocol `name` / `description` into the same `Study Comparison` plot as the comparison table, using the otherwise-unused lower portion of ClearML's fixed-height Plot card. This avoids a separate sparse guide Plot while keeping the evaluation definition adjacent to its results. Immutable `trial-XXXX` / evaluation-coordinate identities remain in metadata and canonical records even when the UI label is human-readable.

The native ClearML Pipeline DAG is the execution-flow view. MLDB does not publish a second custom Sankey/"Execution Flow" plot on the controller; for Existing-Model Studies such a duplicate plot degenerates into disconnected circles and adds no execution information.

## 6. Retry, cache, and rerun policy

Operational retry belongs to ClearML. Multiple retry Tasks remain attempts of the same MLDB logical stage and must be collected as such.

ClearML step caching/reuse is disabled by default for formal execution. `mldb rerun` creates a fresh Study Result/Pipeline Run from the exact immutable source Plan; it does not silently reuse an old green Task.

ClearML UI actions that create an execution without first allocating a canonical MLDB Study Result are not a supported formal MLDB entrypoint.

## 7. Interruption and cancellation

After the Pipeline exists, closing the local `mldb run`/`resume` process does not cancel backend work. `resume` reconnects to the same Study Result/Pipeline and reconciles completed children.

`mldb cancel` first records canonical cancellation intent, then requests Pipeline/active-child cancellation. Final canonical status still comes from normal result/cancellation reconciliation.

## 8. Authority reminder

ClearML is the operational execution engine and observability UI. It is deliberately not the source of truth for formal MLDB results.

If ClearML and canonical MLDB disagree, investigate and reconcile through the formal lifecycle; do not edit canonical YAML to match the UI.

## 9. Implementation status

As of 2026-09-17, T011-02 through T011-04 are implemented and an actual two-trial RTX 3090 ClearML Study has completed through training acceptance, dependent evaluation release, recovery/resume, and terminal canonical Study closure. One Study Result creates/recovers one ClearML controller Task with native Pipeline DAG configuration; released child Tasks are bound to exact Pipeline nodes before enqueue, detailed telemetry remains on child Tasks, and bounded Study-summary values are projected to the controller without requiring the ClearML Fileserver.

The controller Task is an operational/UI ownership container, not a second remote MLDB scheduler loop. MLDB reconciliation opens semantic gates; the ClearML backend performs Task creation/enqueue and ClearML queue/agent infrastructure owns worker/resource execution.

Actual verification found that API-server-2.17+ ClearML UIs discover Pipelines through native hidden `.pipelines/<pipeline-name>` subprojects. Controllers created directly in the Namespace Project remained valid Tasks but were invisible in the Pipelines page. The adapter now uses the native subproject layout while retaining recovery compatibility with the earlier flat placement. Final T011-05 closure still requires human-visible Pipeline-page confirmation and the remaining actual cancellation check.
