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

One Namespace still maps to one ClearML Project. A Study does not get its own Project.

One fresh MLDB Study Result gets one fresh, recoverable Pipeline Run.

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

Opening a child node shows detailed Task telemetry/artifacts/logs. The Pipeline/controller summary may mirror only the small set of values useful for Study-level comparison.

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

As of 2026-09-17, T011-02 through T011-04 have a local implementation and focused conformance coverage. One Study Result creates/recover one ClearML controller Task with native Pipeline DAG configuration; semantically released child Tasks are bound to the exact Pipeline node before ClearML enqueue, and a bounded Study-summary artifact/configuration is mirrored to the controller.

The controller Task is currently an operational/UI ownership container, not a second remote MLDB scheduler loop. MLDB reconciliation opens semantic gates; the ClearML backend performs Task creation/enqueue and ClearML queue/agent infrastructure owns worker/resource execution. This avoids requiring the remote controller to mutate canonical result files that live with MLDB.

Actual ClearML web UI behavior and RTX 3090 execution remain unverified until T011-05. In particular, do not claim final Pipeline UI grouping/retry behavior until the real-server closure check passes.
