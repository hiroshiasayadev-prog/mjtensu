# Overview: MLDB Study execution

- **id**: `spec:mldb.study`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb`

## What this is

Defines the MLDB Study layer that turns one reusable Model-selection and evaluation definition into one immutable Study Run plan and related child execution lineage.

Study owns declarative experiment intent. It either trains new Models from a deterministic grid or selects existing Models, then applies common evaluation stages. Study Run owns one concrete materialization of that intent. Training Run and Evaluation Run remain authoritative for individual execution outcomes.

## Current contract

The Study flow is:

```text
sealed Study
    |
    v
validate complete Study and referenced immutable inputs
    |
    +--> model.train ------> materialize training grid
    |
    +--> model.existing ---> materialize existing Models
    |
    v
allocate Study Run
    |
    v
write immutable plan.jsonl
    |
    v
Queue internal work units
    |
    +--> training-derived trial: Training Run -> Model -> Evaluation Runs
    |
    +--> existing-Model trial: existing Model -------> Evaluation Runs
    |
    v
finalize Study Run
```

Study validation must succeed before Study Run allocation or Queue insertion. Every Study contains at least one evaluation stage.

One child failure does not terminate independent branches. A `running` Study Run is resumable, and retrying terminal child work creates new child Run IDs without changing the plan. Retrying or explicitly repeating a terminal Study Run creates a new Study Run ID.

Queue state is not part of Study or Study Run persistence. A queue may consume the immutable plan, but worker claims, retry timing, leases, heartbeats, and scheduling state remain replaceable operational data.

## Responsibility boundaries

| concern | owner |
|---|---|
| Reusable Model source and evaluation-stage declaration | Study format. |
| Training-grid or existing-Model materialization | Model expansion. |
| One execution event and plan integrity metadata | Study Run format. |
| Exact materialized trial rows | Study Run plan format. |
| Study Run state transitions and child-failure interpretation | Study Run lifecycle. |
| Train Protocol and Evaluation Protocol parameter defaults | `spec:mldb.runtime.public_parameters`. |
| Individual training result | Training Run. |
| Learned result identity | Model. |
| Individual evaluation result | Evaluation Run. |
| Worker claim, lease, heartbeat, retry timer, priority, and queue transport | `spec:mldb.orchestration`. |

## Non-goals

- Define random or Bayesian search.
- Define conditional search spaces or adaptive trial generation.
- Define metric-driven pruning or early termination of unrelated trials.
- Store queue-worker state in Study or Study Run.
- Duplicate child Run metrics into authoritative Study Run state.
- Define production promotion or release selection.
- Define a separate v1 top-level training-only or direct-evaluation execution definition outside Study.

## Topics

| title | kind | ref | summary |
|---|---|---|---|
| Study format | Contract | `spec:mldb.study.study_format` | Versioned Study YAML, exclusive train-or-existing Model source, non-empty evaluation stages, and lifecycle fields. |
| Study Run format | Contract | `spec:mldb.study.study_run_format` | One Study execution record, plan integrity metadata, and optional derived summary. |
| Study Run plan format | Contract | `spec:mldb.study.plan_format` | Immutable JSONL representation of training-derived or existing-Model trials and their fully resolved evaluation stages. |
| Model expansion | Concept | `spec:mldb.study.grid_expansion` | Training-grid expansion, existing-Model ordering, public parameter resolution, trial IDs, and dependency intent. |
| Study Run lifecycle | Concept | `spec:mldb.study.study_run_lifecycle` | Study-level terminal states, failure isolation, blocking, and reconciliation boundaries. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb` | Parent MLDB overview. |
| `spec:mldb.runtime.public_parameters` | Resolves protocol defaults used during plan materialization. |
| `spec:mldb.training` | Executes planned training trials. |
| `spec:mldb.model` | Supplies the learned identity required by dependent evaluations. |
| `spec:mldb.evaluation` | Executes planned evaluation stages. |
| `spec:mldb.orchestration` | Schedules Study plan work through Queue and separate Workers without changing Study persistence semantics. |
