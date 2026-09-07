# Concept: Study Run lifecycle

- **id**: `spec:mldb.study.study_run_lifecycle`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.study`

## What this is

Defines Study Run state transitions, terminal-state meaning, child-failure isolation, and the boundary between durable Study history and operational queue state.

A Study Run is one execution event. Re-running the same sealed Study creates another Study Run rather than revising the earlier event.

## Concept model

Study Run statuses are:

```text
running
completed
completed_with_failures
failed
cancelled
```

Terminal states are:

```text
completed
completed_with_failures
failed
cancelled
```

Conceptually:

```text
          +--> completed
          |
running --+--> completed_with_failures
          |
          +--> failed
          |
          +--> cancelled
```

A terminal Study Run is immutable.

## Rules

### Creation and plan materialization

A Study must pass the validation required to begin execution before its Study Run is allocated.

The Study Run is created as `running` when concrete plan materialization begins.

A complete valid `plan.jsonl` becomes immutable after materialization succeeds and its hash, byte size, and counts are recorded.

If orchestration is interrupted while the Study Run remains `running` and no complete valid plan exists, plan materialization may restart for that same Study Run. Partial unfinalized plan bytes may be discarded or replaced because they are not yet an MLDB plan artifact.

If materialization fails and the Study Run is finalized as `failed` before a complete valid plan can be established, plan metadata may be absent. A partial plan file is not a valid immutable plan.

If a complete valid plan already exists, later resume or retry activity must use that immutable plan rather than regenerate or mutate its execution intent.

### Terminal meanings

| status | meaning |
|---|---|
| `completed` | Orchestration reached the end of the plan and every runnable planned coordinate is fully satisfied, including coordinates recovered by retry. |
| `completed_with_failures` | Orchestration reached the end of the plan, but one or more planned coordinates remain unsatisfied because training or evaluation failed, was cancelled, remained partial, or an evaluation was blocked by a missing upstream Model. |
| `failed` | Study-level planning or orchestration failed in a way that prevented normal processing of the plan. |
| `cancelled` | The Study Run was intentionally stopped before normal completion. |

A single failed child Run does not by itself make the Study Run `failed`.

A historical failed, cancelled, or partial child attempt that is later followed by a successful retry does not by itself prevent the Study Run from reaching `completed`.

### Model-source outcomes

For a `model.train` Study, child Training Run outcomes are:

| child Training Run outcome | Study orchestration consequence |
|---|---|
| `completed` | The planned training coordinate is satisfied and its Model may release dependent evaluation work. |
| `failed` | The attempt remains historical evidence; the coordinate may be retried with a new Training Run ID while independent work continues. |
| `cancelled` | The attempt remains historical evidence; the coordinate may be retried with a new Training Run ID while independent work continues. |

While no successful Training Run exists for a training-derived trial, its evaluation intents remain blocked operationally and no synthetic Evaluation Run is required solely to represent that block.

If a later retry completes successfully, the training coordinate becomes satisfied and its dependent evaluations may proceed. If orchestration ends without a successful training attempt for that coordinate, the Study Run becomes `completed_with_failures` when all other normal processing finishes.

For a `model.existing` Study, each trial's Model dependency is already satisfied by the immutable Model selected in the plan. No Training Run exists for that trial, and evaluation work may become runnable immediately after the Study Run plan is established and concrete evaluation preflight succeeds.

### Evaluation outcomes

| child Evaluation Run outcome | Study orchestration consequence |
|---|---|
| `completed` | Counts as fully successful planned evaluation work. |
| `completed_partial` | Retains usable historical output and does not block siblings; the planned evaluation coordinate may be retried for a complete result. If it remains partial at Study end, the Study Run becomes `completed_with_failures`. |
| `failed` | Does not block independent siblings or other trials; the planned coordinate may be retried with a new Evaluation Run ID. If it remains unsatisfied at Study end, the Study Run becomes `completed_with_failures`. |
| `cancelled` | Does not invalidate the Model or completed Training Run; the planned coordinate may be retried with a new Evaluation Run ID. If it remains unsatisfied at Study end, the Study Run becomes `completed_with_failures`. |

The originating Training Run and Model remain valid for every Evaluation Run outcome.

### Queue boundary

Queue job states such as these are not Study Run lifecycle states:

```text
blocked
ready
active
retry_wait
satisfied
failed
cancelled
```

Worker claims, leases, heartbeat state, attempt counters, and stale-worker conditions are likewise operational Queue data. They must not replace the persistent Study Run, Training Run, Model, or Evaluation Run source of truth.

Queue restart or replacement must not alter the materialized plan or terminal child history.

### Resume, retry, and reconciliation

A `running` Study Run is resumable after scheduler, worker, or process interruption.

| situation | required behavior |
|---|---|
| `running` Study Run has no complete valid plan | Resume the same Study Run and restart plan materialization; unfinalized partial plan bytes are not authoritative. |
| `running` Study Run has a complete valid plan | Resume from the immutable plan and existing child lineage without mutating the plan. |
| Study Run is terminal `failed`, `cancelled`, or `completed_with_failures` | Retrying the Study creates a new Study Run ID; the terminal Study Run is not reopened. |
| Study Run is terminal `completed` and the user explicitly repeats it | Create a new Study Run ID; the completed Study Run is not reopened. |
| Child Training Run is terminal `failed` or `cancelled` and the same planned coordinate is retried | Create a new Training Run ID with the same `study.run` and `study.trial` lineage. |
| Child Evaluation Run is terminal `failed`, `cancelled`, or `completed_partial` and the same planned coordinate is retried | Create a new Evaluation Run ID with the same `study.run`, `study.trial`, and `study.stage` lineage. |

Orchestration should be restartable from:

- the immutable Study Run plan when one exists;
- existing Training Runs with matching Study lineage for training-derived trials;
- automatically generated Models for successful training-derived trials;
- existing Models referenced directly by existing-Model plan rows;
- existing Evaluation Runs with matching Study lineage.

Study-linked child creation must use `study.run`, `study.trial`, and where applicable `study.stage` to avoid accidental duplicate child creation caused only by scheduler restart.

Queue-specific stale-worker detection, attempt counters, retry timing, and lease semantics remain operational state outside the Study Run contract.

Study Run does not need an authoritative mutable array of child Run IDs.

### Summary

A Study Run may persist derived summary counts for convenience.

Summary counts describe final outcomes of planned coordinates rather than raw historical child-attempt counts.

Training summary counts apply only to training-derived Studies. Existing-Model Studies have no planned training coordinates and need only evaluation outcome counts.

A coordinate that failed once and later succeeded through retry contributes to the final successful coordinate count. The failed attempt remains queryable through child Run lineage.

Summary counts may be rebuilt from plan coordinates, child Run records, and orchestration outcomes. Rebuilding a derived summary must not change terminal child facts or plan identity.

## Boundary

| concern | owner |
|---|---|
| Study Run YAML fields | `spec:mldb.study.study_run_format`. |
| Immutable plan rows | `spec:mldb.study.plan_format`. |
| Training Run lifecycle | `spec:mldb.training.training_run_lifecycle`. |
| Model creation invariant | `spec:mldb.model`. |
| Evaluation Run lifecycle | `spec:mldb.evaluation.evaluation_run_lifecycle`. |
| Durable Queue states, Worker leases, retry waiting, and restart reconciliation | `spec:mldb.orchestration.queue_lifecycle`. |
| Aggregate metric ranking or best-model selection | Derived analysis or future selection contract. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.study` | Parent Study overview. |
| `spec:mldb.study.study_run_format` | Persists Study Run status and plan metadata. |
| `spec:mldb.study.plan_format` | Provides the immutable execution intent used for reconciliation. |
| `spec:mldb.training.training_run_lifecycle` | Defines child training terminal states. |
| `spec:mldb.evaluation.evaluation_run_lifecycle` | Defines child evaluation terminal states. |
| `spec:mldb.orchestration` | Owns operational Queue and Worker coordination while preserving this Study Run lifecycle and retry semantics. |
| `spec:mldb.orchestration.queue_lifecycle` | Defines durable SQLite job progress and restart reconciliation against this Study Run plan and child history. |
