# MLDB-ADR-SCHEMA-020: Allow Study existing-Model sources

- **status**: accepted
- **date**: 2026-09-04
- **depends_on**: MLDB-ADR-SCHEMA-006, MLDB-ADR-SCHEMA-007, MLDB-ADR-SCHEMA-009, MLDB-ADR-SCHEMA-010, MLDB-ADR-SCHEMA-013, MLDB-ADR-SCHEMA-017
- **supersedes**:
- **migrated_to_spec**: 2026-09-04

## Context

MLDB-ADR-SCHEMA-009 defines Study as a training grid followed by common evaluation stages.

The project also needs to apply a revised Evaluation Protocol or a new evaluation Corpus to Models that already exist. Requiring those Models to be retrained solely to reach evaluation would waste compute and would create unrelated Training Runs.

Introducing a separate top-level direct-evaluation execution model would add another durable execution-intent path alongside Study Run planning, queue recovery, retry, and reconciliation.

Study can instead express how its evaluation subjects are obtained: either by training new Models or by selecting existing Models. The same Study Run, immutable plan, queue, retry, and Evaluation Run contracts can then cover both cases.

The project does not currently need user-requested training-only execution. A Study is expected to evaluate the Models it produces or selects.

## Decision

Generalize Study v1 so its `model` section chooses exactly one Model source:

```text
model.train
model.existing
```

Exactly one of these alternatives is present.

`evaluations` remains common to both alternatives and must contain at least one evaluation stage.

### Train Model source

A training Study uses:

```yaml
model:
  train:
    corpus: gray35-train-v1
    protocol: tile-classifier-standard-v1
    architectures:
      - plain-cnn-v1
      - mobilenet-v3-small-f8-r1-v1
    parameters:
      learning_rate:
        values: [0.001, 0.0003, 0.0001]
    seeds: [42, 43]
```

`model.train` retains the deterministic Cartesian-product semantics previously defined for the Study `training` section.

Each materialized coordinate creates one planned Training execution. A successful Training Run creates its Model, after which every Study evaluation stage is applied to that Model.

The existing rules for Architecture ordering, parameter-key ordering, parameter-value ordering, seed ordering, duplicate rejection, public parameter resolution, and Task compatibility remain unchanged.

### Existing Model source

An evaluation-only Study uses:

```yaml
model:
  existing:
    - mdl-20260901-001
    - mdl-20260901-002
```

`model.existing` is a non-empty ordered list of existing Model IDs.

Duplicate Model IDs are invalid.

Every selected Model must resolve through a completed Training Run with valid canonical learned weights.

All selected Models must resolve to one common Task because one Study evaluation-stage set is applied uniformly to every selected Model.

Every evaluation Corpus and Evaluation Protocol must reference that same Task.

Existing Model declaration order is semantically significant and determines Study-local trial ordering for this source mode.

An existing-Model Study creates no Training Runs and no new Models. It creates only the planned Evaluation Runs required by its evaluation stages.

### Common evaluation stages

Every Study requires a non-empty `evaluations` list.

Each declared evaluation stage is materialized once for every Study-local trial, regardless of whether that trial obtains its Model from training or from `model.existing`.

Evaluation-stage semantics, fixed parameter resolution, stage uniqueness, output contracts, retry behavior, and failure isolation remain unchanged.

### Study Run plan rows

Study Run retains one `plan.jsonl` row per Study-local trial.

Each row contains exactly one Model-source alternative:

```text
training
model
```

A training-derived row contains `training` with the fully resolved training coordinate and does not contain `model`.

An existing-Model row contains `model` with the exact selected Model ID and does not contain `training`.

Both row forms contain the fully resolved `evaluations` list.

Study-local trial IDs remain:

```text
trial-0001
trial-0002
...
```

For training Studies, ordering follows the canonical Cartesian-product rules already established by MLDB-ADR-SCHEMA-013.

For existing-Model Studies, ordering follows the authored `model.existing` list.

The existing Study Run `plan.trials` field counts plan rows in either source mode.

### Lifecycle and retry

A training-derived trial is fully satisfied according to the existing training-plus-evaluation coordinate rules.

An existing-Model trial has no training coordinate. Its Model dependency is satisfied by the selected immutable Model during Study validation and plan materialization.

Study Run completion for an existing-Model Study therefore depends only on its planned evaluation coordinates.

Evaluation retries create new Evaluation Run IDs with the same Study Run, trial, and stage lineage. Existing Models are never copied or recreated for retry.

Study Run plan immutability and reconciliation semantics remain unchanged.

### Execution entry

Study is sufficient as the durable user-authored execution definition for both train-then-evaluate workflows and re-evaluation of existing Models.

Training Run and Evaluation Run remain concrete execution-history entities rather than reusable user-authored execution definitions.

This decision does not remove their runtime executor interfaces; orchestration may invoke those executors as child work materialized from a Study Run plan.

## Rationale

A single Study abstraction avoids separate persistence and recovery rules for direct evaluation requests.

Making Model source explicit reflects the actual experiment question: either create Models from a controlled training grid or select already learned Models, then apply a common evaluation set.

Keeping one plan-row and trial-lineage model allows Queue and Worker execution to remain identical at the Evaluation job level.

Requiring at least one evaluation stage matches the intended MLDB workflow and avoids a second top-level training-only execution path that currently has no concrete need.

Preserving existing training-grid rules minimizes change to the established Study semantics while adding the missing existing-Model workflow.

## Rejected alternatives

### Retrain an existing Model merely to run a new Evaluation Protocol

This wastes compute and creates a new learned result unrelated to the evaluation change being investigated.

### Add a separate durable direct-evaluation request entity

That would require another source of queued execution intent, retry recovery, and reconciliation beside Study Run plans.

### Put existing Models into the training grid

Existing Models are learned identities, not Architecture or training-parameter values. Treating them as training coordinates would blur the definition/execution boundary.

### Permit both `model.train` and `model.existing` in one Study

The two sources have different dependency semantics and ordering rules. Mixing them in one Study would complicate plan materialization without a current use case.

Separate Studies remain composable and easier to audit.

### Permit a Study with no evaluation stages

The current workflow does not require training-only Studies. An empty evaluation surface would make the Study an unnecessary wrapper around training execution and weaken the single Study execution model.

## Consequences

Study format and materialization must validate exactly one of `model.train` or `model.existing`.

The former `training` Study fields move beneath `model.train` without changing their individual semantics.

Study validation must verify selected existing Models and their common Task before allocating a Study Run.

Study Run plan validation must accept either a resolved `training` object or an existing `model` ID per row, but never both or neither.

Training-derived Study Runs continue to create Training and Evaluation child work.

Existing-Model Study Runs create Evaluation child work only.

Queue and Worker design can therefore treat every user-requested experiment as a Study Run while still scheduling Training and Evaluation as separate internal work units.

## Evidence

Evaluation Protocols are independently versioned executable assets, so evaluation behavior can legitimately change while a Model remains immutable.

Model already provides a stable learned-result identity and resolves canonical weights through its completed Training Run.

Study Run already supplies immutable plan intent, Study-local trial identity, Evaluation Run lineage, retry, and reconciliation semantics needed to evaluate existing Models without another durable request type.
