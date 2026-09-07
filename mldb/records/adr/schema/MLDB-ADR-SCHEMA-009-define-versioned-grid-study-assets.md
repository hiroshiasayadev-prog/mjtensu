# MLDB-ADR-SCHEMA-009: Define versioned grid Study assets

- **status**: accepted
- **date**: 2026-09-04
- **depends_on**: MLDB-ADR-SCHEMA-002, MLDB-ADR-SCHEMA-003, MLDB-ADR-SCHEMA-004, MLDB-ADR-SCHEMA-007
- **supersedes**:
- **migrated_to_spec**:

## Context

MLDB now has reusable Task, Corpus, Architecture, Train Protocol, and Evaluation Protocol definitions plus immutable Training Run, Model, and Evaluation Run records.

Running model experiments one combination at a time would still require repetitive manual commands. The project frequently compares multiple Architectures, learning rates, batch sizes, seeds, and evaluation conditions. These combinations should be declarable once and expanded automatically into queued Training Runs followed by Evaluation Runs for the Models produced by successful training.

The experiment declaration must remain separate from queue state. Queue concepts such as worker leases, attempts, heartbeats, retries, claims, and scheduling priority are operational concerns and should not become part of the durable MLDB ontology.

The initial need is deterministic Cartesian-product exploration rather than Bayesian optimization, random search, conditional search spaces, or early pruning. A small grid Study is sufficient to make current architecture and parameter comparisons substantially easier while keeping the schema understandable.

## Decision

Introduce `Study` as a versioned declarative MLDB asset that defines a reusable grid of training trials and the post-training evaluations to execute for each successfully produced Model.

Study is an authored definition. It does not itself record execution state and does not directly contain Training Run or Evaluation Run IDs.

Study v1 supports deterministic grid expansion over:

- selected Architectures;
- public Train Protocol parameters;
- training seeds.

Each generated training trial may be followed by zero or more fixed Evaluation Protocol stages applied to the Model produced by that trial.

### Physical placement

Study assets live under:

```text
mldb_data/
  studies/
```

Each Study is one YAML file:

```text
mldb_data/studies/<study-id>.yaml
```

Example:

```text
mldb_data/studies/tile-classifier-lr-search-v1.yaml
```

### ID and lifecycle

Every Study ID must end with:

```text
-v<positive-integer>
```

Study status is one of:

```text
draft
sealed
```

A `draft` Study may be edited while the experiment design is being prepared.

A `sealed` Study is immutable as an experiment definition. Executing the same sealed Study again creates another Study Run rather than mutating the Study.

Changes to the training Corpus, Train Protocol, Architecture set, parameter grid, seeds, evaluation stages, or their fixed parameters require a new Study revision after sealing.

### Metadata format

Study metadata uses:

```text
mjtensu.mldb/study/v1
```

Example:

```yaml
schema: mjtensu.mldb/study/v1
id: tile-classifier-lr-search-v1
status: sealed

name: Tile classifier learning-rate search
description: >
  Compare Plain CNN and f8-r1 MobileNet classifiers over learning-rate,
  batch-size, and seed combinations, then evaluate every successful model
  on final and real-capture holdouts.

training:
  corpus: gray35-train-v1
  protocol: tile-classifier-standard-v1

  architectures:
    - plain-cnn-v1
    - mobilenet-v3-small-f8-r1-v1

  parameters:
    learning_rate:
      values: [0.001, 0.0003, 0.0001]
    batch_size:
      values: [512, 1024]

  seeds: [42, 43]

evaluations:
  - stage: final-holdout
    corpus: gray35-final-holdout-v1
    protocol: tile-classifier-standard-eval-v1
    parameters: {}

  - stage: real-holdout
    corpus: gray35-real-holdout-v1
    protocol: tile-classifier-robustness-eval-v1
    parameters:
      batch_size: 2048
```

The required fields are:

- `schema`;
- `id`;
- `status`;
- `name`;
- `description`;
- `training.corpus`;
- `training.protocol`;
- `training.architectures`;
- `training.parameters`;
- `training.seeds`;
- `evaluations`.

`training.architectures` and `training.seeds` must be non-empty lists.

`training.parameters` may be an empty mapping when all Train Protocol public parameters should use their defaults.

`evaluations` may be an empty list when a Study is intended to perform training only.

### Training grid

Study v1 uses Cartesian-product grid expansion.

For every Architecture, seed, and combination of values listed under `training.parameters`, one training trial is generated.

For example:

```text
2 Architectures
x 3 learning-rate values
x 2 batch-size values
x 2 seeds
= 24 training trials
```

Every key under `training.parameters` must exist in the referenced Train Protocol's public parameter interface established by MLDB-ADR-SCHEMA-004.

Each parameter entry has exactly one required `values` list:

```yaml
parameters:
  learning_rate:
    values: [0.001, 0.0003, 0.0001]
```

The list must be non-empty.

Public Train Protocol parameters omitted from the Study use the sealed Train Protocol default during Study Run materialization.

Unknown parameter keys are invalid.

Study does not define arbitrary per-trial overrides beyond this declared grid.

### Seeds

Training seed is a first-class grid axis because Training Run owns the concrete training seed independently of Train Protocol parameters.

Every Study v1 training trial receives exactly one seed from `training.seeds`.

Duplicate seed values are invalid because they would create duplicate grid coordinates within one Study Run.

### Architecture compatibility

Every Architecture listed by the Study must be sealed and must reference the same Task as the selected Train Protocol and training Corpus.

Study validation must reject incompatible combinations before a Study Run is materialized.

A Study may intentionally compare different Architecture families as long as each is compatible with the selected Task, Corpus representation, and Train Protocol contract.

### Evaluation stages

Each entry in `evaluations` defines one post-training evaluation stage that should run for every Model produced by a successful training trial.

Each evaluation stage contains:

- `stage`: a Study-local stable identifier;
- `corpus`: one evaluation Corpus ID;
- `protocol`: one Evaluation Protocol ID;
- `parameters`: fixed caller-supplied values for that Evaluation Protocol's public parameter interface.

`stage` values must be unique within the Study.

Evaluation-stage parameters are fixed within Study v1. They are not an additional Cartesian-product axis. The selected values are resolved against Evaluation Protocol defaults when the Study Run plan is materialized.

A later Study schema may introduce evaluation-parameter grids if concrete use cases justify the added expansion complexity.

Each Evaluation Protocol and Corpus must resolve to the same Task as the Models produced by the training section.

### Dependency semantics

The logical dependency graph for one trial is:

```text
Training trial
  -> completed Training Run
  -> automatically generated Model
  -> Evaluation stage A
  -> Evaluation stage B
  -> ...
```

Evaluation stages for the same Model are independent siblings unless a later orchestration design explicitly introduces additional dependencies.

If training fails or is cancelled, evaluation stages that require that trial's Model cannot run and are blocked for that trial only.

If one Evaluation Run fails, sibling evaluations and unrelated trials remain runnable.

Study itself does not encode fail-fast semantics across independent trials.

### Deterministic expansion

A sealed Study must expand deterministically for a given set of referenced sealed assets.

Study Run materialization resolves:

- every omitted Train Protocol public parameter to its default;
- every fixed Evaluation Protocol parameter to its default when omitted;
- the exact Cartesian-product ordering;
- the exact trial IDs and evaluation stage definitions.

The resulting concrete plan is persisted by Study Run so queue implementation changes cannot alter the historical meaning of an already-started Study Run.

### No queue state in Study

Study does not contain fields such as:

- queued/running job counts;
- worker IDs;
- retry counts;
- lease expiry;
- claim timestamps;
- priorities;
- heartbeats;
- scheduler-specific state.

Those are orchestration implementation details rather than experiment-definition semantics.

### No search optimization in v1

Study v1 does not define:

- random search;
- Bayesian optimization;
- Optuna-style samplers;
- conditional parameter spaces;
- adaptive trial generation;
- early pruning;
- metric-driven stopping of remaining trials.

These features may be added later without changing the meaning of existing deterministic grid Studies.

## Rationale

A declarative Study converts repetitive experiment setup into one versioned artifact while keeping the Train Protocol and Evaluation Protocol responsible for what individual executions actually do.

Using only public protocol parameters preserves the authority boundary already established for Training Run and Evaluation Run: Study can automate intentional axes without introducing an unrestricted override system.

Deterministic grid expansion is sufficient for the project's current architecture, augmentation, learning-rate, batch-size, and seed comparisons and is straightforward to audit.

Separating Study from Study Run allows the same sealed experiment design to be executed more than once while preserving each execution as a separate historical event.

Keeping queue state out of Study prevents operational scheduler details from contaminating durable experiment definitions and leaves the project free to implement the queue with SQLite, Redis, a process pool, or another mechanism later.

## Rejected alternatives

### Put queue state directly in Study

Study is a reusable definition. Worker claims, retries, and progress counters change continuously and would make the definition mutable for operational reasons.

Execution state belongs to Study Run and ephemeral queue infrastructure instead.

### Permit arbitrary parameter overrides

This would bypass the public parameter interfaces of Train Protocol and Evaluation Protocol and recreate hidden experiment configuration at the Study layer.

Study may vary only explicitly published parameters.

### Make every evaluation parameter a grid axis

This would multiply the Study search space and complicate trial identity before the project has demonstrated a need for evaluation-parameter sweeps.

Study v1 keeps evaluation parameters fixed per stage.

### Add adaptive optimization immediately

Adaptive samplers require result feedback, trial scheduling policy, pruning semantics, and additional persistent state. None are required to solve the current repetitive grid-search workflow.

Study v1 therefore starts with deterministic Cartesian-product expansion.

## Consequences

Future Study tooling should be able to:

- discover Study YAML under `mldb_data/studies/`;
- validate terminal `-vN` IDs and `draft` / `sealed` lifecycle;
- resolve the training Corpus, Train Protocol, and all Architectures;
- verify common Task compatibility;
- validate every training parameter key against the Train Protocol public interface;
- validate non-empty grid value lists and seeds;
- resolve evaluation stage Corpora and Evaluation Protocols;
- validate fixed evaluation parameter keys against each Evaluation Protocol public interface;
- calculate the deterministic training-trial count;
- expand the Study into an immutable Study Run plan;
- preserve independent-trial and sibling-evaluation failure isolation;
- avoid embedding queue-worker implementation state into the Study asset.

A Study Run can materialize this declarative definition into concrete trials suitable for a queue-driven train -> Model -> evaluate workflow.

## Evidence

The project repeatedly compares multiple classifier Architectures, learning settings, augmentation choices, seeds, and robustness evaluations. These comparisons are naturally expressible as Cartesian products over a small set of intentional Train Protocol parameters followed by common Evaluation Protocol stages.

MLDB-ADR-SCHEMA-004 already provides a sealed public Train Protocol parameter interface, and MLDB-ADR-SCHEMA-007 provides the corresponding Evaluation Protocol interface. Study can therefore compose existing assets without inventing another configuration language.
