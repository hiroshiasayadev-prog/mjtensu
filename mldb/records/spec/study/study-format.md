# Contract: Study format

- **id**: `spec:mldb.study.study_format`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.study`
- **contract_class**: `format`

## What this is

Defines the YAML format for one reusable MLDB Study.

Study v1 selects the Models to evaluate in exactly one of two ways: train new Models from a finite grid, or select existing Models. Every Study declares one or more fixed evaluation stages and contains no queue state.

## Current contract

Study metadata uses:

```text
mjtensu.mldb/study/v1
```

Canonical placement is:

```text
mldb_data/studies/<study-id>.yaml
```

Training Study example:

```yaml
schema: mjtensu.mldb/study/v1
id: tile-classifier-lr-search-v1
status: sealed
name: Tile classifier learning-rate search
description: >
  Compare selected classifier Architectures over learning rate and seed,
  then evaluate every successful Model.

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

evaluations:
  - stage: final-holdout
    corpus: gray35-final-holdout-v1
    protocol: tile-classifier-standard-eval-v1
    parameters: {}
```

Existing-Model Study example:

```yaml
schema: mjtensu.mldb/study/v1
id: tile-classifier-reeval-v1
status: sealed
name: Re-evaluate existing classifiers
description: >
  Apply the revised evaluator to selected existing Models.

model:
  existing:
    - mdl-20260901-001
    - mdl-20260901-002

evaluations:
  - stage: revised-real-holdout
    corpus: gray35-real-holdout-v2
    protocol: tile-classifier-robustness-eval-v2
    parameters: {}
```

## Rules

Every Study requires:

| field | contract |
|---|---|
| `schema` | Exactly `mjtensu.mldb/study/v1`. |
| `id` | Versioned Study ID ending in terminal `-v<positive-integer>`. |
| `status` | `draft` or `sealed`. |
| `name` | Human-readable Study name. |
| `description` | Human-readable experiment intent. |
| `model` | Exactly one Model-source alternative: `train` or `existing`. |
| `evaluations` | Non-empty list of fixed evaluation stages applied to every materialized trial Model. |

`model` must contain exactly one of:

```text
model.train
model.existing
```

A Study must not contain both alternatives or neither alternative.

### Train source

`model.train` requires:

| field | contract |
|---|---|
| `corpus` | Exactly one training Corpus ID. |
| `protocol` | Exactly one Train Protocol ID. |
| `architectures` | Non-empty list of Architecture IDs. |
| `parameters` | Mapping of published Train Protocol keys to non-empty `values` lists. May be empty. |
| `seeds` | Non-empty list of unique integer training seeds. Boolean values are invalid. |

Each `model.train.parameters.<key>` entry has exactly one required `values` list.

Every parameter value must satisfy the JSON-compatible public-parameter value domain defined by `spec:mldb.runtime.public_parameters`.

Every parameter key must be published by the referenced Train Protocol. Omitted public Train Protocol parameters receive their defaults during Study Run materialization.

The written order of keys beneath `model.train.parameters` is not semantically significant. Study materialization orders parameter axes by the canonical key ordering defined in `spec:mldb.study.grid_expansion`.

Every authored training grid axis must contain unique values.

- `model.train.architectures` must not contain duplicate Architecture IDs.
- Each `model.train.parameters.<key>.values` list must not contain duplicate decoded parameter values.
- `model.train.seeds` must contain only integer values, must not contain boolean values, and must not contain duplicate seed values.

Duplicate parameter values are compared using YAML-decoded, type-sensitive value equality. Values with different decoded types are not duplicates merely because a host language might coerce them to equality.

### Existing Model source

`model.existing` is a non-empty ordered list of Model IDs.

- Duplicate Model IDs are invalid.
- Every Model must resolve through a completed Training Run and valid canonical learned weights.
- Every selected Model must resolve to the same Task.
- Authored list order is semantically significant and determines Study-local trial ordering.

An existing-Model Study creates no Training Runs and no new Models.

### Evaluation stages

Each evaluation stage requires:

| field | contract |
|---|---|
| `stage` | Study-local stable identifier, unique within `evaluations`. |
| `corpus` | Evaluation Corpus ID. |
| `protocol` | Evaluation Protocol ID. |
| `parameters` | Fixed caller values for published Evaluation Protocol parameters. |

Evaluation-stage parameters are not Cartesian grid axes in Study v1.

Every supplied fixed evaluation parameter value must satisfy the JSON-compatible public-parameter value domain. Omitted public Evaluation Protocol parameters receive their defaults during Study Run materialization.

Each evaluation stage is applied to every Model materialized by the Study source.

### Lifecycle

A draft Study may be edited while experiment intent is being prepared.

A sealed Study is immutable as an experiment definition. Re-executing it creates another Study Run.

After sealing, any change to Model source, training grid, selected existing Models, evaluation stages, or fixed evaluation parameters requires a new Study revision.

### Compatibility

Before Study Run allocation, the Study must resolve to a compatible set of assets.

| source mode | required compatibility |
|---|---|
| `model.train` | Training Corpus, Train Protocol, and every Architecture reference one Task; executable assets are sealed and required integrity checks succeed. |
| `model.existing` | Every selected Model resolves through a completed Training Run with valid canonical weights, and all selected Models resolve to one Task. |

For both source modes:

- every evaluation Corpus and Evaluation Protocol must reference the same Task as the Study Models;
- every Evaluation Protocol required for execution must be sealed;
- applicable Corpus and executable-asset integrity checks must succeed;
- evaluation parameter keys must satisfy the common public-parameter contract.

Training parameter validation additionally applies to `model.train`.

Compatibility that requires model-family-specific executable knowledge remains outside static Study YAML validation.

### Queue boundary

Study YAML must not contain queue or Worker fields such as job IDs, attempts, claims, leases, heartbeats, retry timing, priorities, or progress counters.

Study does not contain concrete Training Run or Evaluation Run IDs. `model.existing` intentionally contains stable Model IDs because those Models are authored execution inputs.

## Validation rules

- Reject unsupported `schema`.
- Reject filename basename and `id` disagreement.
- Reject an ID without terminal positive-integer `-vN` revision syntax.
- Reject a status outside `draft` or `sealed`.
- Reject `model` unless exactly one of `train` or `existing` is present.
- Reject an empty `evaluations` list.
- Reject duplicate evaluation `stage` identifiers.
- Reject an evaluation parameter key not published by its referenced Evaluation Protocol.
- Reject a fixed evaluation parameter value outside the JSON-compatible public-parameter domain.
- Reject an empty `model.train.architectures` list.
- Reject duplicate Architecture IDs in `model.train.architectures`.
- Reject an empty `model.train.seeds` list.
- Reject a training seed that is not an integer or is boolean.
- Reject duplicate seed values.
- Reject a training parameter entry whose `values` list is empty.
- Reject a training parameter value outside the JSON-compatible public-parameter domain.
- Reject duplicate decoded values within one training parameter `values` list.
- Reject a training parameter key not published by the referenced Train Protocol.
- Reject an empty `model.existing` list.
- Reject duplicate Model IDs in `model.existing`.
- Reject an existing Model that does not resolve through a completed Training Run with valid canonical learned weights.
- Reject existing Models that do not resolve to one common Task.
- Reject incompatible Task relationships for a Study intended for execution.
- Reject mutation of a sealed Study's executable experiment meaning.

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.study` | Parent Study overview. |
| `spec:mldb.study.grid_expansion` | Materializes the selected Model source into Study-local trials. |
| `spec:mldb.runtime.public_parameters` | Defines default and caller-value resolution. |
| `spec:mldb.model` | Defines existing Model identity and learned-state resolution. |
| `spec:mldb.repository.layout` | Defines canonical Study placement. |
