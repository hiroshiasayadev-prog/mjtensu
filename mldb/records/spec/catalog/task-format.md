# Contract: Task format

- **id**: `spec:mldb.catalog.task_format`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.catalog`
- **contract_class**: `format`

## What this is

Defines the YAML contract for one MLDB Task.

A Task owns semantic prediction meaning. It does not own materialized data representation, training procedure, model structure, evaluation results, or deployment behavior.

## Current contract

Task YAML uses schema identifier:

```text
mjtensu.mldb/task/v1
```

The required top-level fields are:

| field | contract |
|---|---|
| `schema` | Exact Task schema identifier. |
| `id` | Immutable Task identity. |
| `name` | Human-readable Task name. |
| `problem_type` | Prediction-problem identifier. |
| `description` | Human-readable semantic description. |
| `input` | Semantic input contract. |
| `target` | Prediction target contract. |
| `semantics` | Task-specific target meaning. |
| `scope` | Explicit in-scope and out-of-scope meaning. |

For a categorical Task, `target.labels` is the normative ordered label vocabulary.

The zero-based array position of a categorical label is its canonical class index for that Task version.

## Rules

- The Task ID must match the canonical filename basename defined by `spec:mldb.repository.layout`.
- A Task ID becomes immutable once another MLDB record references it.
- Task schema version and Task identity revision are separate concepts.
- Task IDs should expose a human-visible revision suffix such as `-v1`.
- A categorical `target.labels` list must contain unique, non-empty values.
- Categorical label order is normative and must not be treated as presentation-only ordering.
- A semantic target change requires a new Task identity or revision.
- Semantic target changes include label addition, removal, reordering, or meaning changes.
- Changes to the semantic input unit, target type, target-mapping rules, or material scope require a new Task identity or revision.
- Editorial corrections that do not change semantic meaning do not require a new Task identity.
- Task must not maintain reverse lists of dependent Corpora, Architectures, Protocols, Models, or Runs.
- Task may describe an input semantic unit such as `single-tile-image` without defining tensor shape or pixel encoding.
- Task must not define Corpus identity, split membership, pixel shape, channels, normalization, augmentation, optimizer, seed, Architecture, learned weights, evaluation result, runtime latency, export, or deployment behavior.

## Validation rules

| condition | result |
|---|---|
| `schema` is not `mjtensu.mldb/task/v1` | Invalid Task record. |
| Required top-level field is absent | Invalid Task record. |
| `id` disagrees with the canonical filename basename | Invalid Task record. |
| Categorical target omits `target.labels` | Invalid Task record. |
| Categorical label is empty or duplicated | Invalid Task record. |
| Categorical label order changes under an already-referenced Task ID | Semantic mutation; a new Task identity is required. |

Generic Task validation does not impose a global enum for `problem_type`, semantic keys, scope vocabulary, or future non-categorical target structures beyond contracts explicitly added later.

## Boundary

| concern | owner |
|---|---|
| Task physical path | `spec:mldb.repository.layout`. |
| Typed Task resolution | `spec:mldb.runtime.asset_resolution`. |
| Materialized sample representation | Corpus contracts. |
| Model tensor interface and topology | Architecture contracts. |
| Training and evaluation behavior | Training and evaluation topics. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.catalog` | Parent catalog Index. |
| `spec:mldb.repository.layout` | Defines Task file placement. |
| `spec:mldb.runtime.asset_resolution` | Defines typed Task lookup. |
