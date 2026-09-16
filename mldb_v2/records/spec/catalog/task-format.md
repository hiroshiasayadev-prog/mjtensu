# Contract: Task format

- **id**: `spec:mldb.v2.catalog.task_format`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.catalog`
- **contract_class**: `format`

## Meaning

Task defines semantic prediction meaning only. It does not own Corpus representation, model topology,
training behavior, evaluation procedure, deployment behavior, or result values.

## YAML

Schema is `mjtensu.mldb-v2/task/v1`.

Required top-level fields are:

| field | contract |
|---|---|
| `schema` | Exact schema identifier. |
| `id` | Full versioned `<namespace>/<local-id>` identity. |
| `status` | `draft` or `sealed`. |
| `name` | Non-empty human-readable name. |
| `problem_type` | Non-empty prediction-problem identifier. |
| `description` | Human-readable semantic description. |
| `input` | JSON-compatible mapping describing the semantic input unit. |
| `target` | JSON-compatible mapping with required non-empty string `type`. |
| `semantics` | JSON-compatible mapping for Task-specific meaning. |
| `scope` | JSON-compatible mapping for explicit semantic scope. |
For `target.type: categorical`, `target.labels` is required, ordered, non-empty, and contains unique
non-empty strings. Its zero-based order is the canonical class index.

Other target types may define additional versioned contracts later. Generic Task validation does not
invent a universal target vocabulary beyond the required `target.type` discriminator.

## Identity and lifecycle

The local ID ends in `-v<positive-integer>`. Namespace/path/ID consistency follows the repository
contract. Task has no `.py` companion.

Changing prediction meaning, label order/meaning, semantic input unit, target type, or material scope
requires a new Task revision. Representation-only changes belong to Corpus and do not require a new
Task.

A sealed Task is immutable under its ID and never returns to draft.

## Validation

Reject unsupported schema, missing/incorrect field types, ID/path mismatch, invalid lifecycle state,
invalid version suffix, an empty target type, or invalid categorical labels. Additional keys inside
`input`, non-categorical `target`, `semantics`, and `scope` are preserved as JSON-compatible semantic
metadata; unknown top-level keys are invalid in schema v1.
