# Contract: MLDB Model format

- **id**: `spec:mldb.model.model_format`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.model`
- **contract_class**: `format`

## What this is

Defines the complete YAML format for an MLDB Model v1 record.

A Model record identifies the canonical learned result of one completed Training Run without duplicating learned bytes or upstream execution metadata.

## Current contract

Model metadata uses schema identifier:

```text
mjtensu.mldb/model/v1
```

The complete v1 record is:

```yaml
schema: mjtensu.mldb/model/v1
id: mdl-20260904-001
training_run: tr-20260904-001
```

| field | required | contract |
|---|---|---|
| `schema` | yes | Exactly `mjtensu.mldb/model/v1`. |
| `id` | yes | Deterministic Model ID corresponding to `training_run`. |
| `training_run` | yes | Existing completed Training Run ID. |

These are the only Model v1 fields.

## Rules

- Model YAML lives at `mldb_data/models/<model-id>.yaml`.
- The filename basename must equal `id`.
- `id` uses `mdl-YYYYMMDD-NNN` for Training Run v1 identities.
- `training_run` uses `tr-YYYYMMDD-NNN`.
- Model ID date and sequence must exactly match the referenced Training Run date and sequence.
- `training_run` must resolve to a Training Run with `status: completed`.
- The referenced Training Run must contain a valid canonical `result.weights` record.
- Model v1 must not contain `status`, `name`, `description`, `architecture`, `task`, `corpus`, `train_protocol`, `seed`, `artifact`, `metrics`, or deployment fields.
- Model v1 has no sibling `.pt` artifact.
- Existing valid Model YAML is immutable.

## Validation rules

A Model record is invalid when any of the following is true:

| condition | validation result |
|---|---|
| YAML schema identifier differs from `mjtensu.mldb/model/v1` | Reject. |
| Required field is absent | Reject. |
| Undeclared additional Model v1 field is present | Reject. |
| Filename basename differs from `id` | Reject. |
| `id` does not satisfy the Model v1 ID grammar | Reject. |
| `training_run` does not satisfy the Training Run v1 ID grammar | Reject. |
| `id` and `training_run` suffixes differ | Reject. |
| Referenced Training Run does not exist | Reject. |
| Referenced Training Run is not `completed` | Reject. |
| Referenced Training Run lacks valid canonical weight metadata | Reject. |
| Another Model already represents the same completed Training Run | Reject the conflicting record. |

Model validation does not copy or rewrite the referenced Training Run.

## Boundary

| concern | owner |
|---|---|
| Canonical Model path | `spec:mldb.repository.layout`. |
| Training Run format and `result.weights` metadata | `spec:mldb.training.training_run_format`. |
| Canonical `.pt` artifact contents and integrity | `spec:mldb.training.canonical_weights`. |
| Deterministic Model identity and automatic creation | `spec:mldb.model.identity`. |
| Evaluation metrics and result artifacts | Evaluation topic. |
| Promotion, release, production, and deployment status | Future lifecycle topics. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.model` | Parent Model overview. |
| `spec:mldb.model.identity` | Defines the deterministic Model/Training Run relationship. |
| `spec:mldb.training.training_run_format` | Defines the referenced completed Training Run record. |
| `spec:mldb.training.canonical_weights` | Defines the learned artifact reached through the Training Run. |
