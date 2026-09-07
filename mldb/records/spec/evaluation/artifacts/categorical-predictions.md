# Contract: Categorical predictions artifact

- **id**: `spec:mldb.evaluation.artifacts.categorical_predictions`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.evaluation.artifacts`
- **contract_class**: `format`

## What this is

Defines the initial formal per-sample prediction artifact for categorical Tasks.

The schema identifier is `mjtensu.mldb/eval-artifact/categorical-predictions/v1` and the file format is JSONL.

## Current contract

Each non-empty line is one JSON object for one evaluated sample.

Required fields are:

| field | type | meaning |
|---|---|---|
| `sample_id` | string | Corpus-local sample identifier. |
| `target` | string | Ground-truth categorical Task label. |
| `prediction` | string | Predicted categorical Task label. |

Example:

```json
{"sample_id":"sample-000001","target":"5m","prediction":"5m"}
{"sample_id":"sample-000002","target":"6m","prediction":"5m"}
```

Additional object fields are permitted. They have no generic MLDB semantics unless a later schema revision defines them.

## Rules

- The artifact format is JSONL.
- Each non-empty line represents exactly one JSON object.
- `sample_id` uses the selected Corpus's local sample identity namespace.
- `target` must use a label from the selected categorical Task.
- `prediction` must use a label from the selected categorical Task.
- Additional fields may contain protocol-specific confidence or diagnostic information.
- Additional fields must not change the meaning of the three required fields.
- MLDB v1 does not define a universal confidence field or probability-vector representation in this schema.

## Validation rules

- Reject a non-JSONL formal format declaration for this schema.
- Reject a non-empty line that is not valid JSON.
- Reject a line whose JSON value is not an object.
- Reject an object missing `sample_id`, `target`, or `prediction`.
- Reject a required field whose value is not a string.
- Reject `target` or `prediction` values outside the referenced Task label set.
- Tolerate additional object fields.

This schema does not define uniqueness, ordering, confidence semantics, or required coverage of every Corpus sample beyond the required per-record fields.

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.evaluation.artifacts` | Parent artifact schema Index. |
| `spec:mldb.evaluation.result_validation` | Applies this schema before importing a returned formal artifact. |
| `spec:mldb.catalog.task_format` | Defines categorical Task labels and class semantics. |
| `spec:mldb.catalog.image_classification_corpus` | Defines Corpus-local sample identity for the initial image-classification Corpus. |
