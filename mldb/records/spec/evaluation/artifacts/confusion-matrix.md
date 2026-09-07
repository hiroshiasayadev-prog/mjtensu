# Contract: Confusion matrix artifact

- **id**: `spec:mldb.evaluation.artifacts.confusion_matrix`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.evaluation.artifacts`
- **contract_class**: `format`

## What this is

Defines the initial formal confusion-matrix artifact for categorical Tasks.

The schema identifier is `mjtensu.mldb/eval-artifact/confusion-matrix/v1` and the file format is CSV in long form.

## Current contract

Required columns are:

| column | type | meaning |
|---|---|---|
| `target` | string | Ground-truth categorical Task label. |
| `prediction` | string | Predicted categorical Task label. |
| `count` | non-negative integer | Number of evaluated samples in the target/prediction pair. |

Example:

```csv
target,prediction,count
5m,5m,812
5m,6m,3
6m,5m,5
6m,6m,799
```

Additional columns are permitted and have no generic MLDB semantics unless a later schema revision defines them.

## Rules

- The artifact format is CSV.
- The table uses long form rather than a wide class-by-class matrix.
- `target` must use a label from the selected categorical Task.
- `prediction` must use a label from the selected categorical Task.
- `count` must represent a non-negative integer.
- Each `(target, prediction)` pair must appear at most once.
- Additional columns may contain protocol-specific information.
- Additional columns must not change the meaning of the three required columns.
- The schema does not require rows for zero-count Task-label pairs that are absent from the file.

## Validation rules

- Reject a non-CSV formal format declaration for this schema.
- Reject a CSV without `target`, `prediction`, and `count` columns.
- Reject a row whose `target` or `prediction` value is outside the referenced Task label set.
- Reject a `count` value that is not a non-negative integer.
- Reject duplicate `(target, prediction)` pairs.
- Tolerate additional columns.

The schema validates table structure and Task-label usage. It does not independently prove that counts equal another predictions artifact or a Corpus sample count.

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.evaluation.artifacts` | Parent artifact schema Index. |
| `spec:mldb.evaluation.result_validation` | Applies this schema before importing a returned formal artifact. |
| `spec:mldb.catalog.task_format` | Defines the categorical Task label set. |
