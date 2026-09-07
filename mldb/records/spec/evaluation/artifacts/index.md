# Index: Evaluation artifact schemas

- **id**: `spec:mldb.evaluation.artifacts`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.evaluation`

## What this is

Navigation index for versioned structured artifact schemas accepted as formal Evaluation Run outputs.

Each child contract defines one concrete file-content schema. Evaluation Protocol declares which schema applies to each formal artifact key.

## Topics

| title | kind | ref | summary |
|---|---|---|---|
| Categorical predictions | Contract | `spec:mldb.evaluation.artifacts.categorical_predictions` | JSONL per-sample categorical target and prediction records. |
| Confusion matrix | Contract | `spec:mldb.evaluation.artifacts.confusion_matrix` | CSV long-form categorical target/prediction counts. |
