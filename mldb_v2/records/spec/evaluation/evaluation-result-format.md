# Contract: Evaluation Result format

- **id**: `spec:mldb.v2.evaluation.evaluation_result_format`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.evaluation`
- **contract_class**: `format`

## Shape

Schema is `mjtensu.mldb-v2/evaluation-result/v1`. One attempted logical Evaluation coordinate has
exactly one terminal Evaluation Result with deterministic ID from
`spec:mldb.v2.results.study_result_format`.

```yaml
schema: mjtensu.mldb-v2/evaluation-result/v1
id: rotated-fcos/run-abcd-trial-0001-eval-0001
study_result: rotated-fcos/run-abcd
plan: rotated-fcos/spatial-screen-v2-plan-0123456789abcdef
trial: trial-0001
coordinate: eval-0001
stage: final-holdout
model: rotated-fcos/run-abcd-trial-0001-model
task: rotated-fcos/mahjong-obb-v1
corpus: rotated-fcos/human-obb-v3
evaluation_protocol: rotated-fcos/human-obb-v2
parameters: {iou_threshold: 0.5}
source_commit: <full unabbreviated Git commit object id>
attempts: []
status: completed
diagnostic: null
result: null
```
`attempts` is a non-empty ordered list of `spec:mldb.v2.results.attempt_summary`. `parameters` is the
complete resolved mapping copied from the Plan coordinate.

## Status-dependent payload

For `status: completed`:

```yaml
diagnostic: null
result:
  metrics:
    f1: 0.967
    rotated_iou_mean: 0.829
  artifacts:
    predictions:
      uri: s3://mldb-artifacts/...
      bytes: 98765
      sha256: <64 lowercase hex>
      format: jsonl
      schema: mjtensu.example/predictions/v1
```

`metrics` contains exactly the accepted declared formal metric keys; required metrics are present,
optional absent metrics are omitted. `artifacts` contains only produced declared formal artifacts;
required artifacts are present and optional absent artifacts are omitted. Artifact values extend
`spec:mldb.v2.storage.artifact_reference` with declaration-required immutable metadata.
For `status: failed` or `cancelled`, `result` is null and `diagnostic` is non-null according to
`spec:mldb.v2.common.diagnostic`. A planned Evaluation coordinate that never receives a backend
attempt has no Evaluation Result; Study Result records that coordinate as `skipped` instead.

## Invariants

- `study_result`, `plan`, `trial`, `coordinate`, `stage`, Model, definitions, parameters, and
  `source_commit` exactly match the immutable Plan/runtime lineage.
- Completed metrics satisfy `spec:mldb.v2.evaluation.result_validation` and contain no undeclared
  formal values.
- Completed artifact references are integrity-verified before persistence.
- Failed/cancelled records contain no success payload.
- Once persisted, the Evaluation Result is immutable.
- Extra backend scalars, plots, logs, worker identity, queue name, and live telemetry are excluded.
