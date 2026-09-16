# Contract: Training Result format

- **id**: `spec:mldb.v2.training.training_result_format`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.training`
- **contract_class**: `format`

## Shape

Schema is `mjtensu.mldb-v2/training-result/v1`. One attempted logical training stage has exactly one
terminal Training Result with deterministic ID from `spec:mldb.v2.results.study_result_format`.

```yaml
schema: mjtensu.mldb-v2/training-result/v1
id: rotated-fcos/run-abcd-trial-0001-train
study_result: rotated-fcos/run-abcd
plan: rotated-fcos/spatial-screen-v2-plan-0123456789abcdef
trial: trial-0001
task: rotated-fcos/mahjong-obb-v1
architecture: rotated-fcos/stem-s1-v1
corpus: rotated-fcos/train-v3
train_protocol: rotated-fcos/standard-v2
parameters: {batch_size: 32, learning_rate: 0.001}
seed: 42
source_commit: <full unabbreviated Git commit object id>
attempts: []
status: completed
diagnostic: null
result: null
```
`attempts` is a non-empty ordered list of `spec:mldb.v2.results.attempt_summary` because a Training
Result exists only for an actually attempted stage. `parameters` is the exact complete resolved
mapping copied from the Plan trial.

## Status-dependent payload

For `status: completed`:

```yaml
diagnostic: null
result:
  weights:
    uri: s3://mldb-artifacts/...
    bytes: 123456
    sha256: <64 lowercase hex>
    format: pytorch-state-dict/v1
  model: rotated-fcos/run-abcd-trial-0001-model
```

The weight value follows `spec:mldb.v2.storage.artifact_reference` and
`spec:mldb.v2.training.canonical_weights`. `model` is the deterministic Model created by the same
acceptance operation.

For `status: failed` or `cancelled`, `result` is null and `diagnostic` is non-null according to
`spec:mldb.v2.common.diagnostic`. A backend-completed candidate rejected by MLDB becomes `failed`
with an acceptance diagnostic; it is never persisted as completed.
## Invariants

- `study_result`, `plan`, `trial`, definitions, parameters, seed, and `source_commit` exactly match
  the immutable Plan/Study Result lineage.
- `seed` is an integer and boolean is invalid.
- Completed weights are integrity-verified before persistence.
- Failed/cancelled records contain no success payload.
- Once persisted, the Training Result is immutable.
- Backend logs, checkpoints, worker identity, queue name, and live telemetry are not canonical fields.
