# Contract: Backend stage execution input

- **id**: `spec:mldb.v2.backend.stage_input`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.backend`
- **contract_class**: `value`

## Purpose and common shape

A backend admission carries one immutable backend-neutral value with schema
`mjtensu.mldb-v2/stage-input/v1`. It combines Plan-fixed intent with runtime Model lineage that may
become available only after training acceptance.

```yaml
schema: mjtensu.mldb-v2/stage-input/v1
study_result: rotated-fcos/run-abcd
plan: rotated-fcos/spatial-screen-v2-plan-0123456789abcdef
plan_sha256: <Plan content_sha256>
trial: trial-0001
kind: training
coordinate: null
source_commit: <full unabbreviated Git commit object id>
pins: []
stage: {}
runtime_model: null
```

`pins` is the exact Plan pin list, unchanged. Backend-specific queue/resource/config fields are not
part of this value.
## Training variant

For `kind: training`, `coordinate` and `runtime_model` are null. `stage` is exactly the materialized
Plan training source:

```yaml
stage:
  task: rotated-fcos/mahjong-obb-v1
  corpus: rotated-fcos/train-v3
  architecture: rotated-fcos/stem-s1-v1
  train_protocol: rotated-fcos/standard-v2
  parameters: {batch_size: 32, learning_rate: 0.001}
  seed: 42
```

No learned Model is supplied to training.

## Evaluation variant

For `kind: evaluation`, `coordinate` is the exact `eval-NNNN` and `stage` is exactly the materialized
Plan Evaluation coordinate:

```yaml
stage:
  name: final-holdout
  task: rotated-fcos/mahjong-obb-v1
  corpus: rotated-fcos/human-obb-v3
  evaluation_protocol: rotated-fcos/human-obb-v2
  parameters: {iou_threshold: 0.5}
```
`runtime_model` is then non-null and contains the immutable accepted lineage needed to load weights:

```yaml
runtime_model:
  model: rotated-fcos/run-abcd-trial-0001-model
  training_result: rotated-fcos/run-abcd-trial-0001-train
  task: rotated-fcos/mahjong-obb-v1
  architecture: rotated-fcos/stem-s1-v1
  weights:
    uri: s3://mldb-artifacts/...
    bytes: 123456
    sha256: <64 lowercase hex>
    format: pytorch-state-dict/v1
```

This shape is identical for an existing Model and a Model produced earlier in the same Study.
Runtime-produced Model/Training Result records need not exist in `source_commit`; the driver derives
this snapshot only from canonically accepted immutable records.

## Rules

- Construction is allowed only for a coordinate ready under `spec:mldb.v2.study.execution_readiness`.
- The harness verifies Plan digest, pins, IDs, ArtifactRefs, implementation hashes, and manifest
  digests before domain invocation.
- Controller-local paths, credentials, and ClearML SDK objects are forbidden.
- Backend adapters may encode this value into backend configuration but MUST preserve it exactly.
- Stage input is operational transport data, not a canonical entity under `mldb_data/`.
