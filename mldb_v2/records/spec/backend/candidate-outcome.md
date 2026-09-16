# Contract: Backend observation and candidate outcome

- **id**: `spec:mldb.v2.backend.candidate_outcome`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.backend`
- **contract_class**: `value`

## Stage key

Every backend observation is bound to one exact logical stage key:

```yaml
study_result: rotated-fcos/run-abcd
plan: rotated-fcos/spatial-screen-v2-plan-0123456789abcdef
trial: trial-0001
kind: training
coordinate: null
source_commit: <full unabbreviated Git commit object id>
```

For Evaluation, `kind: evaluation` and `coordinate` is the exact `eval-NNNN`. Backend Task names are
never parsed to reconstruct this key.

## Active observation

An active observation has exactly:

```yaml
state: active
stage_key: <exact stage-key value>
backend: clearml
execution_ids: [<opaque backend ids in attempt order>]
```
Active telemetry beyond those fields is backend/UI data and is not consumed as canonical MLDB state.

## Terminal candidate

A terminal candidate has exactly:

```yaml
state: terminal
stage_key: <exact stage-key value>
attempts: []
status: completed
diagnostic: null
result: null
```

`attempts` is a non-empty ordered list of `spec:mldb.v2.results.attempt_summary`. `status` is exactly
`completed`, `failed`, or `cancelled`.

For completed training:

```yaml
result:
  weights:
    uri: s3://mldb-artifacts/...
    bytes: 123456
    sha256: <64 lowercase hex>
    format: pytorch-state-dict/v1
```
For completed Evaluation:

```yaml
result:
  metrics:
    f1: 0.967
  artifacts:
    predictions:
      uri: s3://mldb-artifacts/...
      bytes: 98765
      sha256: <64 lowercase hex>
      format: jsonl
      schema: mjtensu.example/predictions/v1
```

For `failed` or `cancelled`, `result` is null and `diagnostic` is non-null according to
`spec:mldb.v2.common.diagnostic`. Completed candidates require `diagnostic: null`.

## Rules

- Candidate values are backend-neutral data; generic MLDB never consumes ClearML SDK objects.
- Terminal `stage_key` MUST exactly echo the admitted StageInput key/source identity.
- Collection rejects ownership/stage/source mismatch before result acceptance.
- Candidate `completed` is provisional until `spec:mldb.v2.verification.result_acceptance` succeeds.
- Acceptance may convert an invalid completed candidate into a canonical failed child Result but
  never repairs malformed output into success.
- Extra backend scalars, plots, logs, undeclared artifacts, resource telemetry, and queue state are
  not part of this value.
