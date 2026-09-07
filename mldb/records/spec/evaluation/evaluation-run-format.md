# Contract: Evaluation Run format

- **id**: `spec:mldb.evaluation.evaluation_run_format`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.evaluation`
- **contract_class**: `format`

## What this is

Defines `run.yaml` for one concrete Evaluation Run.

The record identifies the exact Model, Corpus, Evaluation Protocol, resolved parameters, execution facts, and accepted formal results for one evaluation attempt.

## Current contract

Evaluation Run metadata uses schema:

```text
mjtensu.mldb/evaluation-run/v1
```

Completed example:

```yaml
schema: mjtensu.mldb/evaluation-run/v1
id: ev-20260904-001
status: completed

model: mdl-20260903-001
corpus: gray35-final-holdout-v1
evaluation_protocol: tile-classifier-standard-eval-v1

study:
  run: sr-20260904-001
  trial: trial-0001
  stage: final-holdout

parameters:
  batch_size: 4096

execution:
  started_at: 2026-09-04T10:00:00+09:00
  finished_at: 2026-09-04T10:00:14+09:00

result:
  metrics:
    accuracy: 0.9976
  artifacts:
    predictions:
      path: artifacts/predictions.jsonl
      format: jsonl
      schema: mjtensu.mldb/eval-artifact/categorical-predictions/v1
      sha256: 0123456789abcdef...
      bytes: 123456
```

Every persisted Evaluation Run contains complete selected inputs and resolved parameters from its first `running` record. Launch preflight succeeds before Run allocation, so v1 has no partially resolved Evaluation Run shape.

Every Evaluation Run requires:

| field | contract |
|---|---|
| `schema` | Exactly `mjtensu.mldb/evaluation-run/v1`. |
| `id` | Event ID `ev-YYYYMMDD-NNN`. |
| `status` | `running`, `completed`, `completed_partial`, `failed`, or `cancelled`. |
| `model` | Exactly one Model ID. |
| `corpus` | Exactly one Corpus ID. |
| `evaluation_protocol` | Exactly one Evaluation Protocol ID. |
| `parameters` | Complete resolved public parameter mapping. |
| `execution.started_at` | Timestamp for the concrete execution attempt. |

Every terminal Run additionally requires `execution.finished_at`.

A `completed` or `completed_partial` Run additionally requires both `result.metrics` and `result.artifacts`. Either mapping may be empty when no accepted output of that class exists.

A `completed_partial` Run additionally requires at least one accepted formal metric or artifact and at least one incompleteness record through `unavailable_outputs` or `validation_issues` for a rejected optional formal artifact.

A `completed` Run has no unavailable declared metric and no returned optional formal artifact rejected from formal results.

Each accepted `result.artifacts.<key>` entry requires:

| field | contract |
|---|---|
| `path` | Run-relative path beneath `artifacts/`. |
| `format` | Format declared by the Evaluation Protocol. |
| `schema` | Artifact schema declared by the Evaluation Protocol. |
| `sha256` | SHA-256 of the accepted immutable artifact bytes. |
| `bytes` | Artifact byte size. |

`environment` is optional free-form execution metadata.

`failure` may record concise `type` and `message` values for failed Runs.

`unavailable_outputs` may record declared formal outputs that the Evaluation Protocol explicitly reported unavailable. Each entry contains `output`, `type`, and `message` according to `spec:mldb.evaluation.evaluate_interface`.

`validation_issues` may record non-fatal formal-output validation problems such as a returned optional artifact rejected by validation. Each issue should identify `output`, a short machine-readable `type`, and a concise `message`.

## Rules

- Evaluation Run does not repeat Task because Model, Corpus, and Evaluation Protocol establish the Task compatibility chain.
- `parameters` contains the complete resolved mapping, not caller overrides only.
- Evaluation Run has no universal seed field.
- A Study-created Run may contain `study.run`, `study.trial`, and `study.stage` lineage.
- When `study` is present, all three lineage fields are required together.
- Study lineage must match one planned evaluation coordinate in the referenced Study Run plan.
- `result.metrics` contains only accepted formal scalar metrics.
- `result.artifacts` contains only accepted formal structured artifacts imported into MLDB-owned `artifacts/` storage.
- Explicitly unavailable metrics are represented only through `unavailable_outputs`; no `null`, NaN, or placeholder metric value is persisted.
- `completed_partial` means evaluation execution succeeded but the formal result surface is incomplete; it is not an alias for `failed`.
- Files that remain only beneath `work/` are not formal result entries.
- Evaluation Run does not duplicate Model training lineage or canonical learned-weight metadata.
- A terminal Run is a historical execution record and receives no `-vN` revision.

## Validation rules

- Reject an unsupported `schema` value.
- Reject an ID outside the Evaluation Run event-ID grammar.
- Reject a status outside `running`, `completed`, `completed_partial`, `failed`, or `cancelled`.
- Reject missing Model, Corpus, Evaluation Protocol, parameters, or start time.
- Reject a terminal Run without `execution.finished_at`.
- Reject `completed` or `completed_partial` without both result mappings.
- Reject `completed_partial` without at least one accepted formal metric or artifact.
- Reject `completed_partial` without at least one incompleteness record in `unavailable_outputs` or `validation_issues` for a rejected optional artifact.
- Reject `completed` with an unavailable declared metric or a rejected returned optional formal artifact.
- Reject a `parameters` mapping containing a value outside the JSON-compatible public-parameter domain.
- Reject Study lineage with only a subset of `run`, `trial`, and `stage`.
- Reject a Study lineage coordinate that does not match the Run inputs and resolved parameters.
- Reject an accepted artifact entry missing path, format, schema, SHA-256, or byte size.
- Reject an accepted artifact path outside the Evaluation Run `artifacts/` directory.
- Reject accepted artifact metadata that disagrees with persisted bytes.
- Reject mutation of terminal Run facts under the lifecycle contract.

Metric presence, unavailable-output semantics, and structured-artifact content validation belong to `spec:mldb.evaluation.result_validation`.

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.evaluation` | Parent evaluation overview. |
| `spec:mldb.evaluation.evaluation_run_lifecycle` | Defines valid state transitions and terminal immutability. |
| `spec:mldb.evaluation.result_validation` | Defines which returned metrics and artifacts may enter the completed record. |
| `spec:mldb.runtime.public_parameters` | Defines the complete `parameters` mapping. |
