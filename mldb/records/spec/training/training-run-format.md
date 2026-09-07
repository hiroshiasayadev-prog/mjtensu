# Contract: Training Run format

- **id**: `spec:mldb.training.training_run_format`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.training`
- **contract_class**: `format`

## What this is

Defines the YAML format for one concrete MLDB Training Run.

A Training Run records one execution attempt. Re-executing the same inputs creates another Run rather than revising an existing terminal Run.

## Current contract

Training Run metadata uses schema identifier:

```text
mjtensu.mldb/training-run/v1
```

Canonical placement is:

```text
mldb_data/training_runs/<training-run-id>/run.yaml
```

Example completed record:

```yaml
schema: mjtensu.mldb/training-run/v1
id: tr-20260904-001
status: completed
corpus: gray35-train-v1
architecture: plain-cnn-v1
train_protocol: tile-classifier-standard-v1
parameters:
  epochs: 150
  learning_rate: 0.001
execution:
  seed: 42
  started_at: 2026-09-04T01:00:00+09:00
  finished_at: 2026-09-04T01:20:00+09:00
result:
  weights:
    format: pytorch-state-dict
    path: artifacts/weights.pt
    sha256: 0123456789abcdef...
    bytes: 1234567
```

## Rules

### Required fields

Every persisted Training Run must contain the complete selected inputs and resolved parameters from its first `running` record. Launch preflight succeeds before Run allocation, so no partially resolved Training Run shape exists in v1.

Every Training Run must contain:

| field | contract |
|---|---|
| `schema` | `mjtensu.mldb/training-run/v1`. |
| `id` | Unique Training Run event ID. |
| `status` | Lifecycle state defined by the Training Run lifecycle spec. |
| `corpus` | Exactly one Corpus ID. |
| `architecture` | Exactly one Architecture ID. |
| `train_protocol` | Exactly one Train Protocol ID. |
| `parameters` | Complete resolved public-parameter mapping. |
| `execution.seed` | Concrete integer training seed. Boolean is invalid. |
| `execution.started_at` | Timestamp for execution start. |

A terminal Run additionally requires `execution.finished_at`.

A `completed` Run additionally requires:

- `result.weights.format`;
- `result.weights.path`;
- `result.weights.sha256`;
- `result.weights.bytes`.

For v1, `result.weights.format` is `pytorch-state-dict` and the canonical path is `artifacts/weights.pt`. The format identifier means the direct CPU tensor-state `torch.save` representation defined by `spec:mldb.training.canonical_weights`.

### Run ID

Training Run v1 IDs use:

```text
tr-YYYYMMDD-NNN
```

The date is the local date on which the Run is allocated. `NNN` is a zero-padded per-date sequence beginning at `001`.

Training Run IDs are event IDs and do not use `-vN` revision syntax.

### Input records

`corpus`, `architecture`, and `train_protocol` record the exact selected reusable assets.

Task is not repeated in Training Run YAML. Before execution, the runtime verifies that all three selected assets reference the same Task.

`parameters` stores the complete mapping produced by `spec:mldb.runtime.public_parameters`, not only caller overrides.

`execution.seed` is a first-class integer execution input. Boolean is invalid, and MLDB v1 defines no universal numeric range beyond the integer type contract.

### Optional fields

`environment` is optional best-effort execution metadata.

`work` is an optional free-form inventory of notable files beneath the Run `work/` directory. Files need not be listed merely to exist as working files.

A failed Run should record concise `failure.type` and `failure.message` when safe and available.

### Study lineage

A Training Run created from a Study Run may contain:

```yaml
study:
  run: sr-20260904-001
  trial: trial-0001
```

When `study` is present, both `run` and `trial` are required.
The selected Corpus, Architecture, Train Protocol, seed, and resolved parameters must match the referenced planned trial.

## Validation rules

| condition | result |
|---|---|
| Unsupported `schema` | Invalid Training Run record. |
| Directory name and `id` disagree | Invalid Training Run record. |
| ID does not match `tr-YYYYMMDD-NNN` | Invalid v1 Training Run record. |
| Missing required base field | Invalid Training Run record. |
| Terminal Run lacks `execution.finished_at` | Invalid terminal Run. |
| `completed` Run lacks canonical weight metadata | Invalid completed Run. |
| Completed weight path differs from `artifacts/weights.pt` | Invalid v1 completed Run. |
| `parameters` does not equal the complete resolved Train Protocol mapping | Invalid Run input record. |
| `parameters` contains a value outside the JSON-compatible public-parameter domain | Invalid Run input record. |
| `execution.seed` is not an integer or is boolean | Invalid Run input record. |
| Study lineage is partial or disagrees with the referenced plan | Invalid Study-linked Run. |

Artifact byte integrity is validated by the canonical-weight contract rather than by parsing YAML alone.

## Boundary

| concern | owner |
|---|---|
| Run directory placement | `spec:mldb.repository.layout`. |
| Lifecycle transitions and terminal immutability | `spec:mldb.training.training_run_lifecycle`. |
| Public parameter resolution | `spec:mldb.runtime.public_parameters`. |
| `train(context)` invocation | `spec:mldb.training.train_interface`. |
| `weights.pt` contents and state compatibility | `spec:mldb.training.canonical_weights`. |
| Automatic Model format and identity | `spec:mldb.model`. |
| Exact environment vocabulary | Not standardized by MLDB v1. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.training` | Parent training overview. |
| `spec:mldb.runtime.public_parameters` | Defines the complete `parameters` mapping. |
| `spec:mldb.repository.layout` | Defines the Training Run directory shape. |
