# Contract: Train Protocol format

- **id**: `spec:mldb.training.train_protocol_format`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.training`
- **contract_class**: `format`

## What this is

Defines the YAML format and lifecycle contract for one reusable executable Train Protocol.

A Train Protocol defines how compatible Architectures for one Task are trained. It does not bind permanently to one Corpus or one Architecture.

## Current contract

Train Protocol metadata uses schema identifier:

```text
mjtensu.mldb/train-protocol/v1
```

Canonical placement is defined by `spec:mldb.repository.layout` as sibling `<id>.yaml` and `<id>.py` files under `mldb_data/train_protocols/`.

Example:

```yaml
schema: mjtensu.mldb/train-protocol/v1
id: tile-classifier-adamw-cosine-v1
status: sealed
task: tile-shape-classification-35-v1
name: Standard tile classifier training
description: Train compatible tile classifiers with the protocol-owned procedure.
implementation:
  entrypoint: train
  sha256: 0123456789abcdef...
parameters:
  epochs:
    default: 150
  learning_rate:
    default: 0.001
notes:
  optimizer: AdamW
  scheduler: cosine
```

## Rules

### Required fields

| field | contract |
|---|---|
| `schema` | Must equal `mjtensu.mldb/train-protocol/v1`. |
| `id` | Immutable Train Protocol ID and sibling-file basename. |
| `status` | `draft` or `sealed`. |
| `task` | Exactly one Task ID. |
| `name` | Non-empty human-readable name. |
| `description` | Human-readable protocol summary. |
| `implementation.entrypoint` | `train` for v1. |
| `parameters` | Mapping of caller-visible Run-varying parameters. Empty mapping is valid. |

`implementation.sha256` is required for `sealed` and may be omitted for `draft`.

`notes` and other human-readable supplemental metadata are optional.

### ID and lifecycle

- Train Protocol IDs must end with `-v<positive-integer>`.
- `draft` assets may be edited while being developed and verified.
- `sealed` assets are immutable as executable training definitions.
- A sealed asset must never return to `draft`.
- A result-affecting implementation change after sealing requires a new Train Protocol revision.
- Adding or removing a public parameter after sealing requires a new revision.
- Changing a published parameter default after sealing requires a new revision.
- Editorial changes that do not alter executable or parameter meaning do not require a revision.

### Public parameters

Each `parameters.<key>` value must be a mapping containing `default`.

Each `default` must be a JSON-compatible public parameter value as defined by `spec:mldb.runtime.public_parameters`.

```yaml
parameters:
  batch_size:
    default: 1024
    description: Training batch size.
```

Only declared keys may vary between Training Runs. Common default and caller-value resolution is owned by `spec:mldb.runtime.public_parameters`.

Additional fields such as `description`, `type`, `minimum`, `maximum`, or `suggested` may be present. MLDB v1 does not assign generic validation semantics to those advisory fields.

### Implementation integrity

For `status: sealed`, `implementation.sha256` is the SHA-256 of the sibling `<train-protocol-id>.py` file.

Result-affecting project-owned training logic must reside in that sibling file. The sealed file must not delegate result-affecting project-owned behavior to mutable project-local implementation modules.

Framework and third-party dependencies may remain external. MLDB infrastructure imports are allowed when they do not define training behavior.

The Python file is authoritative for executable behavior not exposed through public parameters. The YAML is not a universal training configuration language.

## Validation rules

| condition | result |
|---|---|
| Unsupported `schema` | Invalid Train Protocol. |
| YAML `id` differs from canonical basename | Invalid Train Protocol. |
| ID lacks terminal positive-integer `-vN` | Invalid Train Protocol. |
| Unknown `status` | Invalid Train Protocol. |
| Missing required field | Invalid Train Protocol. |
| `parameters` is not a mapping | Invalid Train Protocol. |
| Public parameter entry lacks `default` | Invalid Train Protocol. |
| Public parameter `default` is outside the JSON-compatible value domain | Invalid Train Protocol. |
| `sealed` protocol lacks `implementation.sha256` | Invalid sealed Train Protocol. |
| Recorded implementation hash differs from sibling Python bytes | Invalid sealed Train Protocol. |
| Referenced Task cannot resolve | Invalid Train Protocol. |

Whether a resolved Train Protocol is compatible with one selected Corpus and Architecture belongs to the concrete training-execution contract.

## Boundary

| concern | owner |
|---|---|
| Canonical file placement | `spec:mldb.repository.layout`. |
| Typed Task and Train Protocol resolution | `spec:mldb.runtime.asset_resolution`. |
| Public parameter resolution | `spec:mldb.runtime.public_parameters`. |
| `TrainContext` and `train()` behavior | `spec:mldb.training.train_interface`. |
| Training Run execution facts | `spec:mldb.training.training_run_format`. |
| Asset-specific pytest sealing requirement | `spec:mldb.verification`. |
| Concrete optimizer, loss, augmentation, scheduler, and checkpoint logic | Train Protocol Python implementation. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.training` | Parent training overview. |
| `spec:mldb.runtime.public_parameters` | Shared parameter declaration semantics. |
| `spec:mldb.runtime.asset_resolution` | Resolves Train Protocol and referenced Task metadata. |
