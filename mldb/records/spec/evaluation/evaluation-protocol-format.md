# Contract: Evaluation Protocol format

- **id**: `spec:mldb.evaluation.evaluation_protocol_format`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.evaluation`
- **contract_class**: `format`

## What this is

Defines the YAML contract for one versioned executable Evaluation Protocol.

The record declares reusable post-training evaluation identity, Task binding, public parameters, executable integrity, scalar metric keys, and formal structured artifact outputs.

## Current contract

Evaluation Protocol metadata uses schema:

```text
mjtensu.mldb/evaluation-protocol/v1
```

Example:

```yaml
schema: mjtensu.mldb/evaluation-protocol/v1
id: tile-classifier-standard-eval-v1
status: sealed

task: tile-shape-classification-35-v1
name: Standard tile classifier evaluation
description: Evaluate a learned tile classifier on a compatible Corpus.

implementation:
  entrypoint: evaluate
  sha256: 0123456789abcdef...

parameters:
  batch_size:
    default: 4096

outputs:
  metrics:
    accuracy:
      type: number
      description: Overall categorical accuracy.
  artifacts:
    predictions:
      format: jsonl
      schema: mjtensu.mldb/eval-artifact/categorical-predictions/v1
      required: true
```

Required top-level fields are:

| field | contract |
|---|---|
| `schema` | Exactly `mjtensu.mldb/evaluation-protocol/v1`. |
| `id` | Immutable Evaluation Protocol ID ending in `-v<positive-integer>`. |
| `status` | `draft` or `sealed`. |
| `task` | Exactly one Task ID. |
| `name` | Human-readable name. |
| `description` | Human-readable evaluation summary. |
| `implementation.entrypoint` | `evaluate` in v1. |
| `parameters` | Mapping of public Run-varying parameters; may be empty. |
| `outputs.metrics` | Mapping of declared scalar metric keys; may be empty. |
| `outputs.artifacts` | Mapping of declared formal structured artifact keys; may be empty. |

`implementation.sha256` is required when `status: sealed` and may be omitted while `status: draft`.

Each public parameter entry must contain `default`. Each `default` must be a JSON-compatible public parameter value as defined by `spec:mldb.runtime.public_parameters`. Additional advisory fields may be present.

Each metric declaration must use `type: number`. A human-readable `description` may be included.

Each formal artifact declaration requires:

| field | contract |
|---|---|
| `format` | `jsonl`, `csv`, or `json` in v1. |
| `schema` | Versioned formal artifact schema identifier. |
| `required` | Boolean indicating whether omission or invalidity prevents successful completion. |

## Rules

- Evaluation Protocol defines post-training evaluation of an existing Model.
- One Evaluation Protocol references exactly one Task.
- Model and Corpus are supplied by Evaluation Run rather than permanently bound to the protocol.
- `parameters` publishes the complete caller-varying interface. Fixed behavior remains in the sibling Python implementation.
- Public parameter resolution follows `spec:mldb.runtime.public_parameters`.
- Scalar metric names are protocol-local and do not establish a universal cross-protocol metric ontology.
- Every declared scalar metric is part of the formal result surface. For one Run, the protocol must either return a valid value or explicitly report that metric unavailable through the evaluation interface.
- Structured information must be represented through formal artifacts rather than nested inside scalar metrics.
- A `draft` protocol may be edited while under development.
- A `sealed` protocol must not return to `draft`.
- Any result-affecting executable change after sealing requires a new Evaluation Protocol revision.
- Adding, removing, or changing metric meaning after sealing requires a new revision.
- Adding, removing, or changing public parameter keys or defaults after sealing requires a new revision.
- Adding, removing, changing requiredness, changing format, or changing schema of a formal artifact after sealing requires a new revision.
- A sealed protocol's result-affecting project-owned evaluation logic must remain self-contained in its sibling Python implementation.
- Third-party libraries and non-result-affecting MLDB infrastructure may remain external dependencies.
- Training-time validation used solely for checkpoint selection must not be represented as this protocol's post-training result contract.

## Validation rules

- Reject an unsupported `schema` value.
- Reject an ID that does not match the canonical filename identity or terminal `-v<positive-integer>` grammar.
- Reject a status other than `draft` or `sealed`.
- Reject a missing or unresolved Task reference.
- Reject a sealed protocol with missing or mismatched `implementation.sha256`.
- Reject `implementation.entrypoint` other than `evaluate` in v1.
- Reject a `parameters` value that is not a mapping.
- Reject any public parameter declaration without `default`.
- Reject a public parameter `default` outside the JSON-compatible value domain.
- Reject `outputs.metrics` or `outputs.artifacts` when they are not mappings.
- Reject a metric declaration whose `type` is not `number`.
- Reject an artifact declaration missing `format`, `schema`, or `required`.
- Reject an artifact `format` outside `jsonl`, `csv`, or `json`.
- Reject a non-boolean artifact `required` value.

The executable entrypoint behavior is validated by the evaluation interface contract rather than by YAML shape alone.

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.evaluation` | Parent evaluation overview. |
| `spec:mldb.evaluation.evaluate_interface` | Defines the declared `evaluate` entrypoint. |
| `spec:mldb.evaluation.result_validation` | Consumes the declared metric and artifact output contract. |
| `spec:mldb.runtime.public_parameters` | Resolves the protocol's public parameter mapping. |
| `spec:mldb.repository.layout` | Defines sibling YAML/Python placement. |
