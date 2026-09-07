# Contract: Architecture format

- **id**: `spec:mldb.catalog.architecture_format`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.catalog`
- **contract_class**: `format`

## What this is

Defines the YAML contract for one versioned PyTorch Architecture asset.

Architecture owns one unweighted model structure for one Task. Training behavior, learned weights, and evaluation behavior remain outside this contract.

## Current contract

Architecture YAML uses schema identifier:

```text
mjtensu.mldb/architecture/v1
```

The required fields are:

| field | contract |
|---|---|
| `schema` | Exact Architecture schema identifier. |
| `id` | Versioned Architecture identity and sibling basename. |
| `status` | `draft` or `sealed`. |
| `task` | Referenced Task ID. |
| `name` | Human-readable Architecture name. |
| `family` | Non-empty searchable family identifier. |
| `description` | Human-readable Architecture description. |
| `implementation.framework` | `pytorch` in Architecture v1. |
| `implementation.entrypoint` | `build` in Architecture v1. |
| `interface.input` | Coarse model-input contract. |
| `interface.output` | Coarse model-output contract. |
| `structure.summary` | Human-readable structure summary. |

Conditional and optional fields are:

| field | contract |
|---|---|
| `implementation.sha256` | Required when `status: sealed`; SHA-256 of sibling `.py`. |
| `structure.traits` | Optional searchable descriptive list. |
| `parameters` | Optional Architecture-specific summary metadata. Generic MLDB does not interpret it. |

## Rules

- Architecture ID must match `<architecture-base-id>-v<positive-integer>`.
- Architecture ID must match the `.yaml` and `.py` sibling basename.
- `draft` Architecture may be edited before sealing.
- `sealed` Architecture is immutable as an executable model definition.
- A sealed Architecture must never return to `draft`.
- Any executable behavior change after sealing requires a new Architecture revision.
- `implementation.sha256` for a sealed Architecture identifies the exact sibling Python bytes.
- Architecture v1 uses `implementation.framework: pytorch` and `implementation.entrypoint: build`.
- Each Architecture references exactly one Task.
- Architecture must not duplicate the Task label vocabulary as its semantic authority.
- `interface` is a coarse compatibility contract, not a universal static tensor type system.
- Image inputs should declare applicable representation facts such as layout, channels, and fixed spatial dimensions when required by the Architecture.
- `interface.output.kind` is a non-empty semantic output identifier.
- `structure.summary` describes broad construction without reproducing the full network graph.
- `parameters` may summarize Architecture-specific facts but does not define caller-supplied constructor arguments.
- Architecture metadata must not own loss, optimizer, scheduler, augmentation, epochs, batch size, seed, checkpoint selection, pretrained-weight selection, or training policy.

## Validation rules

| condition | result |
|---|---|
| `schema` is not `mjtensu.mldb/architecture/v1` | Invalid Architecture metadata. |
| Required field is absent | Invalid Architecture metadata. |
| ID does not end in positive-integer `-vN` | Invalid Architecture ID. |
| YAML ID and sibling basename disagree | Invalid Architecture asset. |
| `status` is not `draft` or `sealed` | Invalid Architecture metadata. |
| Referenced Task does not resolve | Invalid Architecture asset. |
| Framework is not `pytorch` in v1 | Unsupported Architecture v1 framework. |
| Entrypoint is not `build` in v1 | Invalid Architecture v1 interface declaration. |
| Sealed Architecture omits `implementation.sha256` | Invalid sealed Architecture. |
| Sealed implementation hash differs from sibling `.py` bytes | Invalid Architecture integrity. |
| Unknown Architecture-specific `parameters` content exists | Valid; generic MLDB does not interpret it. |

Callable loading and return-value validation belong to the Architecture build interface.

## Boundary

| concern | owner |
|---|---|
| Architecture sibling placement | `spec:mldb.repository.layout`. |
| Task semantics | `spec:mldb.catalog.task_format`. |
| `build()` callable behavior | `spec:mldb.catalog.architecture_build`. |
| Executable asset loading | MLDB runtime executable-loader responsibility. |
| Training behavior | Training topic. |
| Learned state | Training Run and Model topics. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.catalog` | Parent catalog Index. |
| `spec:mldb.catalog.task_format` | Defines the referenced Task semantics. |
| `spec:mldb.repository.layout` | Defines Architecture sibling placement. |
| `spec:mldb.runtime.asset_resolution` | Resolves Architecture metadata without executing it. |
