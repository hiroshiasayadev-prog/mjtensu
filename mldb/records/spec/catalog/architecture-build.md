# Contract: Architecture build interface

- **id**: `spec:mldb.catalog.architecture_build`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.catalog`
- **contract_class**: `interface`

## What this is

Defines the executable PyTorch construction boundary for one resolved MLDB Architecture.

The interface standardizes Architecture construction without standardizing internal module classes or one universal `forward()` return structure.

## Request

The executable loader invokes the Architecture entrypoint declared by Architecture v1:

```python
def build() -> torch.nn.Module:
    ...
```

The request has no arguments.

| request rule | contract |
|---|---|
| entrypoint | `build` |
| positional arguments | None. |
| keyword arguments | None. |
| caller-supplied topology parameters | Prohibited. One Architecture ID resolves to one concrete structure. |
| learned-weight input | None. Architecture construction does not select learned state. |

Architecture-specific structural constants are resolved inside the sibling Architecture implementation.

The executable loader must start from an already-resolved Architecture asset. A caller must not bypass typed asset resolution with a second filename or import convention.

## Response

A successful call returns one `torch.nn.Module` representing the unweighted model structure identified by the Architecture.

| response rule | contract |
|---|---|
| type | `torch.nn.Module`. |
| topology identity | Must represent the selected Architecture ID. |
| declared interface | Must be compatible with the coarse input/output contract in Architecture YAML. |
| learned state | Must not silently load a learned or pretrained checkpoint as Architecture identity. |
| internal output shape | Model-family-specific except where the Architecture metadata declares compatibility facts. |

Framework or model implementation may leave parameters in its ordinary initial state. Train Protocol owns any tracked initialization or pretrained-weight policy used during training.

A sealed Architecture implementation must contain its result-affecting project-owned topology and forward behavior in the sibling Architecture Python file.

The implementation may import third-party/framework code and MLDB infrastructure that does not define project-owned model behavior.

## Errors

| condition | interface result |
|---|---|
| Architecture metadata cannot resolve or fails required integrity validation | Do not invoke `build()`. |
| Sibling Python implementation is missing | Executable loading fails. |
| Python implementation cannot be imported | Executable loading fails. |
| Declared `build` entrypoint is missing or not callable | Architecture interface validation fails. |
| `build` requires caller arguments | Architecture interface validation fails. |
| `build()` raises | Construction fails for that caller operation. |
| Return value is not `torch.nn.Module` | Architecture interface validation fails. |

Concrete Python exception classes are implementation-owned.

## Rules

- Generic MLDB runtime must not branch on model family to construct an Architecture.
- Generic runtime must not rewrite the returned module graph.
- Architecture construction must not perform training.
- Architecture construction must not own loss computation or optimizer construction.
- Architecture construction must not load the canonical learned weights of a Model.
- Train Protocol and Model loading must use the same Architecture `build()` boundary when they need a fresh module.
- Architecture-specific pytest may exercise representative forward behavior, but pytest is not an ordinary runtime dependency.

## Boundary

| concern | owner |
|---|---|
| Architecture YAML identity and coarse interface | `spec:mldb.catalog.architecture_format`. |
| Typed Architecture resolution and integrity | `spec:mldb.runtime.asset_resolution`. |
| Python loading responsibility | MLDB runtime executable loader. |
| Training initialization and pretrained-weight policy | Train Protocol. |
| Canonical learned-state loading | Model/training result contracts. |
| Architecture asset pytest | Verification topic. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.catalog` | Parent catalog Index. |
| `spec:mldb.catalog.architecture_format` | Declares the `build` entrypoint and coarse module interface. |
| `spec:mldb.runtime.asset_resolution` | Resolves Architecture metadata before executable loading. |
| `spec:mldb.runtime.component_model` | Separates asset resolution from executable loading. |
